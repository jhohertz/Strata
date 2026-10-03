// igpu-rework P0: does a kernel reading host memory (GTT) match a kernel reading a device
// buffer, and what does the H2D copy that the copy path pays cost, on this APU?
//
// Same kernel, same data, same size in every arm; the fill copy is measured separately and
// excluded from the read timings.  The checksum is an order-independent XOR of every 32-bit
// word, so the kernel's per-block partials and the CPU's sequential pass must match exactly.
//
//   A  anonymous host buffer (malloc + touched), read by the kernel over GTT
//   A' same, with MADV_HUGEPAGE (--thp) - large pages for the GPU's TLB
//   B  hipMalloc buffer, pre-filled by an H2D copy (the copy path's steady state)
//   C  CPU memcpy of the same size - the copy cost itself, for reference
//   F  --file <path>: the kernel reads a whole mmap'd file.  Run it once on a cold page cache
//      (fresh boot, nothing has touched the file): pass 1 faults every page through the KFD/GTT
//      path for the first time (cold, timed), the CPU reference pass then warms the cache, and
//      pass 2 is the warm kernel read (timed).  A hang on pass 1 with a clean pass 2 is the
//      P1 wedge signature (docs/IGPU.md): the GPU-side first touch of a big GTT-mapped file.
//
// Prints GB/s per arm (median and min/max over the iterations) and the checksum verdict.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK(call)                                                                                                  \
    do {                                                                                                             \
        const hipError_t error = (call);                                                                             \
        if (error != hipSuccess) {                                                                                   \
            std::fprintf(stderr, "igpu_gtt_micro: %s: %s (line %d)\n", #call, hipGetErrorString(error), __LINE__);  \
            return 2;                                                                                                \
        }                                                                    \
    } while (0)

namespace {

// Grid-stride read of `n4` float4s; per-block XOR partials (XOR is order-independent).
__global__ void xor_read_kernel(const float4* __restrict__ src, long long n4, uint32_t* partials) {
    const long long stride = (long long) gridDim.x * blockDim.x;
    uint32_t acc = 0u;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = src[i];
        acc ^= (uint32_t) v.x;
        acc ^= (uint32_t) v.y;
        acc ^= (uint32_t) v.z;
        acc ^= (uint32_t) v.w;
    }
    // block reduction
    __shared__ uint32_t warp[32];
    for (int off = 16; off > 0; off >>= 1) acc ^= __shfl_down_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        acc = (threadIdx.x < (blockDim.x >> 5)) ? warp[threadIdx.x] : 0u;
        for (int off = 16; off > 0; off >>= 1) acc ^= __shfl_down_sync(0xffffffffu, acc, off);
        if (threadIdx.x == 0) partials[blockIdx.x] = acc;
    }
}

template <typename T>
long long time_read(const T* src, long long n4, uint32_t* d_partials, int blocks) {
    const hipStream_t s = 0;
    CHECK(hipMemsetAsync(d_partials, 0, blocks * sizeof(uint32_t), s));
    auto t0 = std::chrono::steady_clock::now();
    xor_read_kernel<<<blocks, 256, 0, s>>>(src, n4, d_partials);
    CHECK(hipGetLastError());
    CHECK(hipStreamSynchronize(s));
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

void report(const char* arm, const uint8_t* cpu_src, long long bytes, const uint32_t* partials, int blocks,
            long long best, long long worst, long long median) {
    // CPU reference checksum (sequential; XOR is order-independent)
    uint32_t ref = 0u;
    const uint32_t* w = (const uint32_t*) cpu_src;
    const long long nwords = bytes / 4;
    for (long long i = 0; i < nwords; i += 8192) {
        uint32_t a = 0;
        for (long long k = i; k < i + 8192 && k < nwords; k++) a ^= w[k];
        ref ^= a;
    }
    uint32_t got = 0u;
    for (int b = 0; b < blocks; b++) got ^= partials[b];
    const double gb = (double) bytes / 1e9;
    std::printf("%-4s checksum %-3s  median %7.1f GB/s  min %7.1f  max %7.1f\n", arm,
                got == ref ? "OK" : "MISMATCH", gb / (median / 1000.0), gb / (worst / 1000.0), gb / (best / 1000.0));
}

}  // namespace

// P1 slice arm: one VMA over [off, off+len) of the file, kernel-read once, checksummed.  Maps a
// per-VMA size question (a single 31.6 GiB VMA faults on this APU; do smaller ones, at any file
// position, pass?).  Run several in one boot: a PASS costs ~seconds, a FAULT costs the boot,
// so order them to bracket the limit and stop at the first fault.
int run_slice_arm(const char* path, long long off, long long len) {
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) { std::perror("igpu_gtt_micro: open"); return 2; }
    off &= ~0xfffull;   // page-align the offset
    len &= ~0xfffull;
    uint8_t* map = (uint8_t*) mmap(nullptr, (size_t) len, PROT_READ, MAP_PRIVATE, fd, (off_t) off);
    if (map == MAP_FAILED) { std::perror("igpu_gtt_micro: mmap slice"); return 2; }
    std::printf("igpu_gtt_micro: slice [%.2f GiB, +%.2f GiB) map=%p (one VMA)\n", (double) off / 1073741824.0,
                (double) len / 1073741824.0, (void*) map);
    const int kBlocks = 512;
    uint32_t* h_partials = (uint32_t*) std::malloc(kBlocks * sizeof(uint32_t));
    uint32_t* d_partials = nullptr;
    CHECK(hipMalloc(&d_partials, kBlocks * sizeof(uint32_t)));
    const long long n4 = len / 16;
    CHECK(hipMemsetAsync(d_partials, 0, kBlocks * sizeof(uint32_t), 0));
    const auto t0 = std::chrono::steady_clock::now();
    xor_read_kernel<<<kBlocks, 256, 0, (hipStream_t) 0>>>((const float4*) map, n4, d_partials);
    CHECK(hipGetLastError());
    CHECK(hipStreamSynchronize(0));
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    CHECK(hipMemcpy(h_partials, d_partials, kBlocks * sizeof(uint32_t), hipMemcpyDeviceToHost));
    uint32_t got = 0u;
    for (int b = 0; b < kBlocks; ++b) got ^= h_partials[b];
    // CPU reference over the same slice (single thread is fine: the slice is small)
    uint32_t ref = 0u;
    const uint32_t* w = (const uint32_t*) map;
    for (long long i = 0; i < len / 4; i += 4096) {
        uint32_t a = 0;
        for (long long k = i; k < i + 4096; k++) a ^= w[k];
        ref ^= a;
    }
    std::printf("F-slice checksum %-3s  %7.1f GB/s  (%lld ms)\n", got == ref ? "OK" : "MISMATCH",
                (double) len / 1e9 / (ms / 1000.0), (long long) ms);
    CHECK(hipFree(d_partials));
    munmap(map, (size_t) len);
    return got == ref ? 0 : 3;
}

// P1 fixed arm: an anonymous MAP_FIXED_NOREPLACE VMA at a chosen VA.  Separates the VA-window
// question (a region at a known-good / known-bad VA) from size and physical placement: same size,
// different VA, same allocator-free pages.
int run_fixed_arm(unsigned long long va, long long gib) {
    const long long bytes = gib * (1ll << 30);
    uint8_t* p = (uint8_t*) mmap((void*) va, (size_t) bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS |
                                     MAP_FIXED_NOREPLACE,
                                 -1, 0);
    if (p == MAP_FAILED) { std::perror("igpu_gtt_micro: mmap fixed"); return 2; }
    if ((uintptr_t) p != va) { std::fprintf(stderr, "igpu_gtt_micro: fixed VA moved to %p\n", (void*) p); return 2; }
    std::memset(p, 0x5a, bytes);   // touch: real pages (every byte; the sparse 1-byte/page touch faults on this APU)
    std::printf("igpu_gtt_micro: fixed %lld GiB at %012llx\n", gib, va);
    const int kBlocks = 512;
    uint32_t* d_partials = nullptr;
    CHECK(hipMalloc(&d_partials, kBlocks * sizeof(uint32_t)));
    CHECK(hipMemsetAsync(d_partials, 0, kBlocks * sizeof(uint32_t), 0));
    const auto t0 = std::chrono::steady_clock::now();
    xor_read_kernel<<<kBlocks, 256, 0, (hipStream_t) 0>>>((const float4*) p, bytes / 16, d_partials);
    CHECK(hipGetLastError());
    CHECK(hipStreamSynchronize(0));
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::printf("F-fixed  %7.1f GB/s  (%lld ms for %lld GiB)\n", (double) bytes / 1e9 / (ms / 1000.0), (long long) ms,
                gib);
    CHECK(hipFree(d_partials));
    munmap(p, (size_t) bytes);
    return 0;
}

// P1 chunks arm: the complement's proposed shape - a PROT_NONE VA reservation with `n` committed
// chunks of `gib` GiB each, separated by 64 KiB PROT_NONE gaps (the gaps keep the committed chunks
// separate VMAs, which is what the KFD path appears to map).  The kernel reads every committed byte.
int run_chunks_arm(int n, long long gib) {
    const long long chunk = gib * (1ll << 30);
    const long long gap = 64ll << 10;
    const long long total_va = (long long) n * (chunk + gap);
    uint8_t* reserve = (uint8_t*) mmap(nullptr, (size_t) total_va, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (reserve == MAP_FAILED) { std::perror("igpu_gtt_micro: reserve"); return 2; }
    for (int c = 0; c < n; ++c) {
        uint8_t* at = (uint8_t*) mmap(reserve + (long long) c * (chunk + gap), (size_t) chunk, PROT_READ | PROT_WRITE,
                                      MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (at == MAP_FAILED) { std::perror("igpu_gtt_micro: commit chunk"); return 2; }
        std::memset(at, 0x5a, chunk);   // touch: real pages (every byte)
    }
    std::printf("igpu_gtt_micro: %d chunks x %lld GiB (+64 KiB gaps) in a %.2f GiB reservation at %p\n", n, gib,
                (double) total_va / 1073741824.0, (void*) reserve);
    const int kBlocks = 512;
    uint32_t* d_partials = nullptr;
    uint32_t* h_partials = (uint32_t*) std::malloc(kBlocks * sizeof(uint32_t));
    CHECK(hipMalloc(&d_partials, kBlocks * sizeof(uint32_t)));
    // read each chunk (the gaps are PROT_NONE and must never be touched)
    for (int c = 0; c < n; ++c) {
        const uint8_t* at = reserve + (long long) c * (chunk + gap);
        CHECK(hipMemsetAsync(d_partials, 0, kBlocks * sizeof(uint32_t), 0));
        const auto t0 = std::chrono::steady_clock::now();
        xor_read_kernel<<<kBlocks, 256, 0, (hipStream_t) 0>>>((const float4*) at, chunk / 16, d_partials);
        CHECK(hipGetLastError());
        CHECK(hipStreamSynchronize(0));
        const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
        CHECK(hipMemcpy(h_partials, d_partials, kBlocks * sizeof(uint32_t), hipMemcpyDeviceToHost));
        uint32_t got = 0;
        for (int b = 0; b < kBlocks; ++b) got ^= h_partials[b];
        std::printf("chunk %d %7.1f GB/s  (%lld ms)\n", c, (double) chunk / 1e9 / (ms / 1000.0), (long long) ms);
    }
    CHECK(hipFree(d_partials));
    munmap(reserve, (size_t) total_va);
    std::printf("igpu_gtt_micro: chunks arm done\n");
    return 0;
}

// P1 anon arm: a large anonymous region (the complement-fallback question: does the KFD path map a
// big anonymous VMA at all?).  malloc + touch, one kernel read, no checksum (contents are a pattern).
int run_anon_arm(long long gib) {
    const long long bytes = gib * (1ll << 30);
    uint8_t* host = (uint8_t*) std::malloc(bytes);
    if (!host) { std::fprintf(stderr, "igpu_gtt_micro: cannot allocate %lld GiB\n", gib); return 2; }
    std::memset(host, 0x5a, bytes);   // touch: real pages (identical to the 1 GiB arm that passes)
    std::printf("igpu_gtt_micro: anon %lld GiB map=%p\n", gib, (void*) host);
    const int kBlocks = 512;
    uint32_t* d_partials = nullptr;
    uint8_t* d_pre = nullptr;
    CHECK(hipMalloc(&d_partials, kBlocks * sizeof(uint32_t)));
    // Model under test (every datum so far): the 1 GiB arm that passes does a 1 GiB H2D hipMemcpy (the GPU's
    // DMA engine reads the host VMA) before the first kernel host read; a 64 MiB ping of the same VMA left the
    // kernel read faulting.  Registration appears per touched RANGE, so ping the WHOLE VMA: a 2 GiB device
    // buffer as target, walking the host buffer end to end (one DMA pass, ~1 s at the measured 35.8 GB/s).
    const long long ping_chunk = 2ll << 30;
    CHECK(hipMalloc(&d_pre, (size_t) ping_chunk));
    const auto pf0 = std::chrono::steady_clock::now();
    for (long long off = 0; off < bytes; off += ping_chunk)
        CHECK(hipMemcpy(d_pre, host + off, (size_t) std::min(ping_chunk, bytes - off), hipMemcpyHostToDevice));
    const long long ping_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - pf0).count();
    std::printf("A-anon   H2D ping of the whole VMA: %7.1f GB/s  (%lld ms)\n",
                (double) bytes / 1e9 / (ping_ms / 1000.0), (long long) ping_ms);
    CHECK(hipMemsetAsync(d_partials, 0, kBlocks * sizeof(uint32_t), 0));
    const auto t0 = std::chrono::steady_clock::now();
    xor_read_kernel<<<kBlocks, 256, 0, (hipStream_t) 0>>>((const float4*) host, bytes / 16, d_partials);
    CHECK(hipGetLastError());
    CHECK(hipStreamSynchronize(0));
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::printf("A-anon   %7.1f GB/s  (%lld ms for %lld GiB)\n", (double) bytes / 1e9 / (ms / 1000.0), (long long) ms,
                gib);
    CHECK(hipFree(d_partials));
    CHECK(hipFree(d_pre));
    return 0;
}

// P1 reg-after-dma arm: the engine registers its complement AFTER gigabytes of device DMA (weight
// loads); the micro's reg arm registered before any GPU activity and passed.  Does a register placed
// after a first DMA still work?  If this faults and --reg passes, the engine must register before its
// first GPU traffic (or use the H2D touch instead).
int run_reg_after_dma_arm(long long gib) {
    const long long bytes = gib * (1ll << 30);
    uint8_t* host = (uint8_t*) std::malloc(bytes);
    if (!host) { std::fprintf(stderr, "igpu_gtt_micro: cannot allocate %lld GiB\n", gib); return 2; }
    std::memset(host, 0x5a, bytes);
    uint8_t* d_scratch = nullptr;
    uint8_t* h_scratch = (uint8_t*) std::malloc(1ull << 20);
    CHECK(hipMalloc(&d_scratch, 1ull << 20));
    CHECK(hipMemcpy(d_scratch, h_scratch, 1ull << 20, hipMemcpyHostToDevice));   // the first DMA of the process
    const hipError_t reg = hipHostRegister(host, (size_t) bytes, hipHostRegisterDefault);
    std::printf("igpu_gtt_micro: reg-after-dma %lld GiB map=%p (1 MiB DMA first, then register) -> %s\n", gib, (void*) host,
                hipGetErrorString(reg));
    if (reg != hipSuccess) return 2;
    uint32_t* d_partials = nullptr;
    CHECK(hipMalloc(&d_partials, 512 * sizeof(uint32_t)));
    CHECK(hipMemsetAsync(d_partials, 0, 512 * sizeof(uint32_t), 0));
    xor_read_kernel<<<512, 256, 0, (hipStream_t) 0>>>((const float4*) host, bytes / 16, d_partials);
    CHECK(hipGetLastError());
    CHECK(hipStreamSynchronize(0));
    std::printf("R-after  kernel read after late register: OK\n");
    CHECK(hipFree(d_partials));
    CHECK(hipFree(d_scratch));
    (void) hipHostUnregister(host);
    return 0;
}

// P1 reg arm: does hipHostRegister alone (no DMA touch) make a big anonymous VMA kernel-readable?
// `flags` lets us A/B the exact registration the engine used (Mapped|Portable) against Default.
int run_reg_arm(long long gib, unsigned flags, const char* label) {
    const long long bytes = gib * (1ll << 30);
    uint8_t* host = (uint8_t*) std::malloc(bytes);
    if (!host) { std::fprintf(stderr, "igpu_gtt_micro: cannot allocate %lld GiB\n", gib); return 2; }
    std::memset(host, 0x5a, bytes);
    const hipError_t reg = hipHostRegister(host, (size_t) bytes, flags);
    std::printf("igpu_gtt_micro: %s %lld GiB map=%p hipHostRegister(0x%x) -> %s\n", label, gib, (void*) host, flags,
                hipGetErrorString(reg));
    if (reg != hipSuccess) return 2;
    const int kBlocks = 512;
    uint32_t* d_partials = nullptr;
    CHECK(hipMalloc(&d_partials, kBlocks * sizeof(uint32_t)));
    CHECK(hipMemsetAsync(d_partials, 0, kBlocks * sizeof(uint32_t), 0));
    const auto t0 = std::chrono::steady_clock::now();
    xor_read_kernel<<<kBlocks, 256, 0, (hipStream_t) 0>>>((const float4*) host, bytes / 16, d_partials);
    CHECK(hipGetLastError());
    CHECK(hipStreamSynchronize(0));
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::printf("R-reg    %7.1f GB/s  (%lld ms for %lld GiB, NO h2d ping)\n", (double) bytes / 1e9 / (ms / 1000.0),
                (long long) ms, gib);
    CHECK(hipFree(d_partials));
    (void) hipHostUnregister(host);
    return 0;
}

// P1 scatter arm: the engine's gather shape on a registered region - 24,576 CTAs, each reading its own
// 1.38 MiB blob at a scattered base (the complement's exact block layout), one kernel, the way the prompt
// path's gather_native walks the experts.  A hang/fault here means the KFD path cannot serve that pattern
// (not the pattern-free full reads that pass), and the alias design must bounce the prompt path.
__global__ void scatter_read_kernel(const uint4* __restrict__ region, long long region_words4, long long blob_words4,
                                    long long n_blobs, uint32_t* partials, int n_partials) {
    const long long b = blockIdx.x;
    if (b >= n_blobs) return;
    const long long base = b * blob_words4;
    uint32_t acc = 0u;
    for (long long i = threadIdx.x; i < blob_words4; i += blockDim.x) {
        const uint4 v = region[base + i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    __shared__ uint32_t warp[32];
    for (int off = 16; off > 0; off >>= 1) acc ^= __shfl_down_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        acc = (threadIdx.x < (blockDim.x >> 5)) ? warp[threadIdx.x] : 0u;
        for (int off = 16; off > 0; off >>= 1) acc ^= __shfl_down_sync(0xffffffffu, acc, off);
        if (threadIdx.x == 0) partials[b % (long long) n_partials] ^= acc;
    }
}

int run_scatter_arm(long long gib) {
    const long long bytes = gib * (1ll << 30);
    uint8_t* host = (uint8_t*) std::malloc(bytes);
    if (!host) { std::fprintf(stderr, "igpu_gtt_micro: cannot allocate %lld GiB\n", gib); return 2; }
    std::memset(host, 0x5a, bytes);
    if (hipHostRegister(host, (size_t) bytes, hipHostRegisterDefault) != hipSuccess) {
        std::fprintf(stderr, "igpu_gtt_micro: register: %s\n", hipGetErrorString(hipGetLastError()));
        return 2;
    }
    // the complement's exact block layout: 1,382,400 B blobs, as many as fit
    const long long blob = 1382400;
    const long long n_blobs = bytes / blob;
    const int n_partials = 1024;
    uint32_t* d_partials = nullptr;
    uint32_t* h_partials = (uint32_t*) std::malloc(n_partials * sizeof(uint32_t));
    CHECK(hipMalloc(&d_partials, n_partials * sizeof(uint32_t)));
    CHECK(hipMemsetAsync(d_partials, 0, n_partials * sizeof(uint32_t), 0));
    std::printf("igpu_gtt_micro: scatter %lld GiB registered, %lld blobs x 1,382,400 B, one kernel\n", gib, n_blobs);
    const auto t0 = std::chrono::steady_clock::now();
    scatter_read_kernel<<<(unsigned) n_blobs, 256, 0, (hipStream_t) 0>>>((const uint4*) host, bytes / 16, blob / 16,
                                                                         n_blobs, d_partials, n_partials);
    CHECK(hipGetLastError());
    CHECK(hipStreamSynchronize(0));
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    CHECK(hipMemcpy(h_partials, d_partials, n_partials * sizeof(uint32_t), hipMemcpyDeviceToHost));
    // every byte is 0x5a: each uint4 XORs to a fixed value; the answer is that value repeated (parity of counts)
    uint32_t expect = 0x5a5a5a5a ^ 0x5a5a5a5a ^ 0x5a5a5a5a ^ 0x5a5a5a5a;   // = 0
    std::printf("S-scatter  %7.1f GB/s  (%lld ms, partial-sum %08x)\n", (double) bytes / 1e9 / (ms / 1000.0),
                (long long) ms, h_partials[0] ^ h_partials[1] ^ expect);
    CHECK(hipFree(d_partials));
    (void) hipHostUnregister(host);
    std::printf("igpu_gtt_micro: scatter arm done\n");
    return 0;
}

// P1 file arm: kernel reads of a whole mmap'd file, cold then warm, checksummed against the CPU.
int run_file_arm(const char* path) {
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) { std::perror("igpu_gtt_micro: open"); return 2; }
    struct stat st = {};
    if (fstat(fd, &st) != 0) { std::perror("igpu_gtt_micro: fstat"); return 2; }
    const long long bytes = (long long) st.st_size & ~15LL;
    if (bytes <= 0) { std::fprintf(stderr, "igpu_gtt_micro: file too small\n"); return 2; }
    uint8_t* map = (uint8_t*) mmap(nullptr, (size_t) bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { std::perror("igpu_gtt_micro: mmap"); return 2; }
    std::printf("igpu_gtt_micro: file %s = %.2f GiB mapped read-only (run cold: fresh boot, nothing touched it)\n",
                path, (double) bytes / 1073741824.0);

    const int kBlocks = 1024;
    uint32_t* h_partials = (uint32_t*) std::malloc(kBlocks * sizeof(uint32_t));
    uint32_t* d_partials = nullptr;
    CHECK(hipMalloc(&d_partials, kBlocks * sizeof(uint32_t)));
    const long long n4 = bytes / 16;

    // CPU reference (8 threads over the page cache; also warms it for pass 2)
    uint32_t ref = 0u;
    {
        const long long nwords = bytes / 4, per = (nwords + 7) / 8;
        std::vector<uint32_t> partial(8, 0u);
        std::vector<std::thread> pool;
        for (int t = 0; t < 8; ++t) {
            pool.emplace_back([t, per, nwords, w = (const uint32_t*) map, &partial] {
                uint32_t a = 0u;
                for (long long i = (long long) t * per; i < (t == 7 ? nwords : (long long) (t + 1) * per); i++) a ^= w[i];
                partial[t] = a;
            });
        }
        for (auto& t : pool) t.join();
        for (uint32_t a : partial) ref ^= a;
    }
    auto pass = [&](const char* label) -> bool {
        CHECK(hipMemsetAsync(d_partials, 0, kBlocks * sizeof(uint32_t), 0));
        const auto t0 = std::chrono::steady_clock::now();
        xor_read_kernel<<<kBlocks, 256, 0, (hipStream_t) 0>>>((const float4*) map, n4, d_partials);
        CHECK(hipGetLastError());
        CHECK(hipStreamSynchronize(0));
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        CHECK(hipMemcpy(h_partials, d_partials, kBlocks * sizeof(uint32_t), hipMemcpyDeviceToHost));
        uint32_t got = 0u;
        for (int b = 0; b < kBlocks; ++b) got ^= h_partials[b];
        std::printf("F%s   checksum %-3s  %7.1f GB/s  (%lld ms for %.2f GiB)\n", label, got == ref ? "OK" : "MISMATCH",
                    (double) bytes / 1e9 / (ms / 1000.0), (long long) ms, (double) bytes / 1073741824.0);
        return got == ref;
    };
    const bool ok1 = pass("1-cold");
    const bool ok2 = pass("2-warm");
    CHECK(hipFree(d_partials));
    munmap(map, (size_t) bytes);
    std::printf("igpu_gtt_micro: file arm done (cold %s, warm %s)\n", ok1 ? "OK" : "BAD", ok2 ? "OK" : "BAD");
    return ok1 && ok2 ? 0 : 3;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--file") == 0 && i + 1 < argc) return run_file_arm(argv[i + 1]);
        if (std::strcmp(argv[i], "--slice") == 0 && i + 3 < argc)
            return run_slice_arm(argv[i + 1], (long long) std::atof(argv[i + 2]) * (1ll << 30),
                                 (long long) std::atof(argv[i + 3]) * (1ll << 30));
        if (std::strcmp(argv[i], "--anon") == 0 && i + 1 < argc)
            return run_anon_arm((long long) std::atof(argv[i + 1]));
        if (std::strcmp(argv[i], "--fixed") == 0 && i + 2 < argc)
            return run_fixed_arm((unsigned long long) strtoull(argv[i + 1], nullptr, 16), (long long) std::atof(argv[i + 2]));
        if (std::strcmp(argv[i], "--chunks") == 0 && i + 3 < argc)
            return run_chunks_arm(std::atoi(argv[i + 1]), (long long) std::atof(argv[i + 2]));
        if (std::strcmp(argv[i], "--reg") == 0 && i + 1 < argc)
            return run_reg_arm((long long) std::atof(argv[i + 1]), hipHostRegisterDefault, "reg-default");
        if (std::strcmp(argv[i], "--reg-mapped") == 0 && i + 1 < argc)
            return run_reg_arm((long long) std::atof(argv[i + 1]),
                               (unsigned) (hipHostRegisterMapped | hipHostRegisterPortable), "reg-mapped");
        if (std::strcmp(argv[i], "--reg-after-dma") == 0 && i + 1 < argc)
            return run_reg_after_dma_arm((long long) std::atof(argv[i + 1]));
        if (std::strcmp(argv[i], "--scatter") == 0 && i + 1 < argc)
            return run_scatter_arm((long long) std::atof(argv[i + 1]));
    }
    const bool thp = argc > 1 && std::strcmp(argv[1], "--thp") == 0;
    constexpr long long bytes = 1ll << 30;   // 1 GiB
    constexpr int kBlocks = 512;
    constexpr int kIters = 9;

    uint8_t* host = (uint8_t*) std::malloc(bytes);
    if (!host) { std::fprintf(stderr, "igpu_gtt_micro: cannot allocate 1 GiB host buffer\n"); return 2; }
    std::memset(host, 0x5a, bytes);           // touch: real pages, page cache / anonymous backing
    if (thp && madvise(host, bytes, MADV_HUGEPAGE) != 0)
        std::fprintf(stderr, "igpu_gtt_micro: note: MADV_HUGEPAGE refused (continuing)\n");

    uint32_t* h_partials = (uint32_t*) std::malloc(kBlocks * sizeof(uint32_t));
    uint32_t* d_partials = nullptr;
    uint8_t* d_buf = nullptr;
    CHECK(hipMalloc(&d_partials, kBlocks * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_buf, bytes));

    std::printf("igpu_gtt_micro: 1 GiB, %d blocks x 256 threads, %d timed iterations%s\n", kBlocks, kIters,
                thp ? ", MADV_HUGEPAGE" : "");

    // B: fill the device buffer once (the copy path's steady state reads this afterwards)
    CHECK(hipMemcpy(d_buf, host, bytes, hipMemcpyHostToDevice));
    auto cf0 = std::chrono::steady_clock::now();
    CHECK(hipMemcpy(d_buf, host, bytes, hipMemcpyHostToDevice));   // timed fill
    auto cf1 = std::chrono::steady_clock::now();
    const long long copy_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cf1 - cf0).count();
    std::printf("C    H2D fill copy: %7.1f GB/s  (%lld ms for 1 GiB)\n",
                (double) bytes / 1e9 / (copy_ms / 1000.0), copy_ms);

    const long long n4 = bytes / 16;
    for (const char* arm : {"A", "B"}) {
        const auto src = (arm[0] == 'A') ? (const float4*) host : (const float4*) d_buf;
        long long best = LLONG_MAX, worst = 0, median = 0;
        long long samples[kIters];
        for (int i = 0; i < kIters; i++) {
            samples[i] = time_read(src, n4, d_partials, kBlocks);
            best = std::min(best, samples[i]);
            worst = std::max(worst, samples[i]);
        }
        std::sort(samples, samples + kIters);
        median = samples[kIters / 2];
        CHECK(hipMemcpy(h_partials, d_partials, kBlocks * sizeof(uint32_t), hipMemcpyDeviceToHost));
        report(arm, host, bytes, h_partials, kBlocks, best, worst, median);
    }
    CHECK(hipFree(d_partials));
    CHECK(hipFree(d_buf));
    std::printf("igpu_gtt_micro: done\n");
    return 0;
}
