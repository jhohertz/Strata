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

#include "strata/kernels/iq_kernels.hpp"

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

// P1 alias arm: a kernel reading through the cudaHostGetDevicePointer ALIAS of a cudaHostAlloc region
// (the engine's grp_mapped path does exactly this: copy_i32 kernels read/write m.grp_dev, the device alias
// of a small pinned host buffer).  The --reg arm tested the host pointer; this tests the alias.
int run_alias_arm(long long gib) {
    const long long bytes = gib * (1ll << 30);
    uint8_t* host = (uint8_t*) std::malloc(bytes);   // plain memory, then pinned - like cudaHostAlloc's result
    if (!host) { std::fprintf(stderr, "igpu_gtt_micro: cannot allocate %lld GiB\n", gib); return 2; }
    std::memset(host, 0x5a, bytes);
    if (hipHostRegister(host, (size_t) bytes, hipHostRegisterDefault) != hipSuccess) {
        std::fprintf(stderr, "igpu_gtt_micro: register: %s\n", hipGetErrorString(hipGetLastError()));
        return 2;
    }
    void* alias = nullptr;
    if (hipHostGetDevicePointer(&alias, host, 0) != hipSuccess || alias == nullptr) {
        std::fprintf(stderr, "igpu_gtt_micro: no device alias: %s (host=%p)\n", hipGetErrorString(hipGetLastError()),
                     (void*) host);
        return 2;
    }
    std::printf("igpu_gtt_micro: alias %lld GiB host=%p device-alias=%p%s\n", gib, (void*) host, alias,
                alias == (void*) host ? " (identity)" : " (distinct mapping)");
    const int kBlocks = 512;
    uint32_t* d_partials = nullptr;
    CHECK(hipMalloc(&d_partials, kBlocks * sizeof(uint32_t)));
    CHECK(hipMemsetAsync(d_partials, 0, kBlocks * sizeof(uint32_t), 0));
    const auto t0 = std::chrono::steady_clock::now();
    xor_read_kernel<<<kBlocks, 256, 0, (hipStream_t) 0>>>((const float4*) alias, bytes / 16, d_partials);
    CHECK(hipGetLastError());
    CHECK(hipStreamSynchronize(0));
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::printf("A-alias   kernel read through the device alias: %7.1f GB/s  (%lld ms)\n",
                (double) bytes / 1e9 / (ms / 1000.0), (long long) ms);
    CHECK(hipFree(d_partials));
    (void) hipHostUnregister(host);
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

// igpu-rework P1 run 26/27: the engine's alias runs IMA in the expert section (dmesg: NULL-base
// reads, GPU VA 0x0-0x6000) in iq_dequant_gu_f16, while this arm's identical workload passes.
// Run 27: every complement read so far has been on the LEGACY stream (the micro, the engine's
// prefault DMA, the BLAS warmup) - the engine's dequant is the first kernel to read the registered
// complement on a NON-BLOCKING stream.  Optional args: "nb" runs the dequants on a non-blocking
// stream; a number holds that many GiB of device memory (the engine's ~8 GiB session scratch
// pressure on the iGPU's 16 GiB budget).
int run_dequant_arm(long long gib, bool nb, long long pressure_gib, bool thp, bool dev, bool pinned) {
    // igpu-rework P2.1: the alias profile says the dequant reads 7.05 GiB in 8.0 s (0.95 GB/s) while
    // a flat read of the same region runs at 54.7-82.6 GB/s.  This arm now TIMES the 512-expert
    // dequant and can (thp) map the region with 2 MiB pages before registration, and (dev) copy the
    // 512 blobs to a device buffer and dequant from there - the three numbers that say whether the
    // slowness is the GTT (thp closes the gap), the TLB/page size, or the kernel's access pattern
    // (dev matches GTT: the pattern is fine, the mapping is not; dev is much faster: the pattern is
    // fine too and the mapping is the problem).
    constexpr int kExperts = 512;
    constexpr int kType = 42;   // Q2_0, this pack's gu_type and d_type
    constexpr int64_t n_ff = 640, n_embd = 2560;
    constexpr size_t up_off = 460800, down_off = 921600, blob = 1382400;
    const long long need = (long long) kExperts * (long long) blob + (1ll << 30);
    if (gib * (1ll << 30) < need) { std::fprintf(stderr, "igpu_gtt_micro: --dequant needs %.2f GiB\n",
                                                 (double) need / 1073741824.0);
        return 2;
    }
    const long long bytes = gib * (1ll << 30);
    uint8_t* host = nullptr;
    bool host_pinned_alloc = false;
    if (pinned) {
        // the engine's complement: cudaHostAlloc(Mapped|Portable) - driver page-locked memory
        const hipError_t e = hipHostAlloc((void**) &host, (size_t) bytes, hipHostAllocMapped | hipHostAllocPortable);
        if (e != hipSuccess || host == nullptr) {
            std::fprintf(stderr, "igpu_gtt_micro: dequant arm: cudaHostAlloc %lld GiB: %s\n", gib, hipGetErrorString(e));
            return 2;
        }
        host_pinned_alloc = true;
        std::printf("igpu_gtt_micro: dequant arm: %lld GiB via cudaHostAlloc (the engine's mapping)\n", gib);
    } else {
        host = (uint8_t*) std::malloc(bytes);
        if (!host) { std::fprintf(stderr, "igpu_gtt_micro: cannot allocate %lld GiB\n", gib);
            return 2;
        }
    }
    for (long long off = 0; off < bytes; off += 4096)   // touch + a deterministic per-page pattern
        std::memset(host + off, (int) (((off >> 12) & 0xff) ^ 0x5a), 4096);
    if (thp) {
        if (madvise(host, (size_t) bytes, MADV_HUGEPAGE) != 0)
            std::fprintf(stderr, "igpu_gtt_micro: dequant arm: MADV_HUGEPAGE refused (continuing)\n");
        else
            std::printf("igpu_gtt_micro: dequant arm: region mapped with 2 MiB pages (THP)\n");
    }
    if (!host_pinned_alloc &&
        hipHostRegister(host, (size_t) bytes, hipHostRegisterDefault) != hipSuccess) {
        std::fprintf(stderr, "igpu_gtt_micro: register: %s\n", hipGetErrorString(hipGetLastError()));
        return 2;
    }
    void* alias = nullptr;
    CHECK(hipHostGetDevicePointer(&alias, host, 0));
    std::printf("igpu_gtt_micro: dequant arm: %lld GiB %s, alias=%p%s, %d experts x (gu %dx%d + d %dx%d)\n",
                gib, host_pinned_alloc ? "pinned" : "registered", alias, alias == (void*) host ? " (identity)" : "",
                kExperts, (int) n_ff, (int) n_embd, (int) n_embd, (int) n_ff);
    uint16_t* dq_gu = nullptr, *dq_d = nullptr, *d_ping = nullptr;
    CHECK(hipMalloc(&dq_gu, 1280 * 2560 * 2));
    CHECK(hipMalloc(&dq_d, 2560 * 640 * 2));
    CHECK(hipMalloc(&d_ping, 4 << 20));
    // the engine's VRAM state: ~8 GiB of session scratch held while the experts run (16 GiB budget)
    void* pressure = nullptr;
    if (pressure_gib > 0) {
        const long long pb = pressure_gib * (1ll << 30);
        if (hipMalloc(&pressure, (size_t) pb) != hipSuccess) {
            std::fprintf(stderr, "igpu_gtt_micro: dequant arm: cannot hold %lld GiB: %s\n", pressure_gib,
                         hipGetErrorString(hipGetLastError()));
            return 2;
        }
        std::printf("igpu_gtt_micro: dequant arm: holding %lld GiB of device memory (engine-like pressure)\n",
                    pressure_gib);
    }
    hipStream_t s = 0;
    if (nb) {
        if (hipStreamCreateWithFlags(&s, hipStreamNonBlocking) != hipSuccess) {
            std::fprintf(stderr, "igpu_gtt_micro: dequant arm: stream: %s\n", hipGetErrorString(hipGetLastError()));
            return 2;
        }
        std::printf("igpu_gtt_micro: dequant arm: non-BLOCKING stream (the engine's m.cs class)\n");
    } else {
        std::printf("igpu_gtt_micro: dequant arm: legacy stream (as in run 26)\n");
    }
    // the engine's prefault: one full DMA read pass (4 MiB chunks) before any kernel touches the region
    const size_t chunk = 4ull << 20;
    for (long long off = 0; off < bytes; off += chunk)
        CHECK(hipMemcpy(d_ping, host + off, (size_t) std::min<long long>(chunk, bytes - off), hipMemcpyHostToDevice));
    CHECK(hipDeviceSynchronize());
    // dev: the same blobs in device memory - the copy path's read source, the kernel unchanged
    const uint8_t* base = (const uint8_t*) alias;
    uint8_t* d_blobs = nullptr;
    if (dev) {
        const long long bl = (long long) kExperts * (long long) blob;
        CHECK(hipMalloc(&d_blobs, (size_t) bl));
        CHECK(hipMemcpy(d_blobs, host, (size_t) bl, hipMemcpyHostToDevice));
        base = d_blobs;
        std::printf("igpu_gtt_micro: dequant arm: dequanting from a DEVICE buffer (H2D copy, %lld MiB)\n",
                    bl >> 20);
    }
    std::printf("igpu_gtt_micro: dequant arm: running the %d experts (timed)\n", kExperts);
    const auto dq0 = std::chrono::steady_clock::now();
    for (int e = 0; e < kExperts; ++e) {
        const uint8_t* b = base + (long long) e * (long long) blob;
        strata::kernels::iq_dequant_gu_f16(kType, b, b + up_off, n_ff, n_embd, dq_gu, (void*) s);
        strata::kernels::iq_dequant_f16(kType, b + down_off, n_embd * n_ff, dq_d, (void*) s);
        const hipError_t err = hipStreamSynchronize(s);
        if (err != hipSuccess) {
            std::fprintf(stderr, "igpu_gtt_micro: dequant arm: expert %d failed: %s (blob at %p)\n", e,
                         hipGetErrorString(err), (const void*) b);
            return 1;
        }
    }
    const long long dq_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - dq0).count();
    const double dq_gb = (double) kExperts * (double) blob / 1e9 / (dq_ms / 1000.0);
    std::printf("igpu_gtt_micro: dequant arm: %d experts dequantized in %lld ms: %.2f GB/s read from %s%s%s\n",
                kExperts, dq_ms, dq_gb, dev ? "device" : "GTT", thp ? "+THP" : "", nb ? "+nb" : "");
    // spot-check the outputs: the first 64 of each must be finite (the patterned input is not valid Q2_0,
    // so values are junk - this only proves the writes landed)
    uint16_t* h = (uint16_t*) std::malloc(2);
    uint16_t gu0 = 0, d0 = 0;
    CHECK(hipMemcpy(&gu0, dq_gu, 2, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&d0, dq_d, 2, hipMemcpyDeviceToHost));
    std::printf("igpu_gtt_micro: dequant arm: all %d experts dequantized (gu[0]=0x%04x d[0]=0x%04x)\n", kExperts,
                (unsigned) gu0, (unsigned) d0);
    CHECK(hipFree(dq_gu));
    CHECK(hipFree(dq_d));
    CHECK(hipFree(d_ping));
    if (d_blobs) CHECK(hipFree(d_blobs));
    if (pressure) CHECK(hipFree(pressure));
    if (nb) (void) hipStreamDestroy(s);
    if (host_pinned_alloc) (void) hipFreeHost(host);
    else {
        (void) hipHostUnregister(host);
        free(host);
    }
    free(h);
    return 0;
}

// igpu-rework P2.15: the fused path's ceiling.  The engine's two expert paths meet at the same
// ~39 tok/s wall from opposite sides: the dequant+BLAS path pays a 7x write amplification (64
// fp16 values written per 18-byte Q2_0 block read) at ~50 GB/s combined, and the fused dp4a path
// reads without amplification but computes at ~152 GFLOPS.  This arm reads the same blocks through
// the engine's mapping and does one dp4a per 4-byte code word - no LDS, no dequantized writes, no
// per-expert launch rhythm - so the number it prints is the pure dp4a ceiling: near the emulated
// peak (~3.6 TFLOPS) means the engine's 152 GFLOPS is the tile shape; near 150-300 GFLOPS means
// the dp4a emulation itself is the wall and the fused path cannot beat the dequant path here.
__global__ void dp4a_ceiling_kernel(const uint8_t* __restrict__ blocks, long long nblocks, uint32_t* partials) {
    const int64_t lane = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t a0 = 0x01020304u + threadIdx.x, a1 = 0x05060708u + threadIdx.x;
    const uint32_t a2 = 0x090a0b0cu + threadIdx.x, a3 = 0x0d0e0f10u + threadIdx.x;
    int64_t acc = 0;
    for (int64_t b = lane; b < nblocks; b += (int64_t) gridDim.x * blockDim.x) {
        const uint8_t* p = blocks + b * 18;   // type 42: 16 code bytes + 2 scale bytes per 64 values
        uint32_t w0, w1, w2, w3;
        __builtin_memcpy(&w0, p, 4); __builtin_memcpy(&w1, p + 4, 4);
        __builtin_memcpy(&w2, p + 8, 4); __builtin_memcpy(&w3, p + 12, 4);
        acc += __dp4a(w0, a0, 0) + __dp4a(w1, a1, 0) + __dp4a(w2, a2, 0) + __dp4a(w3, a3, 0);
    }
    if (acc < 0) acc = -acc;
    atomicAdd(&partials[blockIdx.x], (uint32_t) acc);
}

// igpu-rework P2.15b: the compute-bound dp4a peak.  The --dp4a arm above is read-bound: Q2_0
// fixes 32 FLOP per 18-byte block, so at the full 82.6 GB/s read it cannot exceed 147 GFLOPS -
// that number is the floor, not the peak, and the engine's gemm (280-380 GFLOPS, blobs reused
// across ~14 tokens per row-group) already sits above it.  This arm keeps the codes in registers
// (no host read at all) and runs the same dp4a inner loop: whatever it prints is the pure dp4a
// peak for this APU, and the engine's 300 GFLOPS is a percentage of it.  If the peak is ~4
// TFLOPS (native or lightly emulated dp4a), the tile shape has a 10x headroom; if it is ~500
// GFLOPS, the engine is already near the wall and the expert phase cannot move much.
__global__ void dp4a_peak_kernel(uint32_t seed, long long iters, uint32_t* partials) {
    const uint32_t a0 = seed + threadIdx.x, a1 = seed + threadIdx.x * 3u, a2 = seed + threadIdx.x * 5u,
                   a3 = seed + threadIdx.x * 7u;
    uint32_t acc = 0;
    for (long long i = 0; i < iters; ++i) {
        const uint32_t w0 = (uint32_t) (i * 2654435761u) ^ a0, w1 = (uint32_t) (i * 40503u) ^ a1;
        const uint32_t w2 = (uint32_t) (i * 2246822519u) ^ a2, w3 = (uint32_t) (i * 3266489917u) ^ a3;
        acc += __dp4a(w0, a0, 0) + __dp4a(w1, a1, 0) + __dp4a(w2, a2, 0) + __dp4a(w3, a3, 0);
    }
    atomicAdd(&partials[blockIdx.x], acc);
}

int run_dp4a_peak_arm(int blocks, long long iters) {
    uint32_t* d_partials = nullptr;
    CHECK(hipMalloc(&d_partials, (size_t) blocks * sizeof(uint32_t)));
    CHECK(hipMemsetAsync(d_partials, 0, (size_t) blocks * sizeof(uint32_t), 0));
    dp4a_peak_kernel<<<blocks, 256, 0, (hipStream_t) 0>>>(0x5a5a5a5au, iters, d_partials);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());   // warm the launch; the timed pass follows
    const auto t0 = std::chrono::steady_clock::now();
    dp4a_peak_kernel<<<blocks, 256, 0, (hipStream_t) 0>>>(0x5a5a5a5au, iters, d_partials);
    CHECK(hipStreamSynchronize(0));
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    const double dp4a_count = (double) blocks * 256.0 * (double) iters * 4.0;
    std::printf("igpu_gtt_micro: dp4a peak arm: %d blocks x 256 threads x %lld x 4 dp4a in %lld ms: %.0f GFLOPS\n",
                blocks, iters, ms, dp4a_count * 8.0 / 1e9 / (ms / 1000.0));
    CHECK(hipFree(d_partials));
    return 0;
}

int run_dp4a_arm(long long gib, bool pinned) {
    const long long bytes = gib * (1ll << 30);
    uint8_t* host = nullptr;
    const bool host_pinned_alloc = pinned;
    if (pinned) {
        const hipError_t e = hipHostAlloc((void**) &host, (size_t) bytes, hipHostAllocMapped | hipHostAllocPortable);
        if (e != hipSuccess || host == nullptr) {
            std::fprintf(stderr, "igpu_gtt_micro: dp4a arm: cudaHostAlloc %lld GiB: %s\n", gib, hipGetErrorString(e));
            return 2;
        }
    } else {
        host = (uint8_t*) std::malloc(bytes);
        if (!host) { std::fprintf(stderr, "igpu_gtt_micro: dp4a arm: cannot allocate %lld GiB\n", gib); return 2; }
    }
    for (long long off = 0; off < bytes; off += 4096)
        std::memset(host + off, (int) (((off >> 12) & 0xff) ^ 0x5a), 4096);
    if (!host_pinned_alloc && hipHostRegister(host, (size_t) bytes, hipHostRegisterDefault) != hipSuccess) {
        std::fprintf(stderr, "igpu_gtt_micro: dp4a arm: register: %s\n", hipGetErrorString(hipGetLastError()));
        return 2;
    }
    void* alias = nullptr;
    CHECK(hipHostGetDevicePointer(&alias, host, 0));
    const int kBlocks = 512;
    uint32_t* d_partials = nullptr;
    uint8_t* d_ping = nullptr;
    CHECK(hipMalloc(&d_partials, kBlocks * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_ping, 4 << 20));
    // the engine's prefault: one full DMA pass before the first kernel read (the passing model)
    const size_t chunk = 4ull << 20;
    for (long long off = 0; off < bytes; off += chunk)
        CHECK(hipMemcpy(d_ping, host + off, (size_t) std::min<long long>(chunk, bytes - off), hipMemcpyHostToDevice));
    CHECK(hipDeviceSynchronize());
    const long long nblocks = bytes / 18;
    CHECK(hipMemsetAsync(d_partials, 0, kBlocks * sizeof(uint32_t), 0));
    const auto t0 = std::chrono::steady_clock::now();
    dp4a_ceiling_kernel<<<kBlocks, 256, 0, (hipStream_t) 0>>>((const uint8_t*) alias, nblocks, d_partials);
    CHECK(hipGetLastError());
    CHECK(hipStreamSynchronize(0));
    const long long ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    const double flops = (double) nblocks * 32.0 / 1e9 / (ms / 1000.0);   // 16 MACs x 2 per block
    std::printf("igpu_gtt_micro: dp4a arm: %lld GiB %s read in %lld ms: %.1f GB/s, %.0f GFLOPS dp4a\n",
                gib, host_pinned_alloc ? "pinned" : "registered", ms,
                (double) bytes / 1e9 / (ms / 1000.0), flops);
    CHECK(hipFree(d_partials));
    CHECK(hipFree(d_ping));
    if (host_pinned_alloc) (void) hipFreeHost(host);
    else { (void) hipHostUnregister(host); free(host); }
    return 0;
}


// igpu-rework P2.4: the engine's gather phase, in a process without the engine.  The 0.1.36 alias
// profile's "dequant" phase (8.0 s, 26 %) actually contains the MMQ-on gather: per expert, one
// copy16_kernel launch (gate+up -> group gu slot, down -> group d slot), GTT read + VRAM write.
// This arm replays the exact kernel (copied from src/prefill/moe_mmq.cu) over 512 contiguous
// 1.38 MB blobs in the engine's pinned complement mapping, back-to-back launches as the engine's
// compute() does, and times it - plus one flat copy of the same total bytes.  Per-expert ~= flat:
// the 8.0 s is host launch gaps (batch the gathers); per-expert ~= 1.8 GB/s: the pattern is the
// problem.
__global__ void micro_copy16_kernel(const uint4* __restrict__ a, int64_t na, const uint4* __restrict__ b, int64_t nb,
                                    uint4* __restrict__ ab_dst, const uint4* __restrict__ c, int64_t nc,
                                    uint4* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

// igpu-rework P2.7: the engine's 8 expert-pool worker threads, as busy spinners - if the CPU
// contention (driver workqueue, doorbell flush, scheduler) changes the per-launch cost, the
// gather rate drops when these are on.
std::atomic<bool> g_thr_stop{false};
void thr_spin() {
    while (!g_thr_stop.load(std::memory_order_relaxed)) { std::atomic_thread_fence(std::memory_order_seq_cst); }
}

int run_gather_arm(long long gib, bool pinned, bool pin_reg, bool nb, long long pressure_gib, bool scatter,
                   bool two_streams, bool sync_rhythm, bool file_cache, int spin_threads) {
    // file_cache: the engine's coexistence state - the 31.64 GiB experts.bin is ALSO resident in
    // the page cache (it was just read into the pinned complement), so the machine carries ~2x
    // 31.6 GiB of the same data and the kernel's page-cache/THP activity runs during the gathers.
    // If the gather rate drops with the file cache present, the coexistence (reclaim/THP churn on
    // the GTT path) is the engine's 2.3 ms/kernel gap.
    long long fc_bytes = 0;
    uint8_t* fc = nullptr;
    if (file_cache) {
        const char* fp = "/home/jhohertz/co/Strata/packs/qwen38-flash-next-q2_0/experts.bin";
        const int f = open(fp, O_RDONLY);
        if (f < 0) { std::fprintf(stderr, "igpu_gtt_micro: gather arm: cannot open %s\n", fp); return 2; }
        fc_bytes = (long long) lseek(f, 0, SEEK_END);
        fc = (uint8_t*) mmap(nullptr, (size_t) fc_bytes, PROT_READ, MAP_PRIVATE, f, 0);
        close(f);
        if (fc == MAP_FAILED) { std::fprintf(stderr, "igpu_gtt_micro: gather arm: mmap of experts.bin failed\n"); return 2; }
        // read it in (page cache fill, like the engine's complement copy just did)
        volatile uint8_t sink = 0;
        for (long long off = 0; off < fc_bytes; off += 1ll << 20) sink += fc[off];
        (void) sink;
        std::printf("igpu_gtt_micro: gather arm: experts.bin in the page cache (%.2f GiB) alongside the pinned region\n",
                    (double) fc_bytes / 1073741824.0);
    }
    constexpr int kExperts = 512;
    constexpr size_t half = 460800, blob = 1382400;   // gate | up | down, Q2_0 640x2560 / 2560x640
    const long long bytes = gib * (1ll << 30);
    if (bytes < (long long) kExperts * (long long) blob + (1ll << 30)) {
        std::fprintf(stderr, "igpu_gtt_micro: --gather needs >= 1 GiB\n");
        return 2;
    }
    uint8_t* host = nullptr;
    if (pinned) {
        const hipError_t e = hipHostAlloc((void**) &host, (size_t) bytes, hipHostAllocMapped | hipHostAllocPortable);
        if (e != hipSuccess || host == nullptr) {
            std::fprintf(stderr, "igpu_gtt_micro: gather arm: cudaHostAlloc %lld GiB: %s\n", gib, hipGetErrorString(e));
            return 2;
        }
    } else {
        host = (uint8_t*) std::malloc(bytes);
        if (!host) { std::fprintf(stderr, "igpu_gtt_micro: cannot allocate %lld GiB\n", gib); return 2; }
    }
    for (long long off = 0; off < bytes; off += 4096) std::memset(host + off, (int) (((off >> 12) & 0xff) ^ 0x5a), 4096);
    if ((!pinned || pin_reg) && hipHostRegister(host, (size_t) bytes, hipHostRegisterDefault) != hipSuccess) {
        std::fprintf(stderr, "igpu_gtt_micro: gather arm: register: %s\n", hipGetErrorString(hipGetLastError()));
        return 2;
    }
    void* alias = nullptr;
    CHECK(hipHostGetDevicePointer(&alias, host, 0));
    const uint8_t* base = (const uint8_t*) alias;
    uint8_t* d_gu = nullptr;   // group gu slot per expert: 2*half
    uint8_t* d_d = nullptr;    // group d slot per expert: half
    CHECK(hipMalloc(&d_gu, 2 * half * kExperts));
    CHECK(hipMalloc(&d_d, half * kExperts));
    uint8_t* d_flat = nullptr; // one full blob per expert: the flat copy's destination
    CHECK(hipMalloc(&d_flat, blob * kExperts));
    // the engine's VRAM state: ~8 GiB of session scratch held while the experts run (16 GiB budget)
    void* pressure = nullptr;
    if (pressure_gib > 0) {
        const long long pb = pressure_gib * (1ll << 30);
        if (hipMalloc(&pressure, (size_t) pb) != hipSuccess) {
            std::fprintf(stderr, "igpu_gtt_micro: gather arm: cannot hold %lld GiB: %s\n", pressure_gib,
                         hipGetErrorString(hipGetLastError()));
            return 2;
        }
        std::printf("igpu_gtt_micro: gather arm: holding %lld GiB of device memory (engine-like VRAM pressure)\n",
                    pressure_gib);
    }
    hipStream_t s = 0;
    if (nb) CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    // sync_rhythm: the engine's per-layer cadence - a small D2H (the routed ids) + a FULL stream
    // sync after every 16 experts, then a small H2D (slot/src) before the next batch
    uint8_t* h_ids = nullptr;
    uint8_t* d_ids = nullptr;
    if (sync_rhythm) {
        h_ids = (uint8_t*) std::malloc(4096);
        CHECK(hipMalloc(&d_ids, 4096));
        std::printf("igpu_gtt_micro: gather arm: engine rhythm - D2H + full sync every 16 experts\n");
    }
    // the engine runs its small D2H/H2D copies on a SECOND non-blocking stream (m.copy) while the
    // gathers run on m.cs - if the second queue changes the driver's doorbell path, the gather
    // launch cost should jump when this is on
    hipStream_t s2 = 0;
    uint8_t* s2buf = nullptr;
    if (two_streams) {
        CHECK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
        CHECK(hipMalloc(&s2buf, 4096));
        std::printf("igpu_gtt_micro: gather arm: second non-blocking stream active (the engine's m.copy)\n");
    }
    // the engine's prefault: one full DMA read pass over the expert region (the WHOLE region when
    // scattered, as the engine's routing picks experts from across the 31.64 GiB complement)
    uint8_t* ping = nullptr;
    CHECK(hipMalloc(&ping, 4 << 20));
    const long long pf_span = scatter ? bytes : (long long) kExperts * (long long) blob;
    for (long long off = 0; off < pf_span; off += 4ull << 20)
        CHECK(hipMemcpy(ping, host + off, (size_t) std::min<long long>(4ll << 20, pf_span - off), hipMemcpyHostToDevice));
    CHECK(hipDeviceSynchronize());
    // scattered offsets: deterministic LCG over the whole region, blob-aligned (the engine's per-layer
    // routing reads ~36 experts at spread positions per section)
    std::vector<long long> sc_off(kExperts, 0);
    if (scatter) {
        unsigned long long rng = 0x9e3779b97f4a7c15ull;
        const long long slots = bytes / (long long) blob;
        for (int e = 0; e < kExperts; ++e) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            sc_off[e] = (long long) ((rng >> 17) % (unsigned long long) slots) * (long long) blob;
        }
        std::printf("igpu_gtt_micro: gather arm: %d experts at SCATTERED offsets across %.1f GiB (the engine's routing pattern)\n",
                    kExperts, (double) bytes / 1073741824.0);
    }
    const int64_t na = half / 16, nc = half / 16;
    std::printf("igpu_gtt_micro: gather arm: %d experts x 1.38 MB, %s mapping%s%s, timing per-expert launches\n", kExperts,
                pinned ? (pin_reg ? "pinned+registered (the engine's exact state)" : "pinned (no userptr)")
                       : "registered (userptr, not pinned)",
                nb ? " + nb" : "", scatter ? " + scattered" : "");
    std::vector<std::thread> thrs;
    if (spin_threads > 0) {
        g_thr_stop = false;
        for (int t = 0; t < spin_threads; ++t) thrs.emplace_back(thr_spin);
        std::printf("igpu_gtt_micro: gather arm: %d busy-spin worker threads (the engine's pool)\n", spin_threads);
    }
    const auto g0 = std::chrono::steady_clock::now();
    for (int e = 0; e < kExperts; ++e) {
        const uint8_t* b = scatter ? base + sc_off[e] : base + (long long) e * (long long) blob;
        micro_copy16_kernel<<<(unsigned) ((2 * na + nc + 255) / 256), 256, 0, s>>>((const uint4*) b, na, (const uint4*) (b + half),
                                                                                  na, (uint4*) (d_gu + (long long) e * 2 * half),
                                                                                  (const uint4*) (b + 2 * half), nc,
                                                                                  (uint4*) (d_d + (long long) e * half));
        if (two_streams && (e % 16) == 15)   // the engine's per-layer small copies on the other stream
            CHECK(hipMemcpyAsync(s2buf, host + (size_t) e * 4096, 4096, hipMemcpyHostToDevice, s2));
        if (sync_rhythm && (e % 16) == 15) {   // the engine's per-layer ids readback + full drain
            CHECK(hipMemcpyAsync(d_ids, host + (size_t) e * 4096, 4096, hipMemcpyHostToDevice, s));
            CHECK(hipMemcpyAsync(h_ids, d_ids, 4096, hipMemcpyDeviceToHost, s));
            CHECK(hipStreamSynchronize(s));   // the engine's per-layer full drain
            CHECK(hipMemcpyAsync(d_ids, h_ids, 4096, hipMemcpyHostToDevice, s));
        }
    }
    CHECK(hipStreamSynchronize(s));
    const long long g_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - g0).count();
    const double total_bytes = (double) kExperts * (double) blob;
    std::printf("igpu_gtt_micro: gather arm: %d per-expert gathers in %lld ms: %.2f GB/s (GTT read + VRAM write)\n",
                kExperts, g_ms, total_bytes / 1e9 / (g_ms / 1000.0));
    // the same total bytes in ONE flat copy: the rate the hardware can sustain on this mapping
    const auto f0 = std::chrono::steady_clock::now();
    const int64_t flat = ((long long) kExperts * (long long) blob) / 16;
    micro_copy16_kernel<<<(unsigned) ((flat + 255) / 256), 256, 0, s>>>((const uint4*) host, flat, nullptr, 0, (uint4*) d_flat,
                                                                        nullptr, 0, nullptr);
    CHECK(hipStreamSynchronize(s));
    const long long f_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - f0).count();
    std::printf("igpu_gtt_micro: gather arm: one flat copy of the same bytes in %lld ms: %.2f GB/s\n", f_ms,
                total_bytes / 1e9 / (f_ms / 1000.0));
    if (two_streams) {
        CHECK(hipStreamSynchronize(s2));
        CHECK(hipFree(s2buf));
        (void) hipStreamDestroy(s2);
    }
    if (sync_rhythm) {
        CHECK(hipFree(d_ids));
        free(h_ids);
    }
    if (fc) (void) munmap(fc, (size_t) fc_bytes);
    g_thr_stop = true;
    for (auto& t : thrs) t.join();
    CHECK(hipFree(ping));
    CHECK(hipFree(d_gu));
    CHECK(hipFree(d_d));
    CHECK(hipFree(d_flat));
    if (pressure) CHECK(hipFree(pressure));
    if (nb) (void) hipStreamDestroy(s);
    if (pin_reg) (void) hipHostUnregister(host);
    if (pinned) (void) hipFreeHost(host);
    else {
        (void) hipHostUnregister(host);
        free(host);
    }
    return 0;
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
        if (std::strcmp(argv[i], "--alias") == 0 && i + 1 < argc)
            return run_alias_arm((long long) std::atof(argv[i + 1]));
        if (std::strcmp(argv[i], "--dp4a-peak") == 0 && i + 2 < argc)
            return run_dp4a_peak_arm(std::atoi(argv[i + 1]), (long long) std::atof(argv[i + 2]));
        if (std::strcmp(argv[i], "--dp4a") == 0 && i + 1 < argc) {
            bool pinned = false;
            int j = i + 2;
            while (j < argc) { if (std::strcmp(argv[j], "pinned") == 0) pinned = true; ++j; }
            return run_dp4a_arm((long long) std::atof(argv[i + 1]), pinned);
        }
        if (std::strcmp(argv[i], "--dequant") == 0 && i + 1 < argc) {
            bool nb = false, thp = false, dev = false, pinned = false;
            long long pressure = 0;
            int j = i + 2;
            while (j < argc) {
                if (std::strcmp(argv[j], "nb") == 0) nb = true;
                else if (std::strcmp(argv[j], "thp") == 0) thp = true;
                else if (std::strcmp(argv[j], "dev") == 0) dev = true;
                else if (std::strcmp(argv[j], "pinned") == 0) pinned = true;
                else if (pressure == 0 && argv[j][0] >= '0' && argv[j][0] <= '9') pressure = (long long) std::atof(argv[j]);
                ++j;
            }
            return run_dequant_arm((long long) std::atof(argv[i + 1]), nb, pressure, thp, dev, pinned);
        }
        if (std::strcmp(argv[i], "--gather") == 0 && i + 1 < argc) {
            bool pinned = true, pin_reg = false, nb = false, scatter = false, two_streams = false,
                 sync_rhythm = false, file_cache = false;
            int spin_threads = 0;
            long long pressure = 0;
            for (int j = i + 2; j < argc; ++j) {
                if (std::strcmp(argv[j], "thr") == 0 && j + 1 < argc) spin_threads = std::atoi(argv[++j]);
                if (std::strcmp(argv[j], "reg") == 0) pinned = false;
                else if (std::strcmp(argv[j], "pr") == 0) pin_reg = true;   // pinned AND registered: the engine's state
                else if (std::strcmp(argv[j], "nb") == 0) nb = true;
                else if (std::strcmp(argv[j], "sc") == 0) scatter = true;
                else if (std::strcmp(argv[j], "2s") == 0) two_streams = true;
                else if (std::strcmp(argv[j], "sr") == 0) sync_rhythm = true;
                else if (std::strcmp(argv[j], "fc") == 0) file_cache = true;
                else if (argv[j][0] >= '0' && argv[j][0] <= '9') pressure = (long long) std::atof(argv[j]);
            }
            return run_gather_arm((long long) std::atof(argv[i + 1]), pinned, pin_reg, nb, pressure, scatter,
                                  two_streams, sync_rhythm, file_cache, spin_threads);
        }
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
