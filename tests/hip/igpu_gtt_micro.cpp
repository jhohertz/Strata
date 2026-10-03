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
#include <sys/mman.h>

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

int main(int argc, char** argv) {
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
