#include "hip/hip_runtime.h"
#include "../GpuCacheWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <stdexcept>
#include <string>

static int  g_cache_sm_count   = 0;
static bool g_cache_topo_ready = false;

static void cache_query_topology() {
    if (g_cache_topo_ready) return;
    int dev;
    hipGetDevice(&dev);
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, dev);
    g_cache_sm_count   = p.multiProcessorCount;
    g_cache_topo_ready = true;
}

static int cache_get_iters(size_t N) {
    return static_cast<int>(1000000000 / N) + 2;
}

template <int N>
struct CacheWorkloadParams {
    static constexpr int BLOCKSIZE =
        (N % 512 == 0 && N >= 512) ? 512 :
        (N % 256 == 0 && N >= 256) ? 256 : 128;
    static constexpr int ITERS = 1000000000 / N + 2;
};

template <int N, int ITERS, int BLOCKSIZE>
__global__ void sumKernel_wl(cache_dtype* __restrict__ A,
                             const cache_dtype* __restrict__ B, int zero) {
    cache_dtype localSum = (cache_dtype)0;
    B += threadIdx.x;

#pragma unroll N / BLOCKSIZE > 32 ? 1 : 32 / (N / BLOCKSIZE)
    for (int iter = 0; iter < ITERS; iter++) {
        B += zero;
        auto B2 = B + N;
#pragma unroll N / BLOCKSIZE >= 64 ? 32 : N / BLOCKSIZE
        for (int i = 0; i < N; i += BLOCKSIZE)
            localSum += B[i] * B2[i];
        localSum *= (cache_dtype)1.3;
    }
    if (localSum == (cache_dtype)1233)
        A[threadIdx.x] += localSum;
}

__global__ void initKernel_wl(cache_dtype* A, size_t N) {
    size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (size_t idx = tidx; idx < N; idx += blockDim.x * gridDim.x)
        A[idx] = (cache_dtype)1.1;
}

template <int N>
static void launch_cache_kernel(cache_dtype* bufA, cache_dtype* bufB, hipStream_t s) {
    constexpr int ITERS     = CacheWorkloadParams<N>::ITERS;
    constexpr int BLOCKSIZE = CacheWorkloadParams<N>::BLOCKSIZE;
    sumKernel_wl<N, ITERS, BLOCKSIZE><<<g_cache_sm_count, BLOCKSIZE, 0, s>>>(bufA, bufB, 0);
}

#define DISPATCH_CACHE(N_VAL) case N_VAL: launch_cache_kernel<N_VAL>(bufA, bufB, s); break;

static void launch_sum_kernel_wl(size_t N, cache_dtype* bufA, cache_dtype* bufB, hipStream_t s) {
    switch (N) {

        DISPATCH_CACHE(128)     DISPATCH_CACHE(256)
        DISPATCH_CACHE(512)     DISPATCH_CACHE(768)
        DISPATCH_CACHE(1024)    DISPATCH_CACHE(1536)
        DISPATCH_CACHE(2048)    DISPATCH_CACHE(2560)
        DISPATCH_CACHE(3072)    DISPATCH_CACHE(3584)
        DISPATCH_CACHE(4096)    DISPATCH_CACHE(4608)
        DISPATCH_CACHE(5120)    DISPATCH_CACHE(5632)
        DISPATCH_CACHE(6144)    DISPATCH_CACHE(6656)
        DISPATCH_CACHE(7168)    DISPATCH_CACHE(7680)
        DISPATCH_CACHE(8192)    DISPATCH_CACHE(8704)
        DISPATCH_CACHE(9216)    DISPATCH_CACHE(9728)
        DISPATCH_CACHE(10240)   DISPATCH_CACHE(10752)
        DISPATCH_CACHE(11264)   DISPATCH_CACHE(11776)
        DISPATCH_CACHE(12288)   DISPATCH_CACHE(12800)
        DISPATCH_CACHE(13312)   DISPATCH_CACHE(13824)
        DISPATCH_CACHE(14336)   DISPATCH_CACHE(14848)
        DISPATCH_CACHE(15360)   DISPATCH_CACHE(15872)
        DISPATCH_CACHE(16384)

        DISPATCH_CACHE(18944)   DISPATCH_CACHE(22016)
        DISPATCH_CACHE(26112)   DISPATCH_CACHE(30208)
        DISPATCH_CACHE(35840)   DISPATCH_CACHE(41984)
        DISPATCH_CACHE(49152)   DISPATCH_CACHE(57344)
        DISPATCH_CACHE(67072)   DISPATCH_CACHE(78336)
        DISPATCH_CACHE(91648)   DISPATCH_CACHE(107520)
        DISPATCH_CACHE(125952)  DISPATCH_CACHE(147456)
        DISPATCH_CACHE(201728)  DISPATCH_CACHE(236032)
        DISPATCH_CACHE(276480)  DISPATCH_CACHE(323072)
        DISPATCH_CACHE(378368)  DISPATCH_CACHE(442368)
        DISPATCH_CACHE(517632)  DISPATCH_CACHE(605696)
        DISPATCH_CACHE(709120)  DISPATCH_CACHE(829440)
        DISPATCH_CACHE(970752)  DISPATCH_CACHE(1135616)
        DISPATCH_CACHE(1328640) DISPATCH_CACHE(1554944)
        DISPATCH_CACHE(1819136) DISPATCH_CACHE(2128384)
        DISPATCH_CACHE(2490368) DISPATCH_CACHE(2913792)
        DISPATCH_CACHE(3408896) DISPATCH_CACHE(3988480)
        DISPATCH_CACHE(4666368) DISPATCH_CACHE(5459968)
        DISPATCH_CACHE(6388224) DISPATCH_CACHE(7474176)
        DISPATCH_CACHE(8744960) DISPATCH_CACHE(10231808)
        DISPATCH_CACHE(11971072) DISPATCH_CACHE(14006272)
        DISPATCH_CACHE(16387072) DISPATCH_CACHE(19172864)
        DISPATCH_CACHE(22432768) DISPATCH_CACHE(26246144)
        DISPATCH_CACHE(30707712) DISPATCH_CACHE(35928576)

        DISPATCH_CACHE(24576)   DISPATCH_CACHE(32768)
        DISPATCH_CACHE(65536)   DISPATCH_CACHE(131072)
        DISPATCH_CACHE(262144)  DISPATCH_CACHE(524288)
        DISPATCH_CACHE(1048576) DISPATCH_CACHE(2097152)
        DISPATCH_CACHE(4194304)
        default:
            throw std::runtime_error(
                "cache: unsupported working_set_elements value: " +
                std::to_string(N));
    }
}

#undef DISPATCH_CACHE

using HipCache = GpuCacheWorkload<Baseliner::Hardware::HipBackend>;

template <>
void HipCache::setup_device(typename backend::stream_t stream) {
    cache_query_topology();
    m_sm_count = g_cache_sm_count;
    size_t N          = m_working_set_elements;
    size_t buf_count  = 2ULL * N + 1024;
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_a), buf_count * sizeof(cache_dtype), stream));
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_b), buf_count * sizeof(cache_dtype), stream));
    initKernel_wl<<<52, 256, 0, stream>>>(m_device_buffer_a, buf_count);
    initKernel_wl<<<52, 256, 0, stream>>>(m_device_buffer_b, buf_count);
    CHECK_HIP(hipStreamSynchronize(stream));
}

template <>
auto HipCache::run(typename backend::stream_t stream) -> std::monostate {
    launch_sum_kernel_wl(m_working_set_elements, m_device_buffer_a, m_device_buffer_b, stream);
    return {};
}

template <>
auto HipCache::number_of_bytes() -> std::optional<size_t> {
    cache_query_topology();
    size_t N     = m_working_set_elements;
    int    iters = cache_get_iters(N);
    return 2ULL * N * sizeof(cache_dtype) * g_cache_sm_count * iters;
}

template <>
void HipCache::fetch_results(typename backend::stream_t stream) {
    if (m_device_buffer_a) {
        CHECK_HIP(hipFreeAsync(m_device_buffer_a, stream));
        m_device_buffer_a = nullptr;
    }
    if (m_device_buffer_b) {
        CHECK_HIP(hipFreeAsync(m_device_buffer_b, stream));
        m_device_buffer_b = nullptr;
    }
}

BASELINER_REGISTER_WORKLOAD(HipCache);
