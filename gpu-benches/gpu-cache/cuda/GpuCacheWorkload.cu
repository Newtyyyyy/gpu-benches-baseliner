#include "../GpuCacheWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

static int  g_cache_sm_count   = 0;
static bool g_cache_topo_ready = false;

static void cache_query_topology() {
    if (g_cache_topo_ready) return;
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    g_cache_sm_count   = p.multiProcessorCount;
    g_cache_topo_ready = true;
}

static int cache_get_iters(size_t N) {
    return static_cast<int>(100000000 / N) + 2;
}

// ---------------------------------------------------------------------------
// CacheParams: compile-time BLOCKSIZE and ITERS for each N
// ---------------------------------------------------------------------------

template <int N>
struct CacheWorkloadParams {
    static constexpr int BLOCKSIZE =
        (N % 512 == 0 && N >= 512) ? 512 :
        (N % 256 == 0 && N >= 256) ? 256 : 128;
    static constexpr int ITERS = 100000000 / N + 2;
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

template <int N>
static void launch_cache_kernel(cache_dtype* bufA, cache_dtype* bufB, cudaStream_t s) {
    constexpr int ITERS     = CacheWorkloadParams<N>::ITERS;
    constexpr int BLOCKSIZE = CacheWorkloadParams<N>::BLOCKSIZE;
    sumKernel_wl<N, ITERS, BLOCKSIZE><<<g_cache_sm_count, BLOCKSIZE, 0, s>>>(bufA, bufB, 0);
}

#define DISPATCH_CACHE(N_VAL) case N_VAL: launch_cache_kernel<N_VAL>(bufA, bufB, s); break;

static void launch_sum_kernel_wl(size_t N, cache_dtype* bufA, cache_dtype* bufB, cudaStream_t s) {
    switch (N) {
        DISPATCH_CACHE(128)    DISPATCH_CACHE(256)    DISPATCH_CACHE(512)
        DISPATCH_CACHE(768)    DISPATCH_CACHE(1024)   DISPATCH_CACHE(2048)
        DISPATCH_CACHE(3072)   DISPATCH_CACHE(4096)   DISPATCH_CACHE(5120)
        DISPATCH_CACHE(6144)   DISPATCH_CACHE(7168)   DISPATCH_CACHE(8192)
        DISPATCH_CACHE(10240)  DISPATCH_CACHE(12288)  DISPATCH_CACHE(14336)
        DISPATCH_CACHE(16384)  DISPATCH_CACHE(24576)  DISPATCH_CACHE(32768)
        DISPATCH_CACHE(49152)  DISPATCH_CACHE(65536)  DISPATCH_CACHE(131072)
        DISPATCH_CACHE(262144) DISPATCH_CACHE(524288) DISPATCH_CACHE(1048576)
        DISPATCH_CACHE(2097152) DISPATCH_CACHE(4194304)
        default:
            throw std::runtime_error(
                "cache: unsupported working_set_elements value: " +
                std::to_string(N));
    }
}

#undef DISPATCH_CACHE

// ---------------------------------------------------------------------------
// IWorkload specializations
// ---------------------------------------------------------------------------

using CudaCache = GpuCacheWorkload<Baseliner::Hardware::CudaBackend>;

template <>
void CudaCache::setup_device(typename backend::stream_t stream) {
    cache_query_topology();
    m_sm_count = g_cache_sm_count;
    size_t N          = m_working_set_elements;
    size_t buf_count  = 2ULL * N + 1024;
    CHECK_CUDA(cudaMallocAsync(&m_device_buffer_a, buf_count * sizeof(cache_dtype), stream));
    CHECK_CUDA(cudaMallocAsync(&m_device_buffer_b, buf_count * sizeof(cache_dtype), stream));
    initKernel_wl<<<52, 256, 0, stream>>>(m_device_buffer_a, buf_count);
    initKernel_wl<<<52, 256, 0, stream>>>(m_device_buffer_b, buf_count);
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <>
auto CudaCache::run(typename backend::stream_t stream) -> std::monostate {
    launch_sum_kernel_wl(m_working_set_elements, m_device_buffer_a, m_device_buffer_b, stream);
    return {};
}

template <>
auto CudaCache::number_of_bytes() -> std::optional<size_t> {
    cache_query_topology();
    size_t N     = m_working_set_elements;
    int    iters = cache_get_iters(N);
    return 2ULL * N * sizeof(cache_dtype) * g_cache_sm_count * iters;
}

template <>
void CudaCache::fetch_results(typename backend::stream_t stream) {
    if (m_device_buffer_a) {
        CHECK_CUDA(cudaFreeAsync(m_device_buffer_a, stream));
        m_device_buffer_a = nullptr;
    }
    if (m_device_buffer_b) {
        CHECK_CUDA(cudaFreeAsync(m_device_buffer_b, stream));
        m_device_buffer_b = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

BASELINER_REGISTER_WORKLOAD(CudaCache);
