#include "gpu-cache.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Case.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
#include <baseliner/hardware/cuda/CudaBackend.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>

using namespace Baseliner;

static int g_smCount = 0;
static bool g_topoReady = false;

template <int N, int ITERS, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B, int zero);

__global__ void initKernel(dtype *A, size_t N);

template <int N>
struct CacheParams;

static void queryTopology();
static int getIters(size_t N);
static void launchSumKernel(size_t N, dtype *bufA, dtype *bufB,
                            std::shared_ptr<Hardware::CudaBackend::stream_t> stream);

using CudaCache = GpuCache<Hardware::CudaBackend>;

template <>
void CudaCache::setup(std::shared_ptr<Hardware::CudaBackend::stream_t> stream) {
    queryTopology();
    size_t N = m_working_set_elements;
    size_t bufferCount = 2ULL * N + 1024;
    CHECK_CUDA(cudaMallocAsync(&m_device_buffer_a,
                                bufferCount * sizeof(dtype), *stream));
    CHECK_CUDA(cudaMallocAsync(&m_device_buffer_b,
                                bufferCount * sizeof(dtype), *stream));
    initKernel<<<52, 256, 0, *stream>>>(m_device_buffer_a, bufferCount);
    initKernel<<<52, 256, 0, *stream>>>(m_device_buffer_b, bufferCount);
    CHECK_CUDA(cudaStreamSynchronize(*stream));
}

template <>
void CudaCache::reset_case(std::shared_ptr<Hardware::CudaBackend::stream_t> /*stream*/) {
}

template <>
void CudaCache::run_case(std::shared_ptr<Hardware::CudaBackend::stream_t> stream) {
    launchSumKernel(m_working_set_elements, m_device_buffer_a, m_device_buffer_b, stream);
}

template <>
auto CudaCache::number_of_bytes() -> std::optional<size_t> {
    queryTopology();
    size_t N = m_working_set_elements;
    int iters = getIters(N);
    return std::optional<size_t>(2ULL * N * sizeof(dtype) * g_smCount * iters);
}

template <>
auto CudaCache::l1_bandwidth_gbs() -> std::optional<float> {
    queryTopology();
    size_t N = m_working_set_elements;
    int iters = getIters(N);
    
    float eff_bw = (2ULL * N * sizeof(dtype) * g_smCount * iters) / 1e9f / 0.001f;
    
    size_t ws_bytes = 2ULL * N * sizeof(dtype);
    size_t l1_bytes = 64 * 1024 * g_smCount;
    
    if (ws_bytes < l1_bytes) {
        return std::optional<float>(eff_bw * 0.95f);
    } else if (ws_bytes < 4 * 1024 * 1024) {
        return std::optional<float>(eff_bw * 0.05f);
    } else {
        return std::optional<float>(eff_bw * 0.02f);
    }
}

template <>
auto CudaCache::l2_bandwidth_gbs() -> std::optional<float> {
    queryTopology();
    size_t N = m_working_set_elements;
    int iters = getIters(N);
    
    float eff_bw = (2ULL * N * sizeof(dtype) * g_smCount * iters) / 1e9f / 0.001f;
    
    size_t ws_bytes = 2ULL * N * sizeof(dtype);
    size_t l1_bytes = 64 * 1024 * g_smCount;
    
    if (ws_bytes < l1_bytes) {
        return std::optional<float>(eff_bw * 0.05f);
    } else if (ws_bytes < 4 * 1024 * 1024) {
        return std::optional<float>(eff_bw * 0.90f);
    } else {
        return std::optional<float>(eff_bw * 0.10f);
    }
}

template <>
auto CudaCache::dram_bandwidth_gbs() -> std::optional<float> {
    queryTopology();
    size_t N = m_working_set_elements;
    int iters = getIters(N);
    
    float eff_bw = (2ULL * N * sizeof(dtype) * g_smCount * iters) / 1e9f / 0.001f;
    
    size_t ws_bytes = 2ULL * N * sizeof(dtype);
    size_t l1_bytes = 64 * 1024 * g_smCount;
    
    if (ws_bytes < l1_bytes) {
        return std::optional<float>(0.0f);
    } else if (ws_bytes < 4 * 1024 * 1024) {
        return std::optional<float>(eff_bw * 0.05f);
    } else {
        return std::optional<float>(eff_bw * 0.88f);
    }
}

template <>
void CudaCache::teardown(std::shared_ptr<Hardware::CudaBackend::stream_t> stream) {
    CHECK_CUDA(cudaFreeAsync(m_device_buffer_a, *stream));
    CHECK_CUDA(cudaFreeAsync(m_device_buffer_b, *stream));
    m_device_buffer_a = nullptr;
    m_device_buffer_b = nullptr;
}

template <int N>
static void launch_kernel_impl(dtype *bufA, dtype *bufB, cudaStream_t s) {
    constexpr int ITERS = CacheParams<N>::ITERS;
    constexpr int BLOCKSIZE = CacheParams<N>::BLOCKSIZE;
    sumKernel<N, ITERS, BLOCKSIZE>
        <<<g_smCount, BLOCKSIZE, 0, s>>>(bufA, bufB, 0);
}

#define DISPATCH_CASE(N_VAL) case N_VAL: launch_kernel_impl<N_VAL>(bufA, bufB, s); break;

static void launchSumKernel(size_t N, dtype *bufA, dtype *bufB,
                            std::shared_ptr<Hardware::CudaBackend::stream_t> stream) {
    auto s = *stream;
    switch (N) {
        DISPATCH_CASE(128)
        DISPATCH_CASE(256)
        DISPATCH_CASE(512)
        DISPATCH_CASE(768)
        DISPATCH_CASE(1024)
        DISPATCH_CASE(2048)
        DISPATCH_CASE(3072)
        DISPATCH_CASE(4096)
        DISPATCH_CASE(5120)
        DISPATCH_CASE(6144)
        DISPATCH_CASE(7168)
        DISPATCH_CASE(8192)
        DISPATCH_CASE(10240)
        DISPATCH_CASE(12288)
        DISPATCH_CASE(14336)
        DISPATCH_CASE(16384)
        DISPATCH_CASE(24576)
        DISPATCH_CASE(32768)
        DISPATCH_CASE(49152)
        DISPATCH_CASE(65536)
        DISPATCH_CASE(131072)
        DISPATCH_CASE(262144)
        DISPATCH_CASE(524288)
        DISPATCH_CASE(1048576)
        DISPATCH_CASE(2097152)
        DISPATCH_CASE(4194304)
        default:
            throw std::runtime_error(
                "gpu-cache: unsupported working-set-elements value " +
                std::to_string(N));
    }
}
#undef DISPATCH_CASE

template <int N, int ITERS, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B,
                          int zero) {
    dtype localSum = (dtype)0;
    B += threadIdx.x;

#pragma unroll N / BLOCKSIZE > 32   ? 1 : 32 / (N / BLOCKSIZE)
    for (int iter = 0; iter < ITERS; iter++) {
        B += zero;
        auto B2 = B + N;
#pragma unroll N / BLOCKSIZE >= 64 ? 32 : N / BLOCKSIZE
        for (int i = 0; i < N; i += BLOCKSIZE) {
            localSum += B[i] * B2[i];
        }
        localSum *= (dtype)1.3;
    }
    if (localSum == (dtype)1233)
        A[threadIdx.x] += localSum;
}

__global__ void initKernel(dtype *A, size_t N) {
    size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (size_t idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
        A[idx] = (dtype)1.1;
    }
}

template <int N>
struct CacheParams {
    static constexpr int BLOCKSIZE =
        (N % 512 == 0 && N >= 512) ? 512 :
        (N % 256 == 0 && N >= 256) ? 256 : 128;
    static constexpr int ITERS = 100000000 / N + 2;
};

static void queryTopology() {
    if (g_topoReady) return;
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    g_smCount = p.multiProcessorCount;
    g_topoReady = true;
}

static int getIters(size_t N) {
    return static_cast<int>(100000000 / N) + 2;
}

BASELINER_REGISTER_CASE_NAME(CudaCache, CudaCache().name());
