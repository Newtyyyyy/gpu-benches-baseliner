#include "../GpuIncoreWorkload.hpp"
#include <adapters/NVBenchRegister.hpp>
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

template <typename T, int N, int M>
__global__ void incore_FMA_mixed(T p, T* A, int iters) {
#pragma unroll(1)
    for (int iter = 0; iter < iters; iter++) {
        T t[M];
#pragma unroll
        for (int m = 0; m < M; m++) t[m] = p + threadIdx.x + iter + m;
#pragma unroll
        for (int n = 0; n < N / M; n++) {
#pragma unroll
            for (int m = 0; m < M; m++) t[m] = t[m] * (T)0.9 + (T)0.5;
        }
#pragma unroll
        for (int m = 0; m < M; m++) if (t[m] > (T)22313.0) A[0] = t[m];
    }
}

template <typename T, int N, int M>
__global__ void incore_FMA_separated(T p, T* A, int iters) {
    for (int iter = 0; iter < iters; iter++) {
#pragma unroll
        for (int m = 0; m < M; m++) {
            T t = p + threadIdx.x + iter + m;
            for (int n = 0; n < N; n++) t = t * (T)0.9 + (T)0.5;
            if (t > (T)22313.0) A[0] = t;
        }
    }
}

template <typename T, int N, int M>
__global__ void incore_DIV_separated(T p, T* A, int iters) {
#pragma unroll(1)
    for (int iter = 0; iter < iters; iter++) {
        for (int m = 0; m < M; m++) {
            T t = p + threadIdx.x + iter + m;
            for (int n = 0; n < N; n++) t = (T)0.1 / (t + (T)0.2);
            A[threadIdx.x + iter] = t;
        }
    }
}

template <typename T, int N, int M>
__global__ void incore_SQRT_separated(T p, T* A, int iters) {
#pragma unroll(1)
    for (int iter = 0; iter < iters; iter++) {
        for (int m = 0; m < M; m++) {
            T t = p + threadIdx.x + iter + m;
            for (int n = 0; n < N; n++) t = sqrt(t + (T)0.2);
            A[threadIdx.x + iter] = t;
        }
    }
}

// ---------------------------------------------------------------------------
// IWorkload specializations
// ---------------------------------------------------------------------------

using CudaIncore = GpuIncoreWorkload<Baseliner::Hardware::CudaBackend>;

template <>
void CudaIncore::setup_device(typename backend::stream_t stream) {
    int    block_size = 32 * m_warp_count;
    size_t elem_size  = (m_precision == "double") ? sizeof(double) : sizeof(float);
    size_t buf_count  = static_cast<size_t>(block_size + ITERS);
    CHECK_CUDA(cudaMallocAsync(&m_d_output, buf_count * elem_size, stream));
    CHECK_CUDA(cudaMemsetAsync(m_d_output, 0, buf_count * elem_size, stream));
}

template <>
auto CudaIncore::run(typename backend::stream_t stream) -> std::monostate {
    int block_size = 32 * m_warp_count;
    constexpr int block_count = 1;

    if (m_precision == "double") {
        if (m_kernel_type == "fma-mixed") {
            if      (m_ilp == 1) incore_FMA_mixed<double, N_FMA, 1><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else if (m_ilp == 2) incore_FMA_mixed<double, N_FMA, 2><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else if (m_ilp == 4) incore_FMA_mixed<double, N_FMA, 4><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else                 incore_FMA_mixed<double, N_FMA, 8><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
        } else if (m_kernel_type == "fma-separated") {
            if      (m_ilp == 1) incore_FMA_separated<double, N_FMA, 1><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else if (m_ilp == 2) incore_FMA_separated<double, N_FMA, 2><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else if (m_ilp == 4) incore_FMA_separated<double, N_FMA, 4><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else                 incore_FMA_separated<double, N_FMA, 8><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
        } else if (m_kernel_type == "div") {
            if      (m_ilp == 1) incore_DIV_separated<double, N_OTHER, 1><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else if (m_ilp == 2) incore_DIV_separated<double, N_OTHER, 2><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else if (m_ilp == 4) incore_DIV_separated<double, N_OTHER, 4><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else                 incore_DIV_separated<double, N_OTHER, 8><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
        } else {
            if      (m_ilp == 1) incore_SQRT_separated<double, N_OTHER, 1><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else if (m_ilp == 2) incore_SQRT_separated<double, N_OTHER, 2><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else if (m_ilp == 4) incore_SQRT_separated<double, N_OTHER, 4><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
            else                 incore_SQRT_separated<double, N_OTHER, 8><<<block_count, block_size, 0, stream>>>((double)0.32, (double*)m_d_output, ITERS);
        }
    } else {
        if (m_kernel_type == "fma-mixed") {
            if      (m_ilp == 1) incore_FMA_mixed<float, N_FMA, 1><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else if (m_ilp == 2) incore_FMA_mixed<float, N_FMA, 2><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else if (m_ilp == 4) incore_FMA_mixed<float, N_FMA, 4><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else                 incore_FMA_mixed<float, N_FMA, 8><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
        } else if (m_kernel_type == "fma-separated") {
            if      (m_ilp == 1) incore_FMA_separated<float, N_FMA, 1><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else if (m_ilp == 2) incore_FMA_separated<float, N_FMA, 2><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else if (m_ilp == 4) incore_FMA_separated<float, N_FMA, 4><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else                 incore_FMA_separated<float, N_FMA, 8><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
        } else if (m_kernel_type == "div") {
            if      (m_ilp == 1) incore_DIV_separated<float, N_OTHER, 1><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else if (m_ilp == 2) incore_DIV_separated<float, N_OTHER, 2><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else if (m_ilp == 4) incore_DIV_separated<float, N_OTHER, 4><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else                 incore_DIV_separated<float, N_OTHER, 8><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
        } else {
            if      (m_ilp == 1) incore_SQRT_separated<float, N_OTHER, 1><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else if (m_ilp == 2) incore_SQRT_separated<float, N_OTHER, 2><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else if (m_ilp == 4) incore_SQRT_separated<float, N_OTHER, 4><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
            else                 incore_SQRT_separated<float, N_OTHER, 8><<<block_count, block_size, 0, stream>>>((float)0.32, (float*)m_d_output, ITERS);
        }
    }
    return {};
}

template <>
void CudaIncore::fetch_results(typename backend::stream_t stream) {
    if (m_d_output) {
        CHECK_CUDA(cudaFreeAsync(m_d_output, stream));
        m_d_output = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

NVBENCH_REGISTER_WORKLOAD_AXES(CudaIncore);
