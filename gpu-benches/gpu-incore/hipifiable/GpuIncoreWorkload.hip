#include "hip/hip_runtime.h"
#include "../GpuIncoreWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <baseliner/core/stats/Stats.hpp>

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
// Kernel dispatch helpers
// ---------------------------------------------------------------------------

namespace {
    template <typename T, int N>
    void dispatch_ilp(const std::string& kernel_type, int ilp, int block_count, int block_size,
                      T* d_output, int iters, typename Baseliner::Hardware::HipBackend::stream_t stream) {
        if (kernel_type == "fma-mixed") {
            if      (ilp == 1) incore_FMA_mixed<T, N, 1><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else if (ilp == 2) incore_FMA_mixed<T, N, 2><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else if (ilp == 4) incore_FMA_mixed<T, N, 4><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else               incore_FMA_mixed<T, N, 8><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
        } else if (kernel_type == "fma-separated") {
            if      (ilp == 1) incore_FMA_separated<T, N, 1><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else if (ilp == 2) incore_FMA_separated<T, N, 2><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else if (ilp == 4) incore_FMA_separated<T, N, 4><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else               incore_FMA_separated<T, N, 8><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
        } else if (kernel_type == "div") {
            if      (ilp == 1) incore_DIV_separated<T, N, 1><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else if (ilp == 2) incore_DIV_separated<T, N, 2><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else if (ilp == 4) incore_DIV_separated<T, N, 4><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else               incore_DIV_separated<T, N, 8><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
        } else {
            if      (ilp == 1) incore_SQRT_separated<T, N, 1><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else if (ilp == 2) incore_SQRT_separated<T, N, 2><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else if (ilp == 4) incore_SQRT_separated<T, N, 4><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
            else               incore_SQRT_separated<T, N, 8><<<block_count, block_size, 0, stream>>>((T)0.32, d_output, iters);
        }
    }
}

// ---------------------------------------------------------------------------
// IWorkload specializations
// ---------------------------------------------------------------------------

using HipIncore = GpuIncoreWorkload<Baseliner::Hardware::HipBackend>;

template <>
void HipIncore::setup_device(typename backend::stream_t stream) {
    int    block_size = 32 * m_warp_count;
    size_t elem_size  = (m_precision == "double") ? sizeof(double) : sizeof(float);
    size_t buf_count  = static_cast<size_t>(block_size + ITERS);
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_d_output), buf_count * elem_size, stream));
    CHECK_HIP(hipMemsetAsync(m_d_output, 0, buf_count * elem_size, stream));
}

template <>
auto HipIncore::run(typename backend::stream_t stream) -> std::monostate {
    int block_size  = 32 * m_warp_count;
    constexpr int block_count = 1;
    bool is_special = (m_kernel_type == "div" || m_kernel_type == "sqrt");

    if (m_precision == "double") {
        is_special
            ? dispatch_ilp<double, N_OTHER>(m_kernel_type, m_ilp, block_count, block_size, (double*)m_d_output, ITERS, stream)
            : dispatch_ilp<double, N_FMA>  (m_kernel_type, m_ilp, block_count, block_size, (double*)m_d_output, ITERS, stream);
    } else {
        is_special
            ? dispatch_ilp<float, N_OTHER>(m_kernel_type, m_ilp, block_count, block_size, (float*)m_d_output, ITERS, stream)
            : dispatch_ilp<float, N_FMA>  (m_kernel_type, m_ilp, block_count, block_size, (float*)m_d_output, ITERS, stream);
    }
    return {};
}

template <>
void HipIncore::fetch_results(typename backend::stream_t stream) {
    if (m_d_output) {
        CHECK_HIP(hipFreeAsync(m_d_output, stream));
        m_d_output = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

BASELINER_REGISTER_WORKLOAD(HipIncore);
