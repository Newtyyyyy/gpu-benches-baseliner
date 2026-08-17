#include "hip/hip_runtime.h"
#include "../GpuL2StreamWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void l2s_initKernel(double* A, size_t N) {
    size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (size_t idx = tidx; idx < N; idx += blockDim.x * gridDim.x)
        A[idx] = 0.23;
}

__global__ void l2s_read(double* A, const double* __restrict__ B, size_t length) {
    size_t tidx = (threadIdx.x + (size_t)blockIdx.x * blockDim.x) % length;
    double temp = B[tidx];
    if (temp == -1.0) // never true; keeps the load from being optimized away
        A[tidx] = temp;
}

__global__ void l2s_write(double* A, size_t length) {
    size_t tidx = (threadIdx.x + (size_t)blockIdx.x * blockDim.x) % length;
    A[tidx] = 0.23;
}

__global__ void l2s_scale(double* A, const double* __restrict__ B, size_t length) {
    size_t tidx = (threadIdx.x + (size_t)blockIdx.x * blockDim.x) % length;
    A[tidx] = B[tidx] * 1.2;
}

__global__ void l2s_triad(double* A, const double* __restrict__ B,
                          const double* __restrict__ C, const double* __restrict__ D,
                          size_t length) {
    size_t tidx = (threadIdx.x + (size_t)blockIdx.x * blockDim.x) % length;
    A[tidx] = B[tidx] * D[tidx] + C[tidx];
}

// ---------------------------------------------------------------------------
// IWorkload specializations
// ---------------------------------------------------------------------------

using HipL2Stream = GpuL2StreamWorkload<Baseliner::Hardware::HipBackend>;

template <>
void HipL2Stream::setup_device(typename backend::stream_t stream) {
    size_t bytes = m_length * sizeof(double);
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_a), bytes, stream));
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_b), bytes, stream));
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_c), bytes, stream));
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_d), bytes, stream));
    int grid = static_cast<int>(m_length / 1024) + 1;
    l2s_initKernel<<<grid, 1024, 0, stream>>>(m_device_buffer_a, m_length);
    l2s_initKernel<<<grid, 1024, 0, stream>>>(m_device_buffer_b, m_length);
    l2s_initKernel<<<grid, 1024, 0, stream>>>(m_device_buffer_c, m_length);
    l2s_initKernel<<<grid, 1024, 0, stream>>>(m_device_buffer_d, m_length);
    CHECK_HIP(hipStreamSynchronize(stream));
}

template <>
auto HipL2Stream::run(typename backend::stream_t stream) -> std::monostate {
    int grid = static_cast<int>(ITERATION_COUNT / BLOCKSIZE) + 1;
    if (m_kernel_type == "read") {
        l2s_read<<<grid, BLOCKSIZE, 0, stream>>>(m_device_buffer_a, m_device_buffer_b, m_length);
    } else if (m_kernel_type == "write") {
        l2s_write<<<grid, BLOCKSIZE, 0, stream>>>(m_device_buffer_a, m_length);
    } else if (m_kernel_type == "scale") {
        l2s_scale<<<grid, BLOCKSIZE, 0, stream>>>(m_device_buffer_a, m_device_buffer_b, m_length);
    } else {
        l2s_triad<<<grid, BLOCKSIZE, 0, stream>>>(m_device_buffer_a, m_device_buffer_b,
                                                   m_device_buffer_c, m_device_buffer_d, m_length);
    }
    return {};
}

template <>
void HipL2Stream::fetch_results(typename backend::stream_t stream) {
    if (m_device_buffer_a) { CHECK_HIP(hipFreeAsync(m_device_buffer_a, stream)); m_device_buffer_a = nullptr; }
    if (m_device_buffer_b) { CHECK_HIP(hipFreeAsync(m_device_buffer_b, stream)); m_device_buffer_b = nullptr; }
    if (m_device_buffer_c) { CHECK_HIP(hipFreeAsync(m_device_buffer_c, stream)); m_device_buffer_c = nullptr; }
    if (m_device_buffer_d) { CHECK_HIP(hipFreeAsync(m_device_buffer_d, stream)); m_device_buffer_d = nullptr; }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

BASELINER_REGISTER_WORKLOAD(HipL2Stream);
