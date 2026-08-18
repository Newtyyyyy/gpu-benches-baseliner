#include "hip/hip_runtime.h"
#include "../GpuSmallKernelsWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>

__global__ void small_init_kernel(double* A, size_t N, double val) {
    size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (size_t idx = tidx; idx < N; idx += blockDim.x * gridDim.x)
        A[idx] = val;
}

__global__ void small_scale(double* __restrict__ A, const double* __restrict__ B, int size) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= size) return;
    A[tidx] = B[tidx] * 0.25;
}

using HipSmallKernels = GpuSmallKernelsWorkload<Baseliner::Hardware::HipBackend>;

template <>
void HipSmallKernels::setup_device(typename backend::stream_t stream) {
    size_t bytes = static_cast<size_t>(m_size) * sizeof(double);
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_a), bytes, stream));
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_b), bytes, stream));
    int grid = m_size / 1024 + 1;
    small_init_kernel<<<grid, 1024, 0, stream>>>(m_device_buffer_a, m_size, 0.23);
    small_init_kernel<<<grid, 1024, 0, stream>>>(m_device_buffer_b, m_size, 1.44);
    CHECK_HIP(hipStreamSynchronize(stream));
}

template <>
auto HipSmallKernels::run(typename backend::stream_t stream) -> std::monostate {
    int grid = m_size / m_block_size + 1;
    small_scale<<<grid, m_block_size, 0, stream>>>(m_device_buffer_a, m_device_buffer_b, m_size);
    return {};
}

template <>
void HipSmallKernels::fetch_results(typename backend::stream_t stream) {
    if (m_device_buffer_a) { CHECK_HIP(hipFreeAsync(m_device_buffer_a, stream)); m_device_buffer_a = nullptr; }
    if (m_device_buffer_b) { CHECK_HIP(hipFreeAsync(m_device_buffer_b, stream)); m_device_buffer_b = nullptr; }
}

BASELINER_REGISTER_WORKLOAD(HipSmallKernels);
