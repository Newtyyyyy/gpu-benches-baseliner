#include "../GpuUmstreamWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>

__global__ void umstream_triad(double* A, double* B, double* C, size_t N) {
    size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x)
        A[i] = B[i] + C[i] * 1.3;
}

using CudaUmstream = GpuUmstreamWorkload<Baseliner::Hardware::CudaBackend>;

template <>
void CudaUmstream::setup_device(typename backend::stream_t stream) {
    size_t nb_bytes = m_item_count * sizeof(double);
    CHECK_CUDA(cudaMallocManaged(&m_A, nb_bytes));
    CHECK_CUDA(cudaMallocManaged(&m_B, nb_bytes));
    CHECK_CUDA(cudaMallocManaged(&m_C, nb_bytes));

    int device_id;
    CHECK_CUDA(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    int max_active_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, umstream_triad, m_block_size, 0));
    int dynamic_block_count = prop.multiProcessorCount * max_active_blocks;
    m_block_count = (m_block_count_override > 0) ? m_block_count_override : dynamic_block_count;

    if (m_prefetch) {
        // Note: cudaMemPrefetchAsync is deprecated in CUDA 12.x
        // Skipping prefetch for now
    }
}

template <>
void CudaUmstream::reset_device(typename backend::stream_t /*stream*/) {}

template <>
auto CudaUmstream::run(typename backend::stream_t stream) -> std::monostate {
    umstream_triad<<<m_block_count, m_block_size, 0, stream>>>(m_A, m_B, m_C, m_item_count);
    return {};
}

template <>
void CudaUmstream::free() {
    if (m_A) { cudaFree(m_A); m_A = nullptr; }
    if (m_B) { cudaFree(m_B); m_B = nullptr; }
    if (m_C) { cudaFree(m_C); m_C = nullptr; }
}

BASELINER_REGISTER_WORKLOAD(CudaUmstream);

namespace Baseliner::Stats {
    BASELINER_REGISTER_STAT(Spread);
}
