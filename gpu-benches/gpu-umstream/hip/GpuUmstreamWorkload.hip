#include "../GpuUmstreamWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>

namespace {
__global__ void umstream_triad(double* A, double* B, double* C, size_t N) {
    size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x)
        A[i] = B[i] + C[i] * 1.3;
}
} // namespace

using HipUmstream = GpuUmstreamWorkload<Baseliner::Hardware::HipBackend>;

template <>
void HipUmstream::setup_device(typename backend::stream_t stream) {
    size_t nb_bytes = m_item_count * sizeof(double);
    CHECK_HIP(hipMallocManaged(&m_A, nb_bytes));
    CHECK_HIP(hipMallocManaged(&m_B, nb_bytes));
    CHECK_HIP(hipMallocManaged(&m_C, nb_bytes));

    int device_id;
    CHECK_HIP(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, device_id));

    int max_active_blocks = 0;
    CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, umstream_triad, m_block_size, 0));
    m_block_count = prop.multiProcessorCount * max_active_blocks;

    if (m_prefetch) {
        CHECK_HIP(hipMemPrefetchAsync(m_A, nb_bytes, device_id, stream));
        CHECK_HIP(hipMemPrefetchAsync(m_B, nb_bytes, device_id, stream));
        CHECK_HIP(hipMemPrefetchAsync(m_C, nb_bytes, device_id, stream));
    }
}

template <>
auto HipUmstream::run(typename backend::stream_t stream) -> std::monostate {
    umstream_triad<<<m_block_count, m_block_size, 0, stream>>>(m_A, m_B, m_C, m_item_count);
    return {};
}

template <>
void HipUmstream::free() {
    if (m_A) { hipFree(m_A); m_A = nullptr; }
    if (m_B) { hipFree(m_B); m_B = nullptr; }
    if (m_C) { hipFree(m_C); m_C = nullptr; }
}

BASELINER_REGISTER_WORKLOAD(HipUmstream);
