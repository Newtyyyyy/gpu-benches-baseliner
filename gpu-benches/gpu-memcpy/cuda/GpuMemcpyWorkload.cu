#include "../GpuMemcpyWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
#include <cstring>

using CudaMemcpy = GpuMemcpyWorkload<Baseliner::Hardware::CudaBackend>;

template <>
void CudaMemcpy::setup_device(typename backend::stream_t stream) {
    CHECK_CUDA(cudaMallocAsync(&m_device_buffer, m_item_count, stream));
    if (m_pin_memory)
        CHECK_CUDA(cudaMallocHost(&m_host_buffer, m_item_count));
    else
        m_host_buffer = new char[m_item_count];
}

template <>
void CudaMemcpy::reset_device(typename backend::stream_t ) {
    std::memset(m_host_buffer, 0, m_item_count);
}

template <>
auto CudaMemcpy::run(typename backend::stream_t stream) -> std::monostate {
    CHECK_CUDA(cudaMemcpyAsync(m_device_buffer, m_host_buffer,
                               m_item_count, cudaMemcpyDefault, stream));
    return {};
}

template <>
void CudaMemcpy::fetch_results(typename backend::stream_t ) {
    if (m_device_buffer) {
        cudaFree(m_device_buffer);
        m_device_buffer = nullptr;
    }
}

template <>
void CudaMemcpy::free() {
    if (m_host_buffer) {
        if (m_pin_memory)
            cudaFreeHost(m_host_buffer);
        else
            delete[] m_host_buffer;
        m_host_buffer = nullptr;
    }
}

BASELINER_REGISTER_WORKLOAD(CudaMemcpy);
