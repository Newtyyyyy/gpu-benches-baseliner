#include "../GpuMemcpyWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <cstring>

using HipMemcpy = GpuMemcpyWorkload<Baseliner::Hardware::HipBackend>;

template <>
void HipMemcpy::setup_device(typename backend::stream_t stream) {
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer), m_item_count, stream));
    if (m_pin_memory)
        CHECK_HIP(hipHostMalloc(&m_host_buffer, m_item_count));
    else
        m_host_buffer = new char[m_item_count];
}

template <>
void HipMemcpy::reset_device(typename backend::stream_t /*stream*/) {
    std::memset(m_host_buffer, 0, m_item_count);
}

template <>
auto HipMemcpy::run(typename backend::stream_t stream) -> std::monostate {
    CHECK_HIP(hipMemcpyAsync(m_device_buffer, m_host_buffer,
                               m_item_count, hipMemcpyDefault, stream));
    return {};
}

template <>
void HipMemcpy::fetch_results(typename backend::stream_t /*stream*/) {
    if (m_device_buffer) {
        hipFree(m_device_buffer);
        m_device_buffer = nullptr;
    }
}

template <>
void HipMemcpy::free() {
    if (m_host_buffer) {
        if (m_pin_memory)
            hipHostFree(m_host_buffer);
        else
            delete[] m_host_buffer;
        m_host_buffer = nullptr;
    }
}

BASELINER_REGISTER_WORKLOAD(HipMemcpy);
