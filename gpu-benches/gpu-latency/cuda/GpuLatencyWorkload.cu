#include "../GpuLatencyWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
#include <algorithm>
#include <random>
#include <vector>

static constexpr int LAT_CACHE_LINE_INTS = 128 / sizeof(int64_t);
static constexpr int LAT_SKIP_FACTOR     = 1;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <typename T>
__global__ void lat_pointer_chasing(T* buf, T* dummy_out, int64_t iterations) {
    T* idx = buf;
    const int UNROLL = 32;
#pragma unroll 1
    for (int64_t n = 0; n < iterations; n += UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) idx = (T*)*idx;
    }
    if (threadIdx.x + blockIdx.x > 12000) dummy_out[0] = (T)idx;
}

// ---------------------------------------------------------------------------
// Pointer-chain builder
// ---------------------------------------------------------------------------

static void lat_build_pointer_chain(int64_t* buf, int64_t chain_len) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::vector<int64_t> order(chain_len);
    for (int64_t i = 0; i < chain_len; i++) order[i] = i + 1;
    order[chain_len - 1] = 0;
    std::shuffle(order.begin(), order.end() - 1, rng);

    for (int cl = 0; cl < LAT_CACHE_LINE_INTS; cl++) {
        int64_t idx = 0;
        for (int64_t i = 0; i < chain_len; i++) {
            buf[(idx * LAT_CACHE_LINE_INTS + cl) * LAT_SKIP_FACTOR] =
                LAT_SKIP_FACTOR * (order[i] * LAT_CACHE_LINE_INTS + cl +
                                   (order[i] == 0 ? 1 : 0));
            idx = order[i];
        }
    }
    buf[LAT_SKIP_FACTOR *
        (order[chain_len - 2] * LAT_CACHE_LINE_INTS + LAT_CACHE_LINE_INTS - 1)] = 0;

    int64_t total = LAT_SKIP_FACTOR * LAT_CACHE_LINE_INTS * chain_len;
    for (int64_t n = 0; n < total; n++)
        buf[n] = (int64_t)(buf) + buf[n] * sizeof(int64_t);
}

// ---------------------------------------------------------------------------
// IWorkload specializations
// ---------------------------------------------------------------------------

using CudaLatency = GpuLatencyWorkload<Baseliner::Hardware::CudaBackend>;

template <>
void CudaLatency::setup_device(typename backend::stream_t /*stream*/) {
    size_t buf_bytes = m_buffer_size_kb * 1024;
    m_chain_length   = static_cast<int64_t>(
        buf_bytes / (LAT_SKIP_FACTOR * LAT_CACHE_LINE_INTS * sizeof(int64_t)));
    if (m_chain_length < 2) m_chain_length = 16;
    m_iterations = std::max(m_chain_length, (int64_t)100000);
    CHECK_CUDA(cudaMallocManaged(&m_device_buffer, buf_bytes));
    CHECK_CUDA(cudaMallocManaged(&m_dummy_buffer, sizeof(int64_t)));
    lat_build_pointer_chain(m_device_buffer, m_chain_length);
}

template <>
void CudaLatency::reset_device(typename backend::stream_t stream) {
    // Two warm-up passes to prime the cache state before the timed run
    lat_pointer_chasing<int64_t>
        <<<1, 1, 0, stream>>>(m_device_buffer, m_dummy_buffer, m_iterations);
    lat_pointer_chasing<int64_t>
        <<<1, 1, 0, stream>>>(m_device_buffer, m_dummy_buffer, m_iterations);
}

template <>
auto CudaLatency::run(typename backend::stream_t stream) -> std::monostate {
    lat_pointer_chasing<int64_t>
        <<<1, 1, 0, stream>>>(m_device_buffer, m_dummy_buffer, m_iterations);
    return {};
}

template <>
void CudaLatency::free() {
    if (m_device_buffer) { cudaFree(m_device_buffer); m_device_buffer = nullptr; }
    if (m_dummy_buffer)  { cudaFree(m_dummy_buffer);  m_dummy_buffer  = nullptr; }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

BASELINER_REGISTER_WORKLOAD(CudaLatency);
