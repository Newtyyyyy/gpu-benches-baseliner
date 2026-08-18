#include "hip/hip_runtime.h"
#include "../GpuRooflineWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <stdexcept>
#include <string>

template <typename T, int N, int M, int BLOCKSIZE>
__global__ void roofline_testfun(T* const __restrict__ dA, T* const __restrict__ dB, T* dC) {
    T* sA = dA + threadIdx.x + blockIdx.x * BLOCKSIZE * M;
    T* sB = dB + threadIdx.x + blockIdx.x * BLOCKSIZE * M;

    T sum = 0;

    for (int i = 0; i < M; i += 2) {
        T a  = sA[i * BLOCKSIZE];
        T b  = sB[i * BLOCKSIZE];
        T v  = a - b;
        T a2 = sA[(i + 1) * BLOCKSIZE];
        T b2 = sB[(i + 1) * BLOCKSIZE];
        T v2 = a2 - b2;
        for (int n = 0; n < N; n++) {
            v  = v * a - b;
            v2 = v2 * a - b;
        }
        sum += v + v2;
    }
    if (threadIdx.x == 0)
        dC[blockIdx.x] = sum;
}

template <typename T>
__global__ void roofline_initKernel(T* data, size_t data_len) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tidx; idx < data_len; idx += gridDim.x * blockDim.x)
        data[idx] = (T)idx;
}

using HipRoofline = GpuRooflineWorkload<Baseliner::Hardware::HipBackend>;
static constexpr int ROOFLINE_M         = HipRoofline::M;
static constexpr int ROOFLINE_BLOCKSIZE = HipRoofline::BLOCKSIZE;

namespace {
    auto roofline_block_count(int n, int sm_count) -> int {
        int maxActiveBlocks = 0;
#define DISPATCH_N(N_VAL)                                                                     \
    case N_VAL:                                                                              \
        CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(                             \
            &maxActiveBlocks, roofline_testfun<float, N_VAL, ROOFLINE_M, ROOFLINE_BLOCKSIZE>,  \
            ROOFLINE_BLOCKSIZE, 0));                                                          \
        break;

        switch (n) {
        DISPATCH_N(0) DISPATCH_N(1) DISPATCH_N(2) DISPATCH_N(4) DISPATCH_N(6) DISPATCH_N(8)
        DISPATCH_N(10) DISPATCH_N(12) DISPATCH_N(14) DISPATCH_N(16) DISPATCH_N(18) DISPATCH_N(20)
        DISPATCH_N(22) DISPATCH_N(24) DISPATCH_N(28) DISPATCH_N(32) DISPATCH_N(36) DISPATCH_N(40)
        DISPATCH_N(44) DISPATCH_N(48) DISPATCH_N(54) DISPATCH_N(60) DISPATCH_N(66) DISPATCH_N(72)
        DISPATCH_N(80) DISPATCH_N(88) DISPATCH_N(96) DISPATCH_N(106) DISPATCH_N(116) DISPATCH_N(126)
        DISPATCH_N(138) DISPATCH_N(150) DISPATCH_N(164) DISPATCH_N(178) DISPATCH_N(194) DISPATCH_N(212)
        DISPATCH_N(230) DISPATCH_N(250) DISPATCH_N(272) DISPATCH_N(296) DISPATCH_N(322) DISPATCH_N(350)
        DISPATCH_N(380) DISPATCH_N(412) DISPATCH_N(448) DISPATCH_N(486) DISPATCH_N(528) DISPATCH_N(574)
        DISPATCH_N(622) DISPATCH_N(674) DISPATCH_N(732) DISPATCH_N(794) DISPATCH_N(862) DISPATCH_N(934)
        DISPATCH_N(1012)
        default:
            throw std::runtime_error("roofline: unsupported n value: " + std::to_string(n));
        }
#undef DISPATCH_N
        return sm_count * maxActiveBlocks;
    }

    void roofline_launch(int n, float* dA, float* dB, float* dC, int block_count,
                         typename Baseliner::Hardware::HipBackend::stream_t stream) {
#define DISPATCH_N(N_VAL)                                                                     \
    case N_VAL:                                                                              \
        roofline_testfun<float, N_VAL, ROOFLINE_M, ROOFLINE_BLOCKSIZE>                        \
            <<<block_count, ROOFLINE_BLOCKSIZE, 0, stream>>>(dA, dB, dC);                     \
        break;

        switch (n) {
        DISPATCH_N(0) DISPATCH_N(1) DISPATCH_N(2) DISPATCH_N(4) DISPATCH_N(6) DISPATCH_N(8)
        DISPATCH_N(10) DISPATCH_N(12) DISPATCH_N(14) DISPATCH_N(16) DISPATCH_N(18) DISPATCH_N(20)
        DISPATCH_N(22) DISPATCH_N(24) DISPATCH_N(28) DISPATCH_N(32) DISPATCH_N(36) DISPATCH_N(40)
        DISPATCH_N(44) DISPATCH_N(48) DISPATCH_N(54) DISPATCH_N(60) DISPATCH_N(66) DISPATCH_N(72)
        DISPATCH_N(80) DISPATCH_N(88) DISPATCH_N(96) DISPATCH_N(106) DISPATCH_N(116) DISPATCH_N(126)
        DISPATCH_N(138) DISPATCH_N(150) DISPATCH_N(164) DISPATCH_N(178) DISPATCH_N(194) DISPATCH_N(212)
        DISPATCH_N(230) DISPATCH_N(250) DISPATCH_N(272) DISPATCH_N(296) DISPATCH_N(322) DISPATCH_N(350)
        DISPATCH_N(380) DISPATCH_N(412) DISPATCH_N(448) DISPATCH_N(486) DISPATCH_N(528) DISPATCH_N(574)
        DISPATCH_N(622) DISPATCH_N(674) DISPATCH_N(732) DISPATCH_N(794) DISPATCH_N(862) DISPATCH_N(934)
        DISPATCH_N(1012)
        default:
            throw std::runtime_error("roofline: unsupported n value: " + std::to_string(n));
        }
#undef DISPATCH_N
    }
}

template <>
void HipRoofline::setup_device(typename backend::stream_t stream) {
    int dev;
    CHECK_HIP(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, dev));
    m_sm_count = prop.multiProcessorCount;

    m_block_count = roofline_block_count(m_n, m_sm_count);
    m_data_len    = static_cast<size_t>(m_block_count) * BLOCKSIZE * M;

    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_a), m_data_len * sizeof(float), stream));
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_b), m_data_len * sizeof(float), stream));
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_c), m_data_len * sizeof(float), stream));
    roofline_initKernel<<<m_block_count, 256, 0, stream>>>(m_device_buffer_a, m_data_len);
    roofline_initKernel<<<m_block_count, 256, 0, stream>>>(m_device_buffer_b, m_data_len);
    roofline_initKernel<<<m_block_count, 256, 0, stream>>>(m_device_buffer_c, m_data_len);
    CHECK_HIP(hipStreamSynchronize(stream));
}

template <>
auto HipRoofline::run(typename backend::stream_t stream) -> std::monostate {
    roofline_launch(m_n, m_device_buffer_a, m_device_buffer_b, m_device_buffer_c, m_block_count,
                    stream);
    return {};
}

template <>
void HipRoofline::fetch_results(typename backend::stream_t stream) {
    if (m_device_buffer_a) { CHECK_HIP(hipFreeAsync(m_device_buffer_a, stream)); m_device_buffer_a = nullptr; }
    if (m_device_buffer_b) { CHECK_HIP(hipFreeAsync(m_device_buffer_b, stream)); m_device_buffer_b = nullptr; }
    if (m_device_buffer_c) { CHECK_HIP(hipFreeAsync(m_device_buffer_c, stream)); m_device_buffer_c = nullptr; }
}

BASELINER_REGISTER_WORKLOAD(HipRoofline);
