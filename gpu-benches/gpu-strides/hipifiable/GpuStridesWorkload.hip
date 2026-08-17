#include "hip/hip_runtime.h"
#include "../GpuStridesWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <array>
#include <string>

// CUPTI/Perfworks is NVIDIA-only; there is no ROCm equivalent here. It is
// enabled only when the HIP build defines STRIDES_ENABLE_CUPTI (see
// hip/CMakeLists.txt). The mechanically-hipified variant is compiled WITHOUT
// that define — even under HIP-on-NVIDIA, where __HIP__ is not defined — so it
// simply reports 0 for the three L1/L2 counters (the hand-tuned hip/ port can
// add rocprofiler). An explicit build flag is used rather than a compiler
// macro because none reliably distinguishes native-HIP from HIP-on-NVIDIA.
#ifdef STRIDES_ENABLE_CUPTI
#include "cupti/measureMetricPW.hpp"   // L1/L2 hardware counters (ported from RRZE gpu-metrics)
#endif

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

template <typename T>
__global__ void strides_initKernel(T* A, size_t N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x)
        A[i] = T(0.1);
}

template <typename T, int N>
__global__ __launch_bounds__(1024) void strides_blockKernel(T* A, T* B, int zero, int pitch) {
    int tidx = threadIdx.x % 64;
    int tidy = threadIdx.y % 32;

    T sum = T(0);
    T* A2 = A + tidx + tidy * pitch + pitch;

#pragma unroll 1
    for (int n = 0; n < N; n += 8) {
        A2 += zero;
        sum += A2[0] * A2[4];
        sum += A2[8] * A2[12];
        sum += A2[20] * A2[16];
        sum += A2[24] * A2[28];
    }
    if (sum == T(123123.23))
        B[tidx] = sum;
}

template <typename T, int N>
__global__ void strides_strideKernel(T* A, T* B, int zero, int stride) {
    int tidx = threadIdx.x;

    T sum = T(0);
    T* A2 = A + (tidx % 64) * stride;

#pragma unroll 1
    for (int n = 0; n < N; n += 8) {
        A2 += zero;
        sum += A2[0] * A2[4];
        sum += A2[8] * A2[12];
        sum += A2[20] * A2[16];
        sum += A2[24] * A2[28];
    }

    if (sum == T(123123.23))
        B[tidx] = sum;
}

// ---------------------------------------------------------------------------
// IWorkload specializations
// ---------------------------------------------------------------------------

using HipStrides = GpuStridesWorkload<Baseliner::Hardware::HipBackend>;

namespace {
    auto divisor_of_1024_at_most(int w) -> int {
        if (w < 1) w = 1;
        if (w > 1024) w = 1024;
        while (1024 % w != 0) w--;
        return w;
    }

    constexpr int STRIDES_PITCH = 4098;

    // Single kernel launch, shared by run() and the CUPTI-profiled launches so
    // both measure the exact same work. Async on the given stream.
    void strides_dispatch(const std::string& kernel_type, const std::string& precision, int arg,
                          void* bufA, void* bufB, hipStream_t stream) {
        constexpr int N = HipStrides::N;
        if (kernel_type == "block") {
            int w = divisor_of_1024_at_most(arg == 0 ? 1 : arg);
            dim3 block(w, 1024 / w, 1);
            if (precision == "double")
                strides_blockKernel<double, N><<<1, block, 0, stream>>>((double*)bufA, (double*)bufB, 0, STRIDES_PITCH);
            else
                strides_blockKernel<float, N><<<1, block, 0, stream>>>((float*)bufA, (float*)bufB, 0, STRIDES_PITCH);
        } else {
            dim3 block(1024, 1, 1);
            if (precision == "double")
                strides_strideKernel<double, N><<<1, block, 0, stream>>>((double*)bufA, (double*)bufB, 0, arg);
            else
                strides_strideKernel<float, N><<<1, block, 0, stream>>>((float*)bufA, (float*)bufB, 0, arg);
        }
    }
} // namespace

template <>
void HipStrides::setup_device(typename backend::stream_t stream) {
    size_t elem_size = (m_precision == "double") ? sizeof(double) : sizeof(float);
    size_t bytes = static_cast<size_t>(BUFFER_ELEMS) * elem_size;
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_a), bytes, stream));
    CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&m_device_buffer_b), bytes, stream));
    if (m_precision == "double") {
        strides_initKernel<double><<<256, 400, 0, stream>>>((double*)m_device_buffer_a, BUFFER_ELEMS);
        strides_initKernel<double><<<256, 400, 0, stream>>>((double*)m_device_buffer_b, BUFFER_ELEMS);
    } else {
        strides_initKernel<float><<<256, 400, 0, stream>>>((float*)m_device_buffer_a, BUFFER_ELEMS);
        strides_initKernel<float><<<256, 400, 0, stream>>>((float*)m_device_buffer_b, BUFFER_ELEMS);
    }
    CHECK_HIP(hipStreamSynchronize(stream));

    // --- CUPTI L1/L2 hardware counters (matches the original gpu-strides) ---
    // Each counter needs its own profiled kernel launch (CUPTI replay), so this
    // is slow — the same cost the original pays. Values are normalized per
    // warp-instruction, exactly as upstream does. Skipped on non-CUPTI backends.
#ifdef STRIDES_ENABLE_CUPTI
    static bool s_metric_inited = false;
    if (!s_metric_inited) { initMeasureMetric(); s_metric_inited = true; }

    static const std::array<const char*, 3> METRIC_NAMES = {
        "l1tex__data_pipe_lsu_wavefronts.sum",
        "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum",
        "lts__t_sectors.sum"};

    for (size_t i = 0; i < METRIC_NAMES.size(); i++) {
        measureMetricsStart({METRIC_NAMES[i]});
        strides_dispatch(m_kernel_type, m_precision, m_arg,
                         m_device_buffer_a, m_device_buffer_b, stream);
        CHECK_HIP(hipStreamSynchronize(stream));
        auto res = measureMetricsStop();
        m_hw_metrics[i] = res.empty() ? 0.0 : res[0] / METRIC_NORM;
    }
#endif
}

template <>
auto HipStrides::run(typename backend::stream_t stream) -> std::monostate {
    strides_dispatch(m_kernel_type, m_precision, m_arg,
                     m_device_buffer_a, m_device_buffer_b, stream);
    return {};
}

template <>
void HipStrides::fetch_results(typename backend::stream_t stream) {
    if (m_device_buffer_a) { CHECK_HIP(hipFreeAsync(m_device_buffer_a, stream)); m_device_buffer_a = nullptr; }
    if (m_device_buffer_b) { CHECK_HIP(hipFreeAsync(m_device_buffer_b, stream)); m_device_buffer_b = nullptr; }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

BASELINER_REGISTER_WORKLOAD(HipStrides);
