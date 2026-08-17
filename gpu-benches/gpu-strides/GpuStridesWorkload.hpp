#pragma once
#include <baseliner/core/Workload.hpp>
#include <baseliner/core/stats/IStats.hpp>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>

// Hardware-counter metrics measured via CUPTI (CUDA backend only), matching
// the three L1/L2 counters the original RRZE gpu-strides reports, normalized
// per warp-instruction. On backends without CUPTI they stay 0.
class StridesL1LsuWavefronts : public Baseliner::Stats::Imetric<StridesL1LsuWavefronts, float> {
public:
    using Baseliner::Stats::Imetric<StridesL1LsuWavefronts, float>::Imetric;
    [[nodiscard]] auto name() const -> std::string override { return "l1_lsu_wavefronts_per_warp"; }
    [[nodiscard]] auto unit() const -> std::string override { return ""; }
    [[nodiscard]] auto granularity() const -> Baseliner::MetricGranularity override {
        return Baseliner::MetricGranularity::ONCE;
    }
};

class StridesL1LdWavefronts : public Baseliner::Stats::Imetric<StridesL1LdWavefronts, float> {
public:
    using Baseliner::Stats::Imetric<StridesL1LdWavefronts, float>::Imetric;
    [[nodiscard]] auto name() const -> std::string override { return "l1_ld_wavefronts_per_warp"; }
    [[nodiscard]] auto unit() const -> std::string override { return ""; }
    [[nodiscard]] auto granularity() const -> Baseliner::MetricGranularity override {
        return Baseliner::MetricGranularity::ONCE;
    }
};

class StridesL2Sectors : public Baseliner::Stats::Imetric<StridesL2Sectors, float> {
public:
    using Baseliner::Stats::Imetric<StridesL2Sectors, float>::Imetric;
    [[nodiscard]] auto name() const -> std::string override { return "l2_sectors_per_warp"; }
    [[nodiscard]] auto unit() const -> std::string override { return ""; }
    [[nodiscard]] auto granularity() const -> Baseliner::MetricGranularity override {
        return Baseliner::MetricGranularity::ONCE;
    }
};

// Ported from RRZE gpu-benches/gpu-strides: probes L1/register-file access
// patterns with two kernels, both launched as a single 1024-thread block:
//  - "stride": each of 64 distinct lanes repeatedly re-reads a small window
//    at `arg`-element stride (arg = 0..64).
//  - "block": a pitched 2D access pattern (blockDim = arg x 1024/arg),
//    fixed pitch, matching upstream's blockKernel sweep over warp counts.
// `arg` is shared by both kernel types for a single sweep option; for
// "block" it is rounded down to the nearest divisor of 1024.
template <typename BackendT>
class GpuStridesWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuStridesWorkload::backend;

    static constexpr int N = 18 * 64 * 1024;
    static constexpr int BUFFER_ELEMS = 1 << 20;

    auto algo() -> std::string override { return "gpu-strides"; }

    auto number_of_bytes() -> std::optional<size_t> override {
        size_t elem_size = (m_precision == "double") ? sizeof(double) : sizeof(float);
        return 1024ULL * N * elem_size;
    }

    void setup_host_random_generated() override {}
    void setup_device(typename backend::stream_t stream) override;
    void reset_device(typename backend::stream_t /*stream*/) override {}
    auto run(typename backend::stream_t stream) -> typename backend::launch_result_t override;
    void fetch_results(typename backend::stream_t stream) override;
    void free() override {}

    // Per-warp-instruction normalization factor, shared with the CUPTI
    // profiling code (1024 threads * N iterations / 32-lane warp).
    static constexpr double METRIC_NORM = 1024.0 * static_cast<double>(N) / 32.0;

    void inner_setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> engine) override {
        engine->register_metric<StridesL1LsuWavefronts>(static_cast<float>(m_hw_metrics[0]));
        engine->register_metric<StridesL1LdWavefronts>(static_cast<float>(m_hw_metrics[1]));
        engine->register_metric<StridesL2Sectors>(static_cast<float>(m_hw_metrics[2]));
    }

    void inner_update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> engine) override {
        engine->update_values<StridesL1LsuWavefronts>(static_cast<float>(m_hw_metrics[0]));
        engine->update_values<StridesL1LdWavefronts>(static_cast<float>(m_hw_metrics[1]));
        engine->update_values<StridesL2Sectors>(static_cast<float>(m_hw_metrics[2]));
    }

protected:
    void register_options() override {
        Baseliner::IWorkload<BackendT>::register_options();
        this->add_option("GpuStrides", "kernel_type",
                         "Access pattern: stride (strided re-reads) or block "
                         "(pitched 2D access)", m_kernel_type)
            .sweep({"stride", "block"});
        this->add_option("GpuStrides", "precision",
                         "Floating-point precision: float or double", m_precision)
            .sweep({"float", "double"});
        this->add_option("GpuStrides", "arg",
                         "Stride (kernel_type=stride) or warp count (kernel_type=block, "
                         "rounded down to a divisor of 1024)", m_arg)
            .sweep({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
                    "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26",
                    "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                    "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52",
                    "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64"});
    }

protected:
    void*       m_device_buffer_a = nullptr;
    void*       m_device_buffer_b = nullptr;
    std::string m_kernel_type     = "stride";
    std::string m_precision       = "float";
    int         m_arg             = 1;

    // L1/L2 hardware counters, normalized per warp-instruction. Filled by the
    // CUDA backend's CUPTI profiling in setup_device; 0 on other backends.
    std::array<double, 3> m_hw_metrics = {0.0, 0.0, 0.0};
};
