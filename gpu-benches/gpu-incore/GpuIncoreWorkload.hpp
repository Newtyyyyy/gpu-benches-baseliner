#pragma once
#include <baseliner/core/Workload.hpp>
#include <baseliner/core/hardware/BackendStats.hpp>
#include <baseliner/core/stats/Stats.hpp>
#include <cstddef>
#include <optional>
#include <string>

class InCoreOps : public Baseliner::Stats::IStat<InCoreOps, size_t> {
public:
    void calculate(size_t & /*val*/) override {}
    auto name() const -> std::string override { return "ops_per_run"; }
    auto unit() const -> std::string override { return "ops"; }
    auto granularity() const -> Baseliner::MetricGranularity override {
        return Baseliner::MetricGranularity::ON_DEMAND;
    }
};

template <typename BackendT>
class RcpThruStat : public Baseliner::Stats::IStat<RcpThruStat<BackendT>, float,
    Baseliner::Stats::Median, InCoreOps, Baseliner::Stats::ClockFrequency<BackendT>> {
public:
    void calculate(float &val, const float &median_ms, const size_t &ops, const float &clock_ghz) override {
        if (ops > 0 && median_ms > 0.0f && clock_ghz > 0.0f) {
            val = static_cast<float>(
                (static_cast<double>(median_ms) * 1e-3) * clock_ghz * 1e9 /
                static_cast<double>(ops));
        }
    }
    auto name() const -> std::string override { return "rcp_throughput"; }
    auto unit() const -> std::string override { return "cycles/op"; }
    auto granularity() const -> Baseliner::MetricGranularity override {
        return Baseliner::MetricGranularity::ON_DEMAND;
    }
};

template <typename BackendT>
class GpuIncoreWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuIncoreWorkload::backend;

    auto algo() -> std::string override { return "gpu-incore"; }

    // No meaningful byte or FLOP count (ops_per_run is measured separately)
    auto number_of_floating_point_operations() -> std::optional<size_t> override {
        return std::nullopt;
    }

    void setup_host_random_generated() override {}
    void setup_device(typename backend::stream_t stream) override;
    void reset_device(typename backend::stream_t /*stream*/) override {}
    auto run(typename backend::stream_t stream) -> typename backend::launch_result_t override;
    void fetch_results(typename backend::stream_t stream) override;
    void free() override {}

protected:
    void register_options() override {
        Baseliner::IWorkload<BackendT>::register_options();
        this->add_option("GpuIncore", "kernel_type",
                         "Kernel type: fma-mixed, fma-separated, div, sqrt",
                         m_kernel_type)
            .sweep({"fma-mixed", "fma-separated", "div", "sqrt"});
        this->add_option("GpuIncore", "ilp",
                         "Instruction-level parallelism (1, 2, 4 or 8)", m_ilp)
            .sweep({"1", "2", "4", "8"});
        this->add_option("GpuIncore", "precision",
                         "Floating-point precision: float or double", m_precision)
            .sweep({"float", "double"});
        this->add_option("GpuIncore", "warp_count",
                         "Number of warps per block (block_size = 32 * warp_count)",
                         m_warp_count)
            .sweep({"1", "2", "4", "8", "16", "32"});
    }

    void inner_setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> engine) override {
        engine->register_stat<InCoreOps>();
        engine->register_stat<Baseliner::Stats::ClockFrequency<BackendT>>();
        engine->register_stat<RcpThruStat<BackendT>>();
    }

    void inner_update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> engine) override {
        size_t N_type = (m_kernel_type == "div" || m_kernel_type == "sqrt")
                            ? static_cast<size_t>(N_OTHER)
                            : static_cast<size_t>(N_FMA);
        engine->update_values<InCoreOps>(N_type * static_cast<size_t>(ITERS) * m_warp_count);
    }

protected:
    void*       m_d_output    = nullptr;
    std::string m_kernel_type = "fma-mixed";
    int         m_ilp         = 4;
    std::string m_precision   = "float";
    int         m_warp_count  = 4;

    static constexpr int ITERS   = 10000;
    static constexpr int N_FMA   = 1024;
    static constexpr int N_OTHER = 128;
};
