#pragma once
#include <baseliner/core/Workload.hpp>
#include <baseliner/core/stats/Stats.hpp>
#include <cstddef>
#include <optional>
#include <string>

class LatencyIterations : public Baseliner::Stats::IStat<LatencyIterations, int64_t> {
public:
    void calculate(int64_t & /*val*/) override {}
    auto name() const -> std::string override { return "iterations"; }
    auto unit() const -> std::string override { return ""; }
    auto granularity() const -> Baseliner::MetricGranularity override {
        return Baseliner::MetricGranularity::ON_DEMAND;
    }
};

class LatencyNsStat : public Baseliner::Stats::IStat<LatencyNsStat, float,
    Baseliner::Stats::Median, LatencyIterations> {
public:
    void calculate(float &val, const float &median_ms, const int64_t &iters) override {
        if (iters > 0 && median_ms > 0.0f)
            val = static_cast<float>(static_cast<double>(median_ms) * 1e6 /
                                     static_cast<double>(iters));
    }
    auto name() const -> std::string override { return "latency_ns"; }
    auto unit() const -> std::string override { return "ns"; }
    auto granularity() const -> Baseliner::MetricGranularity override {
        return Baseliner::MetricGranularity::ON_DEMAND;
    }
};

template <typename BackendT>
class GpuLatencyWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuLatencyWorkload::backend;

    auto algo() -> std::string override { return "gpu-latency"; }

    void setup_host_random_generated() override {}
    void setup_device(typename backend::stream_t stream) override;
    void reset_device(typename backend::stream_t stream) override;
    auto run(typename backend::stream_t stream) -> typename backend::launch_result_t override;
    void fetch_results(typename backend::stream_t /*stream*/) override {}
    void free() override;

protected:
    void register_options() override {
        using Baseliner::SweepPolicy;
        Baseliner::IWorkload<BackendT>::register_options();
        this->add_option("GpuLatency", "buffer_size_kb",
                         "Size of the pointer-chasing buffer in KB", m_buffer_size_kb)
            .sweep(SweepPolicy::PowersOfTwo, "16", "524288");
        this->add_option("GpuLatency", "iterations",
                         "Number of pointer-chase iterations per measurement", m_iterations);
    }

    void inner_setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> engine) override {
        engine->register_stat<LatencyIterations>();
        engine->register_stat<LatencyNsStat>();
    }

    void inner_update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> engine) override {
        engine->update_values<LatencyIterations>(m_iterations);
    }

protected:
    int64_t* m_device_buffer  = nullptr;
    int64_t* m_dummy_buffer   = nullptr;
    size_t   m_buffer_size_kb = 64;
    int64_t  m_iterations     = 100000;
    int64_t  m_chain_length   = 0;
};
