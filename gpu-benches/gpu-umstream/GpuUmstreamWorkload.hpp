#pragma once
#include <baseliner/core/Workload.hpp>
#include <baseliner/core/stats/Stats.hpp>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>

namespace Baseliner::Stats {

class Spread : public IStat<Spread, float, SortedExecutionTimeVector> {
public:
    [[nodiscard]] auto name() const -> std::string override { return "spread"; }
    [[nodiscard]] auto unit() const -> std::string override { return "%"; }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
        return MetricGranularity::ON_DEMAND;
    }
    void calculate(float &value,
                   const typename SortedExecutionTimeVector::type &sorted_vec) override {
        size_t n = sorted_vec.size();
        if (n < 2) { value = 0.0f; return; }
        if (n == 2) {
            float m = (sorted_vec[0].count() + sorted_vec[1].count()) / 2.0f;
            value = (m > 0.0f) ? (std::abs(sorted_vec[1].count() - sorted_vec[0].count()) / m * 100.0f) : 0.0f;
            return;
        }
        // trimmed mean: exclude min and max — same as um-stream's value()
        float sum = 0.0f;
        for (size_t i = 1; i < n - 1; ++i)
            sum += sorted_vec[i].count();
        float trimmed_mean = sum / static_cast<float>(n - 2);
        float range = sorted_vec.back().count() - sorted_vec.front().count();
        value = (trimmed_mean > 0.0f) ? (range / trimmed_mean * 100.0f) : 0.0f;
    }
};

} // namespace Baseliner::Stats

template <typename BackendT>
class GpuUmstreamWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuUmstreamWorkload::backend;

    auto algo() -> std::string override { return "gpu-umstream"; }

    auto number_of_bytes() -> std::optional<size_t> override {
        return m_item_count * sizeof(double) * 3;
    }

    void setup_host_random_generated() override {
        m_item_count = m_transfer_mb * (1024ULL * 1024ULL / sizeof(double)) / 3;
    }

    void setup_device(typename backend::stream_t stream) override;
    void reset_device(typename backend::stream_t stream) override;
    auto run(typename backend::stream_t stream) -> typename backend::launch_result_t override;
    void fetch_results(typename backend::stream_t /*stream*/) override {}
    void free() override;

protected:
    void register_options() override {
        using Baseliner::SweepPolicy;
        Baseliner::IWorkload<BackendT>::register_options();
        this->add_option("GpuUmstream", "transfer_mb",
                         "Total transfer size in megabytes (all 3 arrays)", m_transfer_mb)
            .sweep(SweepPolicy::PowersOfTwo, "1", "512");
        this->add_option("GpuUmstream", "blocksize",
                         "CUDA thread block size", m_block_size);
        this->add_option("GpuUmstream", "block_count",
                         "Number of CUDA blocks (0 = auto from occupancy)", m_block_count_override);
        this->add_option("GpuUmstream", "prefetch",
                         "Prefetch unified memory to GPU before run", m_prefetch)
            .sweep({"false", "true"});
    }

protected:
    double* m_A           = nullptr;
    double* m_B           = nullptr;
    double* m_C           = nullptr;
    size_t  m_item_count  = 1024ULL * 1024ULL / sizeof(double);
    size_t  m_transfer_mb        = 1;
    int     m_block_size         = 256;
    int     m_block_count        = 1;
    int     m_block_count_override = 0;
    bool    m_prefetch           = true;
    int     m_device_id          = 0;
};
