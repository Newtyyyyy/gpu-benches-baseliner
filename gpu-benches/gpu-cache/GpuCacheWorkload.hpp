#pragma once
#include <baseliner/core/Workload.hpp>
#include <baseliner/core/stats/IStats.hpp>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

using cache_dtype = float;

class WorkingSetKb : public Baseliner::Stats::Imetric<WorkingSetKb, float> {
public:
    using Baseliner::Stats::Imetric<WorkingSetKb, float>::Imetric;
    [[nodiscard]] auto name() const -> std::string override { return "working_set_kb"; }
    [[nodiscard]] auto unit() const -> std::string override { return "kB"; }
    [[nodiscard]] auto granularity() const -> Baseliner::MetricGranularity override {
        return Baseliner::MetricGranularity::ONCE;
    }
};

template <typename BackendT>
class GpuCacheWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuCacheWorkload::backend;

    auto algo() -> std::string override { return "gpu-cache"; }

    auto number_of_bytes() -> std::optional<size_t> override;

    void setup_host_random_generated() override {}
    void setup_device(typename backend::stream_t stream) override;
    void reset_device(typename backend::stream_t /*stream*/) override {}
    auto run(typename backend::stream_t stream) -> typename backend::launch_result_t override;
    void fetch_results(typename backend::stream_t stream) override;
    void free() override {}

    void inner_setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> engine) override {
        float kb = static_cast<float>(2ULL * m_working_set_elements * sizeof(cache_dtype)) / 1024.0f;
        engine->register_metric<WorkingSetKb>(kb);
    }

    void inner_update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> engine) override {
        float kb = static_cast<float>(2ULL * m_working_set_elements * sizeof(cache_dtype)) / 1024.0f;
        engine->update_values<WorkingSetKb>(kb);
    }

protected:
    void register_options() override {
        Baseliner::IWorkload<BackendT>::register_options();
        this->add_option("GpuCache", "max_working_set_elements",
                         "Maximum number of float elements in working set (sweep upper bound)",
                         m_max_working_set_elements);
        this->add_option("GpuCache", "working_set_elements",
                         "Number of float elements in working set", m_working_set_elements)
            .sweep(compute_sweep_values(m_max_working_set_elements));
    }

private:
    static auto cache_exp_series(size_t n) -> size_t {
        size_t val = 32 * 512;
        for (size_t i = 0; i < n; ++i)
            val = static_cast<size_t>(static_cast<double>(val) * 1.17);
        return (val / 512) * 512;
    }

    static auto compute_sweep_values(size_t max_n) -> std::vector<std::string> {
        std::vector<size_t> ns;
        for (size_t n : {128UL, 256UL, 512UL, 768UL})
            if (n <= max_n) ns.push_back(n);
        for (size_t k = 2; k <= 32; ++k) {
            size_t n = k * 512;
            if (n <= max_n) ns.push_back(n); else break;
        }
        for (size_t i = 1; i <= 49; ++i) {
            if (i == 15) continue;
            size_t n = cache_exp_series(i);
            if (n <= max_n) ns.push_back(n); else break;
        }
        std::vector<std::string> result;
        result.reserve(ns.size());
        for (size_t n : ns) result.push_back(std::to_string(n));
        return result;
    }

protected:
    cache_dtype* m_device_buffer_a          = nullptr;
    cache_dtype* m_device_buffer_b          = nullptr;
    size_t       m_working_set_elements     = 4096;
    size_t       m_max_working_set_elements = 35928576; // expSeries(49)
    int          m_sm_count                 = 0;
};
