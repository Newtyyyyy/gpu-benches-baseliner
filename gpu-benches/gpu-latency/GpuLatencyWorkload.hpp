#pragma once
#include <baseliner/core/Workload.hpp>
#include <cstddef>
#include <optional>
#include <string>

template <typename BackendT>
class GpuLatencyWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuLatencyWorkload::backend;

    auto algo() -> std::string override { return "gpu-latency"; }

    auto number_of_bytes() -> std::optional<size_t> override {
        return m_chain_length * sizeof(int64_t);
    }

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

protected:
    int64_t* m_device_buffer  = nullptr;
    int64_t* m_dummy_buffer   = nullptr;
    size_t   m_buffer_size_kb = 64;
    int64_t  m_iterations     = 100000;
    int64_t  m_chain_length   = 0;
};
