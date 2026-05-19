#pragma once
#include <baseliner/core/Workload.hpp>
#include <cstddef>
#include <optional>
#include <string>

template <typename BackendT>
class GpuMemcpyWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuMemcpyWorkload::backend;

    auto algo() -> std::string override { return "gpu-memcpy"; }

    auto number_of_bytes() -> std::optional<size_t> override {
        return m_item_count;
    }

    void setup_host_random_generated() override {
        m_item_count = m_transfer_mb * (1024ULL * 1024ULL);
    }

    void setup_device(typename backend::stream_t stream) override;
    void reset_device(typename backend::stream_t stream) override;
    auto run(typename backend::stream_t stream) -> typename backend::launch_result_t override;
    void fetch_results(typename backend::stream_t stream) override;
    void free() override;

protected:
    void register_options() override {
        using Baseliner::SweepPolicy;
        Baseliner::IWorkload<BackendT>::register_options();
        this->add_option("GpuMemcpy", "transfer_mb",
                         "Transfer size in megabytes", m_transfer_mb)
            .sweep(SweepPolicy::PowersOfTwo, "1", "512");
        this->add_option("GpuMemcpy", "pin_memory",
                         "Use pinned (page-locked) host memory", m_pin_memory)
            .sweep({"false", "true"});
    }

protected:
    char*  m_host_buffer   = nullptr;
    char*  m_device_buffer = nullptr;
    bool   m_pin_memory    = true;
    size_t m_transfer_mb   = 1;
    size_t m_item_count    = 1024ULL * 1024ULL;
};
