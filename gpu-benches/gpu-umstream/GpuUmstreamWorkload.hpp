#pragma once
#include <baseliner/core/Workload.hpp>
#include <cstddef>
#include <optional>
#include <string>

template <typename BackendT>
class GpuUmstreamWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuUmstreamWorkload::backend;

    auto algo() -> std::string override { return "gpu-umstream"; }

    auto number_of_bytes() -> std::optional<size_t> override {
        return m_item_count * sizeof(double) * 3;
    }

    void setup_host_random_generated() override {
        m_item_count = m_transfer_mb * (1024ULL * 1024ULL / sizeof(double));
    }

    void setup_device(typename backend::stream_t stream) override;
    void reset_device(typename backend::stream_t /*stream*/) override {}
    auto run(typename backend::stream_t stream) -> typename backend::launch_result_t override;
    void fetch_results(typename backend::stream_t /*stream*/) override {}
    void free() override;

protected:
    void register_options() override {
        using Baseliner::SweepPolicy;
        Baseliner::IWorkload<BackendT>::register_options();
        this->add_option("GpuUmstream", "transfer_mb",
                         "Transfer size in megabytes (per array)", m_transfer_mb)
            .sweep(SweepPolicy::PowersOfTwo, "1", "512");
        this->add_option("GpuUmstream", "blocksize",
                         "CUDA thread block size", m_block_size);
        this->add_option("GpuUmstream", "prefetch",
                         "Prefetch unified memory to GPU before run", m_prefetch)
            .sweep({"false", "true"});
    }

protected:
    double* m_A           = nullptr;
    double* m_B           = nullptr;
    double* m_C           = nullptr;
    size_t  m_item_count  = 1024ULL * 1024ULL / sizeof(double);
    size_t  m_transfer_mb = 1;
    int     m_block_size  = 256;
    int     m_block_count = 1;
    bool    m_prefetch    = true;
};
