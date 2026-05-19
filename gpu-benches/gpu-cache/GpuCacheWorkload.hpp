#pragma once
#include <baseliner/core/Workload.hpp>
#include <cstddef>
#include <optional>
#include <string>

using cache_dtype = float;

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

protected:
    void register_options() override {
        Baseliner::IWorkload<BackendT>::register_options();
        this->add_option("GpuCache", "working_set_elements",
                         "Number of float elements in working set", m_working_set_elements)
            .sweep({"128", "256", "512", "768", "1024", "2048", "3072", "4096",
                    "5120", "6144", "7168", "8192", "10240", "12288", "14336",
                    "16384", "24576", "32768", "49152", "65536", "131072",
                    "262144", "524288", "1048576", "2097152", "4194304"});
    }

protected:
    cache_dtype* m_device_buffer_a      = nullptr;
    cache_dtype* m_device_buffer_b      = nullptr;
    size_t       m_working_set_elements = 4096;
    int          m_sm_count             = 0;
};
