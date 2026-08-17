#pragma once
#include <baseliner/core/Workload.hpp>
#include <cstddef>
#include <optional>
#include <string>

// Ported from RRZE gpu-benches/gpu-roofline: a kernel doing 2 loads + N FMA-ish
// ops per element (arithmetic intensity roofline). Upstream swept N at
// compile time via a Makefile `-DPARN=N` define and rebuilt per point; here N
// is dispatched at runtime over the same set of values via a switch table
// (see cuda/GpuRooflineWorkload.cu), since baseliner sweeps options, not
// rebuilds.
template <typename BackendT>
class GpuRooflineWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuRooflineWorkload::backend;

    static constexpr int M         = 4000;
    static constexpr int BLOCKSIZE = 256;

    auto algo() -> std::string override { return "gpu-roofline"; }

    auto number_of_bytes() -> std::optional<size_t> override {
        return 2ULL * m_data_len * sizeof(float);
    }

    auto number_of_floating_point_operations() -> std::optional<size_t> override {
        return (2ULL + 2ULL * static_cast<size_t>(m_n)) * m_data_len;
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
        this->add_option("GpuRoofline", "n",
                         "Number of inner FMA-ish iterations per 2 loads "
                         "(controls arithmetic intensity)", m_n)
            .sweep({"0", "1", "2", "4", "6", "8", "10", "12", "14", "16", "18", "20", "22",
                    "24", "28", "32", "36", "40", "44", "48", "54", "60", "66", "72", "80",
                    "88", "96", "106", "116", "126", "138", "150", "164", "178", "194", "212",
                    "230", "250", "272", "296", "322", "350", "380", "412", "448", "486",
                    "528", "574", "622", "674", "732", "794", "862", "934", "1012"});
    }

protected:
    float* m_device_buffer_a = nullptr;
    float* m_device_buffer_b = nullptr;
    float* m_device_buffer_c = nullptr;
    int    m_n               = 8;
    int    m_sm_count        = 0;
    int    m_block_count     = 0;
    size_t m_data_len        = 0;
};
