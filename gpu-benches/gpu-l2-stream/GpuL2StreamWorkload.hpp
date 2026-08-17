#pragma once
#include <baseliner/core/Workload.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

// Ported from RRZE gpu-benches/gpu-l2-stream: STREAM-style kernels (read,
// scale, write, triad) launched over a huge fixed number of work-items
// (iteration_count), where each thread's index wraps modulo `length`. Varying
// `length` sweeps the reuse footprint, similarly to gpu-l2-cache. Occupancy
// control via shared-memory "spoiler" padding is dropped (it only existed
// upstream to throttle blocks/SM for an occupancy sweep we don't reproduce).
template <typename BackendT>
class GpuL2StreamWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuL2StreamWorkload::backend;

    static constexpr int64_t ITERATION_COUNT = 1024LL * 1024 * 1024 + 2;
    static constexpr int BLOCKSIZE = 256;

    auto algo() -> std::string override { return "gpu-l2-stream"; }

    auto number_of_bytes() -> std::optional<size_t> override {
        return static_cast<size_t>(stream_count(m_kernel_type)) *
               static_cast<size_t>(ITERATION_COUNT) * sizeof(double);
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
        this->add_option("GpuL2Stream", "kernel_type",
                         "STREAM kernel: read, scale, write or triad", m_kernel_type)
            .sweep({"read", "scale", "write", "triad"});
        this->add_option("GpuL2Stream", "length",
                         "Number of doubles the per-thread index wraps around "
                         "(controls the reuse footprint)", m_length)
            .sweep({"24545", "32737", "40929", "49121", "57313", "65505", "81889", "90081",
                    "98273", "114657", "122849", "139233", "155617", "163809", "180193",
                    "196577", "212961", "229345", "245729", "262113", "286689", "303073",
                    "319457", "344033", "368609", "384993", "409569", "434145", "458721",
                    "491489", "516065", "548833", "573409", "606177", "638945", "671713",
                    "704481", "745441", "778209", "819169", "860129", "901089", "950241",
                    "991201", "1040353", "1089505", "1146849", "1196001", "1253345",
                    "1310689", "1368033", "1433569", "1499105", "1564641", "1638369",
                    "1712097", "1785825", "1867745", "1949665", "2039777", "2129889",
                    "2220001", "2318305", "2416609", "2523105", "2629601", "2744289",
                    "2858977", "2981857", "3112929", "3244001", "3383265", "3522529",
                    "3669985", "3825633", "3989473", "4153313", "4333537", "4513761",
                    "4702177", "4898785", "5103585", "5316577", "5537761", "5767137",
                    "6004705", "6250465", "6504417", "6774753", "7053281", "7348193",
                    "7651297", "7962593", "8290273", "8626145", "8978401", "9347041",
                    "9732065", "10125281", "10543073", "10969057", "11419617", "11878369",
                    "12361697", "12869601", "13393889", "13934561", "14499809", "15089633",
                    "15695841", "16334817", "16998369", "17686497", "18399201", "19144673",
                    "19914721", "20725729", "21561313", "22429665", "23330785", "24272865",
                    "25255905", "26271713", "27328481", "28434401", "29581281", "30769121",
                    "32006113", "33300449", "34635745", "36028385", "37478369", "38985697",
                    "40558561", "42188769", "43884513", "45645793", "47480801", "49381345",
                    "51372001", "53428193", "55574497", "57810913", "60129249", "62537697",
                    "65052641", "67657697", "70377441", "73195489", "76136417", "79183841",
                    "82362337", "85663713", "89096161", "92667873", "96387041"});
    }

private:
    static auto stream_count(const std::string& kernel_type) -> int {
        if (kernel_type == "read") return 1;
        if (kernel_type == "write") return 1;
        if (kernel_type == "scale") return 2;
        return 4; // triad
    }

protected:
    double*     m_device_buffer_a = nullptr;
    double*     m_device_buffer_b = nullptr;
    double*     m_device_buffer_c = nullptr;
    double*     m_device_buffer_d = nullptr;
    std::string m_kernel_type     = "triad";
    size_t      m_length          = 1040353;
};
