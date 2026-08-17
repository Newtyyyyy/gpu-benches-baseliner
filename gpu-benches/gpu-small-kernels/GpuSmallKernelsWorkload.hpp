#pragma once
#include <baseliner/core/Workload.hpp>
#include <cstddef>
#include <optional>
#include <string>

// Ported from RRZE gpu-benches/gpu-small-kernels: repeatedly launches a tiny
// "scale" kernel over a data set of `size` doubles at a given CUDA block
// size, to characterize how launch overhead dominates bandwidth at small
// sizes. Upstream also had -graph/-pta/-pt-gsync launch-batching modes to
// amortize launch overhead; those are dropped here since baseliner already
// measures per-call latency directly via its own timer.
template <typename BackendT>
class GpuSmallKernelsWorkload : public Baseliner::IWorkload<BackendT> {
public:
    using backend = typename GpuSmallKernelsWorkload::backend;

    auto algo() -> std::string override { return "gpu-small-kernels"; }

    auto number_of_bytes() -> std::optional<size_t> override {
        return static_cast<size_t>(m_size) * 2ULL * sizeof(double);
    }

    void setup_host_random_generated() override {}
    void setup_device(typename backend::stream_t stream) override;
    void reset_device(typename backend::stream_t /*stream*/) override {}
    auto run(typename backend::stream_t stream) -> typename backend::launch_result_t override;
    void fetch_results(typename backend::stream_t stream) override;
    void free() override {}

protected:
    void register_options() override {
        using Baseliner::SweepPolicy;
        Baseliner::IWorkload<BackendT>::register_options();
        this->add_option("GpuSmallKernels", "size",
                         "Number of doubles processed by the scale kernel", m_size)
            .sweep({"4096", "4341", "4601", "4877", "5169", "5479", "5807", "6155", "6524", "6915",
                    "7329", "7768", "8234", "8728", "9251", "9806", "10394", "11017", "11678", "12378",
                    "13120", "13907", "14741", "15625", "16562", "17555", "18608", "19724", "20907", "22161",
                    "23490", "24899", "26392", "27975", "29653", "31432", "33317", "35316", "37434", "39680",
                    "42060", "44583", "47257", "50092", "53097", "56282", "59658", "63237", "67031", "71052",
                    "75315", "79833", "84622", "89699", "95080", "100784", "106831", "113240", "120034", "127236",
                    "134870", "142962", "151539", "160631", "170268", "180484", "191313", "202791", "214958", "227855",
                    "241526", "256017", "271378", "287660", "304919", "323214", "342606", "363162", "384951", "408048",
                    "432530", "458481", "485989", "515148", "546056", "578819", "613548", "650360", "689381", "730743",
                    "774587", "821062", "870325", "922544", "977896", "1036569", "1098763", "1164688", "1234569", "1308643",
                    "1387161", "1470390", "1558613", "1652129", "1751256", "1856331", "1967710", "2085772", "2210918", "2343573",
                    "2484187", "2633238", "2791232", "2958705", "3136227", "3324400", "3523864", "3735295", "3959412", "4196976",
                    "4448794", "4715721", "4998664", "5298583", "5616497", "5953486", "6310695", "6689336", "7090696", "7516137",
                    "7967105", "8445131", "8951838", "9488948", "10058284", "10661781", "11301487", "11979576", "12698350", "13460251",
                    "14267866", "15123937", "16031373", "16993255", "18012850", "19093621", "20239238", "21453592", "22740807", "24105255",
                    "25551570", "27084664", "28709743", "30432327", "32258266", "34193761", "36245386", "38420109", "40725315", "43168833",
                    "45758962", "48504499", "51414768", "54499654", "57769633", "61235810", "64909958", "68804555", "72932828", "77308797",
                    "81947324", "86864163", "92076012", "97600572", "103456606", "109664002", "116243842", "123218472", "130611580"});
        this->add_option("GpuSmallKernels", "block_size",
                         "CUDA thread block size for the scale kernel launch", m_block_size)
            .sweep({"32", "64", "128", "256", "512", "1024"});
    }

protected:
    double* m_device_buffer_a = nullptr;
    double* m_device_buffer_b = nullptr;
    int     m_size            = 1048576;
    int     m_block_size      = 256;
};
