#ifndef GPU_CACHE_HPP
#define GPU_CACHE_HPP
#include <baseliner/Case.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
#include <baseliner/hardware/cuda/CudaBackend.hpp>
#include <baseliner/stats/Stats.hpp>
#include <cstddef>     // size_t
#include <memory>      // std::shared_ptr
#include <string>      // std::string
#include <optional>    // std::optional

using dtype = float;

// Define metric tags for bandwidths
namespace Baseliner::Stats {
  class L1BandwidthMetric : public Imetric<L1BandwidthMetric, float> {
  public:
    auto name() const -> std::string override { return "l1_bandwidth_gbs"; }
    auto unit() const -> std::string override { return "GB/s"; }
    auto saving_policy() -> MetricSavingPolicy override { return MetricSavingPolicy::SAVE; }
  };

  class L2BandwidthMetric : public Imetric<L2BandwidthMetric, float> {
  public:
    auto name() const -> std::string override { return "l2_bandwidth_gbs"; }
    auto unit() const -> std::string override { return "GB/s"; }
    auto saving_policy() -> MetricSavingPolicy override { return MetricSavingPolicy::SAVE; }
  };

  class DramBandwidthMetric : public Imetric<DramBandwidthMetric, float> {
  public:
    auto name() const -> std::string override { return "dram_bandwidth_gbs"; }
    auto unit() const -> std::string override { return "GB/s"; }
    auto saving_policy() -> MetricSavingPolicy override { return MetricSavingPolicy::SAVE; }
  };
}

template <typename BackendT>
class GpuCache : public Baseliner::ICase<BackendT> {
public:
  auto name() -> std::string override {
    return "gpu-cache";
  };

  void setup(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void reset_case(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void run_case(std::shared_ptr<typename BackendT::stream_t> stream) override;
  auto number_of_bytes() -> std::optional<size_t> override;
  auto l1_bandwidth_gbs() -> std::optional<float>;
  auto l2_bandwidth_gbs() -> std::optional<float>;
  auto dram_bandwidth_gbs() -> std::optional<float>;
  void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override;

  auto validate_case() -> bool override {
    return true;
  };

  void register_options() override {
    Baseliner::ICase<BackendT>::register_options();
    this->add_option("gpu-cache", "working-set-elements",
                     "Number of float elements in working set (bytes = 2*N*sizeof(float))",
                     m_working_set_elements);
  }

  void case_setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override {
    Baseliner::ICase<BackendT>::case_setup_metrics(engine);
    engine->register_metric<Baseliner::Stats::L1BandwidthMetric>();
    engine->register_metric<Baseliner::Stats::L2BandwidthMetric>();
    engine->register_metric<Baseliner::Stats::DramBandwidthMetric>();
  }

  void case_update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override {
    Baseliner::ICase<BackendT>::case_update_metrics(engine);
    if (auto l1_bw = l1_bandwidth_gbs()) {
      engine->update_values<Baseliner::Stats::L1BandwidthMetric>(l1_bw.value());
    }
    if (auto l2_bw = l2_bandwidth_gbs()) {
      engine->update_values<Baseliner::Stats::L2BandwidthMetric>(l2_bw.value());
    }
    if (auto dram_bw = dram_bandwidth_gbs()) {
      engine->update_values<Baseliner::Stats::DramBandwidthMetric>(dram_bw.value());
    }
  }

protected:
  float *m_device_buffer_a = nullptr;
  float *m_device_buffer_b = nullptr;
  size_t m_working_set_elements = 4096;
};

#endif // GPU_CACHE_HPP
