#ifndef GPU_LATENCY_HPP
#define GPU_LATENCY_HPP
#include <baseliner/Case.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
#include <baseliner/hardware/BackendStats.hpp>
#include <baseliner/stats/StatsEngine.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <optional>

template <typename BackendT>
class GpuLatency : public Baseliner::ICase<BackendT> {
public:
  auto name() -> std::string override {
    return "gpu-latency";
  };

  void setup(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void reset_case(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void run_case(std::shared_ptr<typename BackendT::stream_t> stream) override;
  auto number_of_bytes() -> std::optional<size_t> override;
  void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override;

  auto validate_case() -> bool override {
    return true;
  };

  void register_options() override {
    Baseliner::ICase<BackendT>::register_options();
    this->add_option("gpu-latency", "buffer-size", "Size of buffer for pointer chasing (KB)", m_buffer_size_kb);
    this->add_option("gpu-latency", "iterations", "Number of pointer-chase iterations per measurement", m_iterations);
  }

  void case_setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override;
  void case_update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override;

private:
  int64_t *m_device_buffer = nullptr;
  int64_t *m_dummy_buffer = nullptr;

  size_t m_buffer_size_kb = 64;
  int64_t m_iterations = 100000;
  int64_t m_chain_length = 0;
};

namespace GpuLatencyDefaults {
  constexpr size_t MIN_KB = 16;
  constexpr size_t MAX_KB = 524288;
  constexpr double RATIO = 1.042;
  constexpr size_t THRESHOLD_KB = 122880;
  constexpr double LARGE_RATIO = 1.1;
}

#endif // GPU_LATENCY_HPP

