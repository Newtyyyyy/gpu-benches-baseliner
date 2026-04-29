#include "../gpu-latency.hpp"
#include <baseliner/Case.hpp>
#include <baseliner/hardware/cuda/CudaBackend.hpp>
#include <baseliner/stats/Stats.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

using namespace Baseliner;
using namespace Baseliner::Stats;

constexpr int CACHE_LINE_INTS = 128 / sizeof(int64_t);
constexpr int SKIP_FACTOR = 1;

class LatencyCycles : public IStat<LatencyCycles, float, Median, ClockFrequency<Hardware::CudaBackend>> {
  static inline int64_t s_iters = 100000;
public:
  static void set_iters(int64_t n) { s_iters = std::max(n, int64_t(100)); }
  auto name() const -> std::string override { return "latency_cycles"; }
  auto unit() const -> std::string override { return "cycles/iter"; }
  auto compute_policy() -> StatComputePolicy override { return StatComputePolicy::ON_DEMAND; }
  void calculate(float &val, const float &median_ms, const float &clock_ghz) override {
    val = (median_ms * clock_ghz * 1e6f) / static_cast<float>(s_iters);
  }
};

template <typename T>
__global__ void pointer_chasing_kernel(T *buf, T *dummy_out, int64_t iterations) {
  T *idx = buf;
  const int UNROLL = 32;
#pragma unroll 1
  for (int64_t n = 0; n < iterations; n += UNROLL) {
#pragma unroll
    for (int u = 0; u < UNROLL; u++) idx = (T *)*idx;
  }
  if (threadIdx.x + blockIdx.x > 12000) dummy_out[0] = (T)idx;
}

void build_pointer_chain(int64_t *buf, int64_t chain_len) {
  static thread_local std::mt19937 rng(std::random_device{}());
  std::vector<int64_t> order(chain_len);
  for (int64_t i = 0; i < chain_len; i++) order[i] = i + 1;
  order[chain_len - 1] = 0;
  std::shuffle(order.begin(), order.end() - 1, rng);

  for (int cl = 0; cl < CACHE_LINE_INTS; cl++) {
    int64_t idx = 0;
    for (int64_t i = 0; i < chain_len; i++) {
      buf[(idx * CACHE_LINE_INTS + cl) * SKIP_FACTOR] =
          SKIP_FACTOR * (order[i] * CACHE_LINE_INTS + cl + (order[i] == 0 ? 1 : 0));
      idx = order[i];
    }
  }
  buf[SKIP_FACTOR * (order[chain_len - 2] * CACHE_LINE_INTS + CACHE_LINE_INTS - 1)] = 0;

  int64_t total = SKIP_FACTOR * CACHE_LINE_INTS * chain_len;
  for (int64_t n = 0; n < total; n++)
    buf[n] = (int64_t)(buf) + buf[n] * sizeof(int64_t);
}

template <>
void GpuLatency<Hardware::CudaBackend>::setup(std::shared_ptr<Hardware::CudaBackend::stream_t> /*stream*/) {
  size_t buf_bytes = m_buffer_size_kb * 1024;
  m_chain_length = buf_bytes / (SKIP_FACTOR * CACHE_LINE_INTS * sizeof(int64_t));
  if (m_chain_length < 2) m_chain_length = 16;
  CHECK_CUDA(cudaMallocManaged(&m_device_buffer, buf_bytes));
  CHECK_CUDA(cudaMallocManaged(&m_dummy_buffer, sizeof(int64_t)));
  build_pointer_chain(m_device_buffer, m_chain_length);
}

template <>
void GpuLatency<Hardware::CudaBackend>::reset_case(std::shared_ptr<Hardware::CudaBackend::stream_t> stream) {
    pointer_chasing_kernel<int64_t><<<1, 1, 0, *stream>>>(m_device_buffer, m_dummy_buffer, m_iterations);
    pointer_chasing_kernel<int64_t><<<1, 1, 0, *stream>>>(m_device_buffer, m_dummy_buffer, m_iterations);

}

template <>
void GpuLatency<Hardware::CudaBackend>::run_case(std::shared_ptr<Hardware::CudaBackend::stream_t> stream) {
  pointer_chasing_kernel<int64_t><<<1, 1, 0, *stream>>>(m_device_buffer, m_dummy_buffer, m_iterations);
}

template <>
auto GpuLatency<Hardware::CudaBackend>::number_of_bytes() -> std::optional<size_t> {
  return m_chain_length * sizeof(int64_t);
}

template <>
void GpuLatency<Hardware::CudaBackend>::teardown(std::shared_ptr<Hardware::CudaBackend::stream_t> /*stream*/) {
  CHECK_CUDA(cudaFree(m_device_buffer));
  CHECK_CUDA(cudaFree(m_dummy_buffer));
}

template <>
void GpuLatency<Hardware::CudaBackend>::case_setup_metrics(std::shared_ptr<Stats::StatsEngine> &engine) {
  engine->register_stat<ClockFrequency<Hardware::CudaBackend>>();
  engine->register_stat<LatencyCycles>();
}

template <>
void GpuLatency<Hardware::CudaBackend>::case_update_metrics(std::shared_ptr<Stats::StatsEngine> & /*engine*/) {
  LatencyCycles::set_iters(m_iterations);
}

using CudaLatency = GpuLatency<Hardware::CudaBackend>;
BASELINER_REGISTER_CASE_NAME(CudaLatency, CudaLatency().name());

namespace {
class LenBuffSweepSuite : public Baseliner::ISuite {
public:
  auto name() -> std::string override { return "LenBuffSweepSuite"; }

  [[nodiscard]] auto run_all() -> Baseliner::RunResult override {
    static thread_local std::mt19937 rng(std::random_device{}());

    std::vector<std::string> values;
    double buf_kb = static_cast<double>(m_min_kb);
    while (true) {
      if (buf_kb > static_cast<double>(m_threshold_kb))
        buf_kb *= m_large_ratio;
      size_t rounded = static_cast<size_t>(std::round(buf_kb));
      if (rounded > m_max_kb) break;
      values.push_back(std::to_string(rounded));
      if (rounded >= m_max_kb) break;
      buf_kb = buf_kb * m_ratio + 1.0 + static_cast<double>(rng() % 11);
    }
    if (values.empty() || std::stoull(values.back()) < m_max_kb)
      values.push_back(std::to_string(m_max_kb));

    const Baseliner::OptionsMap basemap = get_benchmark()->gather_axe_options();
    if (basemap.find(m_interface_name) == basemap.end() ||
        basemap.at(m_interface_name).find(m_option_name) == basemap.at(m_interface_name).end()) {
      throw std::runtime_error("LenBuffSweepSuite: option '" + m_option_name +
                               "' not found in interface '" + m_interface_name + "'");
    }

    std::vector<Baseliner::BenchmarkResult> results_v;
    results_v.reserve(values.size());
    bool first = true;
    for (const std::string &val : values) {
      Baseliner::OptionsMap patch;
      patch[m_interface_name][m_option_name].m_value = val;
      get_benchmark()->propagate_axe_options(patch);
      Baseliner::BenchmarkResult result = get_benchmark()->run();
      result.m_options = patch;
      results_v.push_back(result);
      Baseliner::print_benchmark_result(std::cout, result, first);
      if (first) first = false;
      if (Baseliner::ExecutionController::exit_requested()) break;
    }
    return Baseliner::build_run_result(results_v, get_benchmark()->get_device_info());
  }

  void register_options() override {
    this->add_option("LenBuffSweepSuite", "interface_name",
                     "Interface holding the option to sweep", m_interface_name);
    this->add_option("LenBuffSweepSuite", "option_name",
                     "Option to sweep", m_option_name);
    this->add_option("LenBuffSweepSuite", "min_kb",
                     "Start value in KB (default: 16)", m_min_kb);
    this->add_option("LenBuffSweepSuite", "max_kb",
                     "End value in KB (the consigne)", m_max_kb);
    this->add_option("LenBuffSweepSuite", "ratio",
                     "Geometric ratio between steps (default: 1.042)", m_ratio);
    this->add_option("LenBuffSweepSuite", "threshold_kb",
                     "Above this size (KB), apply large_ratio extra multiplier (default: 122880 = 120MB)", m_threshold_kb);
    this->add_option("LenBuffSweepSuite", "large_ratio",
                     "Extra multiplier applied when size > threshold_kb (default: 1.1)", m_large_ratio);
  }

  void register_options_dependencies() override {}

private:
  std::string m_interface_name = "gpu-latency";
  std::string m_option_name    = "buffer-size";
  size_t      m_min_kb         = GpuLatencyDefaults::MIN_KB;
  size_t      m_max_kb         = GpuLatencyDefaults::MAX_KB;
  double      m_ratio          = GpuLatencyDefaults::RATIO;
  size_t      m_threshold_kb   = GpuLatencyDefaults::THRESHOLD_KB;
  double      m_large_ratio    = GpuLatencyDefaults::LARGE_RATIO;
};
BASELINER_REGISTER_SUITE(LenBuffSweepSuite)
} // namespace
