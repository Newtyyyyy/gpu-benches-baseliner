#ifndef GPU_INCORE_HPP
#define GPU_INCORE_HPP
#include <baseliner/Case.hpp>
#include <cstddef>
#include <memory>
#include <string>


template <typename BackendT>
class GpuIncore : public Baseliner::ICase<BackendT> {

public:
  auto name() -> std::string override {
    return "gpu-incore";
  };
  void setup(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void reset_case(std::shared_ptr<typename BackendT::stream_t> stream) override;
  void run_case(std::shared_ptr<typename BackendT::stream_t> stream) override;
  auto number_of_floating_point_operations() -> std::optional<size_t> override;
  void case_setup_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override;
  void case_update_metrics(std::shared_ptr<Baseliner::Stats::StatsEngine> &engine) override;
  void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override;
  auto validate_case() -> bool override {
    return true;
  };
  void register_options() override {
    Baseliner::ICase<BackendT>::register_options();
    this->add_option("gpu-incore", "kernel-type", "Kernel type: fma-mixed, fma-separated, div, sqrt", m_kernel_type);
    this->add_option("gpu-incore", "ilp", "Instruction-level parallelism (1, 2, 4, or 8)", m_ilp);
    this->add_option("gpu-incore", "precision", "Precision: float or double", m_precision);
  }

private:
  void *d_output = nullptr;
  std::string m_kernel_type = "fma-mixed";
  int m_ilp = 4;
  std::string m_precision = "float";
  static constexpr int ITERS = 10000;
  static constexpr int N_FMA = 1024; 
  static constexpr int N_OTHER = 128;
};
#endif // GPU_INCORE_HPP
