#include "baseliner/hardware/cuda/CudaBackend.hpp"
#include "gpu-incore.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Case.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/hardware/BackendStats.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>

using namespace Baseliner;
using namespace Baseliner::Hardware;

//on pourrait peut etre le remplacer par FLOPcount (avec number of float point operations ?)
// mais + compliqué pour comparer les types entre eux FMA/DIV/SQRT
//à deplacer dans stats.hpp ou garder ici ? utile dans d'autres benchmarks ou FLOP count meilleur dans les autres cas ?
class InCoreOps : public Stats::IStat<InCoreOps, size_t> {
public:
  void calculate(size_t &val) override {}
  auto name() const -> std::string override { return "ops_per_run"; }
  auto unit() const -> std::string override { return "ops"; }
  auto compute_policy() -> Stats::StatComputePolicy override {
    return Stats::StatComputePolicy::ON_DEMAND;
  }
};
//pareil à deplacer dans Stats.hpp ou garder ici ? utile dans d'autres benchmarks ou meilleur dans les autres cas ?
class RcpThruStat : public Stats::IStat<RcpThruStat, float, Stats::Median, InCoreOps, Stats::ClockFrequency<CudaBackend>> {
public:
  void calculate(float &val, const float &median_ms, const size_t &ops, const float &clock_ghz) override {
    if (ops > 0 && median_ms > 0.0f && clock_ghz > 0.0f) {
      val = static_cast<float>(
          (static_cast<double>(median_ms) * 1e-3) * clock_ghz * 1e9 /
          static_cast<double>(ops));
    }
  }
  auto name() const -> std::string override { return "rcp_throughput"; }
  auto unit() const -> std::string override { return "cycles/op"; }
  auto compute_policy() -> Stats::StatComputePolicy override {
    return Stats::StatComputePolicy::ON_DEMAND;
  }
};

template <typename T, int N, int M>
__global__ void FMA_mixed(T p, T *A, int iters) {
#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
    T t[M];
#pragma unroll
    for (int m = 0; m < M; m++) {
      t[m] = p + threadIdx.x + iter + m;
    }
#pragma unroll
    for (int n = 0; n < N / M; n++) {
#pragma unroll
      for (int m = 0; m < M; m++) {
        t[m] = t[m] * (T)0.9 + (T)0.5;
      }
    }
#pragma unroll
    for (int m = 0; m < M; m++) {
      if (t[m] > (T)22313.0) {
        A[0] = t[m];
      }
    }
  }
}

template <typename T, int N, int M>
__global__ void FMA_separated(T p, T *A, int iters) {
  for (int iter = 0; iter < iters; iter++) {
#pragma unroll
    for (int m = 0; m < M; m++) {
      T t = p + threadIdx.x + iter + m;
      for (int n = 0; n < N; n++) {
        t = t * (T)0.9 + (T)0.5;
      }
      if (t > (T)22313.0) {
        A[0] = t;
      }
    }
  }
}

template <typename T, int N, int M>
__global__ void DIV_separated(T p, T *A, int iters) {
#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
    for (int m = 0; m < M; m++) {
      T t = p + threadIdx.x + iter + m;
      for (int n = 0; n < N; n++) {
        t = (T)0.1 / (t + (T)0.2);
      }
      A[threadIdx.x + iter] = t;
    }
  }
}

template <typename T, int N, int M>
__global__ void SQRT_separated(T p, T *A, int iters) {
#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
    for (int m = 0; m < M; m++) {
      T t = p + threadIdx.x + iter + m;
      for (int n = 0; n < N; n++) {
        t = sqrt(t + (T)0.2);
      }
      A[threadIdx.x + iter] = t;
    }
  }
}

template <>
void GpuIncore<CudaBackend>::setup(
    std::shared_ptr<CudaBackend::stream_t> stream) {
  int warp_count = static_cast<int>(this->get_work_size());
  int block_size = 32 * warp_count;
  size_t buf_count = static_cast<size_t>(block_size + ITERS);
  size_t elem_size = (m_precision == "double") ? sizeof(double) : sizeof(float);
  CHECK_CUDA(cudaMallocAsync(&d_output, buf_count * elem_size, *stream));
  CHECK_CUDA(cudaMemsetAsync(d_output, 0, buf_count * elem_size, *stream));
}


template <>
void GpuIncore<CudaBackend>::reset_case(
    std::shared_ptr<CudaBackend::stream_t> /*stream*/) {}

template <>
void GpuIncore<CudaBackend>::run_case(
    std::shared_ptr<CudaBackend::stream_t> stream) {
  int block_size = 32 * static_cast<int>(this->get_work_size());
  constexpr int block_count = 1;

//faire template ?<type, N, M> pour eviter les if ?
  if (m_precision == "double") {
    if (m_kernel_type == "fma-mixed") {
      if (m_ilp == 1)
        FMA_mixed<double, N_FMA, 1><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else if (m_ilp == 2)
        FMA_mixed<double, N_FMA, 2><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else if (m_ilp == 4)
        FMA_mixed<double, N_FMA, 4><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else
        FMA_mixed<double, N_FMA, 8><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
    } else if (m_kernel_type == "fma-separated") {
      if (m_ilp == 1)
        FMA_separated<double, N_FMA, 1><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else if (m_ilp == 2)
        FMA_separated<double, N_FMA, 2><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else if (m_ilp == 4)
        FMA_separated<double, N_FMA, 4><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else
        FMA_separated<double, N_FMA, 8><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
    } else if (m_kernel_type == "div") {
      if (m_ilp == 1)
        DIV_separated<double, N_OTHER, 1><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else if (m_ilp == 2)
        DIV_separated<double, N_OTHER, 2><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else if (m_ilp == 4)
        DIV_separated<double, N_OTHER, 4><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else
        DIV_separated<double, N_OTHER, 8><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
    } else {
      if (m_ilp == 1)
        SQRT_separated<double, N_OTHER, 1><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else if (m_ilp == 2)
        SQRT_separated<double, N_OTHER, 2><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else if (m_ilp == 4)
        SQRT_separated<double, N_OTHER, 4><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
      else
        SQRT_separated<double, N_OTHER, 8><<<block_count, block_size, 0, *stream>>>((double)0.32, (double*)d_output, ITERS);
    }
  } else {
    if (m_kernel_type == "fma-mixed") {
      if (m_ilp == 1)
        FMA_mixed<float, N_FMA, 1><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else if (m_ilp == 2)
        FMA_mixed<float, N_FMA, 2><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else if (m_ilp == 4)
        FMA_mixed<float, N_FMA, 4><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else
        FMA_mixed<float, N_FMA, 8><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
    } else if (m_kernel_type == "fma-separated") {
      if (m_ilp == 1)
        FMA_separated<float, N_FMA, 1><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else if (m_ilp == 2)
        FMA_separated<float, N_FMA, 2><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else if (m_ilp == 4)
        FMA_separated<float, N_FMA, 4><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else
        FMA_separated<float, N_FMA, 8><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
    } else if (m_kernel_type == "div") {
      if (m_ilp == 1)
        DIV_separated<float, N_OTHER, 1><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else if (m_ilp == 2)
        DIV_separated<float, N_OTHER, 2><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else if (m_ilp == 4)
        DIV_separated<float, N_OTHER, 4><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else
        DIV_separated<float, N_OTHER, 8><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
    } else {
      if (m_ilp == 1)
        SQRT_separated<float, N_OTHER, 1><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else if (m_ilp == 2)
        SQRT_separated<float, N_OTHER, 2><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else if (m_ilp == 4)
        SQRT_separated<float, N_OTHER, 4><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
      else
        SQRT_separated<float, N_OTHER, 8><<<block_count, block_size, 0, *stream>>>((float)0.32, (float*)d_output, ITERS);
    }
  }
}

template <>
void GpuIncore<CudaBackend>::teardown(
    std::shared_ptr<CudaBackend::stream_t> /*stream*/) {
  CHECK_CUDA(cudaFree(d_output));
  d_output = nullptr;
}


template <>
auto GpuIncore<CudaBackend>::number_of_floating_point_operations()
    -> std::optional<size_t> {
  return {};
}

template <>
void GpuIncore<CudaBackend>::case_setup_metrics(
    std::shared_ptr<Stats::StatsEngine> &engine) {
  engine->register_stat<InCoreOps>();
  engine->register_stat<Stats::ClockFrequency<CudaBackend>>();
  engine->register_stat<RcpThruStat>();
}

template <>
void GpuIncore<CudaBackend>::case_update_metrics(
    std::shared_ptr<Stats::StatsEngine> &engine) {
  size_t N_type = (m_kernel_type == "div" || m_kernel_type == "sqrt")
                       ? static_cast<size_t>(N_OTHER)
                       : static_cast<size_t>(N_FMA);
  engine->update_values<InCoreOps>(N_type * static_cast<size_t>(ITERS) * this->get_work_size());
}

using CudaIncore = GpuIncore<CudaBackend>;
BASELINER_REGISTER_CASE_NAME(CudaIncore, CudaIncore().name());

