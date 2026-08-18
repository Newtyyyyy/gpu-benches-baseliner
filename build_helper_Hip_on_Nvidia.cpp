// Backend<hipStream_t, std::monostate> and Backend<cudaStream_t, std::monostate> are the
// same C++ type under HIP's NVIDIA platform (hipStream_t is a typedef of cudaStream_t), so
// HipBackend's ClockFrequency/DeviceTemperature/DevicePowerUtilization specializations are
// only ever defined by baseliner-src when BASELINER_HAS_AMDSMI is on. On this NVIDIA node
// there is no AMD GPU, so we provide the same specializations backed by NVML instead.
#include <baseliner/core/hardware/BackendStats.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <cuda_runtime_api.h>
#include <nvml.h>

namespace {
  void check_nvml(nvmlReturn_t result) {
    if (result != NVML_SUCCESS) {
      throw std::runtime_error(std::string("NVML error: ") + nvmlErrorString(result));
    }
  }

  auto current_nvml_device() -> nvmlDevice_t {
    static bool initialized = false;
    if (!initialized) {
      check_nvml(nvmlInit());
      initialized = true;
    }
    int cuda_idx = 0;
    cudaGetDevice(&cuda_idx);
    char pci_bus_id[64];
    cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_idx);
    nvmlDevice_t device{};
    check_nvml(nvmlDeviceGetHandleByPciBusId(pci_bus_id, &device));
    return device;
  }
} // namespace

namespace Baseliner::Stats {
  template <>
  void ClockFrequency<Hardware::HipBackend>::calculate(
      typename ClockFrequency<Hardware::HipBackend>::type &value_to_update) {
    unsigned int clock_mhz = 0;
    check_nvml(nvmlDeviceGetClockInfo(current_nvml_device(), NVML_CLOCK_SM, &clock_mhz));
    value_to_update = static_cast<float>(clock_mhz) / 1000.0F;
  }

  template <>
  void DeviceTemperature<Hardware::HipBackend>::calculate(
      typename DeviceTemperature<Hardware::HipBackend>::type &value_to_update) {
    unsigned int temp_c = 0;
    check_nvml(nvmlDeviceGetTemperature(current_nvml_device(), NVML_TEMPERATURE_GPU, &temp_c));
    value_to_update = static_cast<int>(temp_c);
  }

  template <>
  void DevicePowerUtilization<Hardware::HipBackend>::calculate(
      typename DevicePowerUtilization<Hardware::HipBackend>::type &value_to_update) {
    unsigned int power_mw = 0;
    unsigned int limit_mw = 0;
    nvmlDevice_t device = current_nvml_device();
    check_nvml(nvmlDeviceGetPowerUsage(device, &power_mw));
    check_nvml(nvmlDeviceGetEnforcedPowerLimit(device, &limit_mw));
    value_to_update = (limit_mw == 0) ? 0.0F : (static_cast<float>(power_mw) / static_cast<float>(limit_mw)) * 100.0F;
  }
} // namespace Baseliner::Stats
