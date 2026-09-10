# gpu-benches-baseliner

A fork of [RRZE-HPC/gpu-benches](https://github.com/RRZE-HPC/gpu-benches) ported to the
[gpu-kernel-baseliner](https://github.com/comeyrd/gpu-kernel-baseliner) architecture, so the
same micro-benchmarks run on CUDA and HIP through one driver, one protocol format and one
result format.

Every benchmark exists in two flavours:

- `cuda/` — the CUDA source, the reference implementation
- `hipifiable/` — its mechanical HIP translation, produced by `hipify-perl` (see [Hipify_script.sh](Hipify_script.sh))

---

# Setup, from scratch

## 1. Requirements

| Need | Version used | Why | Check with |
|---|---|---|---|
| CMake | 3.28.3 (≥ 3.15 required) | Build system, presets | `cmake --version` |
| `clang++` | 17 | Host compiler hardcoded in the presets, with `lld` as linker | `clang++ --version` |
| GCC toolchain | 11 | Not a compiler here: clang++ takes its C++ standard library (libstdc++) from it. Forced by `--gcc-install-dir` in the presets | `clang++ -v` |
| CUDA Toolkit | 12.4 | Any CUDA build; also CUPTI for `gpu-strides` | `nvcc --version` |
| ROCm | 7.0.1 on AMD, 6.4.4 for HIP-on-NVIDIA headers | Any HIP build needs its headers, even when nvcc does the compiling. `hipify-perl` is only needed for step 6 | `ls $HIP_PATH/include/hip/hip_runtime.h` |

The baseliner itself is **not** a prerequisite: CMake fetches it automatically
(`FetchContent`, pinned to tag `v1.0`).

## 2. Get the code

```bash
git clone --depth 1 --single-branch -b helio https://github.com/Newtyyyyy/gpu-benches-baseliner.git
cd gpu-benches-baseliner
```

## 3. Build

Build directories follow `build/<preset-name>/`, and the target is always `gpu-benches_exec`.

| Preset | Configure command | Needs | Builds |
|---|---|---|---|
| `release-cuda` | `cmake --preset release-cuda` | CUDA Toolkit | `cuda/` |
| `debug-cuda` | `cmake --preset debug-cuda` | CUDA Toolkit | `cuda/`, `-O0 -g -G`, all warnings |
| `release-hip-only` | `cmake --preset release-hip-only` | ROCm only, no CUDA, real AMD GPU | `hipifiable/` |
| `release-hip-nvidia` | `export HIP_PATH=<your ROCm install>` then `cmake --preset release-hip-nvidia` | ROCm headers + CUDA Toolkit, NVIDIA GPU | `hipifiable/` |

`HIP_PATH` must point at a ROCm installation — it does not have to be a system-wide one,
a ROCm unpacked in your home works, since only the headers are used.

Then, for any of them:

```bash
cmake --build build/<preset-name> --target gpu-benches_exec -j"$(nproc)"
```

`release-hip-nvidia` builds the HIP version of the code and runs it on an NVIDIA GPU, so it
can be compared against the CUDA version **on the same hardware**. The gap between the two
then measures what the HIP translation costs, not a difference between two GPUs. It pulls in
[build_helper_Hip_on_Nvidia.cpp](build_helper_Hip_on_Nvidia.cpp), which supplies the clock /
temperature / power stats through NVML, since AMD-SMI is unavailable there.

## 4. First run

Smoke test — no protocol file involved. One measurement point per workload, on every backend
compiled into the binary:

```bash
./build/release-cuda/gpu-benches_exec run --tiny --output-file tiny.json
```

Full sweep, driven by a protocol file:

```bash
./build/release-cuda/gpu-benches_exec run --protocol-files default-protocol.json --output-file result.json
```


## 5. Read the results

The output JSON nests as follows:

```
campaign_runs[]
└── benchmark_runs[<backend>][<workload>]
    └── benchmark_report.results[]
        └── measurements[]  →  { name, data, granularity }
```

Metric names are per benchmark: `memory_bandwidth`, `latency_ns`, `rcp_throughput`,
`arithmetic_bandwidth`, plus `median` / `mean` / `CoV` everywhere. Each benchmark's doc lists
the ones it reports.

## 6. Adding a new benchmark

1. Write `gpu-[name]/Gpu[name]Workload.hpp` and `gpu-[name]/cuda/Gpu[name]Workload.cu`
2. Add `gpu-[name]/cuda/CMakeLists.txt` (copy one from an existing benchmark)
3. Run `./Hipify_script.sh` from the repo root — it regenerates every `hipifiable/*.hip`
4. Add `gpu-[name]/hipifiable/CMakeLists.txt` (again, copy an existing one)
5. Add `gpu-[name]/CMakeLists.txt`, and register it in `gpu-benches/CMakeLists.txt`

Step 3 is a mechanical translation, not a tuned port, and it is **not automatic**: rerun the
script and commit the regenerated `.hip` whenever you touch a `.cu`.
