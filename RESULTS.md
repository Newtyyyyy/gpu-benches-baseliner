# Part 1 - Impact of the benchmark parameters

Each experiment here compares the **same workload on the same GPU**, changing exactly one
option, to justify the defaults and to show what silently breaks a measurement when a knob is
wrong. Two are covered so far: `flush` and `warm_cool`.

## The measurement loop

The options act on this loop, transcribed from `baseliner/core/Benchmark.hpp`:

```
setup_host / setup_device / warmup
repeat until the stopping criterion is satisfied:   ← one iteration = one batch
   ├─ warm_cool       reach the temperature window, then stop checking
   ├─ block           freeze the stream so the whole batch is queued before it runs
   └─ for each of batch_size runs:
         ├─ reset_device
         ├─ flush     empty L2 - before every run, not once per batch
         └─ timed run
fetch_results / validate
```

The two positions matter: **`flush` is inside the batch**, paid once per timed run, while
**`warm_cool` is outside it**, paid once per batch and never re-checked while the batch runs.

## 1.1 `flush` - L2 flush before every timed run

**Expected** No effect once the working set is far larger than L2, since the data could not have
stayed resident anyway. An effect in the L1/L2 region, where a stale cache would make the
benchmark report cache bandwidth instead of memory bandwidth.

**Status** *No result yet - the data was lost.* The campaign of 2026-08-31 ran both experiments,
but they wrote under the same file names (`avec-runNN.json`, `sans-runNN.json`) and `warm_cool`
ran second. The `manifest.csv` describes 40 runs where 20 files remain. Which set survived is
established, not assumed: the manifest gives 722 s constant across `warmcool/sans` against
738–771 s for `flush/sans`, and the files on disk have a standard deviation of 0.1 s.

**To do** Re-run with the outputs prefixed by experiment (`flush-avec-runNN.json`).

---

## 1.2 `warm_cool` - hold the GPU inside a temperature window

**Experiment** `gpu-cache` on the RTX 2080 Ti, backend `cuda`, campaign
`results_flush_warmcool_20260831_163028`: 26 working-set sizes × 10 runs per condition,
`warm_cool` ∈ {0, 1}, window 55–70 °C, everything else identical.

![warm_cool: temperature over the campaign](figures/part1_parameters/warmcool-temperature.png)

**What the timeout does.** `warm_cool_timeout` is not a "wait at most N seconds then measure
anyway". The loop checks the clock on each pass and, if the window is still not reached,
**throws**: `Device did not warm up or cool down in the 60s allocated.` The run fails rather
than producing data taken outside the window. The campaigns raised it from its 3 s default to
60 s for that reason - three seconds is not enough to cool a hot card.

**The regulation only acts between batches.** The temperature draws a sawtooth: the card is
brought back to 63–65 °C before each batch, then climbs freely to 73 °C while it runs. Only
**46 % of the points sit inside the requested window**, and the excursions above `max_gpu_temp`
are not a malfunction - the loop exits as soon as the window is reached and never reads the
sensor again until the next batch.

![warm_cool on versus off, with the difference](figures/part1_parameters/warmcool-bandwidth.png)

**The effect on the measurement is real but small.** Median difference across the 26 sizes:
**−0.02 %**. It is not uniform - below ~64 kB, where the working set is cache-resident,
`warm_cool = 0` reads **0.2 to 0.4 % lower**, and on **12 of the 26 sizes** that gap exceeds the
combined inter-run spread of the two conditions. Past that point the difference falls back into
the noise. The direction fits the mechanism: a hotter card clocks slightly lower, and the
cache-resident region is where the measurement depends most on the core clock. Inter-run CoV
improves from **0.048 %** to **0.038 %**.

**Verdict** Below half a percent, in the cache-resident region only. Small enough not to threaten
the comparisons in Parts 2 to 4, large enough to keep the option on when two configurations are
compared at a fraction of a percent - at the cost of a longer campaign.

**One documentation bug found on the way.** In `Benchmark.hpp`, `max_gpu_temp` is described as
*"the minimum accepted temperature before cooling down the GPU"*. It is the maximum; the string
was copied from `min_gpu_temp`.

**Limitation.** `warm_cool = 0` reports **no temperature at all**: `register_stat<DeviceTemperature>()`
is guarded behind `get_warm_cool()`. The thermal drift of a `warm_cool = 0` campaign can only be
inferred from its effect on the metric, never observed. Reading the sensor independently of the
option is the single change that would make this section conclusive rather than indicative.

---

# Part 2 - Native CUDA on the RTX 2080 Ti

Three configurations, the same source workloads:

| Label | Build | Hardware | What it isolates |
|---|---|---|---|
| `cuda` | `release-cuda` | RTX 2080 Ti | Reference |
| `hipifiable@nvidia` | `release-hip-nvidia` | RTX 2080 Ti | Cost of the hipify translation, same hardware |
| `hipifiable@amd` | `release-hip-only` | MI210 | Behaviour on the target architecture |

The first two run on the **same card**, so any gap between them comes from the translation and
from nvcc compiling HIP, never from a hardware difference. The third changes the hardware, so
it is read against AMD's published figures rather than against the first two.

Each configuration gets its own part: **Part 2** below is the CUDA reference, **Part 3** puts
the two backends side by side on the same card, and **Part 4** is the MI210. Every figure comes
from a campaign of **10 runs**; `logs/` holds the raw output of all three.

Three benchmarks are shown as a dedicated layout rather than a curve, following what the
upstream repository publishes for them: an ILP x TLP table for `gpu-incore`, a 16-column
stride table for `gpu-strides`, and the `T = a + V/b` fit for `gpu-small-kernels`.

---

## 2.1 gpu-cache

- **Measures** the bandwidth of the on-chip caches (L1, then L2).
- **Good for** how fast the caches feed the cores, and reading each cache's size off the curve - bandwidth drops at the working set where the data stops fitting.
- **Method** one thread block per SM re-reads the same buffer in a loop; the buffer size is swept, so the served level shifts L1 → L2 → DRAM as it grows.

![gpu-cache, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_cache.png)

## 2.2 gpu-incore

- **Measures** the latency and throughput of arithmetic instructions (FMA, DIV, SQRT).
- **Good for** the raw cost of each operation, and how much parallelism it takes to hide that latency.
- **Method** runs chains of one operation while sweeping ILP (1–8 independent chains) against TLP (warps per SM); reports cycles per operation.

![gpu-incore, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_incore.png)

## 2.3 gpu-l2-stream

- **Measures** the bandwidth of the shared L2 cache, and of DRAM beyond it.
- **Good for** the sustained L2 vs main-memory bandwidth, and the footprint where the L2 stops helping.
- **Method** the four STREAM kernels (read, write, scale, triad) over a buffer whose size is swept across the L2 capacity.

![gpu-l2-stream, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_l2_stream.png)

## 2.4 gpu-latency

- **Measures** memory access latency - the time for a single dependent load.
- **Good for** how many cycles a load costs at each level (L1 / L2 / DRAM), which sets how much work must be in flight to hide it.
- **Method** pointer chasing: one warp walks a buffer in random order, each load depending on the previous, so nothing can mask the round trip.

![gpu-latency, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_latency.png)

## 2.5 gpu-memcpy

- **Measures** host ↔ device transfer bandwidth, over the PCIe link.
- **Good for** the cost of moving data on and off the GPU, and the transfer size at which PCIe saturates.
- **Method** copies buffers of increasing size between CPU and GPU and times them.

![gpu-memcpy, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_memcpy.png)

## 2.6 gpu-roofline

- **Measures** compute throughput (GFLOP/s) against arithmetic intensity (FLOP per byte moved).
- **Good for** telling whether a kernel is limited by memory or by compute - the roofline model; the elbow is the crossover.
- **Method** sweeps the arithmetic intensity and plots the throughput actually reached.

![gpu-roofline, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_roofline.png)

## 2.7 gpu-small-kernels

- **Measures** the fixed cost of launching a kernel, against its bandwidth.
- **Good for** the smallest data volume worth a kernel launch: below it you pay mostly for the launch, not for the work.
- **Method** enqueues thousands of tiny `scale` kernels of varying size and fits `T = a + V/b` - `a` is the launch overhead, `b` the asymptotic bandwidth.

![gpu-small-kernels, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_small_kernels.png)

## 2.8 gpu-strides

- **Measures** L1 bandwidth as a function of the access stride.
- **Good for** what non-contiguous access costs: strided reads collapse the bandwidth through cache-bank conflicts, and the table shows which strides hurt.
- **Method** a single block reads with strides 1…N; the result is tabulated as bytes per cycle for each stride.

![gpu-strides, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_strides.png)

## 2.9 gpu-umstream

- **Measures** unified (managed) memory bandwidth, with a prefetch to the GPU.
- **Good for** the cost of unified memory against explicit copies, and its behaviour when the dataset exceeds the card's memory.
- **Method** STREAM over a unified-memory array prefetched to the device, sweeping the transfer size past the card's 11 GB.

![gpu-umstream, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_umstream.png)

---

# Part 3 - CUDA versus HIP on the same NVIDIA card

Both campaigns ran on an RTX 2080 Ti of the same node, over **identical sweep points**, so the
comparison is point by point and the only variable is the backend.

Each figure carries the **mean of the 10 runs** of each backend, the shaded band being their
min–max spread, and a lower panel giving the HIP/CUDA ratio. That lower panel is where the
gap is actually readable: on most benchmarks the two means sit on top of each other.

**Read the envelopes before the gap.** Where the two min–max bands overlap, the difference
between the means is inside the run-to-run noise and means nothing.

**One caveat on the hardware.** The two campaigns used different PCI slots of the same node
(`3B:00.0` for CUDA, `5E:00.0` for HIP), recorded in each `metadata.json`. Same GPU model,
same node, but not guaranteed to be the same physical die.

---

## 3.1 gpu-cache

![gpu-cache, CUDA vs HIP on RTX 2080 Ti](figures/part2_cuda_vs_hip_2080ti/p2_cuda_vs_hip_gpu_cache.png)

## 3.2 gpu-incore

![gpu-incore, CUDA vs HIP on RTX 2080 Ti](figures/part2_cuda_vs_hip_2080ti/p2_cuda_vs_hip_gpu_incore.png)

## 3.3 gpu-l2-stream

![gpu-l2-stream, CUDA vs HIP on RTX 2080 Ti](figures/part2_cuda_vs_hip_2080ti/p2_cuda_vs_hip_gpu_l2_stream.png)

## 3.4 gpu-latency

![gpu-latency, CUDA vs HIP on RTX 2080 Ti](figures/part2_cuda_vs_hip_2080ti/p2_cuda_vs_hip_gpu_latency.png)

## 3.5 gpu-memcpy

![gpu-memcpy, CUDA vs HIP on RTX 2080 Ti](figures/part2_cuda_vs_hip_2080ti/p2_cuda_vs_hip_gpu_memcpy.png)

## 3.6 gpu-roofline

![gpu-roofline, CUDA vs HIP on RTX 2080 Ti](figures/part2_cuda_vs_hip_2080ti/p2_cuda_vs_hip_gpu_roofline.png)

## 3.7 gpu-small-kernels

![gpu-small-kernels, CUDA vs HIP on RTX 2080 Ti](figures/part2_cuda_vs_hip_2080ti/p2_cuda_vs_hip_gpu_small_kernels.png)

## 3.8 gpu-strides

![gpu-strides, CUDA vs HIP on RTX 2080 Ti](figures/part2_cuda_vs_hip_2080ti/p2_cuda_vs_hip_gpu_strides.png)

## 3.9 gpu-umstream

![gpu-umstream, CUDA vs HIP on RTX 2080 Ti](figures/part2_cuda_vs_hip_2080ti/p2_cuda_vs_hip_gpu_umstream.png)

---

# Part 4 - HIP on the AMD MI210

The target architecture, 10 runs, backend `hip`.

**This part is read on its own.** Three benchmarks sweep a different range here than on the
2080 Ti - `gpu-cache` covers 40 points against 26, `gpu-memcpy` 24 against 21, `gpu-umstream`
26 against 19 - and the hardware differs anyway. The curves of Parts 2 and 3 do not
superimpose on these.

The power policy also differs, which matters when reading anything clock-related: the MI210
ran under a **230 W cap with free clocks** (`amd-smi`), while both 2080 Ti campaigns had their
**clocks pinned to TDP** (`nvidia-smi`) and no power cap.

---

## 4.1 gpu-cache

![gpu-cache, HIP on MI210](figures/part3_hip_mi210/p3_mi210_gpu_cache.png)

## 4.2 gpu-incore

![gpu-incore, HIP on MI210](figures/part3_hip_mi210/p3_mi210_gpu_incore.png)

## 4.3 gpu-l2-stream

![gpu-l2-stream, HIP on MI210](figures/part3_hip_mi210/p3_mi210_gpu_l2_stream.png)

## 4.4 gpu-latency

![gpu-latency, HIP on MI210](figures/part3_hip_mi210/p3_mi210_gpu_latency.png)

## 4.5 gpu-memcpy

![gpu-memcpy, HIP on MI210](figures/part3_hip_mi210/p3_mi210_gpu_memcpy.png)

## 4.6 gpu-roofline

![gpu-roofline, HIP on MI210](figures/part3_hip_mi210/p3_mi210_gpu_roofline.png)

## 4.7 gpu-small-kernels

![gpu-small-kernels, HIP on MI210](figures/part3_hip_mi210/p3_mi210_gpu_small_kernels.png)

## 4.8 gpu-strides

![gpu-strides, HIP on MI210](figures/part3_hip_mi210/p3_mi210_gpu_strides.png)

## 4.9 gpu-umstream

![gpu-umstream, HIP on MI210](figures/part3_hip_mi210/p3_mi210_gpu_umstream.png)

---

# Part 5 - Reproducing this

Every figure in this document comes from one of three campaigns, each kept whole under `logs/`:

| Campaign | Backend | Hardware | Used by |
|---|---|---|---|
| `Log_Test_RTX2080ti_cuda` | `cuda` | RTX 2080 Ti | Parts 2 and 3 |
| `Log_Test_RTX2080ti_hip` | `hip` | RTX 2080 Ti | Part 3 |
| `Log_Test_MI210` | `hip` | AMD Instinct MI210 | Part 4 |

Each directory holds 191 files:

```
logs/Log_Test_<target>/
├── metadata.json                the conditions of the campaign
├── <benchmark>.proto.json       the protocol actually used          (9)
├── <benchmark>.run<NN>.json     one file per repetition, NN = 01..10 (90)
├── <benchmark>.run<NN>.json.log the workload's own output           (90)
└── run.log                      the orchestrator log
```

Run numbers are **zero-padded** so alphabetical order matches numerical order.

`metadata.json` records the machine, the GPU and its PCI address, the backend, the build preset,
the baseliner version, the throttling policy and the number of runs. Two limits are worth
stating plainly rather than discovering later:

- **The commit is not recorded.** `git_version` is `not-provided` in all three campaigns: the
  build did not populate it. The binary that produced these numbers cannot be identified from
  the results alone.
- **Driver and toolkit versions are not recorded either.** Neither the campaign files nor the
  result JSON carry them.

The throttling policy is not the same on both cards, and it is a field to read before comparing
anything clock-related: the MI210 ran under a **230 W cap with free clocks** (`amd-smi`), while
both 2080 Ti campaigns had their **clocks pinned to TDP** with no power cap (`nvidia-smi`).

One more thing the result files do not carry: the options of the `Benchmark` section. A report
holds only `id`, `results` and `hardware`, so the value of `flush` or `warm_cool` for a given
run exists **only** in that campaign's `.proto.json` and `metadata.json`. Anything varied there
has to be either promoted to a sweep axis, where it lands in each result's `sweep_point`, or
recorded by hand in the metadata.

The figures under `figures/` are produced from these campaigns by a notebook that is **not yet
in this repository**, so they cannot currently be regenerated from a clone alone.

Build and run instructions are in the [README](README.md). The quickest check that a build is
sane at all:

```bash
./build/release-cuda/gpu-benches_exec run --tiny --output-file tiny.json
```
