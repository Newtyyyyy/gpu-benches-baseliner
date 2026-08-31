# Results

Measurements collected with this repository, kept on the `helio_log` branch alongside the raw
runs that produced them. Two things are shown here:

1. **What each benchmark knob actually changes** — measured, not assumed.
2. **How the same benchmark behaves across architectures** — native CUDA, hipified HIP on the
   same NVIDIA card, and HIP on a real AMD card.

The methodology of each benchmark lives in its own `gpu-benches/<name>/GPU-<name>.md`. This
document only reports what was observed.

```
RESULTS.md          this file
logs/               raw campaign output, one directory per campaign
figures/            the plots referenced below, as PNG
```

---

# Part 1 — Impact of the benchmark parameters

Every plot in this part compares the **same workload on the same GPU**, changing exactly one
option. The point is to justify the defaults, and to make visible what silently breaks a
measurement when a knob is wrong.

Two options are covered here, the two that were actually varied during this work: the L2 flush
and the thermal control. Both belong to the `Benchmark` section of the protocol. The measurement
loop applies them in this order:

```
setup_device
  └─ warmup            one untimed run
  repeat until max_nb_repetition:
     ├─ warm_cool      hold the GPU inside a temperature window
     ├─ block          freeze the stream, queue the whole batch
     └─ for each of batch_size runs:
          ├─ reset_device
          ├─ flush     empty L2 — before every repetition, not every batch
          └─ timed run
     └─ dynamic_batch  resize batch_size from the batch duration
```

---

## 1.1 `flush` — L2 flush before every repetition

**Default** `1`. Empties the L2 cache before each timed run, so every repetition starts from a
cold cache. Note it happens before *every repetition*, not between batches.

**Why it matters** Without it, data left in L2 by the previous repetition is still resident when
the next one starts. The kernel then reads from cache what it was supposed to read from memory,
and the reported bandwidth is not the bandwidth of the level being probed.

**Expected** No effect once the working set is far larger than L2 — the data could not have
stayed resident anyway. A large effect in the L1/L2 region.

**Experiment** `gpu-cache`, full `working_set_elements` sweep, with `Benchmark.flush` as a
second sweep axis over {0, 1}, so both curves come from a single campaign under identical
thermal conditions.

<!-- figure: figures/flush-gpu-cache.svg -->
> _(to fill: bandwidth vs working set, one curve per flush value)_

**Observed** _(to fill)_

**Verdict** _(to fill)_

---

## 1.2 `warm_cool` — hold the GPU inside a temperature window

**Default** `0`, window `45–60 °C`, `warm_cool_timeout` `3 s`. Before each batch, a warming
kernel runs until the GPU reaches `min_gpu_temp`, or the GPU is cooled until it drops below
`max_gpu_temp`. If the window is not reached within the timeout, the run throws.

**Why it matters** A campaign is long. Its first benchmarks run on a cold GPU and its last ones
on a hot, possibly throttled one. That drift is indistinguishable from a real difference between
the benchmarks unless the temperature is held.

**Expected** The effect should appear as a trend across the runs of a campaign, not inside a
single run. With `warm_cool=0`, the metric drifts with run index; with `warm_cool=1`, it should
stay flat.

**Experiment** The same campaign twice, 10 runs each, `warm_cool=0` then `warm_cool=1`, plotting
the metric against run index. Unlike `flush`, this cannot be a sweep axis: it is a
time-dependent effect, so the two conditions have to be separate campaigns.

<!-- figure: figures/warm-cool-drift.svg -->
> _(to fill: metric vs run index, one curve per warm_cool value)_

**Observed** _(to fill)_

**Verdict** _(to fill)_

---

# Part 2 — Native CUDA on the RTX 2080 Ti

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

_(to fill: bandwidth per memory level, three curves; L1/L2 thresholds vs specs)_

![gpu-cache, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_cache.png)

## 2.2 gpu-incore

_(to fill: rcp_throughput vs theoretical 32/N; FP32/FP64 ratio)_

![gpu-incore, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_incore.png)

## 2.3 gpu-l2-stream

![gpu-l2-stream, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_l2_stream.png)

## 2.4 gpu-latency

![gpu-latency, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_latency.png)

## 2.5 gpu-memcpy

![gpu-memcpy, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_memcpy.png)

## 2.6 gpu-roofline

![gpu-roofline, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_roofline.png)

## 2.7 gpu-small-kernels

![gpu-small-kernels, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_small_kernels.png)

## 2.8 gpu-strides

![gpu-strides, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_strides.png)

## 2.9 gpu-umstream

![gpu-umstream, CUDA on RTX 2080 Ti](figures/part1_cuda_2080ti/p1_cuda2080_gpu_umstream.png)

---

# Part 3 — CUDA versus HIP on the same NVIDIA card

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

# Part 4 — HIP on the AMD MI210

The target architecture, 10 runs, backend `hip`.

**This part is read on its own.** Three benchmarks sweep a different range here than on the
2080 Ti — `gpu-cache` covers 40 points against 26, `gpu-memcpy` 24 against 21, `gpu-umstream`
26 against 19 — and the hardware differs anyway. The curves of Parts 2 and 3 do not
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

# Part 5 — Reproducing this

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
