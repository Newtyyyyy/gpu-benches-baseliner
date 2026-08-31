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
figures/            the plots referenced below, as SVG
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

# Part 2 — Cross-architecture comparison

Three configurations, the same source workloads:

| Label | Build | Hardware | What it isolates |
|---|---|---|---|
| `cuda` | `release-cuda` | RTX 2080 Ti | Reference |
| `hipifiable@nvidia` | `release-hip-nvidia` | RTX 2080 Ti | Cost of the hipify translation, same hardware |
| `hipifiable@amd` | `release-hip-only` | MI210 | Behaviour on the target architecture |

The first two run on the **same card**, so any gap between them comes from the translation and
from nvcc compiling HIP, never from a hardware difference. The third changes the hardware, so
it is read against AMD's published figures rather than against the first two.

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

# Part 3 — Reproducing this

Each campaign directory under `logs/` carries a `metadata.json` recording the machine, the GPU,
the driver and toolkit versions, the preset used and the commit the binary was built from, so a
figure can always be traced back to the conditions that produced it.

```
logs/<date>_<machine>_<variant>/
├── metadata.json
├── <benchmark>.proto.json      the protocol actually used
├── <benchmark>.run<N>.json     one file per repetition of the campaign
└── run.log
```

Build and run instructions are in the [README](README.md). The quickest check that a build is
sane at all:

```bash
./build/release-cuda/gpu-benches_exec run --tiny --output-file tiny.json
```
