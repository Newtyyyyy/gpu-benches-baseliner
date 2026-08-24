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

The options come from the `Benchmark`, `Backend` and `StoppingCriterion` sections of the
protocol. The measurement loop applies them in this order:

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
cold cache.

**Expected** No effect once the working set is far larger than L2, since the data could not
have stayed resident anyway. A large effect in the L1/L2 region, where a warm cache from the
previous repetition would inflate the bandwidth.

**Experiment** `gpu-cache`, full `working_set_elements` sweep, `flush=0` vs `flush=1`.

<!-- figure: figures/flush-gpu-cache.svg -->
> _(to fill: bandwidth vs working set, two curves)_

**Observed** _(to fill)_

**Verdict** _(to fill)_

---

## 1.2 `warmup` — one untimed run first

**Default** `1`. Runs the workload once without timing it, and reports the duration separately
as `warmup_time`.

**Expected** Removes module loading and JIT from the first measurement. The effect should be
largest where a single run is short, so `gpu-small-kernels` at the low end of its `size` sweep.

**Experiment** `gpu-small-kernels`, `warmup=0` vs `warmup=1`; compare `warmup_time` against
the `median`.

<!-- figure: figures/warmup-small-kernels.svg -->
> _(to fill)_

**Observed** _(to fill)_

**Verdict** _(to fill)_

---

## 1.3 `block` — queue the whole batch before running it

**Default** `1`. A blocking kernel holds the stream while the whole batch is enqueued, then
releases it, so per-launch enqueue cost does not land inside the timed section. The stream is
released every `block_queue_size` (64) runs to keep the queue bounded.

**Expected** Matters where launch overhead is comparable to the kernel itself — again
`gpu-small-kernels`, and any benchmark at a small sweep point.

**Experiment** `gpu-small-kernels`, `block=0` vs `block=1`, across the `size` sweep.

<!-- figure: figures/block-small-kernels.svg -->
> _(to fill)_

**Observed** _(to fill)_

**Verdict** _(to fill)_

---

## 1.4 `batch_size` and `dynamic_batch`

**Defaults** `batch_size=25`, `dynamic_batch=0`, `minimal_batch_duration=10 ms`. With the
dynamic mode on, the batch size doubles when a batch runs shorter than the target and halves
when it runs more than twice as long.

**Expected** Larger batches amortize timer granularity, so the coefficient of variation should
fall as `batch_size` grows, then flatten once the timer stops being the limit.

**Experiment** One short workload, `batch_size` swept over 1, 5, 25, 100, 400; plot `CoV`.

<!-- figure: figures/batch-size-cov.svg -->
> _(to fill)_

**Observed** _(to fill)_

**Verdict** _(to fill)_

---

## 1.5 `lock_clock` — pin the GPU clocks

**Default** `0`. Locks the clocks, optionally between `min_clock_value` and `max_clock_value`.

**Expected** The most consequential knob of all for `gpu-incore`, where the measured clock
enters the `rcp_throughput` formula directly: a drifting clock does not add noise, it biases
the metric. Elsewhere it mostly narrows the spread between repetitions.

**Experiment** `gpu-incore`, `lock_clock=0` vs `lock_clock=1`, plotting both
`clock_frenquency` and `rcp_throughput` over the sweep.

<!-- figure: figures/lock-clock-incore.svg -->
> _(to fill)_

**Observed** _(to fill)_

**Verdict** _(to fill)_

---

## 1.6 `warm_cool` — hold a temperature window

**Defaults** `0`, window `45–60 °C`, `warm_cool_timeout=3 s`. Before each batch, a warming
kernel runs until the GPU reaches `min_gpu_temp`, or the GPU is cooled until it drops under
`max_gpu_temp`. Throws if the window is not reached within the timeout.

**Expected** Matters on long campaigns, where the first benchmarks run on a cold GPU and the
last ones on a hot, possibly throttled one. Should show up as drift across the 10 runs of a
campaign rather than within one run.

**Experiment** Same campaign twice, `warm_cool=0` then `1`; plot the metric against run index.

<!-- figure: figures/warm-cool-drift.svg -->
> _(to fill)_

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

<!-- figure: figures/arch-gpu-cache.svg -->

## 2.2 gpu-incore

_(to fill: rcp_throughput vs theoretical 32/N; FP32/FP64 ratio)_

<!-- figure: figures/arch-gpu-incore.svg -->

## 2.3 gpu-l2-stream

<!-- figure: figures/arch-gpu-l2-stream.svg -->

## 2.4 gpu-latency

<!-- figure: figures/arch-gpu-latency.svg -->

## 2.5 gpu-memcpy

<!-- figure: figures/arch-gpu-memcpy.svg -->

## 2.6 gpu-roofline

<!-- figure: figures/arch-gpu-roofline.svg -->

## 2.7 gpu-small-kernels

<!-- figure: figures/arch-gpu-small-kernels.svg -->

## 2.8 gpu-strides

<!-- figure: figures/arch-gpu-strides.svg -->

## 2.9 gpu-umstream

<!-- figure: figures/arch-gpu-umstream.svg -->

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
