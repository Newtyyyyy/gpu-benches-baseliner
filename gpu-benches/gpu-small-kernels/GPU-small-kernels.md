# gpu-small-kernels — Research & Analysis

## Goal

Characterize how **kernel launch overhead dominates at small problem sizes**. A trivial scale kernel is run over data sets ranging from a few thousand to over a hundred million doubles. At the small end the fixed cost of a launch outweighs the work; at the large end the kernel saturates DRAM bandwidth. The crossover tells you the smallest problem worth sending to the GPU.

---

## Measurement principle

The kernel is deliberately minimal — one load, one multiply, one store, one bounds check:

```cpp
__global__ void small_scale(double* A, const double* B, int size) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= size) return;
    A[tidx] = B[tidx] * 0.25;
}
```

Grid size is `size / block_size + 1`. Bandwidth counts one read and one write per element:

```
number_of_bytes = size × 2 × sizeof(double)
bandwidth (GB/s) = number_of_bytes / time_s / 1e9
```

At small sizes the measured time is essentially launch latency, so the computed bandwidth collapses toward zero — that collapse *is* the measurement. As `size` grows, bandwidth climbs and eventually flattens at the DRAM peak.

---

## Sweep parameters

| Parameter | Values | Description |
|---|---|---|
| `size` | 179 values from 4 096 to 130 611 580 doubles (~32 kB to ~1 GB per buffer) | Number of doubles processed |
| `block_size` | 32, 64, 128, 256, 512, 1024 | CUDA thread block size |

The `size` series is geometric with a ratio near 1.06, dense enough to resolve the knee precisely. The full cartesian product is **179 × 6 = 1074 points**.

---

## Reported metrics

| Metric | Unit | Description |
|---|---|---|
| `median` | ms | Median kernel execution time |
| `mean` | ms | Mean |
| `CoV` | % | Coefficient of variation (stability) |
| `memory_bandwidth` | GB/s | Effective bandwidth (2 × size × 8 bytes / time) |

The interesting reading is `median` at the small end — it plateaus at the launch overhead floor, typically a few microseconds — and `memory_bandwidth` at the large end.

---

## Configuration parameters (protocol JSON)

| Parameter | Section | Effect |
|---|---|---|
| `size` | `sweep > enumerated` | Range of data set sizes |
| `block_size` | `sweep > enumerated` | Block sizes to test |
| `batch_size` | `Benchmark` | Repetitions per batch. Keep it high: individual runs are very short |
| `warmup` | `Benchmark` | Essential — the first launch of a kernel is slower than the rest |
| `lock_clock` | `cuda > Backend` | Recommended `1` |

---

## Caveats

- Timing resolution is the limiting factor at the small end. A launch takes single-digit microseconds; make sure `batch_size` is large enough that a batch is comfortably longer than the timer's granularity.
- `warmup` is not optional here. The very first launch of a kernel pays module loading and JIT costs that are orders of magnitude above the steady-state overhead being measured.
- `block_size` matters far less than `size` for this benchmark. Its main visible effect is at the large end, where too small a block limits occupancy.
- Upstream had `-graph`, `-pta` and `-pt-gsync` launch-batching modes to amortize overhead across many launches. They are dropped here: the baseliner already times each call directly.
- Both buffers are allocated at `size × sizeof(double)`, so the top of the sweep needs about 2 GB of device memory.
