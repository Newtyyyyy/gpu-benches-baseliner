# gpu-l2-stream — Research & Analysis

## Goal

Measure the **bandwidth of STREAM-style kernels** as a function of the reuse footprint. A fixed, very large number of work items is launched, but each thread's index wraps modulo `length`. Sweeping `length` moves the working set across the cache hierarchy, so the resulting curve shows where L2 stops holding the data and DRAM takes over.

---

## Measurement principle

Four STREAM kernels are available. All of them index into their buffers with the same wrapping expression:

```
tidx = (threadIdx.x + blockIdx.x * blockDim.x) % length
```

The launch always covers `ITERATION_COUNT = 1024³ + 2` work items with `BLOCKSIZE = 256`, no matter the value of `length`. The amount of work is therefore constant, while the footprint touched by that work is not.

| Kernel | Operation | Streams counted |
|---|---|---|
| `read` | `temp = B[tidx]`, stored only under an impossible condition | 1 |
| `write` | `A[tidx] = 0.23` | 1 |
| `scale` | `A[tidx] = B[tidx] * 1.2` | 2 |
| `triad` | `A[tidx] = B[tidx] * D[tidx] + C[tidx]` | 4 |

The `read` kernel guards its store with `if (temp == -1.0)`, a condition that is never true, so the load cannot be optimized away while no store traffic is generated.

```
number_of_bytes = streams × ITERATION_COUNT × sizeof(double)
bandwidth (GB/s) = number_of_bytes / time_s / 1e9
```

Note that the byte count uses `ITERATION_COUNT`, not `length`: it counts the traffic the kernel *requests*, not the distinct memory it touches. That is the point — comparing requested traffic against elapsed time is what exposes the caching effect.

---

## Sweep parameters

| Parameter | Values | Description |
|---|---|---|
| `kernel_type` | `read`, `scale`, `write`, `triad` | STREAM kernel to run |
| `length` | 155 values from 24 545 to 96 387 041 doubles (~192 kB to ~736 MB per buffer) | Number of doubles the per-thread index wraps around |

The `length` values are deliberately odd, non-power-of-two numbers. Powers of two would map onto a small number of cache sets and produce conflict misses that hide the real capacity effect.

---

## Reported metrics

| Metric | Unit | Description |
|---|---|---|
| `median` | ms | Median kernel execution time |
| `mean` | ms | Mean |
| `CoV` | % | Coefficient of variation (stability) |
| `memory_bandwidth` | GB/s | Requested traffic divided by elapsed time |

---

## Configuration parameters (protocol JSON)

| Parameter | Section | Effect |
|---|---|---|
| `kernel_type` | `sweep > enumerated` | Which STREAM kernels to test |
| `length` | `sweep > enumerated` | Range of reuse footprints |
| `lock_clock` | `cuda > Backend` | Recommended `1`: small footprints let the GPU boost |
| `batch_size` | `Benchmark` | Repetitions per batch |
| `flush` | `Benchmark` | Flush L2 before every repetition (recommended `1`) |

---

## Differences with gpu-cache

Both sweep a footprint and report bandwidth, but they do it from opposite directions:

| Aspect | gpu-cache | gpu-l2-stream |
|---|---|---|
| **Work per point** | Scales with N (`ITERS ≈ 1e9 / N`) | Constant (`ITERATION_COUNT` fixed) |
| **Access pattern** | Two buffers read in sequence | One of four STREAM patterns, index wrapped |
| **Bytes counted** | Data actually streamed | Traffic requested by the launch |
| **Focus** | Capacity and thresholds of each level | Behaviour of realistic STREAM kernels around L2 |

---

## Caveats

- The four buffers are allocated at `length × sizeof(double)` each, so the largest sweep point needs roughly 3 GB of device memory for `triad`. Trim the top of the sweep on smaller cards.
- `reset_device` is a no-op: the buffers keep whatever the previous run left in them. Since the kernels never depend on the values, this only matters for cache state, which `flush` handles.
- Occupancy control is intentionally absent. Upstream used a shared-memory "spoiler" allocation to throttle blocks per SM for an occupancy sweep; that sweep is not reproduced here.
- `write` and `read` both count a single stream, but they are not symmetric: write-allocate policies can make the `write` kernel move twice the traffic you would expect on some architectures.
