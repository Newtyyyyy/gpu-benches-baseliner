# gpu-latency — Research & Analysis

## Goal

Measure the GPU's **memory access latency** in nanoseconds per memory level (L1, L2, DRAM) using a *pointer chasing* technique. Unlike gpu-cache which measures bandwidth, this benchmark forces dependent sequential accesses in order to isolate pure latency.

---

## Measurement principle

The kernel walks a chain of randomly shuffled pointers inside a buffer of size `buffer_size_kb`. Each memory access depends on the result of the previous one (pointer chase), which prevents speculative or out-of-order execution. After `iterations` accesses, the total time gives:

```
latency_ns = (median_time_ms × 1e6) / iterations
```

The buffer is initialized as a random permutation of indices (a linked list in memory), which defeats the hardware prefetcher.

---

## Sweep parameters

| Parameter | Values | Description |
|---|---|---|
| `buffer_size_kb` | 16, 32, 64, ... 524288 (powers of 2) | Pointer chase buffer size (16 kB → 512 MB) |
| `iterations` | 100 000 (fixed) | Number of chained accesses per measurement |

---


## Reported metrics

| Metric | Unit | Description |
|---|---|---|
| `median` | ms | Median total time for `iterations` accesses |
| `iterations` | — | Number of chained accesses (set via JSON) |
| `latency_ns` | ns | **Main metric**: latency per access |

`latency_ns` is computed in `LatencyNsStat` from the median and the iteration count.

---

## Configuration parameters (protocol JSON)

| Parameter | Section | Effect |
|---|---|---|
| `buffer_size_kb` | `sweep > PowersOfTwo` | Range of buffer sizes |
| `iterations` | `GpuLatency` | Number of pointer chases (default: 100 000) |
| `lock_clock` | `cuda > Backend` | Recommended `1` for stability |
| `warmup` | `Benchmark` | Warmup run to prime the cache |

---

## Differences with gpu-cache

| Aspect | gpu-cache | gpu-latency |
|---|---|---|
| **Metric** | Bandwidth (GB/s) | Latency (ns) |
| **Accesses** | Sequential (streaming) | Randomly chained (pointer chase) |
| **Prefetcher** | Exploited | Defeated |
| **Parallelism** | All SMs at once | Generally 1 active thread per chain |
| **What is measured** | Aggregate throughput | Latency of a single access |

---

## Caveats

- `m_dummy_buffer` is allocated separately to prevent compiler optimizations on the final result (the kernel writes into it so the pointer chase is not eliminated).
- `reset_device` regenerates the permutation between runs to guarantee consistent random accesses.
- With too few `iterations` (< 10 000), kernel launch noise dominates the measurement.
- The chain length (`m_chain_length`) is computed in `setup_device` from the buffer size.
