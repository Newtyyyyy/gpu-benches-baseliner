# gpu-umstream — Research & Analysis

## Goal

Measure the **Unified Memory bandwidth** with a TRIAD-style kernel (`C = A + B`) over arrays allocated with `cudaMallocManaged`. This benchmark evaluates the behavior of the GPU/CPU page migration system, with and without explicit prefetching.

---

## Measurement principle

The kernel performs a TRIAD operation over three arrays of doubles (`A`, `B`, `C`) allocated in Unified Memory:

```
C[i] = A[i] + B[i]    for all i
```

The data may live in RAM or in VRAM depending on the access history. The effective bandwidth is computed over the 3 arrays (read A, read B, write C):

```
number_of_bytes = item_count × sizeof(double) × 3
bandwidth (GB/s) = number_of_bytes / time_s / 1e9
```

---

## Sweep parameters

| Parameter | Values | Description |
|---|---|---|
| `transfer_mb` | 1, 2, 4, ..., 512 (powers of 2) | Size per array in MB (1 → 512 MB) |
| `prefetch` | `false`, `true` | Prefetch UM to the GPU before the run |

The cartesian product gives **20 points** (10 sizes × 2 prefetch modes).

---

## Unified Memory: behavior without vs with prefetch

| Mode | Mechanism | Performance impact |
|---|---|---|
| `prefetch=false` | Pages migrated on demand on first GPU access (page fault) | High latency, low apparent bandwidth (~PCIe) |
| `prefetch=true` | `cudaMemPrefetchAsync` moves the pages to the GPU before the kernel | DMA migration, bandwidth close to on-GPU HBM/GDDR |

---

## Reported metrics

| Metric | Unit | Description |
|---|---|---|
| `median` | ms | Median kernel execution time |
| `mean` | ms | Mean |
| `CoV` | % | Stability (high when page faults are irregular) |
| `memory_bandwidth` | GB/s | Effective bandwidth (3 × transfer_mb × 1024² / time) |

---

## Configuration parameters (protocol JSON)

| Parameter | Section | Effect |
|---|---|---|
| `transfer_mb` | `sweep > PowersOfTwo` | Range of per-array sizes |
| `prefetch` | `sweep > enumerated` | Enables/disables UM prefetch |
| `blocksize` | `GpuUmstream` | CUDA block size (default: 256) |
| `lock_clock` | `cuda > Backend` | Recommended `1` |
| `warmup` | `Benchmark` | Warmup matters: the first run without prefetch triggers the migrations |

---

## Comparison with gpu-memcpy

| Aspect | gpu-memcpy | gpu-umstream |
|---|---|---|
| **Memory** | Allocated separately (host + device) | Unified Memory (cudaMallocManaged) |
| **Transfer** | Explicit `cudaMemcpy` | Implicit (migration) or prefetch |
| **Compute** | None (pure transfer) | TRIAD `C = A + B` |
| **What is measured** | Raw PCIe bandwidth | Effective UM bandwidth (migration + compute) |
| **Use case** | Calibrate the PCIe link | Assess UM overhead on a real workload |

---

## Caveats

- `reset_device` is a no-op (`{}`): UM arrays are not reinitialized between runs. The `warmup` guarantees the pages are already in their final state before measuring.
- `m_block_count` is not part of the sweep — it is computed implicitly from `item_count` and `block_size`.
