# gpu-incore — Research & Analysis

## Goal

Measure the GPU's **in-core compute throughput** in cycles per operation (`rcp_throughput`), for different arithmetic instruction types, ILP levels and precisions. This benchmark characterizes the GPU functional units without ever saturating memory.

---

## Measurement principle

Each kernel runs `ITERS = 10 000` passes over a chain of independent arithmetic operations (FMA, DIV, SQRT) whose results feed into each other so the compiler cannot eliminate them. Measuring the GPU time yields the reciprocal throughput:

```
rcp_throughput (cycles/op) = (median_time_ms × 1e-3) × frequency_GHz × 1e9 / ops_per_run
```

where `ops_per_run = N_type × ITERS × warp_count` with:
- `N_FMA = 1024` FMA operations per pass
- `N_OTHER = 128` operations per pass for DIV and SQRT

---

## Sweep parameters

| Parameter | Values | Description |
|---|---|---|
| `kernel_type` | `fma-mixed`, `fma-separated`, `div`, `sqrt` | Arithmetic instruction type |
| `ilp` | 1, 2, 4, 8 | *Instruction-Level Parallelism*: number of independent chains per thread |
| `precision` | `float`, `double` | Floating-point precision |
| `warp_count` | 1, 2, 4, 8, 16, 32 | Number of warps per block (block_size = 32 × warp_count) |

The full sweep amounts to **4 × 4 × 2 × 6 = 192 measurement points**.

---

## Kernel types

| `kernel_type` | Description | Targeted functional unit |
|---|---|---|
| `fma-mixed` | FMA with dependencies between operations (chained latency) | FP ALU + pipeline |
| `fma-separated` | FMA with independent operations (pure ILP) | FP ALU (throughput) |
| `div` | Floating-point division | SFU (Special Function Unit) |
| `sqrt` | Floating-point square root | SFU |

---

## Reported metrics

| Metric | Unit | Description |
|---|---|---|
| `median` | ms | Median execution time |
| `ops_per_run` | ops | Total number of operations computed (`N_type × ITERS × warp_count`) |
| `clock_frequency` | GHz | GPU clock measured during the run |
| `rcp_throughput` | cycles/op | Reciprocal throughput — **main metric** |

---

## Configuration parameters (protocol JSON)

| Parameter | Section | Effect |
|---|---|---|
| `kernel_type` | `sweep > enumerated` | Operation types to test |
| `ilp` | `sweep > enumerated` | ILP levels to test |
| `precision` | `sweep > enumerated` | Precisions to test |
| `warp_count` | `sweep > enumerated` | Occupancies to test |
| `lock_clock` | `cuda > Backend` | Recommended `1`: the clock is part of the computation |

---

## Caveats

- `lock_clock` is **critical** here: the GPU clock is used directly in the `rcp_throughput` computation. A varying clock skews the metric.
- `number_of_floating_point_operations()` returns `nullopt` on purpose — the benchmark does not count FLOP/s but cycles/op.
- DIV and SQRT use `N_OTHER = 128` (vs `N_FMA = 1024`) because they are intrinsically slower: this keeps execution times comparable.
