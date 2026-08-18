# gpu-strides — Research & Analysis

## Goal

Probe **L1 and register-file access efficiency** as a function of the access pattern. Two kernels sweep a stride or a thread-block geometry, and on the CUDA backend three hardware counters are read through CUPTI, normalized per warp instruction. The result shows how many wavefronts and L2 sectors a given pattern costs compared with the ideal.

This is the only benchmark in the suite that reports hardware performance counters rather than derived timings.

---

## Measurement principle

Both kernels run as a **single 1024-thread block**, so everything happens on one SM with no inter-SM contention. Each performs `N = 18 × 64 × 1024` iterations of eight strided reads:

```cpp
for (int n = 0; n < N; n += 8) {
    A2 += zero;                    // zero is always 0, defeats hoisting
    sum += A2[0]  * A2[4];
    sum += A2[8]  * A2[12];
    sum += A2[20] * A2[16];
    sum += A2[24] * A2[28];
}
if (sum == T(123123.23)) B[tidx] = sum;
```

The `A2 += zero` is a compile-time-opaque no-op that prevents the compiler from hoisting the loads out of the loop, and the impossible final comparison keeps `sum` alive.

| `kernel_type` | Base pointer | What `arg` controls |
|---|---|---|
| `stride` | `A + (tidx % 64) * stride` | The stride itself. 64 distinct lanes re-read a small window |
| `block` | `A + tidx + tidy * pitch + pitch` | Block geometry: `dim3(w, 1024/w)`, pitch fixed at 4098 |

For `block`, `arg` is rounded **down** to the nearest divisor of 1024, so 1..64 collapses onto {1, 2, 4, 8, 16, 32, 64}. Several consecutive `arg` values therefore produce identical configurations.

### Hardware counters

On a native CUDA build, `setup_device` runs the kernel three extra times under CUPTI/Perfworks, one per counter:

| Metric reported | CUPTI counter |
|---|---|
| `l1_lsu_wavefronts_per_warp` | `l1tex__data_pipe_lsu_wavefronts.sum` |
| `l1_ld_wavefronts_per_warp` | `l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum` |
| `l2_sectors_per_warp` | `lts__t_sectors.sum` |

Each raw count is divided by the normalization factor:

```
METRIC_NORM = 1024 threads × N iterations / 32 lanes per warp
```

so the reported numbers are **per warp instruction**, directly comparable across configurations. A perfectly coalesced access costs 1 wavefront per warp; strided patterns cost more.

---

## Sweep parameters

| Parameter | Values | Description |
|---|---|---|
| `kernel_type` | `stride`, `block` | Access pattern |
| `precision` | `float`, `double` | Element type |
| `arg` | 0 to 64 | Stride, or warp count for `block` |

The full sweep is **2 × 2 × 65 = 260 points**.

---

## Reported metrics

| Metric | Unit | Description |
|---|---|---|
| `median` | ms | Median kernel execution time |
| `mean` | ms | Mean |
| `CoV` | % | Coefficient of variation (stability) |
| `l1_lsu_wavefronts_per_warp` | — | LSU pipe wavefronts, per warp instruction |
| `l1_ld_wavefronts_per_warp` | — | Global-load output wavefronts, per warp instruction |
| `l2_sectors_per_warp` | — | L2 sectors touched, per warp instruction |

The three counters have `ONCE` granularity: they are measured during setup, not per repetition, so they carry no statistical spread.

---

## Configuration parameters (protocol JSON)

| Parameter | Section | Effect |
|---|---|---|
| `kernel_type` | `sweep > enumerated` | Access patterns to test |
| `precision` | `sweep > enumerated` | Element types to test |
| `arg` | `sweep > enumerated` | Strides / warp counts to test |
| `lock_clock` | `cuda > Backend` | Recommended `1` for the timing part |
| `batch_size` | `Benchmark` | Repetitions per batch |

---

## Caveats

- **The counters are CUDA-only.** They require `STRIDES_ENABLE_CUPTI`, which is defined solely for the native CUDA build. The hipified variant compiles without the CUPTI engine and reports **0** for all three — that is expected, not a failure. There is no ROCm equivalent wired up; a hand-tuned `hip/` port could add rocprofiler.
- The combined executable links CUPTI at the top level, not in the benchmark's own target: object libraries do not propagate their usage requirements through `$<TARGET_OBJECTS:…>`. If you build with `COMBINED_BUILD=OFF`, the per-benchmark target carries them itself.
- For `kernel_type=block`, `arg` values that are not divisors of 1024 are silently rounded down. Expect plateaus of identical results rather than 65 distinct points.
- `arg=0` with `stride` means every lane reads the same address — a broadcast, which is the cheapest possible pattern and a useful baseline. With `block`, `arg=0` is treated as 1.
- The CUPTI engine uses `SCOPE_EXIT` RAII macros that trip `-Wunused-value`; the build relaxes that one warning for this object only, since the project compiles with `-Werror`.
- Counter collection re-runs the kernel three times during setup. That extra work is not part of the timing measurement, but it does make setup noticeably slower for this benchmark.
