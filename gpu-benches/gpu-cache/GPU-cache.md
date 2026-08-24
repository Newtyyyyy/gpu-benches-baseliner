# gpu-cache — Research & Analysis

## Goal

Measure the **effective bandwidth** of the different GPU memory levels (L1, L2, DRAM) by varying the *working set* size per SM. The bandwidth transitions in the resulting curve reveal the capacity and thresholds of each cache level.

---

## Measurement principle

The `sumKernel` kernel launches as many blocks as there are SMs on the GPU. Each SM loads two buffers of `N` floats (`bufA`, `bufB`), multiplies them element-wise and accumulates the result into a local variable. The accumulator is used in an impossible condition (`localSum == 1233`) to force the compiler to keep the loads.

```
bandwidth (GB/s) = (2 × N × sizeof(float) × smCount × ITERS) / time_s / 1e9
```

The factor `2` comes from the two buffers read in sequence. `ITERS ≈ 1e9 / N` is calibrated to keep ~1 s of GPU work per point.

---

## Sweep parameters

| Parameter | Values | Description |
|---|---|---|
| `working_set_elements` (N) | Dense: 128, 256, k×512 (k=1..32), then an exponential series ×1.17 up to ~137 MB | Size in floats of **one** buffer; total memory = 2×N×4 bytes |

The exponential series (`cache_exp_series`) generates values rounded to the nearest multiple of 512, spaced by a factor of ×1.17.


---

## Reported metrics

| Metric | Unit | Description |
|---|---|---|
| `median` | ms | Median execution time per batch |
| `mean` | ms | Mean |
| `CoV` | % | Coefficient of variation (stability) |
| `memory_bandwidth` | GB/s | Computed bandwidth |
| `working_set_kb` | kB | Total size of both buffers (2×N×4 / 1024) |

---

## Configuration parameters (protocol JSON)

| Parameter | Section | Effect |
|---|---|---|
| `lock_clock` | `cuda > Backend` | Locks the GPU to its base clock for reproducibility |
| `working_set_elements` | `sweep > enumerated` | List of N values to test |
| `batch_size` | `Benchmark` | Repetitions per batch |
| `max_nb_repetition` | `StoppingCriterion` | Overall repetition cap |
| `warmup` | `Benchmark` | Warmup run before measuring (recommended: `1`) |
| `flush` | `Benchmark` | Flush L2 before every repetition (recommended: `1`) |

---

## Caveats

- Any value of N missing from the precompiled list raises `std::runtime_error: cache: unsupported working_set_elements value`.
- Without `lock_clock`, the GPU may boost its clock on small working sets (L1), skewing the comparison against the DRAM points.
- Without `flush`, L2 residue from one repetition can artificially improve the next one.
- `BLOCKSIZE` is picked automatically: 512 if N is a multiple of 512, 256 if a multiple of 256, otherwise 128.
