# gpu-roofline — Research & Analysis

## Goal

Trace the GPU's **roofline curve** by sweeping arithmetic intensity. The kernel performs a fixed amount of memory traffic and a configurable amount of arithmetic per element, so increasing `n` walks the workload from memory-bound to compute-bound. The knee of the curve marks the machine balance.

---

## Measurement principle

Each thread iterates over `M = 4000` elements in steps of 2. Per pair it loads two values from `A` and two from `B`, forms two differences, and then runs `n` inner FMA iterations on each:

```cpp
T a = sA[i * BLOCKSIZE], b = sB[i * BLOCKSIZE];
T v = a - b;
for (int k = 0; k < N; k++)
    v = v * a - b;
```

The accumulated `sum` is written out by thread 0 only, which is enough to keep the compiler from eliminating the chain.

```
number_of_bytes = 2 × data_len × sizeof(float)
number_of_flops = (2 + 2 × n) × data_len
arithmetic intensity = flops / bytes = (2 + 2n) / 8   FLOP/byte
```

At `n = 0` the intensity is 0.25 FLOP/byte — firmly memory-bound. At `n = 1012` it reaches ~253 FLOP/byte, deep into the compute-bound regime.

### Occupancy-driven sizing

Grid size is not fixed. For each `n`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is queried for that exact kernel instantiation, and:

```
block_count = sm_count × max_active_blocks
data_len    = block_count × BLOCKSIZE × M
```

This matters: register pressure grows with `n`, occupancy drops, and the buffer is resized accordingly. Every point runs at the best occupancy its own kernel can achieve.

### Runtime dispatch

Upstream swept `n` at compile time with `-DPARN=N` and rebuilt the binary per point. Since the baseliner sweeps options rather than rebuilds, `n` is dispatched here through a `switch` over the 55 precompiled template instantiations. A value outside that list raises `std::runtime_error: roofline: unsupported n value`.

---

## Sweep parameters

| Parameter | Values | Description |
|---|---|---|
| `n` | 55 values: 0, 1, 2, 4, 6, … 934, 1012 | Inner FMA iterations per 2 loads |

Spacing is dense at the low end, where the knee usually sits, and geometric afterwards.

---

## Reported metrics

| Metric | Unit | Description |
|---|---|---|
| `median` | ms | Median kernel execution time |
| `mean` | ms | Mean |
| `CoV` | % | Coefficient of variation (stability) |
| `memory_bandwidth` | GB/s | Dominant on the left of the knee |
| `arithmetic_bandwidth` | GFLOP/s | Dominant on the right of the knee |

Both are reported at every point. Plotting `arithmetic_bandwidth` against intensity gives the roofline; the plateau on the right is the compute peak, the rising slope on the left is bandwidth-limited.

---

## Configuration parameters (protocol JSON)

| Parameter | Section | Effect |
|---|---|---|
| `n` | `sweep > enumerated` | Arithmetic intensity points to test |
| `lock_clock` | `cuda > Backend` | Recommended `1`: the compute plateau is clock-sensitive |
| `batch_size` | `Benchmark` | Repetitions per batch |
| `warmup` | `Benchmark` | Warmup run before measuring |

---

## Caveats

- Any `n` outside the precompiled list throws. To add a value you must extend both `DISPATCH_N` switch tables in `cuda/GpuRooflineWorkload.cu` **and** the sweep list in the header, then rebuild.
- Buffer size varies across the sweep because it follows occupancy. Two points with different `n` do not touch the same amount of memory, which is intended but worth remembering when comparing raw times rather than bandwidths.
- The kernel is `float`-only. There is no precision sweep, unlike `gpu-incore`.
- `M = 4000` and `BLOCKSIZE = 256` are compile-time constants; changing them means editing the header.
- The inner loop reuses `a` and `b` across iterations, so the FMA chain is dependency-bound. The compute plateau reflects that dependency chain, not the peak FMA throughput a fully independent stream would reach — compare with `gpu-incore` `fma-separated` for the latter.
