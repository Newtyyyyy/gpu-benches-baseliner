# GPU In-Core Benchmark

## Goal

Measure the **pure arithmetic capability** of the GPU, independently of memory bandwidth.
All data lives in registers (in-core). The benchmark isolates and characterizes the compute units: FP32 cores, FP64 cores, SFUs.

---

## Kernels

Four arithmetic operation types, two swept parameters:

| Kernel | Operation | Behavior |
|--------|-----------|----------|
| `fma-mixed` | `t[m] = t[m] * 0.9 + 0.5` | M accumulators interleaved in an inner loop — measures throughput |
| `fma-separated` | M independent chains of N sequential FMAs | Chains executed sequentially — measures latency |
| `div` | `t = 0.1 / (t + 0.2)` | Float: SFUs. Double: a sequence of DFMAs (Newton-Raphson) |
| `sqrt` | `t = sqrt(t + 0.2)` | Same as div |

**ILP** `{1, 2, 4, 8}`: number of independent chains → hides latency instruction by instruction  
**TLP** `{1, 2, 4, 8, 16, 32}`: number of warps in the block → hides latency across warps  
**Precision** `{float, double}`: compares FP32 vs FP64 units  
**A single block** is launched → everything runs on 1 SM, no inter-SM contention

**Constants:**
- `ITERS = 10 000`
- `N_FMA = 1024` operations per iteration for fma
- `N_OTHER = 128` operations per iteration for div/sqrt

---

## Metric: RCP Throughput

### Formula in the code

```cpp
val = (median_ms * 1e-3) * (clock_ghz * 1e9) / ops;
// = total_cycles / counted_ops   [cycles/op]
```

The clock is measured dynamically (actual clock under load, not the nominal one).

### What ops counts

```cpp
N_type * ITERS * warp_count
```

**No ×32 factor** — ops counts **warp instructions**, not thread operations.
Each warp instruction processes 32 threads simultaneously.

### Deriving the theoretical value

The SM has **N units**, each processing **1 thread per cycle** (in pipelined throughput).

```
thread_ops   = counted_ops × 32          (1 warp instr. = 32 threads)
total_cycles = thread_ops / N            (N units in parallel)
             = counted_ops × 32 / N

RCP = total_cycles / counted_ops
    = (counted_ops × 32 / N) / counted_ops
    = 32 / N
```

The **32** comes from the warp size (NVIDIA hardware constant).  
The **N** comes from the TU102 whitepaper (number of units per SM on the RTX 2080 Ti).

**Note:** "1 thread per cycle" refers to throughput, not latency. A CUDA core FMA produces 2 FLOPS per cycle (fused multiply + add) but processes 1 thread — which is why peak FLOP/s = N × **2** × freq, while the benchmark counts 1 FMA = 1 op.

---

## Validity condition: pipeline saturation

The `32/N` formula is only reached when **all cores are busy at all times**.

On Turing, the SM is split into **4 SMSPs** of 16 FP32 cores each. A warp is assigned to 1 SMSP:

```
TLP = 1 warp   →  16 FP32 cores active  →  RCP ≈ 32/16 = 2.0 cycles/op
TLP ≥ 4 warps  →  64 FP32 cores active  →  RCP ≈ 32/64 = 0.5 cycles/op  ✓
```

ILP and TLP both contribute to hiding latency:
- **ILP**: several independent chains within the same warp
- **TLP**: several warps ready to execute on different SMSPs
- The `32/N` formula is reached as soon as ILP × TLP provides enough independent instructions

---

## Theoretical values — RTX 2080 Ti (SM75 / TU102)

Sources: TU102 whitepaper (unit counts), CUDA Programming Guide (throughput tables), the `32/N` formula.

| Unit | N / SM | Theoretical RCP = 32/N |
|------|--------|------------------------|
| FP32 cores | 64 | **0.500 cycles/op** |
| FP64 cores | 2 | **16.0 cycles/op** |
| SFUs | 16 | **2.000 cycles/op** |

**FP32/FP64 ratio:** 64/2 = **1:32** — confirmed by the product specs (13.45 TFLOPS FP32 / 420 GFLOPS FP64).

---

## Measured vs theoretical results

### Float — throughput (TLP saturated)

| Kernel | Theoretical | Measured | Ratio |
|--------|-------------|----------|-------|
| fma (FP32 cores) | 0.500 | 0.502 | **99.6%** |
| div / sqrt (SFUs) | 2.000 | 2.016 | **99.2%** |

### Double — throughput (TLP saturated)

| Kernel | Theoretical | Measured | Ratio |
|--------|-------------|----------|-------|
| fma (FP64 cores) | 16.0 | 19.07 | 84.6% |
| div / sqrt | ~160 | 162 | **98.8%** |

### Measured latencies (ILP=1, TLP=1)

| Kernel | Float | Double |
|--------|-------|--------|
| fma-separated | **4.1 cycles** | **48.2 cycles** |
| div | **21.4 cycles** | **502.8 cycles** |
| sqrt | **21.3 cycles** | **433.2 cycles** |

The 4-cycle FP32 FMA latency is confirmed by: Jia et al. (2019), *Dissecting the NVidia Turing T4 GPU via Microbenchmarking*, arXiv:1903.07486.

---

## Finding: DDIV and DSQRT emulated via Newton-Raphson

SFUs only exist in FP32. For FP64, the CUDA compiler emits a sequence of **dependent DFMAs** (Newton-Raphson) instead.

```
DDIV  :  502.8 cycles / 48.2 cycles (DFMA latency) ≈ 10 serial DFMAs
DSQRT :  433.2 cycles / 48.2 cycles                ≈ 10 serial DFMAs
```

Cross-check via the throughput floor (TLP ≥ 8):

```
162 cycles / 16 cycles (DFMA throughput) ≈ 10 DFMAs
```

Both computations agree → **~10 DFMAs per DDIV/DSQRT** on this card.  
This step count is not documented by NVIDIA for consumer GPUs.

---

## Benchmark validation

| Test | What it proves |
|------|----------------|
| FP32 FMA at 99.6% of theoretical | Op counting and timing are correct |
| SFU at 99.2% of theoretical | Same for the special function units |
| FP32/FP64 ratio = 1:32 | Consistent with the hardware specs |
| ILP knee at ILP=2-4 | Physical signature of the Turing pipeline is visible |
| Memory traffic ≈ 0 (Nsight) | Truly in-core, not memory-bound |
| 10 DFMAs for DDIV (latency and throughput) | Consistent measurement across two distinct regimes |
