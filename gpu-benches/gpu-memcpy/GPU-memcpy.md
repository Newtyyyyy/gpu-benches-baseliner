# gpu-memcpy — Research & Analysis

## Goal

Measure the **memory transfer bandwidth** between the CPU (host) and the GPU (device) over PCIe, varying the transfer size and the host memory type (pageable vs pinned).

---

## Measurement principle

The benchmark allocates a host-side buffer and a device-side buffer, then performs a `host → device` transfer (or `device → host` depending on the implementation). The measured time covers only the transfer, not the allocation.

```
bandwidth (GB/s) = transfer_kb × 1024 / time_s / 1e9
```

---

## Sweep parameters

| Parameter | Values | Description |
|---|---|---|
| `transfer_kb` | 128, 256, 512, ... 524288 (powers of 2) | Transfer size in kB (128 kB → 512 MB) |
| `pin_memory` | `false`, `true` | Pageable vs pinned host memory |

The cartesian product of both sweeps is run, i.e. **20 measurement points** (10 sizes × 2 memory modes).

---

## Pageable vs pinned memory

| Mode | Mechanism | Performance impact |
|---|---|---|
| **Pageable** (`pin_memory=false`) | The CUDA driver internally allocates a temporary pinned staging area and double-copies | Reduced bandwidth (~50% of the theoretical PCIe value) |
| **Pinned** (`pin_memory=true`) | Memory locked in physical RAM, direct DMA to the GPU | Peak PCIe bandwidth (≈ 16–32 GB/s depending on generation) |

---

## Reported metrics

| Metric | Unit | Description |
|---|---|---|
| `median` | ms | Median transfer time |
| `mean` | ms | Mean |
| `CoV` | % | Measurement stability |
| `memory_bandwidth` | GB/s | Computed bandwidth (`number_of_bytes / time`) |

`number_of_bytes()` returns `m_item_count = transfer_kb × 1024`.

---

## Configuration parameters (protocol JSON)

| Parameter | Section | Effect |
|---|---|---|
| `transfer_kb` | `sweep > PowersOfTwo` | Range of transfer sizes |
| `pin_memory` | `sweep > enumerated` | Enables/disables pinned memory |
| `batch_size` | `Benchmark` | Repetitions per batch |
| `warmup` | `Benchmark` | Warmup run (recommended to bring up the PCIe link) |

---

## Caveats

- The first PCIe transfer after boot is slower (link initialization). The `warmup` is essential for stable measurements.
- `reset_device` is implemented (unlike gpu-cache): the device buffer is zeroed between runs to avoid GPU-side PCIe caching effects.
