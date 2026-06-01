# GPU In-Core Benchmark

## Objectif

Mesurer la **capacité arithmétique pure** du GPU, indépendamment de la bande passante mémoire.
Toutes les données vivent dans les registres (in-core). Le benchmark isole et caractérise les unités de calcul : FP32 cores, FP64 cores, SFUs.

---

## Kernels

Quatre types d'opérations arithmétiques, deux paramètres sweepés :

| Kernel | Opération | Comportement |
|--------|-----------|--------------|
| `fma-mixed` | `t[m] = t[m] * 0.9 + 0.5` | M accumulateurs interleaved dans une boucle interne — mesure le throughput |
| `fma-separated` | M chaînes indépendantes de N FMAs séquentielles | Chaînes exécutées séquentiellement — mesure la latence |
| `div` | `t = 0.1 / (t + 0.2)` | Float : SFUs. Double : séquence de DFMA (Newton-Raphson) |
| `sqrt` | `t = sqrt(t + 0.2)` | Idem div |

**ILP** `{1, 2, 4, 8}` : nombre de chaînes indépendantes → cache la latence instruction par instruction  
**TLP** `{1, 2, 4, 8, 16, 32}` : nombre de warps dans le bloc → cache la latence entre warps  
**Precision** `{float, double}` : compare les unités FP32 vs FP64  
**1 seul bloc** lancé → tout s'exécute sur 1 SM, pas de contention inter-SM

**Constantes :**
- `ITERS = 10 000`
- `N_FMA = 1024` opérations par itération pour fma
- `N_OTHER = 128` opérations par itération pour div/sqrt

---

## Métrique : RCP Throughput

### Formule dans le code

```cpp
val = (median_ms * 1e-3) * (clock_ghz * 1e9) / ops;
// = cycles_totaux / ops_comptés   [cycles/op]
```

La fréquence est mesurée dynamiquement (clock réelle sous charge, pas nominale).

### Ce que ops compte

```cpp
N_type * ITERS * warp_count
```

**Pas de facteur ×32** — ops compte des **instructions warp**, pas des opérations thread.
Chaque instruction warp traite 32 threads simultanément.

### Dérivation du théorique

Le SM a **N unités**, chacune traite **1 thread par cycle** (en throughput pipeliné).

```
ops_thread    = ops_comptés × 32          (1 instr. warp = 32 threads)
cycles_totaux = ops_thread / N            (N unités en parallèle)
              = ops_comptés × 32 / N

RCP = cycles_totaux / ops_comptés
    = (ops_comptés × 32 / N) / ops_comptés
    = 32 / N
```

Le **32** vient du warp size (constante hardware NVIDIA).  
Le **N** vient du whitepaper TU102 (nombre d'unités par SM sur la RTX 2080 Ti).

**Note :** "1 thread par cycle" désigne le throughput, pas la latence. Un CUDA core FMA produit 2 FLOPS par cycle (multiply + add fusionnés) mais traite 1 thread — c'est pourquoi le peak FLOP/s = N × **2** × freq, tandis que le benchmark compte 1 FMA = 1 op.

---

## Condition de validité : saturation du pipeline

La formule `32/N` est atteinte seulement quand **tous les cores sont occupés en permanence**.

Sur Turing, le SM est découpé en **4 SMSPs** de 16 FP32 cores chacun. Un warp est assigné à 1 SMSP :

```
TLP = 1 warp   →  16 FP32 cores actifs  →  RCP ≈ 32/16 = 2.0 cycles/op
TLP ≥ 4 warps  →  64 FP32 cores actifs  →  RCP ≈ 32/64 = 0.5 cycles/op  ✓
```

ILP et TLP contribuent tous les deux à cacher la latence :
- **ILP** : plusieurs chaînes indépendantes dans le même warp
- **TLP** : plusieurs warps prêts à exécuter sur des SMSPs différents
- La formule `32/N` est atteinte dès que ILP × TLP fournit assez d'instructions indépendantes

---

## Valeurs théoriques — RTX 2080 Ti (SM75 / TU102)

Sources : whitepaper TU102 (nb d'unités), CUDA Programming Guide (throughput tables), formule `32/N`.

| Unité | N / SM | RCP théorique = 32/N |
|-------|--------|----------------------|
| FP32 cores | 64 | **0.500 cycles/op** |
| FP64 cores | 2 | **16.0 cycles/op** |
| SFUs | 16 | **2.000 cycles/op** |

**Ratio FP32/FP64 :** 64/2 = **1:32** — confirmé par les specs produit (13.45 TFLOPS FP32 / 420 GFLOPS FP64).

---

## Résultats mesurés vs théoriques

### Float — throughput (TLP saturé)

| Kernel | Théorique | Mesuré | Ratio |
|--------|-----------|--------|-------|
| fma (FP32 cores) | 0.500 | 0.502 | **99.6%** |
| div / sqrt (SFUs) | 2.000 | 2.016 | **99.2%** |

### Double — throughput (TLP saturé)

| Kernel | Théorique | Mesuré | Ratio |
|--------|-----------|--------|-------|
| fma (FP64 cores) | 16.0 | 19.07 | 84.6% |
| div / sqrt | ~160 | 162 | **98.8%** |

### Latences mesurées (ILP=1, TLP=1)

| Kernel | Float | Double |
|--------|-------|--------|
| fma-separated | **4.1 cycles** | **48.2 cycles** |
| div | **21.4 cycles** | **502.8 cycles** |
| sqrt | **21.3 cycles** | **433.2 cycles** |

La latence FP32 FMA de 4 cycles est confirmée par : Jia et al. (2019), *Dissecting the NVidia Turing T4 GPU via Microbenchmarking*, arXiv:1903.07486.

---

## Découverte : DDIV et DSQRT émulés par Newton-Raphson

Les SFUs n'existent qu'en FP32. Pour FP64, le compilateur CUDA génère une séquence de **DFMA dépendants** (Newton-Raphson) à la place.

```
DDIV  :  502.8 cycles / 48.2 cycles (latence DFMA) ≈ 10 DFMAs en série
DSQRT :  433.2 cycles / 48.2 cycles                ≈  10 DFMAs en série
```

Vérification via le throughput floor (TLP ≥ 8) :

```
162 cycles / 16 cycles (throughput DFMA) ≈ 10 DFMAs
```

Les deux calculs convergent → **~10 DFMAs par DDIV/DSQRT** sur cette carte.  
Ce nombre d'étapes n'est pas documenté par NVIDIA pour les GPUs consumer.

---

## Validation du benchmark

| Test | Ce que ça prouve |
|------|-----------------|
| FP32 FMA à 99.6% du théorique | Le comptage d'ops et le timing sont corrects |
| SFU à 99.2% du théorique | Idem pour les unités spéciales |
| Ratio FP32/FP64 = 1:32 | Cohérent avec les specs hardware |
| Genou ILP à ILP=2-4 | Signature physique du pipeline Turing visible |
| Trafic mémoire ≈ 0 (Nsight) | Vraiment in-core, pas memory-bound |
| 10 DFMAs pour DDIV (latence et throughput) | Mesure cohérente entre deux régimes distincts |


