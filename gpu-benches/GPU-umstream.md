# gpu-umstream — Recherche & Analyse

## Objectif

Mesurer la **bande passante de la mémoire unifiée (Unified Memory)** via un kernel de type TRIAD (`C = A + B`) sur des tableaux alloués avec `cudaMallocManaged`. Ce benchmark évalue le comportement du système de migration de pages GPU/CPU, avec et sans prefetching explicite.

---

## Principe de mesure

Le kernel effectue une opération TRIAD sur trois tableaux de doubles (`A`, `B`, `C`) alloués en Unified Memory :

```
C[i] = A[i] + B[i]    pour tout i
```

Les données peuvent résider en RAM ou en VRAM selon l'historique d'accès. La bande passante effective est calculée sur les 3 tableaux (lecture A, lecture B, écriture C) :

```
number_of_bytes = item_count × sizeof(double) × 3
bandwidth (GB/s) = number_of_bytes / temps_s / 1e9
```

---

## Paramètres de sweep

| Paramètre | Valeurs | Description |
|---|---|---|
| `transfer_mb` | 1, 2, 4, ..., 512 (puissances de 2) | Taille par tableau en MB (1 → 512 MB) |
| `prefetch` | `false`, `true` | Prefetch UM vers le GPU avant le run |

Le produit cartésien donne **20 points** (10 tailles × 2 modes prefetch).

---

## Mémoire unifiée : comportement sans vs avec prefetch

| Mode | Mécanisme | Impact perf |
|---|---|---|
| `prefetch=false` | Pages migrées à la demande lors du premier accès GPU (page fault) | Latence élevée, bande passante apparente faible (~PCIe) |
| `prefetch=true` | `cudaMemPrefetchAsync` déplace les pages vers le GPU avant le kernel | Migration par DMA, bande passante proche de HBM/GDDR sur GPU |

---

## Métriques reportées

| Métrique | Unité | Description |
|---|---|---|
| `median` | ms | Temps d'exécution médian du kernel |
| `mean` | ms | Moyenne |
| `CoV` | % | Stabilité (élevé si page faults irréguliers) |
| `memory_bandwidth` | GB/s | Bande passante effective (3 × transfer_mb × 1024² / temps) |

---

## Paramètres de configuration (protocol JSON)

| Paramètre | Section | Effet |
|---|---|---|
| `transfer_mb` | `sweep > PowersOfTwo` | Plage de tailles par tableau |
| `prefetch` | `sweep > enumerated` | Active/désactive le prefetch UM |
| `blocksize` | `GpuUmstream` | Taille du bloc CUDA (défaut : 256) |
| `lock_clock` | `cuda > Backend` | Recommandé `1` |
| `warmup` | `Benchmark` | Chauffe importante : le 1er run sans prefetch déclenche les migrations |

---

## Comparaison avec gpu-memcpy

| Aspect | gpu-memcpy | gpu-umstream |
|---|---|---|
| **Mémoire** | Allouée séparément (host + device) | Unified Memory (cudaMallocManaged) |
| **Transfert** | Explicite `cudaMemcpy` | Implicite (migration) ou prefetch |
| **Calcul** | Aucun (transfert pur) | TRIAD `C = A + B` |
| **Ce qui est mesuré** | Bande passante PCIe brute | Bande passante UM effective (migration + calcul) |
| **Cas d'usage** | Calibrer le lien PCIe | Évaluer l'overhead de l'UM sur un workload réel |

---

## Points d'attention

- `reset_device` ne fait rien (`{}`) : les tableaux UM ne sont pas réinitialisés entre les runs. Le `warmup` garantit que les pages sont déjà à leur état final avant la mesure.
- `m_block_count` n'est pas dans le sweep — il est calculé implicitement selon `item_count` et `block_size`.

