# gpu-incore — Recherche & Analyse

## Objectif

Mesurer le **débit de calcul in-core** du GPU en cycles par opération (`rcp_throughput`), pour différents types d'instructions arithmétiques, niveaux d'ILP et précisions. Ce benchmark caractérise les unités fonctionnelles du GPU sans jamais saturer la mémoire.

---

## Principe de mesure

Chaque kernel exécute `ITERS = 10 000` passes sur une chaîne d'opérations arithmétiques indépendantes (FMA, DIV, SQRT) dont les résultats s'enchaînent pour empêcher le compilateur de les éliminer. La mesure du temps GPU permet de déduire le débit reciproque :

```
rcp_throughput (cycles/op) = (temps_médian_ms × 1e-3) × fréquence_GHz × 1e9 / ops_per_run
```

où `ops_per_run = N_type × ITERS × warp_count` avec :
- `N_FMA = 1024` opérations FMA par passe
- `N_OTHER = 128` opérations par passe pour DIV et SQRT

---

## Paramètres de sweep

| Paramètre | Valeurs | Description |
|---|---|---|
| `kernel_type` | `fma-mixed`, `fma-separated`, `div`, `sqrt` | Type d'instruction arithmétique |
| `ilp` | 1, 2, 4, 8 | *Instruction-Level Parallelism* : nombre de chaînes indépendantes par thread |
| `precision` | `float`, `double` | Précision virgule flottante |
| `warp_count` | 1, 2, 4, 8, 16, 32 | Nombre de warps par bloc (block_size = 32 × warp_count) |

Le sweep complet représente **4 × 4 × 2 × 6 = 192 points** de mesure.

---

## Types de kernel

| `kernel_type` | Description | Unité fonctionnelle ciblée |
|---|---|---|
| `fma-mixed` | FMA avec dépendances entre opérations (latence chaînée) | FP ALU + pipeline |
| `fma-separated` | FMA avec opérations indépendantes (ILP pur) | FP ALU (débit) |
| `div` | Division flottante | SFU (Special Function Unit) |
| `sqrt` | Racine carrée flottante | SFU |

---

## Métriques reportées

| Métrique | Unité | Description |
|---|---|---|
| `median` | ms | Temps d'exécution médian |
| `ops_per_run` | ops | Nombre total d'opérations calculé (`N_type × ITERS × warp_count`) |
| `clock_frequency` | GHz | Fréquence GPU mesurée pendant le run |
| `rcp_throughput` | cycles/op | Débit réciproque — **métrique principale** |

---

## Paramètres de configuration (protocol JSON)

| Paramètre | Section | Effet |
|---|---|---|
| `kernel_type` | `sweep > enumerated` | Types d'opérations à tester |
| `ilp` | `sweep > enumerated` | Niveaux d'ILP à tester |
| `precision` | `sweep > enumerated` | Précisions à tester |
| `warp_count` | `sweep > enumerated` | Occupations à tester |
| `lock_clock` | `cuda > Backend` | Recommandé `1` : la fréquence entre dans le calcul |

---

## Points d'attention

- `lock_clock` est **critique** ici : la fréquence GPU est utilisée directement dans le calcul de `rcp_throughput`. Une fréquence variable fausse la métrique.
- `number_of_floating_point_operations()` retourne `nullopt` volontairement — le benchmark ne compte pas de FLOP/s mais des cycles/op.
- DIV et SQRT utilisent `N_OTHER = 128` (vs `N_FMA = 1024`) car ils sont intrinsèquement plus lents : cela maintient des temps d'exécution comparables.
