# gpu-cache — Recherche & Analyse

## Objectif

Mesurer la **bande passante effective** des différents niveaux de mémoire du GPU (L1, L2, DRAM) en faisant varier la taille du *working set* par SM. Les transitions de bande passante dans la courbe résultante révèlent les capacités et les seuils de chaque niveau de cache.

---

## Principe de mesure

Le kernel `sumKernel` lance autant de blocs que de SM sur le GPU. Chaque SM charge deux buffers de `N` floats (`bufA`, `bufB`), les multiplie élément par élément et accumule le résultat dans une variable locale. L'accumulateur est utilisé dans une condition impossible (`localSum == 1233`) pour forcer le compilateur à conserver les loads.

```
bandwidth (GB/s) = (2 × N × sizeof(float) × smCount × ITERS) / temps_s / 1e9
```

Le facteur `2` vient des deux buffers lus en séquence. `ITERS ≈ 1e9 / N` est calibré pour maintenir ~1 s de travail GPU par point.

---

## Paramètres de sweep

| Paramètre | Valeurs | Description |
|---|---|---|
| `working_set_elements` (N) | Dense : 128, 256, k×512 (k=1..32) puis série exponentielle ×1.17 jusqu'à ~137 MB | Taille en floats d'**un** buffer ; mémoire totale = 2×N×4 octets |

La série exponentielle (`cache_exp_series`) génère des valeurs arrondies au multiple de 512 le plus proche, espacées d'un facteur ×1.17.


---

## Métriques reportées

| Métrique | Unité | Description |
|---|---|---|
| `median` | ms | Temps d'exécution médian par batch |
| `mean` | ms | Moyenne |
| `CoV` | % | Coefficient de variation (stabilité) |
| `memory_bandwidth` | GB/s | Bande passante calculée |
| `working_set_kb` | kB | Taille totale des deux buffers (2×N×4 / 1024) |

---

## Paramètres de configuration (protocol JSON)

| Paramètre | Section | Effet |
|---|---|---|
| `lock_clock` | `cuda > Backend` | Verrouille le GPU à la fréquence de base pour reproductibilité |
| `working_set_elements` | `sweep > enumerated` | Liste des valeurs N testées |
| `batch_size` | `Benchmark` | Répétitions par batch |
| `max_nb_repetition` | `StoppingCriterion` | Plafond total de répétitions |
| `warmup` | `Benchmark` | Run de chauffe avant mesure (recommandé : `1`) |
| `flush` | `Benchmark` | Flush L2 entre batches (recommandé : `1`) |

---

## Points d'attention

- Toute valeur de N absente de la liste précompilée lève `std::runtime_error: cache: unsupported working_set_elements value`.
- Sans `lock_clock`, le GPU peut booster sa fréquence sur les petits working sets (L1), faussant la comparaison avec les points DRAM.
- Sans `flush`, le L2 résiduel d'un batch peut artificiellement améliorer le batch suivant.
- `BLOCKSIZE` est choisi automatiquement : 512 si N multiple de 512, 256 si multiple de 256, sinon 128.
