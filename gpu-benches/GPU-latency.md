# gpu-latency — Recherche & Analyse

## Objectif

Mesurer la **latence d'accès mémoire** du GPU en nanosecondes par niveau de mémoire (L1, L2, DRAM) via une technique de *pointer chasing*. Contrairement à gpu-cache qui mesure la bande passante, ce benchmark force les accès séquentiels dépendants pour isoler la latence pure.

---

## Principe de mesure

Le kernel parcourt une chaîne de pointeurs mélangés aléatoirement dans un buffer de taille `buffer_size_kb`. Chaque accès mémoire dépend du résultat du précédent (pointer chase), ce qui empêche toute exécution spéculative ou out-of-order. Après `iterations` accès, le temps total permet de calculer :

```
latency_ns = (temps_médian_ms × 1e6) / iterations
```

Le buffer est initialisé comme une permutation aléatoire d'indices (liste chaînée en mémoire), forçant le hardware prefetcher à échouer.

---

## Paramètres de sweep

| Paramètre | Valeurs | Description |
|---|---|---|
| `buffer_size_kb` | 16, 32, 64, ... 524288 (puissances de 2) | Taille du buffer de pointer chase (16 kB → 512 MB) |
| `iterations` | 100 000 (fixe) | Nombre d'accès chaînés par mesure |

---


## Métriques reportées

| Métrique | Unité | Description |
|---|---|---|
| `median` | ms | Temps total médian pour `iterations` accès |
| `iterations` | — | Nombre d'accès chaînés (configuré via JSON) |
| `latency_ns` | ns | **Métrique principale** : latence par accès |

`latency_ns` est calculé dans `LatencyNsStat` à partir de la médiane et du nombre d'itérations.

---

## Paramètres de configuration (protocol JSON)

| Paramètre | Section | Effet |
|---|---|---|
| `buffer_size_kb` | `sweep > PowersOfTwo` | Plage de tailles de buffer |
| `iterations` | `GpuLatency` | Nombre de pointer chases (défaut : 100 000) |
| `lock_clock` | `cuda > Backend` | Recommandé `1` pour stabilité |
| `warmup` | `Benchmark` | Run de chauffe pour initialiser le cache |

---

## Différences avec gpu-cache

| Aspect | gpu-cache | gpu-latency |
|---|---|---|
| **Métrique** | Bande passante (GB/s) | Latence (ns) |
| **Accès** | Séquentiels (streaming) | Chaînés aléatoires (pointer chase) |
| **Prefetcher** | Exploité | Contourné |
| **Parallélisme** | Tous les SM en même temps | Généralement 1 thread actif par chaîne |
| **Ce qui est mesuré** | Débit agrégé | Latence d'un accès unique |

---

## Points d'attention

- `m_dummy_buffer` est alloué séparément pour empêcher les optimisations du compilateur sur le résultat final (le kernel écrit dedans pour éviter que le pointer chase ne soit éliminé).
- `reset_device` réinitialise la permutation entre les runs pour garantir des accès aléatoires cohérents.
- Avec `iterations` trop faibles (< 10 000), le bruit de lancement du kernel domine la mesure.
- La longueur de la chaîne (`m_chain_length`) est calculée au `setup_device` selon la taille du buffer.
