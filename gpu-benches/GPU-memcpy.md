# gpu-memcpy — Recherche & Analyse

## Objectif

Mesurer la **bande passante de transfert mémoire** entre le CPU (host) et le GPU (device) via PCIe, en faisant varier la taille du transfert et le type de mémoire hôte (paginée vs verrouillée).

---

## Principe de mesure

Le benchmark alloue un buffer côté host et un buffer côté device, puis effectue un transfert `host → device` (ou `device → host` selon l'implémentation). Le temps mesuré couvre uniquement le transfert, pas l'allocation.

```
bandwidth (GB/s) = transfer_kb × 1024 / temps_s / 1e9
```

---

## Paramètres de sweep

| Paramètre | Valeurs | Description |
|---|---|---|
| `transfer_kb` | 128, 256, 512, ... 524288 (puissances de 2) | Taille du transfert en kB (128 kB → 512 MB) |
| `pin_memory` | `false`, `true` | Mémoire hôte paginée vs verrouillée (pinned) |

Le produit cartésien des deux sweeps est exécuté, soit **20 points de mesure** (10 tailles × 2 modes mémoire).

---

## Mémoire paginée vs Pinned

| Mode | Mécanisme | Impact perf |
|---|---|---|
| **Paginée** (`pin_memory=false`) | Le driver CUDA alloue en interne une zone pinned temporaire et double-copie | Bande passante réduite (~50% de la valeur théorique PCIe) |
| **Pinned** (`pin_memory=true`) | Mémoire verrouillée en RAM physique, DMA direct vers GPU | Bande passante maximale PCIe (≈ 16–32 GB/s selon génération) |

---

## Métriques reportées

| Métrique | Unité | Description |
|---|---|---|
| `median` | ms | Temps de transfert médian |
| `mean` | ms | Moyenne |
| `CoV` | % | Stabilité de la mesure |
| `memory_bandwidth` | GB/s | Bande passante calculée (`number_of_bytes / temps`) |

`number_of_bytes()` retourne `m_item_count = transfer_kb × 1024`.

---

## Paramètres de configuration (protocol JSON)

| Paramètre | Section | Effet |
|---|---|---|
| `transfer_kb` | `sweep > PowersOfTwo` | Plage de tailles de transfert |
| `pin_memory` | `sweep > enumerated` | Active/désactive la mémoire verrouillée |
| `batch_size` | `Benchmark` | Répétitions par batch |
| `warmup` | `Benchmark` | Run de chauffe (recommandé pour initialiser le lien PCIe) |

---

## Points d'attention

- Le premier transfert PCIe après le boot est plus lent (initialisation du lien). Le `warmup` est essentiel pour des mesures stables.
- `reset_device` est implémenté (contrairement à gpu-cache) : le buffer device est remis à zéro entre les runs pour éviter les effets de cache PCIe côté GPU.
