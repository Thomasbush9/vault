# 2026-07-13 — Zero-shot transfer across the difficulty ladder (eval only)

## Question

Where do the current grid_iid-trained games break? Run existing checkpoints in
eval mode (argmax GS, no training) on the val splits of the new d1–d4 datasets
and read transfer as a function of original competence.

## Setup

`experiments/difficulty_eval.py`. 9 checkpoints spanning original val acc
0.28 → 0.94 (all five seed-sweep games, stuck ones included, + 4 basin models
spread over their range). Datasets, in increasing difficulty (val split only,
1000 examples each):

| dataset | change vs previous | mean objects | occupied-cell baseline |
|---|---|---|---|
| grid_iid | training distribution | 3.5 | 0.292 |
| d1_iid | 6–10 objects, IID distractors | 8.0 | 0.128 |
| d2_hardneg | + 2 hard negatives (every scene has same-shape AND same-colour distractors) | 8.0 | 0.129 |
| d3_size | + variable object size (3 bins) | 8.0 | 0.129 |
| d4_free | + free placement (not grid-aligned) | 8.0 | 0.129 |

Outputs: `runs/difficulty_eval/` (= `.../language/data/language_emergence/runs/difficulty_eval/`)
— `results.json` + 4 figures (transfer_curves, transfer_scatter,
overlap_breakdown, protocol_degradation).

## Results (best checkpoint = basin/annealed_seed5, orig 0.94)

| | grid_iid | d1_iid | d2_hardneg | d3_size | d4_free |
|---|---|---|---|---|---|
| best acc | 0.940 | 0.840 | 0.755 | 0.375 | 0.248 |
| NMI(msg; colour) | 1.00 | 1.00 | 1.00 | 0.84 | 0.86 |
| NMI(msg; shape) | 0.48 | 0.48 | 0.48 | 0.14 | 0.12 |

- **Transfer scales with original competence**: checkpoint ordering is
  preserved on every dataset; stuck models (0.28–0.31) drop to the new
  occupied-cell baseline (~0.13) immediately. No model beats its original acc.
- **Crowding + hard negatives degrade gracefully** (0.94 → 0.84 → 0.76 for the
  best model): the channel keeps working, messages stay fully colour-informative.
- **Size variation is the cliff** (0.76 → 0.38), free placement adds more
  (→ 0.25). Both vision modules were trained on fixed-size, grid-aligned
  objects.
- **The failure at d3/d4 is mostly receiver-side**: colour NMI stays 0.84–0.86
  while accuracy collapses — the sender still says the right colour, but the
  receiver's per-cell scene embeddings can't parse variable-size / off-grid
  scenes. Sender shape information also degrades (0.48 → 0.14) since crops
  change scale.
- **Colour-first protocols cap d2+ performance**: even in-distribution, a
  same-colour distractor costs the best model ~0.2 acc (1.00 no-overlap vs
  0.83 same-colour / 0.78 both), and shape NMI is only 0.48. In d2+ every
  scene contains a same-colour hard negative by construction.

## Implications / next

- Beating d2 needs a protocol that genuinely encodes shape — connects directly
  to the dual-sender line (colour monopolises the receiver).
- Beating d3/d4 needs size/position-robust vision — retrain (or augment) the
  vision modules on the harder distributions before asking anything of the
  channel; the channel itself transfers fine up to d2.
- Vision pretrained on d3/d4 + frozen, channel re-seeded (the basin recipe)
  is the obvious next experiment.

## Addendum (same day): oracle-decodability ceilings

Added `oracle_acc` to `lang_emergence/evaluation.py`: accuracy of pointing
uniformly among objects matching the target on an attribute — the ceiling for
any protocol encoding exactly that information.

| dataset | colour only | shape only | shape+colour |
|---|---|---|---|
| grid_iid | 0.829 | 0.841 | 1.000 |
| d1_iid | 0.615 | 0.621 | 1.000 |
| d2_hardneg | 0.397 | 0.402 | 1.000 |
| d3_size | 0.287 | 0.290 | **0.505** |
| d4_free | 0.287 | 0.286 | **0.488** |

Two corrections to the reading above:

1. Best grid-trained models (0.91–0.94) EXCEED the single-attribute ceiling on
   grid_iid (0.83) — they already encode some shape (consistent with shape
   NMI 0.48). "Colour-first" ≠ "colour-only".
2. On d3/d4 even a PERFECT shape+colour code caps at ~0.5: hard negatives
   there can match the target on both attributes and differ only in size, so
   size becomes a necessary third channel dimension. Relative to ceiling, the
   best model does 0.76 (d2), 0.74 (d3), 0.51 (d4) — the "d3 cliff" is
   substantially an information ceiling of the shape+colour channel, not only
   vision failure; the extra drop on d4 (free placement) is the vision part.
