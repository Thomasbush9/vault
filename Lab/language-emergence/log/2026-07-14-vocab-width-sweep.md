# 2026-07-14 — Channel-width sweep: vocab k ∈ {3,4,5,7,10} on d2 (+ k=7 on d1)

## Question

d2 accuracy tracks surviving message inventory (Spearman 0.98 across the
2026-07-13 suite), and vocab-10 runs keep only ~5–21 of 100 possible
messages. Does a channel with no slack — vocab 7 = 6 usable symbols + EOS,
capacity 36 two-symbol messages = exactly the number of (shape,colour)
combos — (a) resist inventory collapse by forcing fuller use of the space,
or (b) collapse harder because there is no redundancy to explore through?
And where does performance break as capacity drops below the task (k=5 →
25, k=4 → 9, k=3 → 4 messages)?

## Setup

`experiments/channel_game.py` (new agnostic runner: `--data --vocab --seeds
--out`; `d2_k7.py` was the first instance, then generalized). GS, frozen
seed_1 grid_iid vision, 250 epochs, standard config apart from vocab.
Arms: k=7 x 10 seeds on d2 (`runs/data2_k7/`), k=5/4/3 x 5 seeds on d2
(`runs/data2_k{5,4,3}/`), k=7 x 10 seeds on d1_iid (`runs/d1_k7/`), against
the existing k=10 x 15 baseline (`runs/d2_channel/`). vocab_size is now
recorded in run jsons and `resolve_game` rebuilds the right architecture,
so all checkpoints load through `experiments/evaluate.py`.

## Results (d2, figure `figures/d2_vocab_sweep.png`)

| k | capacity | success | mean succ acc | best | msgs (succ) | NMI shape (succ) |
|---|---|---|---|---|---|---|
| 10 | 100 | 11/15 | 0.683 | 0.777 | 8–15 | 0.44 |
| 7 | 49 | 7/10 | 0.671 | 0.757 | 8–15 | 0.42 |
| 5 | 25 | 2/5 | 0.569 | 0.628 | 6–10 | 0.26 |
| 4 | 9 | 0/5 | — | 0.487 | — | — |
| 3 | 4 | 0/5 | — | 0.390 | — | — |

1. **Tight channels do NOT resist collapse — answer (b), with a twist.**
   k=7 is statistically indistinguishable from k=10 (same success rate,
   same accuracy, same 8–15 surviving messages, same shape NMI): the
   inventory equilibrium is set by the learning dynamics, not by capacity,
   as long as capacity is comfortably above it. Removing slack bought
   nothing; runs did not spread out to use the 36-combo space.
2. **The inventory law survives width changes**: inventory-size vs accuracy
   across all 40 runs of the five arms stays on one curve, Spearman
   r = 0.98 (0.96 excluding full collapses). Width only moves where runs
   land on it.
3. **Capacity binds below k≈5**: at k=4 (9 messages < 36 combos) and k=3
   (4 messages) no run clears 0.5; best runs saturate all available
   messages and land at/below the colour-only ceiling (0.397). A k=4 seed
   produced a balanced NMI 0.50/0.50 4-message code that still scored
   exactly 0.397 — encoding a little of both attributes is worth no more
   than all-of-one when the code is this small. k=5 is marginal (2/5).
4. **Erosion can be catastrophic in tight channels**: d2_k7 seed 2 peaked
   0.82 with 19 messages at epoch 100 and collapsed to 0.33 / 6 messages by
   250 — the largest peak→final drop observed so far in any arm. Small
   vocabularies don't slow the late colour-crowding drift; anecdotally they
   amplify its worst case (n=1, but the drop is 4x the vocab-10 mean).
5. **d1 control (k=7, 10 seeds): colour-dominance is dataset-driven, not
   width-driven.** 10/10 clear 0.5, mean 0.631 — but d1's colour-only
   ceiling is 0.615, and mean shape NMI is just 0.19 (vs 0.42 on d2 at the
   same width): without hard negatives the same channel relaxes back to
   colour-first codes. Matched-vocab confirmation of the d2-hardneg effect.

## Implications / next

- Channel width is the wrong lever: everything ≥ ~1.4x the inventory
  equilibrium behaves the same, and anything near/below the combo count is
  strictly worse. The binding constraint remains WHY inventories settle at
  ~8–15 messages. Next levers stay: temperature annealing on d2, early
  stopping at the peak, and diagnosing which messages get merged first
  during erosion (seed 2 of data2_k7 is a clean specimen: 19 → 6 messages
  in 150 epochs with per-epoch metrics saved).
- Infrastructure: `experiments/channel_game.py` runs any
  dataset x vocab x seeds channel ablation in one command.
