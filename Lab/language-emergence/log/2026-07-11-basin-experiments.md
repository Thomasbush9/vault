# 2026-07-11 — Basin experiments (interventions vs the init lottery)

## Setup

`basin_experiments.py` (marimo notebook, runs headless). Vision held fixed
(seed_1's pretrained checkpoints, ~0.99 on both pretext tasks); only the channel
is re-seeded. 3 arms x 20 game seeds x 40 epochs:

- **baseline** — GS temperature 1.0, single Adam lr 1e-3 (current setup)
- **annealed** — temperature 2.0 → 0.5, geometric decay over 30 epochs, then flat
- **asym_lr** — receiver lr 1e-2, sender lr 1e-3

Success = final val acc (mean of last 3 epochs) > 0.5 (bimodal gap makes the
threshold insensitive). Per-epoch instrumentation: val acc, distinct-message
count, NMI(message; shape), NMI(message; color) — the early-warning metrics.

Per-run caching: `runs/basin_experiments/runs/{arm}_seed{s}.json`; models saved
for successful runs only.

## Status

Done. Launched ~17:00, finished ~19:25 (~140 s/run, 60 runs). The early signal
held: baseline with fixed seed_1 vision reached 80% vs the 40% of the
full-pipeline sweep — vision init is a bigger driver of the lottery than the
channel init. See Results.

## Results

All 60 runs completed (finished ~19:25). Success = mean val acc of last 3
epochs > 0.5.

| arm | success rate | mean acc (successful) |
|---|---|---|
| baseline | **16/20 (80%)** | 0.823 |
| annealed | **18/20 (90%)** | 0.869 |
| asym_lr | **6/20 (30%)** | 0.832 |

Stuck seeds still land exactly on the pooling baseline (~0.29); the gap stays
bimodal, so the 0.5 threshold is insensitive.

### Takeaways

1. **Fixing the vision is the big lever.** Baseline here is 80% vs 40% (2/5) in
   the full-pipeline sweep — same channel setup, only difference is the vision
   modules are frozen to seed_1's checkpoints. A large share of the "init
   lottery" lived in vision pretraining, not the channel. This is the headline.
2. **Temperature annealing helps** (90% vs 80%), and also lifts mean accuracy of
   the successful runs (0.869 vs 0.823). Modest but consistent — higher early
   temperature keeps the sender's symbol distribution soft long enough to avoid
   the self-sealing collapse, matching the Havrylov & Titov temperature-
   sensitivity story.
3. **Asymmetric lr HURTS** (30%, worse than baseline). Cranking the receiver lr
   to 1e-2 seems to let the receiver lock into its "point at an occupied cell"
   best-response before the sender's message becomes informative — accelerating
   the pooling collapse rather than escaping it. Interesting failure, not a fix.
   Note the successful asym_lr runs still reach normal accuracy (0.832 mean), so
   it's a basin-selection effect, not a ceiling effect.
4. Mean-acc-when-successful is similar across arms (0.82–0.87): the interventions
   move *how often* you land in the good basin, not how good the good basin is.

### Early-warning metrics

The per-epoch instrumentation (distinct-message count, NMI(msg;shape),
NMI(msg;color)) separates the basins early — stuck runs collapse to 1 distinct
message with ~0 NMI within the first ~10 epochs, exactly as in the 2026-07-10
diagnosis. See `figures/basin_experiments_early_warning.*`.

### Artifacts

- `runs/basin_experiments/summary.json` (per-arm rates, successful seeds)
- `runs/basin_experiments/runs/{arm}_seed{s}.json` (per-run per-epoch histories)
- `runs/basin_experiments/models/{arm}_seed{s}.pt` (successful runs only)
- `figures/basin_experiments_curves.*`, `figures/basin_experiments_early_warning.*`

## Housekeeping (same day)

- Moved `runs/` + `figures/` out of the repo to
  `.../language/data/language_emergence/`, symlinked back, gitignored,
  committed (`ba455f1`). See decisions.md.
- Created this vault project folder from the template.
