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

- Launched ~17:00; ~90 s/run, ~90 min total expected.
- Early signal: baseline with FIXED seed_1 vision went 5/5 on the first five
  seeds — much better than the 2/5 of the full-pipeline sweep. If this holds,
  vision initialization (not channel initialization) may be a bigger driver of
  the lottery than assumed. Watch this.

## Results

_(to be filled when the run completes — success rates per arm, learning curves
`figures/basin_experiments_curves.*`, early-warning scatter
`figures/basin_experiments_early_warning.*`, `runs/basin_experiments/summary.json`)_

## Housekeeping (same day)

- Moved `runs/` + `figures/` out of the repo to
  `.../language/data/language_emergence/`, symlinked back, gitignored,
  committed (`ba455f1`). See decisions.md.
- Created this vault project folder from the template.
