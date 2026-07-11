# Repo

GitHub repo + setup notes for this project.

- **URL:** `https://github.com/Thomasbush9/language_emergence`
- **Cluster path:** `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/language_emergence`

## Setup

```bash
# cluster env (plain venv, shared with the `language` data-generation project)
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/language/envs/lang/bin/activate
# key deps: torch, egg (EGG toolkit), marimo, scikit-learn, matplotlib
```

## Entry points

- `egg_start.py` — marimo notebook, the main pipeline: sender/receiver vision
  pretraining, the GS pointing game, protocol analysis. Launch with
  `marimo edit egg_start.py --watch --headless`.
- `seed_sweep.py` — standalone seeded sweep (5 seeds x 200 game epochs); also
  the shared module (datasets, models, config) imported by other scripts.
- `basin_experiments.py` — marimo notebook: intervention experiments
  (baseline / temperature-annealed / asymmetric-lr, 20 game seeds each,
  fixed vision). Cached per-run; safe to reopen.

## Output convention

`runs/` and `figures/` in the repo are **symlinks** into
`/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/language/data/language_emergence/`
and are gitignored — heavy artifacts never go into git (decided 2026-07-11).
