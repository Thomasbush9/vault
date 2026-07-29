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

## Structure (refactored 2026-07-13)

- `lang_emergence/` — shared library: `config` (paths, channel hyperparams),
  `data` (shard loading + datasets), `models`, `pretrain`, `game`
  (build/train/evaluate), `runs` (enumerate + load past runs/checkpoints),
  `plotting`.
- `experiments/` — runnable scripts: `seed_sweep.py`, `basin_experiments.py`,
  `dual_sender.py` (same run-cache paths as before, so old results still
  load), `d2_channel.py` / `d2_full.py` / `d2_reinforce.py` (d2_hardneg
  retraining suite), `evaluate.py` (agnostic checkpoint x dataset eval CLI
  with oracle-ceiling baselines), `difficulty_eval.py` (d1–d4 transfer
  figures), `inspect_runs.py` (table of every recorded run),
  `validate_checkpoints.py` (reload checkpoints + smoke training run).
- `notebooks/` — the old marimo notebooks (`egg_start.py`,
  `basin_experiments.py`), archived/frozen; the basin notebook's
  `import seed_sweep` no longer resolves.

Quick start for analysis:

```python
from lang_emergence.runs import list_runs, load_history, load_sweep_game
from lang_emergence.game import evaluate_game
```

## Output convention

`runs/` and `figures/` in the repo are **symlinks** into
`/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/language/data/language_emergence/`
and are gitignored — heavy artifacts never go into git (decided 2026-07-11).
