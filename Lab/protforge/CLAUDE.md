# protforge — Claude context

ProtForge is the pipeline / tooling used by `protein-interp` for sequence generation, MSA handling, and structure scoring. This folder tracks ProtForge-the-tool: features to add, fixes, design notes. The actual code lives in the GitHub repo (see `repo.md`).

## Where things are

- `agenda.md` — features to add / TODOs / current focus.
- `repo.md` — GitHub repo + setup.
- `data.md` — cluster paths and run IDs (e.g. active calibration outputs).
- `decisions.md` — design choices that shouldn't get re-litigated.
- `calibration.md` — durable reference for the resource-calibration workflow (subsample → calibrate.sh → analyze → refit). Reads alongside repo `slurm_scripts/calibrate.sh` and `scripts/calibrate/`.
- `log/` — experimental log entries, one per discrete experiment (`YYYY-MM-DD-<slug>.md`).
- `lit.md` — empty stub.

## Default load

Autoload `agenda.md` and the most recent `log/` entry. For calibration work, also pull `calibration.md` and `data.md` (cluster paths). Code-level state lives in the repo (`/Users/thomasbush/Documents/ML/ProtForge/`).
