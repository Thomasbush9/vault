# Decisions

Design choices that shouldn't get re-litigated. New entries go on top.

## 2026-05-04 — Calibration design choices (h100 sweep v1)

- **Option A (single DAG: MSA+Boltz+ESM+ESMFold) over Option B (two-phase MSA-then-rest)**. Simpler for one GPU type. If we sweep multiple GPU types in the future and MSA cost becomes a problem, switch to B (MSA once, then per-GPU calibrations against the same YAML tree). For now, A.

- **`max_files_per_job: 1` for calibration**. Production runs 25 seqs/chunk, but Snakemake's `benchmark:` directive aggregates per-job. We need one TSV row per length value to fit `runtime ~ L²`. Tradeoff: per-job startup overhead is no longer amortized → ESM wall times calibrate slightly high (negligible for Boltz). For sizing safety margins, that's the safe direction. See `calibration.md`.

- **Calibration-generous Boltz/ESMFold resource ceilings (80 GB, 4 / 2 hours)** vs rule defaults (16 GB / 60 min). SLURM allocates what the job uses, not what we ask, so over-asking is free; under-asking kills long-tail data. Calibration is for measuring, not for production sizing.

- **`SLURM_ACCOUNT` defaulted to `kempner_bsabatini_lab` in `calibrate.sh`** (not probed via `sacctmgr`). Kempner partitions need the `kempner_*_lab` account; `sacctmgr` returns the lab's plain `bsabatini_lab` which works on default partitions but rejects on `kempner_h100`. Hardcoding the right one makes calibrate.sh just work.

- **`calibrate.sh` is fully self-contained** — writes its own `config.yaml` from hardcoded shared Kempner paths (template) plus `$USER`-derived user paths. Does NOT merge from repo `config.yaml`, because the webapp owns that file and the cluster checkout may not have it.

- **ESMFold env default = shared lab env** at `/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/envs/esmfold` (matching `webapp/app.py:1383`), not the user-specific path in `config.template.yaml`. The template is misleading on this; most users don't set up their own ESMFold env. ESM keeps the user-specific default since users do typically have their own ESM env.

- **Lab notes go in vault, not repo**. The 4-stage pipeline calibration is a recurring lab activity with experimental logs accumulating over time — those belong in `Lab/protforge/log/`. Durable reference (workflow, troubleshooting) goes in `Lab/protforge/calibration.md`. The repo keeps only the code and a one-line CLAUDE.md pointer.
