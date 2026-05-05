# Data

ProtForge processes data but doesn't own datasets. This file tracks paths and run IDs for ProtForge's own experiments (calibration, etc.). For consuming-project data see `Lab/protein-interp/data.md`.

## H100 calibration sweep — 2026-05-04

Cluster output (full run + benchmarks + sequences):
- `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/calibration/predictions/`

Local mirror (rsynced subset):
- `/Users/thomasbush/tmp-data/tmp_data/benchmark/predictions/`

Calibration FASTA set: `tests/calibration_inputs/fastas/` in the repo (18 sequences after dropping the 2.7k outlier; lengths 76–2039, stratified by length quantile from a 7k working dir).

Per-stage benchmarks land at `<output>/run/benchmarks/<stage>/*.tsv`; per-stage length stats at `<output>/run/<stage>_chunks/chunk_stats.tsv`. Join key is `chunk_id`. See `calibration.md` for schema.

Snakemake driver logs (3 attempts on 2026-05-04):
- `2026-05-04T113141.731577.snakemake.log` — v1a, Boltz chunk_3 OOM (2.7k seq)
- `2026-05-04T155109.820659.snakemake.log` — v1b, organize chunks 6+10 fail
- `2026-05-04T201030.497228.snakemake.log` — v1c, ESMFold env-path mismatch (every chunk)

All on `kempner_h100` partition under `kempner_bsabatini_lab` account. Full writeup: `log/2026-05-04-h100-calibration-v1.md`.
