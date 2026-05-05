# Resource Calibration — workflow and reference

How to run a real cluster sweep on a small sample of inputs and turn the
output into refit coefficients for `webapp/scaling_models.yaml`.

The actual code lives in the repo (`slurm_scripts/calibrate.sh`,
`scripts/calibrate/{subsample,analyze}.py`, `workflow/scripts/chunk_*.py`).
This doc is the durable reference; experimental log entries go in
`log/YYYY-MM-DD-*.md`.

## When to (re)calibrate

- Before processing a large input batch you've never run at this scale (e.g. 7k FASTAs of varied length).
- After changing a Boltz parameter that affects compute (`recycling_steps`, `diffusion_samples`, `num_runs`).
- When adding a new GPU type so the estimator can route to it correctly.

You do **not** need to recalibrate for routine runs — calibrated
coefficients persist in `webapp/scaling_models.calibrated.yaml` until
regenerated.

## High-level flow

```
            (one-time, login node)
1. Subsample          tests/calibration_inputs/fastas/  (≈20 stratified FASTAs)
                                |
                                v
2. Sweep on cluster   /n/holylfs06/.../calib_<ts>/run/
   (calibrate.sh)        ├── benchmarks/<stage>/*.tsv     (wall time, host RSS)
                         ├── <stage>_chunks/chunk_stats.tsv  (length stats)
                         └── sequences/<seq>/...
                                |
                                v
3. Analyze            scripts/calibrate/analyze.py
                                |
                                v
4. Refit              webapp/scaling_models.calibrated.yaml
   (recalibrate_from_benchmarks — pending)
```

Steps 1–3 are wired today. Step 4 (`sklearn.LinearRegression` refit into
`scaling_models.calibrated.yaml`) is the next planned change to
`webapp/estimator.py`.

## Step 1 — Subsample

Stratified by length quantile with extra weight on the upper tail (where
O(L²) memory diverges). Random sampling under-represents the long tail.

```bash
python scripts/calibrate/subsample.py \
    --input_dir /path/to/your/7k_fastas \
    --output_dir tests/calibration_inputs/fastas \
    --n 20 \
    --seed 42
```

Writes:
```
tests/calibration_inputs/fastas/
  <picked_filename_1>.fasta
  ...
  manifest.csv          filename, length, bin (e.g. q75-q90), source_path
```

Inspect the spread before running:
```bash
column -t -s, tests/calibration_inputs/fastas/manifest.csv | sort -k2 -n
```

If a FASTA is known to OOM on the target GPU (e.g. ≥2.5k residues on H100
Boltz at standard recycling/samples), drop it from the dir before the
sweep. Calibration is for the working range, not the failure regime.

## Step 2 — Run the sweep

From a **login node**, in tmux. The Snakemake driver does DAG scheduling
and `sbatch` submissions; actual GPU work runs on compute nodes.

```bash
mamba activate snakemake
tmux new -s calib

bash slurm_scripts/calibrate.sh all h100 \
    tests/calibration_inputs/fastas \
    /n/holylfs06/LABS/bsabatini_lab/Everyone/$USER/calib_$(date +%Y%m%d_%H%M%S)
# Detach: Ctrl-b d
# Reattach: tmux attach -t calib
```

### `all` mode

Enables `msa + boltz + esm + esmfold` in one Snakemake invocation. The
DAG sequences them via `.msa_complete` / `.boltz_complete` sentinels —
Boltz waits for all MSAs, ESM/ESMFold waits for Boltz. ES is excluded
(CPU-only, calibrate separately if needed).

### Concurrency

`CALIB_MAX_JOBS=10` (current default) caps simultaneous SLURM jobs.
Override:
```bash
CALIB_MAX_JOBS=20 bash slurm_scripts/calibrate.sh all h100 ...
```

With cap=10 and 18 sequences:
- MSA wave: ~2 batches of 10 jobs each
- Boltz wave: ~2 batches (slowest = longest FASTA × `recycling × samples`)
- ESM + ESMFold: ~2 batches each (cheap relative to Boltz)

Wall-clock is bounded by Boltz, typically 1–4 hours on H100.

### Env vars

| Var | Default | Purpose |
|---|---|---|
| `SLURM_ACCOUNT` | `kempner_bsabatini_lab` | Kempner GPU partitions need the `kempner_*_lab` account, not the lab's plain account. |
| `PROTFORGE_LOG_DIR` | `/n/home06/$USER/job_logs` | SLURM job logs. |
| `PROTFORGE_ESM_ENV` | `/n/home06/$USER/envs/esm` | ESM conda env (user-specific by convention). |
| `PROTFORGE_ESMFOLD_ENV` | `/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/envs/esmfold` | ESMFold conda env. **Shared lab env**, matching `webapp/app.py:1383`. The user-specific path in `config.template.yaml` is misleading. |

Shared MSA/Boltz paths (mmseq2_db, colabfold_db, boltz cache, etc.) are
hardcoded in `calibrate.sh` from `config.template.yaml` and don't need
overriding for Kempner.

### Why `max_files_per_job: 1`

Production runs with `max_files_per_job: 25` produce one benchmark TSV
*per chunk* (aggregating 25 sequences). Calibration wants one TSV row
*per length value* so we can fit `runtime ~ L²` per sequence:

- `msa.max_files_per_job: 1`
- `boltz.max_files_per_job: 1`
- `esm.num_chunks: 100`     (chunker caps at file count → 1 seq/chunk)
- `esmfold.num_chunks: 100` (same)

**Tradeoff:** each job re-pays its setup overhead (model load, env
activation). For Boltz (~minutes per seq) that's negligible; for ESM
(~3 s/seq + ~30 s startup) the calibrated wall times are slightly
overestimated vs production, where startup amortizes across 25 seqs.
For sizing SLURM resources with safety margins, over-estimating is the
safe direction.

To get production-mode aggregate timings later, run a separate sweep
with `max_files_per_job: 25` and join against `chunk_stats.tsv`.

## Step 3 — Output layout

```
$OUT_ROOT/
├── config.yaml                             # rendered calibration config (self-contained)
├── logs/                                   # SLURM job logs
├── summary.txt                             # post-run per-stage TSV listing
└── run/
    ├── benchmarks/
    │   ├── msa/colabfold_search_<chunk_id>.tsv
    │   ├── boltz/predict_<chunk_id>_run_<run_id>.tsv
    │   ├── esm/esm_chunk_<chunk_id>.tsv
    │   └── esmfold/esmfold_chunk_<chunk_id>.tsv
    ├── msa_chunks/chunk_stats.tsv          # chunk_id, num_seqs, mean_len, ...
    ├── boltz_chunks/chunk_stats.tsv
    ├── esm_chunks/chunk_stats.tsv
    ├── sequences/<seq>/{<seq>.yaml, msa/, boltz/, esm*, esmfold/}
    └── .{msa,boltz,esm,esmfold}_complete
```

### Benchmark TSV schema (Snakemake's `benchmark:` directive)

```
s    h:m:s    max_rss    max_vms    max_uss    max_pss    io_in    io_out    mean_load    cpu_time
```

Key columns:
- `s` — wall-clock seconds
- `max_rss` — peak resident memory (MB) — see caveat below
- `cpu_time` — total CPU seconds across cores

### chunk_stats.tsv schema

```
chunk_id    num_seqs    mean_len    min_len    p95_len    max_len    total_residues
```

### Caveat: `max_rss` is host RSS, not GPU memory

Snakemake's `benchmark:` directive uses `psutil`/rusage which captures
the host process's resident set size. For Boltz/ESM/ESMFold the actual
compute lives on the GPU; the TSV's `max_rss` is just PyTorch's
host-side process allocation (typically 14–18 GB regardless of L).

For MSA, `max_rss` is *also* misleading — it shows 110–145 GB, but most
of that is mmap'd mmseq2 database files. SLURM's cgroup-based accounting
excludes mapped files, which is why MSA jobs run fine on 48 GB allocations.

For partition routing decisions (a100 40 GB vs h100 80 GB) we need real
GPU memory. To capture it: wrap each rule shell with `nvidia-smi` polling
or insert `torch.cuda.max_memory_allocated()` reporting at end of rule.
Not yet implemented.

### Joining benchmarks × chunk_stats

`scripts/calibrate/analyze.py` does this and prints per-stage measured
vs heuristic tables:

```bash
python scripts/calibrate/analyze.py \
    --calib_dir /Users/thomasbush/tmp-data/tmp_data/benchmark/predictions/run \
    --output_dir /tmp/calib_analysis
```

The CSV output (per stage) has columns: `chunk_id, num_seqs, mean_len,
p95_len, max_len, wall_s, host_rss_mb, cpu_time_s`.

## Step 4 — Refit (pending)

`webapp/estimator.py::recalibrate_from_benchmarks` is currently a stub.
Plan: walk `<calib>/run/benchmarks/<stage>/*.tsv` and matching
`chunk_stats.tsv`, fit `runtime ~ L + L²` and `mem ~ L + L²` per
`(stage, gpu_type)` via `sklearn.LinearRegression`, write
`webapp/scaling_models.calibrated.yaml`. The estimator already prefers
the calibrated YAML over the heuristic when present.

The schema currently has `runtime_sec_per_seq: {base, alpha_L2}` for
Boltz — empirical data from the first H100 sweep clearly wants a linear
term. Schema needs an `alpha_L` field (and `_eval_time_per_seq` updated
to read it) before refitting.

## Troubleshooting

### `Invalid account or account/partition combination`

`$SLURM_ACCOUNT` doesn't have access to the chosen partition. Check:
```bash
sacctmgr -nP show assoc user=$USER format=account,partition | sort -u
```
Then:
```bash
SLURM_ACCOUNT=kempner_<your_lab> bash slurm_scripts/calibrate.sh all h100 ...
```

### `/tmp` doesn't work as `output_dir`

`/tmp` on a login node is node-local — compute nodes can't see it.
Always pass an explicit `output_dir` on a shared filesystem
(`/n/holylfs06/...` or `/n/home06/...`).

### A single sequence OOMs

Drop it from `tests/calibration_inputs/fastas/` and rerun. Calibration
is for the working range; the estimator's job is to warn before
submission, not to model the OOM threshold itself.

### `organize_boltz_chunk` failed

Boltz exited with status 0 but produced no `_model_N.cif` files (seen on
1801aa and 2039aa H100 inputs in the first sweep — likely the same
issue motivating a dedicated big-protein Boltz config). Unblock the rest
of the DAG by manually marking the failed chunks as organized:
```bash
cd /n/holylfs06/.../<output>/run/boltz_chunks/
touch chunk_<id>_run_0.organized
```
Then resume — Snakemake will write `.boltz_complete` and continue with
ESM/ESMFold. You lose those Boltz datapoints; OK for first-pass
calibration.

### Resuming after a crash

`calibrate.sh` is idempotent against an existing output dir — Snakemake
skips rules whose outputs (sentinel files) already exist:
```bash
bash slurm_scripts/calibrate.sh all h100 <fasta_dir> <existing_output_dir>
```

## Cross-references

- Repo: `slurm_scripts/calibrate.sh` — entry point.
- Repo: `scripts/calibrate/subsample.py` — stratified picker.
- Repo: `scripts/calibrate/analyze.py` — joiner + heuristic comparison.
- Repo: `webapp/scaling_models.yaml` — current heuristic coefficients.
- Repo: `webapp/estimator.py` — pure-Python module the webapp uses for estimates.
- Repo: `docs/SNAKEMAKE_GUIDE.md` — Snakemake/SLURM machinery.
- Repo: `docs/CLUSTER_SETUP.md` — Kempner shared paths and env setup.
- Vault: `Lab/protforge/log/YYYY-MM-DD-*.md` — experimental logs of individual sweeps.
- Vault: `Lab/protforge/data.md` — cluster paths for active calibration data.
- Vault: `Lab/protforge/decisions.md` — design choices and their rationale.
