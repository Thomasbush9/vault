# 2026-05-07 — Bin-aware chunking landed; ga200 MSA OOM diagnosed

Two threads today:

1. Diagnosed the ga200 production-run failure as an MSA OOM (not a wall-time kill, contrary to my initial hunch). Bumped MSA mem heuristic.
2. Implemented the bin-aware chunking feature (agenda line 11, plan in `bin-aware-chunking.md`). All four stages.
3. Iterated the binning schema once after first try landed: chunk size per bin → single `chunks_per_bin` + per-bin mem/runtime.

## 1. ga200 MSA OOM — what happened

Run: `/n/holylfs06/.../tbush/ga200/`. Six MSA chunks all died with the same signature in `logs_ga200/logs/msa/colabfold_search_*.log`:

```
mmseqs expandaln ... -- threads 4
... died with <Signals.SIGKILL: 9>
```

That's the SLURM cgroup OOM-killer signal. A wall-time kill would say "DUE TO TIME LIMIT" in the slurm scheduler log; SIGKILL on `mmseqs` is the kernel/cgroup terminating the process for exceeding its memory cap. `expandaln` is the step that scans `colabfold_envdb_202108_db_aln` (~140 GB) — its working set blows past whatever mem the job had.

**Root cause:** the v2 calibration (`log/2026-05-05-h100-calibration-v2.md`) noted MSA host RSS hit 110–145 GB and concluded "that's mmap'd DB pages, cgroup excludes them — keep heuristic at 16 GB". That conclusion was wrong on Kempner: the cgroup *does* charge mmseqs's working set against the colabfold DB scan. So `mem_mb=16000` killed every chunk.

**Fix landed (uncommitted, on `webapp` branch):**
- `webapp/scaling_models.yaml`: `msa.{a100,h100}.mem_mb.base: 16000 → 256000` (256 GB, headroom over the 110–145 GB observed peak). Comment block updated to record the correction.
- `workflow/rules/msa.smk:71`: fallback default `48000 → 256000` so users without webapp resource overrides still get safe mem.

**To resume the ga200 run** (still failed on the cluster as of writing):
1. `git pull` on the cluster's ProtForge checkout.
2. In the webapp's MSA Settings → "Memory (MB)" widget shows 256000 directly (new manual-override knob — see §3).
3. Save → `snakemake --profile profiles/slurm/ --rerun-incomplete`. Snakemake re-submits only the 6 dead chunks.

The Kempner partition (`kempner_h100`, `kempner_requeue`) has H100 nodes with ≥1.5 TB RAM, so 256 GB is comfortably below capacity.

## 2. Bin-aware chunking — implementation

Plan was in `Lab/protforge/bin-aware-chunking.md` (written 2026-05-06). Implementation landed today on `webapp` branch (uncommitted).

### Architecture

- **Shared library**: `workflow/scripts/binning.py` — quantile / threshold computation, bin assignment, even-distribution chunk packing, `chunks.tsv` writer/reader, CLI flag definitions, rule-side `chunk_resource()` lookup.
- **Chunkers** (`chunk_yamls_for_esm.py`, `prepare_boltz_chunks.py`, `chunk_fastas.py`): when `--enable_binning` is set, partition by length, split each non-empty bin into `chunks_per_bin` chunks, write `<output>/<stage>_chunks/chunks.tsv` alongside the existing `manifest.txt` + `chunk_stats.tsv`.
- **Snakemake rules** (`msa.smk`, `boltz.smk`, `esm.smk`, `esmfold.smk`): each rule's `mem_mb` and `runtime` are now lambdas that look up `wildcards.chunk_id` in `chunks.tsv`. Falls back to `stage_resource(...)` defaults when chunks.tsv missing (binning disabled). Snakefile-level helper `binning_args(stage_cfg, stage_name=...)` renders config → CLI flags.
- **Estimator** (`webapp/estimator.py`): `BinSpec` / `BinPlan` dataclasses, `compute_bin_plan()` evaluates scaling model per bin (mem from bin's max length, runtime from chunk_size × per_seq + base setup). `apply_estimate_to_config` writes `<stage>.binning.bins:` recipe.
- **Webapp** (`webapp/app.py`): per-stage **Bin-aware chunking** toggle + 3 columns (mode, num_bins/thresholds, chunks_per_bin) + preview table.

### Schema (after iteration — see §2.b)

```yaml
boltz:
  binning:
    enabled: true
    mode: quantile               # quantile | thresholds
    num_bins: 6
    chunks_per_bin: 5            # how many parallel chunks each non-empty bin gets
    thresholds: [...]            # quantile mode: echoed; thresholds mode: required input
    bins:                        # per-bin SLURM resources, smallest L first
      - {mem_mb: 16000, runtime_min: 30}
      - {mem_mb: 24000, runtime_min: 45}
      - {mem_mb: 32000, runtime_min: 60}
      - {mem_mb: 50000, runtime_min: 90}
      - {mem_mb: 80000, runtime_min: 180}
      - {mem_mb: 80000, runtime_min: 240}
```

### Bin packing math

For each non-empty bin with `n` items and `chunks_per_bin = k`:
- `n_chunks = min(k, n)` — sparse bins produce one chunk per item, no empties.
- `base, extra = divmod(n, n_chunks)` — first `extra` chunks get `base+1` items, rest get `base`.

Example: `n=4, k=3` → `n_chunks=3, base=1, extra=1` → sizes `(2, 1, 1)`. For `n=10, k=3` → `n_chunks=3, base=3, extra=1` → sizes `(4, 3, 3)`. Even.

Items inside a bin are sorted by length before splitting, so each chunk's L range is internally contiguous and the `mem_mb` / `runtime_min` allocated for that bin's max length is a tight fit.

### chunks.tsv shape

Columns: `chunk_id  bin  num_seqs  min_len  max_len  p95_len  mem_mb  runtime_min`.

Numeric chunk_ids (0, 1, 2, …) sequential across all bins. The rule's resource callable reads this and returns the per-chunk values. Calibrate analyzer (`scripts/calibrate/analyze.py`) still parses chunk_id as int — works.

### 2.b — Schema iteration during the day

First version had `chunk_size` per bin in the recipe — derived from `target_chunk_runtime_min / per_seq_runtime`. User pushed back: "there should be two numbers: bins to divide between them and then chunks per bin". Refactored:

- Removed `chunk_size` from the recipe.
- Added top-level `chunks_per_bin: int` (single number across bins).
- Chunker derives chunk size at run time from `bin_count / chunks_per_bin`.

Cleaner mental model — user controls `num_bins × chunks_per_bin` directly.

### Failure-mode softening

First version: `binning.enabled: true` with empty `bins:` raised `ValueError` at workflow load → blocked even `snakemake -n`. Real failure on cluster: user toggled binning on in webapp without clicking "Apply to session config".

Soft-fail now: stderr warning + fall back to non-binning mode for that stage. Rule still loads. Fully-formed `bins[i]` missing keys still raise (real config bug).

## 3. Webapp memory + runtime knobs (added before binning)

User asked to make MSA / Boltz / ESM / ESMFold / ES mem visible per stage in the settings dialog. Added `render_slurm_resources(cfg, stage)` helper:

- 3-column row at top of each stage's expander: **Memory (MB)**, **Runtime (min)**, **CPUs per task**.
- Reads/writes `cfg["slurm"]["resources"][stage]` — same slot the estimator's "Apply" button uses.
- Defaults match the rule fallbacks (MSA 256000 / 60 / 4, Boltz 16000 / 60 / 8, ESM 32000 / 60 / 16, ESMFold 32000 / 120 / 8, ES 16000 / 120 / 8).

These knobs are the safety-net resources used when binning is OFF or chunks.tsv is missing. When binning is ON the per-chunk values from `chunks.tsv` win.

## 4. Files touched (all uncommitted on `webapp` branch)

| File | What |
|---|---|
| `Snakefile` | New `chunk_resource(...)` and `binning_args(stage_cfg, stage_name)` helpers. |
| `workflow/scripts/binning.py` | New shared lib. |
| `workflow/scripts/chunk_yamls_for_esm.py` | Bin-aware path + chunks.tsv emission (legacy mode preserved). |
| `workflow/scripts/prepare_boltz_chunks.py` | Same. |
| `workflow/scripts/chunk_fastas.py` | Same (MSA). |
| `workflow/rules/{msa,boltz,esm,esmfold}.smk` | Per-chunk `mem_mb` / `runtime` lambdas. Each rule's `binning_args(...)` invocation. |
| `workflow/rules/msa.smk:71` | Mem fallback 48000 → 256000. |
| `webapp/scaling_models.yaml` | MSA mem base 16000 → 256000 + corrected note. |
| `webapp/estimator.py` | `BinSpec`/`BinPlan`, `compute_bin_plan`, apply path writes `<stage>.binning`. `InputStats.lengths_sorted` retained. |
| `webapp/app.py` | `render_slurm_resources` + `render_binning_controls` + wiring into all 5 stage expanders. |
| `config.template.yaml` | Documents new `binning:` block + `max_seq_len` cutoffs (Boltz / ESMFold). |

## 5. To pick up tomorrow / from home

**Cluster work:**
1. `git pull` ProtForge on `webapp` branch.
2. Resume ga200: just `--rerun-incomplete` after the mem bump should work. Verify the 6 MSA chunks now finish.
3. Once MSA passes, the ga200 run continues into Boltz/ESM/ESMFold. Watch for the long-protein Boltz organize failure (chunks containing 1801aa+) — known issue, agenda "Later" item.

**Bin-aware sanity check:**
1. Use the 200-FASTA dataset (sampled with `scripts/calibrate/subsample.py`).
2. Webapp: toggle "Bin-aware chunking" ON for ESMFold, set `num_bins=6, chunks_per_bin=5`, click "Apply to session config" to populate the recipe.
3. `snakemake --profile profiles/slurm/ -n` first to inspect the DAG. Should see ~30 chunks for ESMFold with varying `mem_mb` / `runtime`.
4. Real run: monitor `<output>/esmfold_chunks/chunks.tsv` for the per-chunk recipe and SLURM `MaxRSS` per chunk after completion.

**Known issues / sharp edges:**
- ESMFold mem heuristic at L=2039aa returns ~115 GB after safety margin — exceeds H100's 80 GB. The `mem_mb: {base: 14000, alpha_L2: 0.018}` formula in `scaling_models.yaml` overshoots in the long tail. Manual workaround: trim the highest bin's `mem_mb` in session config before submitting. Real fix: GPU memory capture + recalibrate (separate Later items).
- Webapp toggle alone doesn't populate the recipe — must click "Apply to session config" after enabling. Fall-back warning prevents hard fail but prints to stderr.
- Calibration analyzer (`scripts/calibrate/analyze.py:62`) does `int(chunk_id)` — still works with binning since chunk_ids are numeric. Doesn't yet parse the new `bin` column in `chunks.tsv` (TODO if we want per-bin calibration analytics).

**Not yet committed.** Everything still on `webapp` branch local. Next session: smoke test on cluster, then commit (one PR or stage-by-stage commits — unfinished call).

## 6. Provenance

- ga200 logs: `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/ga200/outputs/logs/msa/colabfold_search_*.log` (also mirrored at `~/tmp_data/tmp_data/protforge/logs/logs_ga200/logs/msa/`).
- ga200 outputs: `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/ga200/outputs/`.
- Plan doc: `Lab/protforge/bin-aware-chunking.md`.
- Calibration v3 input set (used for all local smoke tests): `~/tmp_data/tmp_data/protforge/data/predictions/run/sequences/` (18 sequences, L=97–2039aa).
