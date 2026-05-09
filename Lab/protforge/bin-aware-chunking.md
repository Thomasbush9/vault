# Bin-aware chunking — implementation plan

Status: **planned, not implemented.** Tracks `agenda.md` line 11 ("Change chunks division: …map sequence length distribution into bins…").

## Context

Current chunkers (`workflow/scripts/chunk_yamls_for_esm.py`, `prepare_boltz_chunks.py`, `chunk_fastas.py`) split sequences by *count* — round-robin or ceil-division on alphabetically-sorted paths. This wastes resources in two directions:

- **Heterogeneous chunks waste mem/time budgets.** A chunk containing one 1800aa sequence and twenty 200aa sequences must be sized for the 1800aa worst case, so the 20 short sequences pay 9× the memory they need and run far below the SLURM time limit.
- **Single-knob chunk size doesn't match runtime curves.** Boltz/ESMFold runtime is O(L²); ESM is startup-dominated. The optimal `seqs/chunk` for L=200 is wildly different from L=1500. One `max_files_per_job` value can't be right for both.

The existing 2-pool ESMFold split (`bin_by_length: true`, short=L<1200, long=L≥1200) is a special case of this. User decision (2026-05-07): generalize to N bins, all four sequence stages (MSA, Boltz, ESM, ESMFold), quantile-based by default.

## Recommended approach

### 1. Bin definition — quantile-based (auto-adaptive)

Default: 5 bins at quantile boundaries `[0, q25, q50, q75, q90, q95, q100]` of the input length distribution. The upper tail gets two narrow bins (q90–q95, q95–q100) because that's where O(L²) memory blows up and where calibration showed ESMFold OOM-cliffs sit. Same shape as `scripts/calibrate/subsample.py:DEFAULT_BINS`.

Explicit-thresholds override (advanced): if config sets `bin_thresholds: [400, 800, 1200, 1800]` the chunker uses those instead. Plan exposes both modes; quantile is the default.

### 2. Per-bin chunk-size policy

For each bin `b` with representative length `L_b` (use bin's p95 length to stay safe):

```
per_seq_runtime_s = scaling_model.eval_time_per_seq(stage, L_b)
chunk_size_b = floor(target_chunk_runtime_min * 60 / per_seq_runtime_s)
clamped to [min_chunk_size, max_chunk_size]
```

Result: small-L bins → many seqs/chunk (amortize startup); large-L bins → few seqs/chunk (avoid timeout). `target_chunk_runtime_min`, `min/max_chunk_size` already exist per-stage in `webapp/scaling_models.yaml`.

### 3. Per-bin SLURM resources

For each bin, evaluate the scaling model at the bin's max length:

```
mem_mb_b      = scaling_model.eval_mem(stage, L_max_b) * safety_margin
runtime_min_b = ceil(chunk_size_b * per_seq_runtime_s / 60) + startup_overhead_min
```

Both numbers go into a per-chunk metadata sidecar (next section) so the Snakemake rule can request them.

### 4. Per-chunk metadata sidecar

Today the chunk_id is encoded in the filename (`id_0.txt`, `id_short_0.txt`). Generalizing to N bins via prefix encoding (`id_b0_3.txt`) is OK, but **a sidecar TSV is cleaner**: chunker writes `chunks.tsv` next to `manifest.txt` with one row per chunk:

| chunk_id | bin | num_seqs | min_len | max_len | p95_len | mem_mb | runtime_min |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 60 | 50  | 280  | 270  | 16000 | 30  |
| 1 | 0 | 60 | 50  | 280  | 270  | 16000 | 30  |
| 5 | 4 | 4  | 1800| 2039 | 2039 | 80000 | 240 |

Chunk IDs stay numeric (works with the existing calibrate analyzer's `int(chunk_id)` assumption). The Snakemake rule's `mem_mb` / `runtime` callables read this TSV (cached at module load) and look up by `wildcards.chunk_id`. Same TSV is what the webapp displays in a "binning preview" before run.

### 5. Config schema

Per stage (under `msa:`, `boltz:`, `esm:`, `esmfold:`):

```yaml
binning:
  enabled: false                    # opt-in; default off keeps current behavior
  mode: quantile                    # quantile | thresholds
  num_bins: 5                       # quantile mode only
  thresholds: [400, 800, 1200]      # thresholds mode only
  target_chunk_runtime_min: 30      # already in scaling_models.yaml; this is per-stage override
  safety_margin: 1.3                # mem multiplier above scaling-model prediction
  startup_overhead_min: 5           # added to per-bin runtime estimate
```

When `binning.enabled: false` (default), behavior is exactly as today — no migration needed for existing configs. When `true`, the chunker computes bins, chunk sizes, and per-chunk resources internally; user-set `num_chunks` / `max_files_per_job` are ignored for that stage.

### 6. Estimator integration (webapp)

`webapp/estimator.py::estimate_stage` extends to optionally produce `BinPlan`:

```python
@dataclass
class BinPlan:
    bins: list[BinSpec]               # per-bin counts, length range, chunk_size, mem, runtime
    total_chunks: int
    total_runtime_min_estimate: int
```

`BinSpec`:
```python
@dataclass
class BinSpec:
    bin_id: int
    num_seqs: int
    len_range: tuple[int, int]
    chunk_size: int
    num_chunks: int
    mem_mb: int
    runtime_min: int
```

The webapp's "Apply to session config" button writes `binning.enabled: true` plus the computed `num_bins` (or `thresholds`) plus a snapshot of the plan into `slurm.resources.<stage>.binning_preview` (advisory; chunker recomputes from input distribution at run time so the plan stays in sync if input changes).

### 7. Webapp UI (Settings dialog, per stage)

Inside each stage's expander:

- New toggle: **"Bin-aware chunking"**.
- When on: select **Mode** (quantile / thresholds), **Number of bins** (or thresholds list).
- A read-only preview table (via `render_estimate_panel`) showing per-bin: count, length range, chunk_size, mem_mb, runtime_min — populated from the most recent `BinPlan`.
- The single-knob `mem_mb` / `runtime` overrides (just landed) become inactive when binning is on; UI shows them as "managed by binning" and disables the inputs.

## Files to modify

| File | Change |
|---|---|
| `workflow/scripts/chunk_yamls_for_esm.py` | Replace 2-pool short/long with N-bin loop. Output `chunks.tsv` sidecar. Quantile + thresholds modes. |
| `workflow/scripts/prepare_boltz_chunks.py` | Same. Boltz still produces directories of symlinks per chunk; binning just changes which file ends up in which chunk. |
| `workflow/scripts/chunk_fastas.py` | Same generalization. MSA's case is mild (DB-bound, constant in L) but adding it keeps the model uniform across stages. |
| `workflow/rules/{msa,boltz,esm,esmfold}.smk` | Each rule's `mem_mb` / `runtime` resource fields become callables that look up `wildcards.chunk_id` in `chunks.tsv` (cached at module load with `functools.lru_cache`). Same callable for all four — extract a helper. |
| `webapp/estimator.py` | `BinPlan` / `BinSpec`. `estimate_stage` returns a plan when `binning.enabled`. |
| `webapp/scaling_models.yaml` | Already has `target_chunk_runtime_min`, `min_chunk_size`, `max_chunk_size` per stage — no changes needed. |
| `webapp/app.py` | Per-stage binning toggle + preview table. Disable the manual mem/runtime inputs when binning is on. |
| `config.template.yaml` | Document `binning:` block under each stage with comments + a worked example. |
| `tests/` (new) | Unit tests for the binning math: quantile boundaries, chunk_size derivation, `chunks.tsv` round-trip. |

## Migration path

The existing ESMFold-only `bin_by_length` schema (`bin_by_length`, `length_threshold`, `num_chunks_short/long`, `mem_short/long_mb`, `time_short/long_min`) is **subsumed** by the new scheme. Plan:

1. Land the generic binning behind `binning.enabled: false`.
2. Prove parity: run the calibration set under `binning.enabled: true, num_bins: 2, thresholds: [1200]` and confirm same chunk layout / same wall-time as the current `bin_by_length: true`.
3. Mark the old keys deprecated in `config.template.yaml`. Both schemas coexist for one cycle.
4. Remove the old keys after one production cycle. Touch `webapp/app.py` (drop the smart-binning sub-section) and `workflow/rules/esmfold.smk` (drop `_esmfold_mem_mb` / `_esmfold_runtime` callables, drop the wildcard_constraints for `short_/long_` IDs).

## Verification

1. **Unit tests** (`tests/test_chunkers.py`, new):
   - Quantile mode on a synthetic 1000-seq dataset with bimodal length distribution → 5 bins with expected counts at each quantile.
   - `chunks.tsv` round-trip: write, parse from rule, verify mem/runtime match the generator.
   - Thresholds mode produces the right chunks for explicit cuts.

2. **Snakemake dry-run** with `binning.enabled: true` on the v3 calibration YAML dir (18 sequences, lengths 97–2039aa) for ESMFold:
   - DAG includes 5 bins.
   - `--printshellcmds` shows different `--mem` / `--time` per chunk.
   - Smallest-bin chunks request <16 GB; largest-bin chunks request 80+ GB.

3. **Real ESMFold run** comparing `binning.enabled: true (num_bins=5)` vs the existing `bin_by_length: true (2-pool)` baseline on the 200-FASTA test set:
   - Total wall time within ±15% (binning shouldn't make it slower).
   - SLURM `MaxRSS` per chunk within 70–95% of requested mem (no over-provisioning, no OOMs).
   - Per-bin runtime within ±20% of the estimator's `runtime_min` prediction.

4. **Cross-stage consistency**: same dataset, all four stages binning-enabled. Inspect each `<stage>_chunks/chunks.tsv` to confirm the bin assignment is consistent across stages (Boltz and ESMFold should bin the same FASTA into the same bin since lengths are identical).

5. **Long-tail-only test**: 200 FASTAs where 5 are >1800aa. Confirm those 5 land in the top bin alone, with 1–2 seqs/chunk and `mem_mb ≥ 80000`.

## Open design questions to revisit during implementation

- **Bin overhead**: the chunker pays a one-time cost to read every sequence's length. For 10k+ sequences this is non-negligible in a checkpoint; cache lengths in a per-input-dir manifest if it gets slow.
- **MSA's bin payoff**: MSA runtime is constant (~1270s/chunk regardless of L) and mem is dominated by the colabfold DB. Binning may add complexity for little benefit. Could ship MSA without binning and only enable for Boltz/ESM/ESMFold; reconsider after measuring.
- **GPU partition routing**: per-bin mem implies per-bin partition choice (e.g., long-bin → h100, short-bin → a100). Out of scope for v1; uses one partition per stage as today.
- **ESM bin-startup tradeoff**: ESM time at 1-per-job is dominated by Python startup (~50s constant). Binning won't fix that — the existing `max_files_per_job: 25-30` recommendation does. Binning still helps mem (currently 4 GB heuristic is fine for any L) but the win is small.

## Out of scope

- GPU memory capture (separate "Later" item; gates real partition routing).
- A100 calibration sweep (separate "Later" item).
- Cross-stage chunk reuse (binning each stage independently is fine; aligning bin boundaries across stages is a future optimization).
- Auto-tuning `safety_margin` from observed `MaxRSS` (would close the loop with `recalibrate_from_benchmarks` — separate workstream).
