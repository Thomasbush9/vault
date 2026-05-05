# H100 calibration — v2: full sweep, fitted coefficients, OOM ceiling

Continuation of `2026-05-04-h100-calibration-v1.md`. v1 ended with ESMFold broken
(env path bug, then driver mismatch); this entry covers the day-2 fix-and-finish:
all 18 ESMFold chunks completed on GPU, all four stages have benchmark TSVs,
heuristic coefficients fitted, and a real OOM ceiling found.

Plots: `figures/2026-05-05-calibration-fits.png`, `figures/2026-05-05-calibration-memory.png`.

## 1. What we did today

1. **Diagnosed v1c ESMFold failures.** Two distinct modes in `logs/esmfold/*.log`:
   - Stale wrong-env-path attempts left over from the v1c retry-loop (chunks 0,
     2, 17 — `EnvironmentLocationNotFound`).
   - All other chunks ran on **CPU** despite `cuda` partition. Root cause:
     `esmfold.smk` had no `module load cuda/cudnn` step (compare `esm.smk:105`),
     so `torch.cuda.is_available()` returned False and the workflow silently
     fell back to CPU. Of those CPU runs, 6 finished within budget (97–770aa),
     6 OOM-killed by the host kernel at long L.
2. **Patched `esmfold.smk`.** `module load python` →
   `module load python gcc/14.2.0-fasrc01 cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01`
   to bring up a working CUDA runtime. Switched `conda activate` →
   `mamba activate` while we were there.
3. **Found a second blocker via interactive H100 session.** Even after the
   module load, torch still complained "NVIDIA driver too old (found version
   12090)". The env's `torch==2.11.0+cu130` was built against CUDA 13; H100
   nodes max out at CUDA 12.9. Fixed in-place (the env is a personal one):
   `pip install --force-reinstall torch==2.7.0 --index-url
   https://download.pytorch.org/whl/cu126`. Smoke-tested ESMFold end-to-end on a
   58aa sequence; GPU available, fold ran in seconds, peak GPU mem reasonable.
4. **Wiped stale ESMFold sentinels and TSVs** so Snakemake would actually
   re-run the 18 chunks (the 6 CPU-time TSVs were misleading and we did not
   want them in the fit).
5. **Resumed `calibrate.sh all h100 ...`.** Run finished in 7m 13s
   (Snakemake summary: COMPLETED). All 18 ESMFold benchmarks now exist; ESM
   filled in chunks 4, 6, 11 that v1 never submitted. MSA + Boltz untouched.
6. **Built `scripts/calibrate/fit_and_plot.py`.** Loads all four stages, joins
   on `chunk_stats.tsv`, filters cold-cache loads (high `io_in`) and silent
   ESMFold OOMs, fits per-stage models, writes plots + a YAML coefficient
   block ready to merge into `webapp/scaling_models.yaml`.

## 2. Headline finding — silent ESMFold OOM ceiling

`run_esmfold.py:117-120` has a bare `except Exception` that swallows CUDA OOM,
prints to stderr, and continues. Because the Snakemake rule's
`touch chunk_X.done` runs unconditionally afterward, **the rule "succeeds"
even when no `structure.pdb` was written.** Six of the 18 calibration
sequences fall into this pattern:

| Seq    | L (aa) | Wall (s) | Has PDB |
|--------|--------|----------|---------|
| Q92878 | 1312   |   97.2   |   no    |
| Q9H7N4 | 1312   |  191.5   |   no    |
| Q9C0J8 | 1336   |   28.3   |   no    |
| Q9P2E9 | 1410   |  146.9   |   no    |
| Q5VW36 | 1801   |   97.6   |   no    |
| Q92576 | 2039   |   34.9   |   no    |

→ **ESMFold v1 (fp16) OOMs on H100 80 GB somewhere between 1159 and 1312 aa.**
The benchmark TSV's wall_s for these chunks is "model load + failed fold
attempt" not real fold time, so they have to be excluded from any time fit.

This is a real heuristic input: the webapp should refuse / warn when ESMFold
is targeted at sequences ≥ ~1300aa on H100, or route to CPU / a chunk-mode
inference.

## 3. The other big confounder — cold-cache model loads

ESMFold (and to a lesser extent ESM) `wall_s` is dominated by **first-touch
model loading** when the chunk lands on a compute node where the HF cache hasn't
been mmap'd yet. The 17.5 GB of ESMFold weights become measurable I/O. Pattern
in `io_in_mb` from the benchmark TSV:

- ESMFold cold (`io_in > 200 MB`): wall_s ranges 70–1130s.
- ESMFold warm (`io_in < 200 MB`): wall_s ranges 12–82s.

Same length L can land in either bucket. Example: two 1312aa chunks — one cold
(191.5s), one warm (12.5s). Without filtering, this drowns the L signal.

The fitter excludes high-`io_in` rows. Sample sizes after filtering: ESM 6/18
(12 cold dropped), ESMFold 6/18 (6 cold + 6 OOM dropped). MSA and Boltz unaffected
(io_in tells us nothing — MSA's I/O is the mmseq2 DB scan; Boltz's is small).

## 4. Per-stage fits

Plots in `figures/2026-05-05-calibration-fits.png` (runtime),
`figures/2026-05-05-calibration-memory.png` (host memory).

### MSA — DB-bound, basically constant

```
n=18  used=18
fit:  wall_s ≈ 1580
MAE:  179s  (variance dominated by run-to-run mmseq2 scan jitter)
current heuristic: ~50 + 0.015·L  →  60–87s  (≈25× under)
```

Same story as v1. Action: `runtime_sec_per_chunk: {base: 1580, per_seq: 0,
per_residue: 0}` for h100. Set MSA `default_gpu` more aggressively if we
want — its compute really is constant in L, so a smaller GPU class is fine.

### Boltz — clean L+L² fit, was 2.5× over at long L

```
n=18  used=18
fit:  wall_s ≈ 51.6 + 0.167·L + 0.000129·L²
MAE:  51s
current heuristic: 30 + 0·L + 0.0006·L²  (no linear term — schema gap)
```

Confirms v1 finding: the schema needs an `alpha_L` field for Boltz, and the
empirical `alpha_L2` is ~5× smaller than the heuristic. The single 397s outlier
at L=562 is the same Boltz-internal stochasticity we saw in v1 — diffusion
sampler variance with `recycling=10, samples=25`.

### ESM — startup-dominated at `max_files_per_job: 1`

```
n=18  used=6  (12 cold-cache excluded)
fit:  wall_s ≈ 62 - 0.011·L      (slope is noise — read as constant ~62s)
MAE:  19s  on n=6
current heuristic: 2 + 0.05·L    (predicts 7–104s)
```

At calibration's 1-seq-per-job setting, ESM time is dominated by Python +
transformers startup + first-touch model load (~30s) + actual embedding (~3s
even for 2k aa). The data is consistent with "constant 60s startup" plus
imperceptible L-dependence. **In production with `max_files_per_job: 25` this
amortizes**, and the heuristic's per-residue cost would actually matter — so
we can't directly transplant the calibrated coefficient.

Recommendation for h100: keep the heuristic structure (`base + alpha_L*L`) but
adjust toward `base ≈ 5, alpha_L ≈ 0.05` — the calibrated `base=62` is an
artefact of running 1-per-job. The MSA + Boltz updates are the priority; ESM
can wait for a production-batch sweep.

### ESMFold — short range only, OOM ceiling above 1200aa

```
n=18  used=6  (6 cold-cache + 6 silent-OOM excluded)
fit (linear):  wall_s ≈ -33 + 0.094·L
MAE:           4s  on n=6  (lengths 562–1159aa)
current heuristic: 3 + 0·L + 2.2e-5·L²
```

The linear slope is ~0.1 s/aa for the warm-cache range we have. The negative
intercept is a fitting artefact — the polynomial doesn't describe behaviour
below L≈350 (model-load floor) or above L≈1200 (OOM). Pragmatic update:
floor at warm-cache load time, e.g. `{base: 25, alpha_L: 0.1, alpha_L2: 0}`,
and add a hard L≤1200 ceiling in the estimator's partition routing for h100.

### Memory — host RSS, not GPU

Same caveat as v1. Snakemake's `benchmark:` captures host process RSS, not
`torch.cuda.max_memory_allocated()`. Empirical fits (green) overlaid on the
current heuristic (orange dashed) in `figures/2026-05-05-calibration-memory.png`:

| Stage   | Empirical RSS @ h100   | Current heuristic | Reading                                                                 |
|---------|------------------------|-------------------|-------------------------------------------------------------------------|
| MSA     | 110–145 GB (mmap'd DB) | ~16 GB            | RSS is misleading — cgroup excludes mapped files. Don't update from RSS. |
| Boltz   | 14–18 GB (flat in L)   | 8–12 GB curving up | Heuristic shape OK; empirical floor 1.5–2× higher.                      |
| **ESM** | **0.5–2 GB**           | **8–25 GB**       | **Heuristic over-allocates by 10–25×.** Drop ESM `mem_mb` aggressively. |
| ESMFold | 12–22 GB (flat in L)   | 14–90 GB          | Heuristic over-allocates on host. But the silent GPU OOMs at L≥1312aa show GPU activations DO grow O(L²) — host RSS just can't see it. |

**The ESMFold OOM is the proof that host RSS is a poor proxy for GPU memory.**
Chunks 6/10/11/13/14/15 had host RSS 16–22 GB right up to the moment they
crashed trying to allocate 33–126 GiB on the GPU. SLURM allocation tracks
host RSS only — the actual partition-routing constraint sits on the other
side of the PCIe bus and is currently invisible.

Concrete actions:

- ESM h100 `mem_mb`: drop to `{base: 2000, per_residue: 0.5, alpha_L2: 0}` —
  the current `{8000, 8, 0.0002}` is wildly over-provisioned. (Stays comfortably
  above empirical RSS even with safety margin.)
- Boltz h100 `mem_mb`: bump base to ~16000 to match empirical floor; leave
  `alpha_L2` until GPU-mem capture lands.
- MSA h100 `mem_mb`: leave at 16000. The 110+ GB RSS is mmap'd DB; SLURM
  cgroup accounting excludes it.
- ESMFold h100: leave heuristic alone; flag that the real ceiling is the
  GPU OOM at L ≈ 1300aa, which RSS doesn't predict.

The "real" GPU-mem capture (still on the followup list) would close all of
this. Until then we're flying blind on partition routing.

## 5b. Optimal chunk size vs L

`figures/2026-05-05-calibration-chunking.png` — using the calibrated runtime
fits, target `target_chunk_runtime_min` from `scaling_models.yaml`, and a flat
30s SLURM-startup overhead estimate, the recommended sequences-per-chunk per
stage are:

| Stage   | L=250  | L=500  | L=1000 | L=1500 | Note                                      |
|---------|--------|--------|--------|--------|-------------------------------------------|
| MSA     | 5      | 5      | 5      | 5      | Floored by `min_chunk_size`; per-seq runtime is constant ~1580s so chunk size is bounded by `target_chunk_runtime_min / 1580 ≈ 1.1`. **Should bump `max_chunk_size` floor to 1**, or just rebuild the MSA chunk model around per-chunk DB-scan amortization (one scan covers many seqs). |
| Boltz   | 26     | 15     | 7      | 4      | Drops sharply with L because of the L² term. Current `max_chunk_size: 40` is fine; **`min_chunk_size: 1` is correct** for the long tail. |
| ESM     | 29     | 31     | 34     | 38     | Roughly constant — ESM time barely depends on L (in our 1-per-job sweep, dominated by startup). Current `max_chunk_size: 500` is unrealistic; **30–40 is the real target**. Means production should set `max_files_per_job: 30`, not 25 — and we'd recover the per-startup amortization. |
| ESMFold | 200*   | 185    | 43     | 24     | Capped at `max_chunk_size: 200` below L≈500; drops fast above that. **Should also gate at `L < 1300` per the OOM ceiling.** |

\* 200 = `max_chunk_size`. Below ~500aa the model-load + startup dominates and
chunking up reduces the per-seq amortized cost.

Translation for production: at the current default `max_files_per_job: 25` the
actual chunk runtime looks like:

- MSA at L=500: 25 × 1580 = 39500s = 11h. **Way over** any reasonable target.
  → MSA should chunk at 1–2 seqs/job for `max_files_per_job: 1`-style operation,
  OR accept 11h jobs and bump the SLURM time limit. Probably the right thing is
  to drop the per-seq amortization and run MSA per-sequence with a generous
  partition.
- Boltz at L=500: 25 × 168 = 4200s = 1h10m. Under `target_chunk_runtime_min: 45`
  by 50% — **could go up to 16/chunk** at L=500, or stick with 25 and accept
  short jobs.
- ESM at any L: ~1500s/chunk for 25 seqs. Sensible.
- ESMFold at L=500: 25 × 14 = 350s = 6 min. **Way under target** — could go up
  to 200 seqs/chunk.

→ The current production-default `max_files_per_job: 25` is reasonable for
Boltz and ESM, **wrong for MSA** (over-amortizes), and **conservative for
ESMFold short proteins** (could batch much higher to amortize the 30s
model-load).

## 5. Coefficient block

`scripts/calibrate/fit_and_plot.py` writes
`/tmp/calib_fits/scaling_models_h100.yaml`. Raw output, **not yet** merged
into `webapp/scaling_models.yaml`. Suggested edits before merging:

- `msa.h100.runtime_sec_per_chunk`: `{base: 1580, per_seq: 0, per_residue: 0}`.
- `boltz.h100.runtime_sec_per_seq`: `{base: 52, alpha_L: 0.17, alpha_L2: 0.00013}`.
  (Schema also needs the `alpha_L` field added — `webapp/estimator.py::_eval_time_per_seq`
  does not currently read it. Two-line change there.)
- `esm.h100.runtime_sec_per_seq`: keep current `{base: 2, alpha_L: 0.05}` until
  a production-batch sweep — calibrated `base=62` is a 1-per-job artefact.
- `esmfold.h100.runtime_sec_per_seq`: `{base: 25, alpha_L: 0.1, alpha_L2: 0}`,
  + add an `oom_l_ceiling: 1200` knob if we want it explicit.
- Memory blocks: leave alone until GPU-mem capture is in.

## 6. Action items / next probes

- [ ] Open a PR with the four-stage h100 coefficient updates (runtime + memory).
  Single PR, scoped to `webapp/scaling_models.yaml` + the `alpha_L`-for-Boltz
  schema bump in `webapp/estimator.py::_eval_time_per_seq`. Drop ESM `mem_mb`
  to `{base: 2000, per_residue: 0.5}` — current heuristic over-allocates 10–25×.
- [ ] Update `max_chunk_size` per stage in `scaling_models.yaml` to match the
  recommended chunking analysis: ESM 30 (was 500), ESMFold 200 OK (with L≤1200
  gate), Boltz 40 OK, MSA needs a re-think (per-seq runtime > target).
- [ ] Fix `run_esmfold.py:117-120` — the bare `except Exception` is masking
  OOMs. Either re-raise CUDA OOM or write a sentinel file the Snakemake rule
  can detect, so the calibration TSV reflects real fold time.
- [ ] Add a hard `L > 1300` early refusal for ESMFold on h100 in the
  estimator. Maybe also probe lower L on a100 80GB to set the partition's
  ceiling.
- [ ] Production-batch sweep (`max_files_per_job: 25–30`) just for ESM — the
  per-job startup amortizes and the per-residue heuristic should hold.
- [ ] **GPU-mem capture** is still the gating item for partition routing
  decisions. Wrap each rule's shell with a 5-second `nvidia-smi` poller or
  insert `torch.cuda.max_memory_allocated()` reporting. Without it, host RSS
  is essentially useless for ESMFold partition decisions (proven by the silent
  GPU OOMs at L ≥ 1312aa which never showed up in host RSS).

## v3 update — conditional trunk chunking eliminates the OOM ceiling

After the v2 analysis we patched `slurm_scripts/run_esmfold.py` to enable the
ESMFold trunk chunking (`model.trunk.set_chunk_size(64)`) per-sequence whenever
`L >= chunk_size_threshold` (default 1200, just below the v2 OOM cutoff). All
other settings unchanged (still fp16 + TF32). Re-ran the full sweep on the
same 18 sequences.

Result: **all 18 ESMFold chunks now produce a `structure.pdb`**, including
the previously-failing long-tail (1312–2039 aa). No silent OOMs.

Refit (`figures/2026-05-05-calibration-fits-v3.png`,
`-memory-v3.png`, `-chunking-v3.png`):

```
[msa]      n=18  fit ≈ 1270 s constant (different node mix vs v2's 1580 s)
[boltz]    n=18  fit ≈ 214 - 0.27·L + 0.00038·L²  (more long-L data shifts shape)
[esm]      n=14 (4 cold)  fit ≈ 49 - 0.005·L  (still ~constant ~50 s @ 1-per-job)
[esmfold]  n=6  (12 cold)  fit ≈ -43 + 0.155·L  (linear, no quadratic — chunking flattens it)
```

Reading: chunked ESMFold is ~0.15 s/aa wall on h100 (warm cache), so a
2039aa protein folds in roughly 270 s end-to-end. That's actually slower
per-aa than the unchunked path was on shorter proteins (~0.10 s/aa) — the
expected speed/memory tradeoff. The H100 80GB OOM ceiling effectively
disappears for the calibration set with chunking on.

Implication for the heuristic: the v2 "L>1200 hard refuse on h100" gate is
no longer needed. The estimator can predict ESMFold runtime cleanly across
the full L range, with two regimes:

- L < 1200: use the un-chunked path (~0.10 s/aa, more variance from
  cold-cache loads at short L).
- L >= 1200: chunked path (~0.15 s/aa, no OOM).

A schema option is to keep two coefficient sets and pick at runtime, or
just use the chunked-path coefficients everywhere (over-estimating short
sequences a bit) and accept the simpler model.

Provenance: `/Users/thom/tmp_data/tmp_data/data/predictions/run/`
(cluster: `tbush/calibration/predictions/run/`). Run completed 2026-05-05
~17:17, 16m 31s wall.

## 7. Provenance

- Cluster output: `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/calibration/predictions/run/`
- Local mirror: `/Users/thom/tmp_data/tmp_data/protforge/data/predictions/run/`
- Snakemake summary: `run/benchmark_summary.txt` (COMPLETED, 7m 13s)
- Plots + raw fit YAML + per-stage CSVs: `/tmp/calib_fits/`
- Fitter: `scripts/calibrate/fit_and_plot.py` (new in this session)
- Patched files in repo:
  - `workflow/rules/esmfold.smk` — module load + mamba activate
  - `scripts/calibrate/analyze.py` — fixed esmfold regex/chunk-dir bug
  - `scripts/calibrate/fit_and_plot.py` — new
- Env mutation: in-place `pip install` of `torch==2.7.0+cu126` into
  `/n/holylfs06/.../protforge/envs/esmfold` (this env is personal — confirmed).
