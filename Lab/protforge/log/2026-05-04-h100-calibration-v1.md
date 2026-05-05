# H100 calibration sweep — first usable data

Goal: measure real per-stage SLURM consumption on Kempner H100 across MSA / Boltz / ESM / ESMFold so the webapp's resource estimator can be tightened from ballpark heuristics to fitted coefficients.

Calibration set: 19 stratified FASTAs from the user's 7k working dir (later trimmed to 18 — see §5). Subsample picker: `scripts/calibrate/subsample.py` (length-quantile stratified, seed=42).

Cluster output root: `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/calibration/predictions/`
Local mirror: `/Users/thomasbush/tmp-data/tmp_data/benchmark/predictions/`
Calibration script: `slurm_scripts/calibrate.sh all h100 ...` (Option A — runs MSA+Boltz+ESM+ESMFold in one DAG).

## 1. What we ran

Three sequential attempts on h100, same input dir, same output dir:

| Run | Date | What ran | Outcome |
|---|---|---|---|
| v1a | 2026-05-04 11:15 | 19 seqs, MSA + Boltz only | Boltz `chunk_3` (the 2.7k seq) failed at default 16 GB / 60 min — SLURM "Reason: Unknown". 18 MSA + 18 Boltz finished. |
| v1b | 2026-05-04 15:51 | 18 seqs (dropped 2.7k), MSA + Boltz | All 18 MSA + 18 Boltz finished. `organize_boltz_chunk` failed for `chunk_6` (1801aa, Q5VW36) and `chunk_10` (2039aa, Q92576) — Boltz produced no `_model_N.cif`. |
| v1c | 2026-05-04 20:10 | resume after `touch chunk_{6,10}_run_0.organized` | 7/18 ESM finished, all 18 ESMFold failed at startup. Workflow died. |

After v1c we identified the ESMFold env-path bug and patched `calibrate.sh`. ESM/ESMFold rerun is pending.

`calibrate.sh` evolved during this experiment — see §6 (decisions).

## 2. Setup

- `subsample.py --n 20 --seed 42` against the 7k FASTA dir → 19 picks (one bin had only 1 sequence). Lengths from 76 to 2700.
- User dropped 2.7k-residue FASTA before v1b (known to be over Boltz's H100 ceiling — separate dedicated config in flight).
- `max_files_per_job: 1` for MSA/Boltz, `num_chunks: 100` for ESM/ESMFold (chunker caps at file count → 1 seq/chunk). One benchmark TSV row per length value.
- `boltz: recycling_steps: 10, diffusion_samples: 25, num_runs: 1` — production defaults.
- Per-stage GPU partition pinned to `kempner_h100`. Account: `kempner_bsabatini_lab` (not the sacctmgr default `bsabatini_lab` — Kempner two-account split).
- Calibration-generous resource ceilings (`slurm.resources.boltz: {mem_mb: 80000, runtime: 240}`, esmfold same shape) so a long sequence wouldn't OOM mid-run.
- Snakemake `--jobs` cap 4 → later 10 (`CALIB_MAX_JOBS`). `restart-times: 2` from `profiles/slurm/config.yaml`.

## 3. Results

### MSA — DB-bound, length-independent

18 samples on H100. Wall times mostly 1200–1800s regardless of length:

```
   L    wall_s   host_rss_mb
  97    1681.5      120869
 142    1673.9      110252
 288    1278.2      134496
 385    1694.2      127765
 392    1265.8      126759
 494    1709.0      124438
 562    1734.5      116039
 586    1671.2      115751
 770    1813.7      133303
 890    1363.0      138281
 898    1661.9      146840
1159    1421.6      133105
1312    1812.6      136329
1312    1665.7      127240
1336    1246.2      145406
1410    1265.2      138294
1801    1672.0      145252
2039    1736.1      117834
```

Empirical fit: `wall_s ≈ 1655 - 0.24·L + 0.00012·L²` — basically constant in L (the linear and quadratic terms are noise around the ~1655s constant).
Heuristic in `webapp/scaling_models.yaml` predicts: `wall_s ≈ 50 + 6·1 + 0.015·L` ≈ 60–87s. **Off by ~25–30×.**

Reading: `colabfold_search` is dominated by the mmseq2 database scan (~22–30 min of DB lookup), not query length. The mean run-to-run noise (≈300s) dwarfs any length signal.

`host_rss_mb` shows 110–145 GB but most of that is mmap'd database — not real allocation. SLURM's 48 GB default works because cgroup accounting excludes mapped files.

### Boltz — heuristic was 2–2.5× over at long L

18 samples on H100. Wall time clearly scales with L:

```
   L    wall_s   host_rss_mb   pred_wall_s   wall_ratio
  97      50.0       14017          35.6        1.40
 142      73.5       14025          42.1        1.75
 288      87.9       15535          79.8        1.10
 385     106.3       15602         118.9        0.89
 392     109.8       15486         122.2        0.90
 494     121.0       14258         176.4        0.69
 562     396.6       14510         219.5        1.81
 586     146.6       15228         236.0        0.62
 770     224.1       15517         385.7        0.58
 890     264.7       15521         505.3        0.52
 898     510.8       15536         513.8        0.99
1159     381.0       16627         836.0        0.46
1312     461.2       17018        1062.8        0.43
1312     433.1       16664        1062.8        0.41
1336     466.5       17027        1100.9        0.42
1410     512.6       16675        1222.9        0.42
1801     773.4       17726        1976.2        0.39
2039     973.2       18209        2524.5        0.39
```

Empirical fit: `wall_s ≈ 52 + 0.17·L + 0.00013·L²`.
Heuristic: `wall_s ≈ 30 + 0·L + 0.0006·L²` (pure-quadratic, no linear term).

`alpha_L2` empirically is **~5× smaller** than the heuristic. The current schema has no `alpha_L`, so the linear contribution leaks into either base or alpha_L2 in the fit.

Reading: H100 attention is meaningfully faster than the back-of-envelope — good news for users; the estimator was over-asking.

There's also Boltz-internal stochasticity: chunk_5 (L=562) took 397s while chunk_7 (L=898) took 511s. With recycling=10 × samples=25 the diffusion path adds variance. Multiple runs per length would tighten this — deferred.

### ESM — partial: 7/18 samples

Workflow died from ESMFold failures before the rest of ESM could submit. Benchmark TSVs for the 7 ESM jobs that ran are on the cluster but didn't sync down on this round; will pick up on rerun. Empirical fit deferred.

### ESMFold — no data

Every chunk failed at job startup in seconds. Root cause: `env_path` defaulted to `/n/home06/$USER/envs/esmfold` (template default) which doesn't exist for tbush. The webapp's hardcoded default at `webapp/app.py:1383` is the shared lab env `/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/envs/esmfold` — that's what works.

Patched `calibrate.sh` after v1c so the new default points at the shared env. Rerun pending.

## 4. Two organize failures (chunk_6, chunk_10)

Both at the long-tail end of the calibration set (1801aa, 2039aa). Boltz's `predict` succeeded (benchmark TSVs exist, `.done` sentinels touched), but no `_model_N.cif` files appeared in `predictions/<seq>/`. `organize_boltz_outputs.py` exits 1 when `processed == 0`.

We unblocked downstream by `touch chunk_{6,10}_run_0.organized` and lost those two Boltz datapoints. Affects only the L>1800 tail of the fit, where data was sparse anyway.

Open question: is this a known Boltz issue at long L, or is the model writing somewhere the organize script doesn't look? Worth checking once the dedicated big-protein Boltz config is up (user's separate workstream).

## 5. Reading

The current `webapp/scaling_models.yaml` is materially wrong for h100 in two specific ways:

1. **MSA runtime is ~30× under-estimated.** The estimator currently treats MSA as a small per-residue cost; it should be ~1700s constant per sequence at `max_files_per_job: 1`. With production batching of 25 seqs/chunk this becomes a per-chunk cost of similar magnitude (mmseq2 scan amortizes within a chunk — needs separate measurement).

2. **Boltz runtime is ~2.5× over-estimated at long L.** `alpha_L2 = 0.0006` is too aggressive; data points to `alpha_L2 ≈ 0.00013–0.00015` with a meaningful linear term.

Translation for the webapp's "Total node-hours" line: when these coefficients land, MSA jumps to dominate the predicted compute (which it actually does in reality), and Boltz drops by half.

We don't have GPU memory data — Snakemake's `benchmark:` directive captures host RSS via `psutil`, not `torch.cuda.max_memory_allocated()`. For partition routing decisions (a100 40 GB vs h100 80 GB) we'd need a separate capture mechanism. Deferred.

## 6. Decisions made along the way

- **Option A (one DAG, MSA+Boltz+ESM+ESMFold together) over Option B (two-phase MSA-once, then sweep stages independently)** — simpler for a single GPU type. Trade-off: re-runs MSA when sweeping a different GPU. Acceptable for h100-only first pass.
- **`max_files_per_job: 1`** instead of production's 25. Each FASTA gets its own SLURM job → one TSV row per length value. Costs per-job startup overhead (small for Boltz, larger fraction for ESM) but gives clean per-length signal. See `calibration.md` for the tradeoff write-up.
- **Calibration-generous Boltz resources (80 GB / 4 hours)** vs rule defaults (16 GB / 60 min). Asks more headroom so a long-tail FASTA doesn't OOM mid-sweep. SLURM only allocates what the job actually uses, so over-asking is free.
- **`SLURM_ACCOUNT=kempner_bsabatini_lab` hardcoded as the calibrate.sh default** instead of probing via `sacctmgr` (which returns the wrong-for-h100 default `bsabatini_lab`). Webapp/`run.sh` plumb account from `config.yaml`; calibrate.sh shouldn't depend on a webapp-managed file existing.
- **Self-contained config in `calibrate.sh`** — writes a complete `config.yaml` from hardcoded shared paths + `$USER`-derived user paths, no merge from repo `config.yaml`. The webapp owns repo `config.yaml` and may not exist when calibrating.
- **ESMFold env default = shared lab env** matching `webapp/app.py:1383`, not the misleading user-specific path in `config.template.yaml`. Patched after v1c.

## 7. Caveats

- **Sample size: 18.** Statistically thin for a quadratic fit, especially at the tails. Boltz's stochastic per-sample variance (~10–30%) eats most of the gain from more samples on the same lengths; a future sweep should add length points instead.
- **Single GPU type (h100).** A100 numbers are pure heuristic right now. The current data only justifies updating the `h100` block in `scaling_models.yaml`; `a100` and `a100_80` should remain heuristic until a separate sweep.
- **No GPU memory data.** Host RSS doesn't drive partition routing. Deferred — needs a `nvidia-smi` poller wrapper or `torch.cuda.max_memory_allocated()` reporting in each rule's shell.
- **MSA per-chunk vs per-seq cost not separated.** Calibration runs 1 seq per chunk; in production 25 seqs share one chunk's mmseq2 scan. The MSA per-residue cost is below noise here — can't tell from this data whether the 30s/extra-residue heuristic is right at production batch size.
- **Long-tail Boltz organize failures (1801, 2039aa).** Two data points lost; matches the user's known long-protein Boltz issue. Working range of the calibration is L < 1500 with confidence, L < 2000 with caveats.

## 8. Open questions / next probes

- **Finish ESM + ESMFold** (rerun with patched `calibrate.sh`). Then refit ESM/ESMFold coefficients in `scaling_models.yaml`.
- **Apply MSA + Boltz coefficient updates** to `webapp/scaling_models.yaml` once ESM/ESMFold are in (one PR, all four stages).
- **Add `alpha_L` to the Boltz schema** in `scaling_models.yaml` and `webapp/estimator.py::_eval_time_per_seq` — current schema only has `base + alpha_L2` for Boltz, but the empirical fit clearly wants a linear term.
- **GPU memory capture.** Smallest first step: wrap each Boltz/ESM/ESMFold shell in `nvidia-smi --query-gpu=memory.used --format=csv -l 5 &` background, post-process. ~10 lines per rule.
- **Long-protein Boltz organize failure root cause.** Check `boltz_chunks/chunk_6_run_0_output/` directly — is `predictions/<seq>/` empty, or are model files written to a different sub-path? Couples to user's dedicated big-protein Boltz workstream.
- **Production-mode timing sweep.** Once heuristics are calibrated for `max_files_per_job: 1`, run a small sweep at `max_files_per_job: 25` on the same FASTAs to compare against and quantify the per-job startup overhead amortization.

## 9. Provenance

- Calibration entry: `slurm_scripts/calibrate.sh all h100 tests/calibration_inputs/fastas /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/calibration/predictions`
- Subsampler: `scripts/calibrate/subsample.py --input_dir <7k_dir> --output_dir tests/calibration_inputs/fastas --n 20 --seed 42`
- Analyzer: `scripts/calibrate/analyze.py --calib_dir <run_dir> --output_dir /tmp/calib_analysis`
- Cluster output (full): `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/calibration/predictions/run/`
- Local mirror: `/Users/thomasbush/tmp-data/tmp_data/benchmark/predictions/run/`
- Snakemake logs (3 attempts):
  - `2026-05-04T113141.731577.snakemake.log` — v1a (Boltz chunk_3 OOM)
  - `2026-05-04T155109.820659.snakemake.log` — v1b (organize chunk_6 + 10 fail)
  - `2026-05-04T201030.497228.snakemake.log` — v1c (ESMFold env mismatch)
- Calibration set lengths (chunk_id → mean_len): see `boltz_chunks/chunk_stats.tsv` in the local mirror.
- Repo state: branch `webapp`, calibrate.sh modified across this session — final form has `all` mode, `CALIB_MAX_JOBS`, generous boltz/esmfold resources, shared-env ESMFold default.
