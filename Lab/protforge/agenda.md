# Agenda

Current focus and the next 1–3 concrete things. When done, fold into a `log/` entry; when parked, move to `decisions.md`.

## Now — container branch (2026-05-16)

- [ ] Verify yesterday's build landed where expected (`find $TBUSH -name '*.sif'`); the user reported empty `sifs/`+`sing_cache/` at one point — most likely stale stubs at the wrong path. See `log/2026-05-15-build-sh-bugfixes.md`.
- [ ] Stage-1 E2E test (post-build): single real-protein FASTA → MSA + Boltz + ESM + ESMFold through the SIF + bind-mounted DBs. Sketch is `containers/test/e2e.sh` (not built yet — design pending in the convo).
- [ ] **Container audit punch list — 15 items, grouped by priority.** Full list: [[container-audit]]. Suggested first PR bundles H1+H3+H5 (`--cleanenv`, `:ro` on DB binds, `/tmp` bind from `$SLURM_TMPDIR`).
- [ ] Apptainer vs Singularity evaluation — see [[container-audit]] L14 tie-in. Research is in flight as a background agent.

## Now — close out today (2026-05-05)

- [ ] Commit + push `webapp/scaling_models.yaml` + `webapp/estimator.py` (v3 calibrated h100 coefficients, ESMFold two-regime chunking split). Currently uncommitted on `webapp` branch.
- [ ] Wire `--chunk_size_threshold` (and `--chunk_size 64`) into `workflow/rules/esmfold.smk` so the Snakemake rule matches what the estimator predicts. Today the script's defaults (1200, 64) match by accident; should be config-driven to avoid drift.
- [ ] Fix `slurm_scripts/run_esmfold.py:117-120` — bare `except Exception` swallows CUDA OOM. Re-raise OOM (or write a sentinel file) so the rule fails properly instead of silently touching `chunk_*.done` with no PDB.

- [ ] Change chunks division: currently we divide the dataset into chunks randomly, but we should do the following: check sequence length distribution-> map it into bins and create chunks for each bin with bin-specific resources. In this way, chunks mapped to the length 100aa will requires less resources + more sequences than chunks of the bin 1k aa and so on. → detailed plan in [[bin-aware-chunking]].
- [ ] Check whether to use a different msa model/alg such as jackhammer to improve it. 
## Done today (2026-05-05)

- v3 H100 calibration finished cleanly. All 18 ESMFold chunks now produce a `structure.pdb` (incl. 1801aa, 2039aa) thanks to per-seq trunk chunking. See `log/2026-05-05-h100-calibration-v2.md` (incl. v3 update at the end).
- Patches landed in repo: `workflow/rules/esmfold.smk` (cuda module load + mamba activate), `scripts/calibrate/analyze.py` (regex bug for esmfold filenames), `slurm_scripts/run_esmfold.py` (per-seq `--chunk_size_threshold`), new `scripts/calibrate/fit_and_plot.py` (fitter + plots).
- Env mutation: `pip install torch==2.7.0+cu126` into the personal esmfold env (was 2.11.0+cu130, incompatible with H100 driver 12.9).
- Webapp updated locally (uncommitted): `scaling_models.yaml` h100 blocks for all 4 stages + ESMFold two-regime split via `chunk_threshold: 1200` + `runtime_sec_per_seq_chunked`. ESM mem dropped from 8–25 GB to 4 GB. ESM `max_chunk_size` 500 → 30. `estimator.py:322` picks chunked-path coefficients when `p95_len ≥ chunk_threshold`.
- Meeting notes + plots in `~/tmp_data/tmp_data/protforge/meeting_2026-05-05/`.

## Next — once webapp PR is merged

- [ ] Wire `webapp/estimator.py::recalibrate_from_benchmarks` to actually fit (sklearn `LinearRegression` against `runtime ~ L + L²`, `mem ~ L + L²`) and write `webapp/scaling_models.calibrated.yaml`. The estimator already prefers `.calibrated.yaml` when present.
- [ ] Production-batch sweep (`max_files_per_job: 25–30`) just for ESM — the per-job startup amortizes and the per-residue heuristic should hold. Calibrated `base ≈ 50s` is a 1-per-job artefact.

## Later — bigger gaps

- [ ] **GPU memory capture.** Snakemake's `benchmark:` is host RSS only. Wrap Boltz/ESM/ESMFold rules with `nvidia-smi` polling or insert `torch.cuda.max_memory_allocated()` reporting. ~10 lines per rule. Gates real partition routing decisions (a100 vs h100 vs h200).
- [ ] **A100 calibration sweep.** Repeat the v1 process on `kempner` (a100). Expect per-stage runtimes ~1.5× h100. We'd populate the `a100` block in `scaling_models.yaml` from real numbers.
- [ ] **Long-protein Boltz organize failure.** Reproduced again in the v3 sweep — chunk_6 (1801aa) and chunk_10 (2039aa) leave no `_model_N.cif`, so `organize_boltz_chunk` fails and the workflow stalls before ESM/ESMFold. Workaround is `touch chunk_{6,10}_run_0.organized`. Dovetails with the dedicated big-protein Boltz workstream.
- [ ] **Production-mode timing comparison.** A small `max_files_per_job: 25` sweep on the same FASTAs to quantify the per-job startup amortization (mostly relevant for ESM).
- [ ] Add support for non-slurm clusters through a singularity container that we can use on both 

## Older items (not currently active)

1. Add hidden representation options.
2. Add support for the UniProt upload — given the .csv we should extract the sequences, write to a .fasta that can be used for sequence generation.
