# Agenda

Current focus and the next 1–3 concrete things. When done, fold into a `log/` entry; when parked, move to `decisions.md`.

## Now — finish first H100 calibration

- [ ] Resume calibration on the cluster with the patched ESMFold env path. ESM has 7/18 done; ESMFold has 0/18. One command unblocks both:
  ```bash
  bash slurm_scripts/calibrate.sh all h100 \
      tests/calibration_inputs/fastas \
      /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/calibration/predictions
  ```
- [ ] Sync ESM + ESMFold benchmarks down to `/Users/thomasbush/tmp-data/tmp_data/benchmark/predictions/run/benchmarks/{esm,esmfold}/`.
- [ ] Re-run `scripts/calibrate/analyze.py` and capture the ESM/ESMFold tables. Append to `log/2026-05-04-h100-calibration-v1.md` (or open a new `2026-05-05-...` log if it spills past today).

## Next — apply learned coefficients

- [ ] Add `alpha_L` field to the Boltz `runtime_sec_per_seq` schema in `webapp/scaling_models.yaml` and update `webapp/estimator.py::_eval_time_per_seq` to read it.
- [ ] Apply the v1 H100 coefficient updates (single PR covering MSA + Boltz + ESM + ESMFold) — drop into the heuristic YAML for now, not a separate `.calibrated.yaml`. Wait for refitter.
- [ ] Wire `webapp/estimator.py::recalibrate_from_benchmarks` to actually fit (sklearn `LinearRegression` against `runtime ~ L + L²`, `mem ~ L + L²`) and write `webapp/scaling_models.calibrated.yaml`. The estimator already prefers `.calibrated.yaml` when present.

## Later — bigger gaps

- [ ] **GPU memory capture.** Snakemake's `benchmark:` is host RSS only. Wrap Boltz/ESM/ESMFold rules with `nvidia-smi` polling or insert `torch.cuda.max_memory_allocated()` reporting. ~10 lines per rule. Gates real partition routing decisions (a100 vs h100 vs h200).
- [ ] **A100 calibration sweep.** Repeat the v1 process on `kempner` (a100). Expect per-stage runtimes ~1.5× h100. We'd populate the `a100` block in `scaling_models.yaml` from real numbers.
- [ ] **Long-protein Boltz organize failure.** Both organize failures (1801aa, 2039aa) point at Boltz producing no `_model_N.cif` for these sizes. Dovetails with the user's separate dedicated big-protein Boltz workstream.
- [ ] **Production-mode timing comparison.** A small `max_files_per_job: 25` sweep on the same FASTAs to quantify the per-job startup amortization (mostly relevant for ESM).

## Older items (not currently active)

1. Add hidden representation options.
2. Add support for the UniProt upload — given the .csv we should extract the sequences, write to a .fasta that can be used for sequence generation.
