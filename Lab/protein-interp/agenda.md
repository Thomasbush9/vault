# Agenda

Current focus and the next 1–3 concrete things. When done, fold into a `log/` entry; when parked, move to `decisions.md`.

## Now — query occlusion smoke (tonight from MCP)

Pivot from gradient attribution → per-position WT-reversion forward sweeps. Same scientific question (does Boltz use the mutant query?), pure-forward implementation, no autograd. Full context in `log/2026-05-05-query-occlusion.md`.

- [ ] **Launch the smoke.** Interactive H100 session, then:
      ```bash
      cd /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability
      git pull
      bash test_occlusion.sh 2>&1 | tee /tmp/occ.log
      ```
      Targets `seq_00132` from p40, recycling=3, `--keep_distograms`.
- [ ] **Verify the canary**: stdout must NOT contain `Warning: MSA does not match input sequence, creating dummy.` If it does, the per-variant .a3m patch isn't reaching the featurizer — results are confounded (Boltz fell back to dummy MSA).
- [ ] **Eyeball the top-15 table** at the end of `/tmp/occ.log` (`pos / wt->mut / kl_mean / kl_max`). Three rough hypotheses (flat / sparse-at-functional-sites / structure-density-correlated) sketched in the log.
- [ ] **Pull the summary back locally** for plotting:
      `seq_00132_occlusion_summary.pt` is small (~KB); `_distograms.pt` is tens of MB.
      Path: `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/occlusion/occlusion_seq_00132/`.

## Next

- [ ] If smoke is clean, plot per-position `kl_mean` profile and cross-reference against PWA-attention columns from `log/2026-05-04-pwa-conservation.md`.
- [ ] Scale occlusion to a cohort of p40 mutants (orchestrator pattern from `scripts/run_boltz_attention.py`).
- [ ] Return to gradient attribution: residual `cap.distogram.requires_grad=False` after `distogram_module(z_leaf)` despite leaf having grad + global grad enabled. Needs patient hook instrumentation, not more guesses. Pair-rep scaffolding is structurally right — see `log/2026-05-05-gradient-attribution-design.md`.

## Parked / blocked

- [ ] Replicate PWA-conservation finding on 2–3 more proteins (not just GFP).
- [ ] Probe 7: MI / APC-corrected MI per column, correlated with `A_j`.
- [ ] Probe 8: structural-contact density per column, correlated with `A_j`.
- [ ] Chromophore validation of attribution tool (depends on gradient attribution working).
- [ ] No-MSA control.
- [ ] Layer-3 intervention: replace `random_uniform` PWA output at layer 3 with `original`'s.
