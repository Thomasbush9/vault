# Query Occlusion — Pivot from Gradient Attribution

Built and pushed a per-position query-occlusion script for Boltz2 after gradient attribution stalled on a `cap.distogram.requires_grad=False` bug we couldn't pinpoint in the available time. Occlusion answers a similar mechanistic question — *which mutant residues actually shift the predicted contact map?* — with pure forward passes, no autograd, no OOM risk.

## Why pivot

- Gradient attribution on a 600M-param model required leaf-replacement at `distogram_module`'s input + freeze-all-params + `model.training=True` with a flip-back hook to satisfy Boltz's grad gates (`boltz2.py:411,440`). Even with that scaffolding, the captured distogram had `requires_grad=False`. Diagnosing the lost grad inside `distogram_module(z_leaf)` would mean instrumenting hooks deeper, and we don't have a strong signal that gradients vs occlusion would differ on the actual scientific question.
- The science question — *does Boltz use the query, or memorize from the MSA in the presence of mutations?* — admits a direct test: revert mutations one at a time and watch the distogram. If reverting position *i* barely changes the prediction, the model wasn't using mutation *i*. If a few positions dominate the divergence, those are the residues the model "listened to."

## What's built

`scripts/run_query_occlusion.py`. Single-script driver, bootstraps `sys.path` so it runs under the cluster boltz env without `python -m`. For a given mutant YAML + WT YAML:

1. Diffs sequences to find mutated positions.
2. For each mutated position *i*, generates a variant where position *i* is reverted to WT.
3. Per variant: writes a YAML and a **patched copy of the MSA** (`out_dir/_variant_msas/<stem>.a3m`) where the first sequence row matches the variant query — see "MSA bug" below.
4. Runs all variants through Boltz with `model.skip_run_structure = True` (skips diffusion sampling; distogram is computed before, see `boltz2.py:491,498-501` — ~5–10× speedup per variant).
5. Captures the distogram via a forward hook on `model.distogram_module`, computes per-(i,j) `KL(baseline ‖ variant)` on the bin distribution, aggregates over valid token pairs.
6. Saves `<stem>_occlusion_summary.pt` (per-position rows + sequences + token mask) and optionally `<stem>_occlusion_distograms.pt` (raw `(B, N, N, num_bins)` per variant).

## MSA bug — caught after first push

First version pointed every variant YAML at the same MSA file. Boltz's featurizer at `featurizerv2.py:259-290` checks the YAML's `protein.sequence` against the MSA's first row (the reference query) and **silently swaps in a dummy MSA on any non-MET/UNK mismatch**. Result: every reverted variant ran with no MSA → confounded experiment ("drop MSA + revert position *i*" instead of "keep MSA + revert position *i*").

Fix: per variant, copy the .a3m and overwrite the first sequence row with the reverted query; homolog rows untouched. Verified end-to-end on `/Users/thom/tmp_data/tmp_data/protein_interpretability/data/132_protein_.a3m` (238-char first row, 270 homolog lines) — patched copy preserves all homolog rows byte-for-byte, source file untouched, length-mismatch raises cleanly. Tests in `tests/test_query_occlusion.py` (10/10 pass).

This warning is the canary going forward: if `Warning: MSA does not match input sequence, creating dummy.` appears in stdout, the patch isn't reaching the featurizer and results are confounded.

## How to launch (tonight from MCP)

```bash
ssh <cluster>
cd /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_interpretability
git pull
# interactive H100 session, then:
bash test_occlusion.sh 2>&1 | tee /tmp/occ.log
```

Smoke test runs on `seq_00132` from p40 with `--recycling_steps 3 --keep_distograms`. Expected log:
- `[occ] mutant=… stem=seq_00132 length=<N> mutations=<K>`
- `[occ] wrote <K+1> variant YAMLs … + patched MSAs in …`
- One `[occ X/Y] <record_id> distogram shape=(1, N, N, 64)` per variant
- **No** `Warning: MSA does not match input sequence` lines (canary)
- Top-15 sorted table: `pos / wt->mut / kl_mean / kl_max`

Outputs: `/n/holylfs06/.../protein_rsa/occlusion/occlusion_seq_00132/`
- `seq_00132_occlusion_summary.pt` — small, the headline result
- `seq_00132_occlusion_distograms.pt` — `(B, N, N, 64)` per variant; only present with `--keep_distograms`
- `_variants/`, `_variant_msas/` — staged inputs (debuggable)

## What to look for in the result

The summary's `rows` is a list of dicts: `{record_id, kind, position, wt_aa, mut_aa, kl_mean, kl_max, n_pairs}`. Plot `kl_mean` against `position`. Hypotheses, in order of "interesting":
- **Flat profile** (all `kl_mean` ≈ 0): model effectively ignores the mutant query — strong cheat-sheet evidence. The MSA's homolog rows alone determine the prediction.
- **Sparse spikes** at chromophore-adjacent positions (~65, 66, 67, 96, 222) and structural hubs: model uses mutations selectively at functionally-important sites.
- **Smooth profile correlated with structural-contact density**: model uses every mutation, weighted by local structural importance. Less mechanistically interesting but useful sanity check.

## Open / next

- Run the smoke; confirm no MSA warning; eyeball the top-15.
- If results look reasonable, scale to a cohort (orchestrator pattern from `scripts/run_boltz_attention.py`).
- Cross-reference per-position KL with PWA-attention from `log/2026-05-04-pwa-conservation.md` — do "high-occlusion" positions also have high PWA attention, or are they separate axes?
- Return to gradient attribution as a follow-up. The pair-rep refactor (frozen params + leaf at `distogram_module` input) is structurally right; the residual `requires_grad=False` bug is in the last mile and needs patient hook instrumentation.
