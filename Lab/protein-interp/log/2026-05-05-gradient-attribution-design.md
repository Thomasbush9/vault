# Gradient Attribution Tool — Design Log

Session goal: design a robust, reusable input-gradient attribution tool for Boltz-2, complementary to the existing PWA-attention extractor. Inspired by Chakravarty et al. 2024 (Nat Commun, *AlphaFold predictions of fold-switched conformations are driven by structure memorization*), which probes "what info drives the prediction" by recycling-step ablation; we extend that with explicit gradient flow.

## Why this experiment

- PWA-attention finding (`log/2026-05-04-pwa-conservation.md`): Boltz attends to *variable, coevolving* MSA columns regardless of query content. That tells us **where the model looks**.
- Gradient attribution tells us **what actually moved the output**. If attention and gradient disagree (model attends to columns whose gradient is small), that's mechanistic evidence for the cheat-sheet — attention is doing template lookup, not information transport.
- Per-recycle attribution lets us watch the model "forget" the query: prediction at step 10 may attribute almost nothing to the query token even when step 0 still does.

## Tool design (locked-in choices)

**Scalar to differentiate**: distogram logits. Specifically:
- `ContactBinNLL(i, j)` — −log p(d_ij ∈ contact_bin) for a target pair.
- `DistogramKL(p, p_ref)` — KL between predicted and a reference distogram (e.g. WT).
- `PairLogProb(i, j, bin)` — log-prob of a specified bin.

Distogram sits before the diffusion sampler ⇒ no stochastic SDE in the backward graph, deterministic gradients. This was the crux choice: we get clean gradients without diffusion noise, and distograms are already the structural readout the cited paper uses (contact maps).

**Surfaces attributed against**:

V1 (chunk 1, per-step forwards):
1. Query embedding (output of `input_embedder`) — (N, D)
2. MSA embedding rows — (S, N, D)

V2 (chunk 3, opt-in once memory budget is known):
3. Pair representation `z_k` at every recycle step k — (N, N, D) per k. Requires single-pass capture mode (graph alive across all recycles). Deferred from v1 because the full graph at K=10 may be the memory killer; per-step forwards have no z_k available across steps anyway.

The pair-rep capture is the headline addition over a vanilla saliency tool — gives a per-step (N, N) attribution map directly comparable to PWA-attention and contact maps — but it's only meaningful once we've validated the embedding-only path and confirmed memory headroom.

**Capture mode (v1 default)**: **per-step separate forwards**. For each K ∈ {0, 5, 10}, run a fresh `forward(recycling_steps=K)`, backward, save grads, drop the graph. Memory is bounded to a single forward, not K stacked. Maps cleanly onto the paper's R0-vs-RN probing structure (which is exactly the question we're asking).

Single-pass mode (one forward, multi-target backward via `torch.autograd.grad`) stays in the design as an opt-in flag for cases where memory permits and we want pair-rep capture (see below).

**Attribution method**: input × gradient as default; integrated gradients (≈10–20 interpolations) opt-in for the chromophore case study.

**Recycling resolution**: three forward passes at K ∈ {0, 5, 10}. Three points are enough to see the trajectory of query→MSA attribution shift; can densify later if the curve is non-monotonic between R0 and R10.

## Module layout

```
src/protein_interpretability/attribution/
  targets.py    # ContactBinNLL, DistogramKL, PairLogProb
  capture.py    # GradientCapture — hooks for embedder, MSA, pairformer recycles
  runner.py     # AttributionRunner.run(record, model) → AttributionResult
  io.py         # save_result/load_result, versioned schema, provenance
  cli.py        # single-record debug entrypoint
scripts/
  run_boltz_gradients.py            # multi-GPU fan-out (mirrors run_boltz_attention.py)
  boltz_gradients_config.yaml
tests/
  test_targets.py / test_capture.py / test_attribution_smoke.py
```

Pattern mirrors the existing attention pipeline so config + cluster orchestration are familiar.

## Robustness budget

- `model.eval()` + explicit `torch.set_grad_enabled(True)`. Disable `torch.compile` (incompatible with retained-grad tensors).
- Pad-mask zeroing applied at save time only; clean tensors during compute.
- Versioned save schema (`attribution_v1`) with provenance: git SHA, ckpt hash, config snapshot, target spec, peak GPU mem, recycling depth.
- Idempotent re-run: skip records whose provenance matches existing output.
- Multi-target batched backward: many pair losses per forward via `torch.autograd.grad`.
- CPU smoke test on 20-residue dummy with no MSA — full forward+backward in seconds, runs in CI.
- Memory guard: log peak after first record, warn >80% capacity.

## Validation plan (GFP chromophore)

Each mutant paired with WT MSA (only the query line changes). Save distogram at R0 vs R10, plus attribution to chromophore-pocket pair (e.g. 66↔203):

| Mutation | Role |
|---|---|
| WT | control |
| S65T | canonical, chromophore intact |
| Y66H | BFP, blue-shifted, folds |
| Y66W | CFP |
| Y66F | non-fluorescent, structure intact |
| G67A | catalytically dead, structure intact |
| R96M | chromophore electrostatics, structure intact |
| E222Q | proton-wire, structure intact |
| surface random | negative control |

Predicted outcome (consistent with PWA-attention finding): at R0 the mutant residue contributes meaningfully; by R10, attribution at the mutant position has flattened toward the MSA-driven baseline. Cleaner mechanistic version of the cheat-sheet claim.

## Open questions / follow-ups

- IG vs input×gradient: is N=20 interpolations enough on Boltz, or do we need more for stability? Decide empirically on the chromophore case.
- Pair-attribution can blow up at long recycles if the graph through pairformer + MSA module exceeds GPU. Will validate single-pass mode at K=10 on one GPU before committing.
- Need to confirm Boltz uses `torch.compile` in the inference path; if so, where to disable it for grad runs.
- Should we also capture attribution against `b = proj_z(z_n)` (the pair-representation projection feeding MSA module)? That's the input side of the recycle; could pair nicely with the z-attribution we already have.

## Provenance

- Paper: `~/tmp-data/tmp_data/papers/AlphaFold predictions of fold-switched conformations are driven by structure memorization.pdf` (Chakravarty et al. 2024, doi:10.1038/s41467-024-51801-z).
- Existing attention pipeline (template): `scripts/run_boltz_attention.py`, `src/protein_interpretability/extract_attention.py`, `src/protein_interpretability/extractor_boltz.py`.
- Status: design only. No code written this session.
