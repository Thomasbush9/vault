# First Chromophore-Block Layer-Grad Attribution — Working Pipeline + First Numbers

End-to-end gradient attribution for the chromophore-block experiment is now running on H100. One mutant (`seq_00132` from p40, 39.9% mutation density). Pipeline notes, first results, and the open interpretation question are below. Pivot context in `log/2026-05-06-chromophore-attribution-pivot.md`.

## Pipeline brought up

Three scripts now exist and work together:

- `scripts/analyze_chromophore_block.py` — local. Reads predicted .cif's for WT and a mutant, sanity-checks CA-RMSD on B between them, expands B with a 6 Å heavy-atom shell around the chromophore triad, persists the residue list to JSON. For `seq_00132`: tentative B preserved at 0.948 Å CA-RMSD; expanded B (n=36, 6 Å shell) preserved at 1.369 Å. Premise holds.
- `scripts/capture_distogram.py` — cluster. One no-grad forward; saves the predicted distogram + token mask. Used for WT to produce the loss reference.
- `scripts/run_chromophore_attribution.py` — cluster. Loads mutant + WT-distogram + B JSON, runs forward with `activation_checkpointing=True`, computes `L2 on E[d_pred] − E[d_WT]` over WT contacts in B (lower = more similar), backwards, saves per-residue input-embedding gradient and per-Pairformer-layer pair-rep gradient norms.

## Debugging notes from first run (kept for next time)

Three distinct issues hit before grads flowed:

1. **`_freeze_all_params` killed the trunk graph.** The legacy `install()` capture mode needs the trunk's autograd graph alive end-to-end. With params frozen and integer-indexed residue inputs, no tensor in the trunk has `requires_grad=True` → checkpoint warns "None of the inputs have requires_grad=True. Gradients will be None" → distogram has `requires_grad=False`. Fix: drop `_freeze_all_params`. Param grads accumulate (~2.4 GB) but we ignore them. The pair-rep-leaf path could afford to freeze because it injected a leaf at `distogram_module`'s input as a grad source; we don't have one.
2. **`Boltz2.__init__` freezes trunk params at construction time** when the loaded checkpoint has `structure_prediction_training=False` (`boltz2.py:351–358`). The inference checkpoint apparently was saved with that flag False, so loading freezes ~4585 trunk parameter tensors *before* `_train_mode_for_boltz_grads` flips the flag. Fix: explicit `for p in model.parameters(): p.requires_grad_(True)` after `model.eval()`.
3. **Recursive vs top-level `model.training`.** `model.training = True` only flips the top-level module; submodule training stays False, which closes the `activation_checkpointing` gate at `pairformer.py:189` and the analogous MSA gate. Fix: `model.train()` (recursive) on entry to `_train_mode_for_boltz_grads`, with `model.eval()` on exit.

After all three: forward+backward end-to-end works, peak GPU 23.16 GB on H100, 4.6 s wall.

## First numbers — `seq_00132`

- **Loss = 1.215 Å²** (sqrt ≈ 1.10 Å typical pair-distance deviation). Roughly tracks the CA-RMSD on expanded B (1.37 Å).
- **`n_contacts_in_B = 234`** WT contacts under the 8 Å threshold (out of 36×36 = 1296 possible pairs, off-diagonal).
- **All 64 backward hooks fired** (`n_missing_layer_grads = 0`). Backward through `torch.utils.checkpoint` works with `register_full_backward_hook`.
- **Peak GPU 23.16 GB; forward+backward 4.6 s.** Cohort scaling is straightforward.

## Per-layer profile

`pair_grad_norm.mean(dim=(1,2))` per Pairformer layer, plotted in `figs/attribution_seq_00132.png` panels (a)/(b).

- **Argmax layer = 0** for both whole-grid and B×B aggregations.
- Whole-grid range: 2.9e-6 to 7.8e-5 (27× spread).
- B×B range: 6.1e-5 to 3.4e-4 (5.5× spread).

**Caveat — likely vanishing-gradient artefact, not real localisation.** Layer 0 is closest to the input; the gradient there has flowed back through 63 Pairformer layers. Naive expectation is gradient *decay* with depth, not the inverse. The opposite pattern we observe could be:

- Pairformer's residual connections inverting the depth-decay direction (most likely).
- Triangle-attention non-monotonicity.
- Genuine "early-layer signal" finding.

One mutant can't disambiguate. Cohort scale-up is the test: if argmax-layer=0 reproduces across 2–3 more p40 mutants, the layer-localisation is real; if not, vanishing-gradient noise.

## Per-residue input attribution

Top 15 residues by ‖∇input‖, with WT/mut identity and B membership:

| rank | pos | WT→mut | in B |
|-----:|----:|--------|:----:|
| 1 | 72 | S→D | — |
| 2 | 73 | R→D | — |
| 3 | 71 | F→I | — |
| 4 | 69 | Q→K | ✓ |
| 5 | 68 | V→V | ✓ |
| 6 | 84 | F→F | ✓ |
| 7 | 88 | M→K | — |
| 8 | 89 | P→W | — |
| 9 | 37 | A→M | — |
| 10 | 5 | E→E | — |
| 11 | 70 | C→G | ✓ |
| 12 | 64 | F→I | ✓ |
| 13 | 7 | L→Q | — |
| 14 | 85 | K→M | — |
| 15 | 8 | F→W | — |

Two observations worth chewing on:

1. **Top hits cluster in residues 64–92** — the immediate sequence neighborhood of the chromophore (65–67), spanning into the next strand. 5/15 are in B (3D-defined). The rest are sequentially adjacent to the chromophore but *outside* B because they aren't within 6 Å of the triad in 3D.
2. **None of the canonical fluorescence-environment residues appear** — T203, R96, E222, H148 are absent from the top 15. Falsifies the hypothesis "Boltz uses biophysically-meaningful residues" for this mutant.

## Three-hypothesis frame for next steps

The first-pass result rules out the biophysics hypothesis (3 below) for `seq_00132`, but doesn't discriminate between (1) and (2):

1. **MSA-pattern matching, position-relevant residues.** Top-attributed residues should track MSA conservation/coevolution, regardless of mutation status. The homologs vote for the structure at those columns; the query is window-dressing.
2. **Training-set memorisation via high-variability anchors.** Top-attributed residues should be *uncorrelated* with MSA conservation but reproduce across GFP-like proteins. The model has learned diagnostic positions for GFP-like folds in training, regardless of MSA signal at those columns.
3. **Biophysics.** Top residues should match the GFP environment list (T203, R96, E222, H148, …). **Falsified for this mutant.**

Both (1) and (2) are publishable findings if confirmed — qualitatively different mechanisms with different mitigation paths.

## Disambiguating experiment — MSA-perturbation companion

Already on the agenda: per-column MSA conservation knockout + coevolution-pair knockout. Cross-reference with the per-residue input attribution map:

- High input-attribution at *high-conservation* MSA columns → hypothesis (1).
- High input-attribution at *low-conservation / variable* MSA columns → hypothesis (2).

This is the same per-column attribution we'd compute from gradient flow into the MSA-module's output, which the current capture path already supports (we just didn't enable the msa_emb hook for this first run — it's there as `MSA_KEY` in `GradientCapture.install()`).

## Open methodology questions

- Per-layer argmax=0: real or vanishing-gradient artefact? Cohort decides.
- Top input-attribution residues sequentially near the chromophore but *not* in 3D shell: gradient flow shortcut through the 1D `s` representation, or real signal?
- 4585 unfrozen parameter tensors × ~2.4 GB grad accumulation. Acceptable on H100 but a target for cleanup if we move to longer recycling chains.

## Status

- Working pipeline. ✓
- First mutant analysed. ✓
- Three hypotheses framed and one (biophysics) provisionally falsified for this mutant.
- Next: cohort scale-up (2–3 more p40 mutants — same layer pattern? same input-attribution clusters?), then MSA-perturbation companion to discriminate (1) vs (2).

---

# Same-day continuation — block vs whole-loss comparison + MSA-channel + IG pivot

After the first block-loss run, two extensions landed and ran:

1. **MSA-channel attribution** — `LayerCapture` now hooks `msa_module.msa_proj` (the MSA embedding before MSA-block checkpointing, `trunkv2.py:638`). One forward+backward gives per-residue MSA-channel attribution alongside per-residue query attribution. Aggregated as `‖∇msa_emb[0, :, n, :]‖₂` over (S, msa_s) → `(N,)`.
2. **Whole-structure-loss option** — `--whole_structure` flag drops the B restriction in `expected_distance_l2`. Both runs now exist on `seq_00132` (both ~5 s wall after pipeline staging).

## Files

- Block + MSA: `data/attr_seq_00132/attribution_seq_00132_block.pt` (loss=1.219, n_contacts=234).
- Whole + MSA: `data/full_struct/attr_seq_00132/attribution_seq_00132_whole.pt` (loss=1.261, n_contacts=2314).
- Comparison plot: `figs/comparison_seq_00132.png` produced by `scripts/plot_attribution_comparison.py`.

## Key findings from the comparison (one mutant only)

### (i) Argmax-layer = 0 reproduces under both losses

|  | block loss | whole loss |
|--|--|--|
| whole-grid argmax | layer 0 | layer 0 |
| B × B argmax | layer 0 | layer 0 |

Persists when we sum across the whole distogram, not specific to the block restriction. Still consistent with vanishing-gradient artefact, but it's a **robust property** of gradient flow in this setup.

### (ii) Block and whole-loss attribution maps are nearly identical

Pearson(block, whole):
- Per-residue query attribution: **0.799**
- Per-residue MSA attribution: **0.824**
- Per-layer profile (whole grid): **0.988**
- Per-layer profile (B × B): **0.980**

**Implication**: the chromophore block isn't special. The same residues that drive the model's preservation of B's geometry also drive its preservation of the whole protein. The mechanism is uniform; B inherits the result. This is itself a result on (ii) — argues against a hypothesis where the model has a *specific* chromophore-preservation circuit.

### (iii) Query / MSA ratio ≈ 5×, but same residues light up

|  | sum(query) | sum(MSA) | query/MSA |
|--|--|--|--|
| block | 8.61 | 1.65 | 5.22 |
| whole | 8.97 | 1.74 | 5.16 |

The 5× ratio is **partly normalisation**: query is one row, MSA is many rows aggregated; per-row MSA grad is much smaller. So the absolute ratio shouldn't be over-interpreted as "query matters 5× more".

What *is* interpretable is the **shape** of the per-residue distributions, and they're highly correlated between channels but with a clean dissociation in the top hits:

- **Top query residues** (by ‖∇input‖) are *mutated* (72 S→D, 71 F→I, 73 R→D, 69 Q→K, 70 C→G, 64 F→I) — gradient large where the mutation has leverage.
- **Top MSA residues** include **the chromophore itself**: S65 enters the top-10 for MSA but not query, because S65 isn't mutated → query has no perturbation lever there, but MSA at column 65 is strongly informative.

Reasonable read: query channel highlights mutated residues near the chromophore; MSA channel highlights the structurally-anchoring positions including the chromophore. Two complementary projections of the same underlying structural-importance signal.

## Why we're switching to integrated gradients (IG)

Plain gradient is a local linear approximation. Two things it can't distinguish:

- **Saturation**: a residue with a large local gradient at the mutant point but small *path-integrated* gradient (the mutation is past saturation). Plain grad would over-attribute.
- **Vanishing-gradient artefact at layer 0**: layer 0 might dominate due to local curvature at the mutant point even if path-averaged it's not dominant. Real signal vs artefact is exactly what IG can disambiguate.

IG also reformulates the question more cleanly. With WT as baseline, IG[i] = "of the prediction-shift going from WT to mutant, how much does residue i contribute?" That matches the scientific question.

But — better idea raised in conversation — **drop the mutant entirely**:

> "with IG we could extract the contribution to the final structure from each aa. We don't even need the mutants but we can just run it on the wt - baseline."

This reframes from "attribution under perturbation" to **direct structural attribution on the WT itself**. Cleaner test: if IG on WT highlights the chromophore + canonical environment residues, the model uses biophysics-meaningful residues; if it highlights random or sequence-adjacent positions, the same pattern-matching critique applies — and we don't need the mutation comparison at all.

## IG-on-WT design (committed direction)

- **Target**: WT input embedding (from WT YAML).
- **Baseline**: content-free input embedding — three options:
  1. Zero embedding (simplest; risks off-distribution behavior at α=0).
  2. **All-alanine sequence** — construct an all-A YAML, forward through `input_embedder`, capture its output (recommended). Biologically defensible, model has seen all-alanine-like patterns in training.
  3. Per-position mean over 20 AAs (most "neutral" by construction; cheap because `input_embedder` is mostly a lookup).
- **Loss / scalar**: `f(x) = -L2(distogram(x), D_WT)`. At baseline, large negative; at WT, 0. IG[i] = contribution of residue i's embedding to recovering the WT prediction.
- **Region**: whole-distogram for first cut. Asking "what drives the WT prediction overall", not "what preserves B".
- **M (path resolution)**: 20 first; bump to 50 if completeness check (`sum(IG) ≈ f(WT) − f(baseline)`) is poor.
- **Cost**: ~M forward+backward at ~5 s each ≈ 100 s. Fits trivially on H100 alongside the ~24 GB peak.

Key open decision: **baseline choice (alanine recommended; final call pending)**.

## What IG-on-WT would discriminate

Per-residue IG bar chart with annotations:

- **Chromophore triad (S65, Y66, G67) high → biophysics-relevant**, model uses them.
- **Canonical environment (T203, R96, E222, H148) high → biophysics-relevant.** The plain-gradient run on the mutant *did not* light these up; if IG-on-WT also doesn't, the falsification of biophysics is robust.
- **Sequentially-adjacent residues near 65–67 (the 64–92 cluster from plain grad) high → local-sequence pattern-matching**, not 3D structural attribution.
- **Distributed / uniform highlighting → general pattern-matching** with no structural focus.

This is a cleaner test than the WT-vs-mutant comparison because it asks the model directly which residues it depends on, without confounding from mutation effects.

## Scripts to write

- `scripts/run_wt_ig.py` (~350 lines, parallel to `run_chromophore_attribution.py`): args `wt_yaml`, `wt_distogram`, `out_dir`, `--ig_steps M=20`, `--baseline {zero,alanine}`, `--whole_structure` (default true). Builds all-A YAML in-script if needed. Forward both WT and baseline YAMLs once (no grad) to capture embeddings. Loop M times with `input_embedder` hook replacing output with `α·wt_emb + (1−α)·baseline_emb`. Save IG per-residue + completeness check value.
- Plot extension on `plot_attribution_comparison.py`: bar chart of IG with B / mutations / canonical-environment markers.

## Debug notes for future me / future debugging

If IG fires and grads are all None or weird:

1. **Check that the input_embedder hook is actually replacing the output**, not just observing it. Forward hooks return values to override; pre-hooks override args. Use `register_forward_hook` and return the modified tensor.
2. **Check that the replaced embedding has `requires_grad=True`** — the interpolation `α·a + (1-α)·b` produces a tensor that needs grad to flow back. Construct as `(alpha * mut_emb + (1 - alpha) * wt_emb).requires_grad_(True)` and `retain_grad()` to capture .grad after backward.
3. **Same param-unfreezing requirement** as plain attribution — `Boltz2.__init__` freezes trunk params at construction when checkpoint has `structure_prediction_training=False`. After load: `for p in model.parameters(): p.requires_grad_(True)`.
4. **Same recursive train mode** — `_train_mode_for_boltz_grads` must call `model.train()` (recursive), not `model.training=True` (top-level only). Otherwise `pairformer.py:189` activation-checkpointing branch doesn't fire and the trunk graph dies.
5. **Completeness sanity check** is the IG-specific diagnostic. After the loop: `sum(IG)` should approximate `loss(WT) − loss(baseline)`. If it doesn't, M is too small or the path is hitting numerical issues.
6. **`use_reentrant=True` warning** — Boltz's checkpoint call doesn't pass this kwarg explicitly. The warning is benign for plain attribution (we confirmed gradients flow). For IG, if backward hooks behave oddly under recompute, the alternative is monkey-patching each Pairformer layer to call `retain_grad` on its own z output during forward.

## Reproduction

Cluster commands (after `git pull`):

```bash
# WT distogram (already run; only re-run if invalidated)
python scripts/capture_distogram.py /n/.../original/.../original.yaml \
    --out_dir /n/.../distograms --recycling_steps 3 --no_kernels

# Block-restricted attribution
python scripts/run_chromophore_attribution.py /n/.../seq_00132.yaml \
    --wt_distogram /n/.../distograms/original_distogram.pt \
    --b_json /n/.../chromo_block.json --out_dir /n/.../grad_att \
    --recycling_steps 1 --no_kernels

# Whole-structure attribution (same call + --whole_structure)
python scripts/run_chromophore_attribution.py /n/.../seq_00132.yaml \
    --wt_distogram /n/.../distograms/original_distogram.pt \
    --b_json /n/.../chromo_block.json --out_dir /n/.../grad_att \
    --recycling_steps 1 --no_kernels --whole_structure
```

Local plotting:

```bash
.venv/bin/python scripts/plot_attribution_comparison.py \
    --block /Users/thom/tmp_data/.../attribution_seq_00132_block.pt \
    --whole /Users/thom/tmp_data/.../attribution_seq_00132_whole.pt \
    --wt_yaml  /Users/thom/tmp_data/.../original.yaml \
    --mut_yaml /Users/thom/tmp_data/.../seq_00132.yaml \
    --out_png  /Users/thom/tmp_data/.../comparison_seq_00132.png
```

## Status (end of 2026-05-07)

- Plain gradient pipeline working end-to-end ✓
- MSA channel captured ✓
- Block and whole-loss runs both done; comparison plotted ✓
- Three findings (layer-0 robustness, block/whole equivalence, query/MSA dissociation) recorded ✓
- IG-on-WT design committed; baseline choice (alanine recommended) pending decision before implementation
