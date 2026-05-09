# Pivot — Occlusion → Chromophore-Block Gradient Attribution

Today's session was a design discussion, not an experiment. Outcome: occlusion is deprioritised as a *scientific* deliverable; we pivot to gradient attribution targeting a functional motif (the GFP chromophore block) with localisation onto internal model components. The occlusion code/tests stay; they may serve later as causal validators of gradient maps.

## Why drop occlusion

Combining the existing literature (lit.md):

- **Adversarial mutations paper** (lit.md:59–62): output structure is invariant up to ~40% mutation. Established at TM-score / RMSD lens.
- **Porter fold-switch papers** (lit.md:64–72): the model overrules even high-quality MSA coevolution evidence with training-set memorisation. Pattern-matching mechanism established at whole-protein lens.

Per-position query occlusion adds finer resolution on the same output-side axis. Confirmatory, not novel. And occlusion alone *cannot* discriminate MSA-pattern-matching from training-memorisation from biophysics — it only tells us *whether the query channel is being read at all*. The mechanistic question (which internal components do the work?) requires reaching into the model.

## What we're doing instead

**Functional-motif gradient attribution.**

- **Reference**: the WT predicted distogram restricted to a chromophore-block residue set `B` — `D_WT[B, B]`. Constant.
- **Variable**: the mutant's predicted distogram restricted to the same submatrix — `D_mut[B, B]`. Function of the mutant's forward pass.
- **Scalar loss**: L2 on expected pair distances, restricted to WT contact pairs within B (those with `E[d_WT(i,j)] < 8Å`). Units: Å². Stable, interpretable, end-to-end differentiable through the distogram.
- **Backward target**: gradients of this scalar w.r.t. **internal model components** of the mutant forward, at multiple layers:
  1. Input embeddings (per-residue saliency map).
  2. Pair representation `z_ℓ` at each Pairformer layer (`(layer × i × j)` heatmap of where the preservation signal lives).
  3. Attention weights per head (which heads do the preservation work — connects to the PWA-conservation finding).

Together: a mechanistic answer to "what in the model is responsible for keeping the chromophore block intact even when 40% of residues are mutated?" Aligned with lit.md:97–101's identified gap.

## Why distogram, not coordinates

We considered loss directly on the model's coordinate output (Boltz's `structure_module.sample()`, `boltz2.py:498–501`). Rejected because:

- The structure module is a **diffusion sampler** — `num_sampling_steps` defaults to 200 (`diffusionv2.py:295`). Backprop through 200 unrolled denoising steps is infeasible memory-wise on a single H100, even with diffusion-conditioning checkpointing.
- The diffusion sampling is **stochastic** — gradients exist via the reparameterisation trick but are noisy estimators that need averaging.
- For B's *backbone* geometry preservation (the operative question for fold integrity → fluorescence), the distogram already encodes the answer. The structure module's job is to embed the distogram into 3D — same geometric content, just embedded.

Distogram loss is the same scientific question for ~1–5% the cost. Coordinates still get saved (one full inference per query, no grad) for visualization + RMSD sanity checks.

## OOM strategy

The previous gradient attribution attempt (`log/2026-05-05-gradient-attribution-design.md`) hit OOM trying to backward from the full distogram through diffusion. The new framing changes both:

- Scalar loss instead of full distogram tensor target (cheaper backward).
- `skip_run_structure=True` (no diffusion in the gradient path).

Plus the now-ranked ladder:

1. `skip_run_structure=True` — free.
2. `recycling_steps=1` for attribution — ~3× memory drop.
3. **`activation_checkpointing=True`** on Pairformer + MSA + diffusion conditioning. **Boltz already has this wired** (`pairformer.py:189–198`, `boltz2.py:503`) — exposed via `PairformerArgsV2.activation_checkpointing` constructor flag. Gated by `self.training`, which is the *same* gate we already flip for the grad-enabled workaround (project memory: "Boltz2.forward gates grads in eval mode"). So enabling checkpointing is essentially free engineering — pass the flag, the existing training-gate workaround turns it on.
4. bf16 / MSA subsampling — only if 1–3 isn't enough.
5. Per-layer attribution loop — last resort.

Realistic expectation: 1 + 3 (+ maybe 2) is enough on H100. Memory probe will confirm.

## Block B — tentative residue list

For GFP, drawn from the structural-biochemistry literature on chromophore environment:

- **Chromophore triad**: S65/T65, Y66, G67. Note Boltz models the residues but not the post-translational cyclization — geometry around them is what we measure.
- **Direct chromophore environment** (mechanistically required for fluorescence): T203, R96, E222, H148, Y145, F165, F46, S205.

Total ~12–15 residues. Should be pinned down with a focused lit check on GFP fluorescence-killing mutations before locking — the operational definition of "the functional block" is which residues kill fluorescence when mutated.

## What we'd skip the structure-coordinate path for, and what we wouldn't

Distogram-only attribution captures **backbone geometry** of B. It does *not* capture:

- Side-chain orientations.
- Hydrogen-bond geometry (e.g. the E222–T203 proton-wire detail).
- All-atom packing.

If a result from distogram attribution makes us suspect side-chain detail matters (e.g., gradient maps don't align with biophysical intuition), escalate to reduced-sampling-step coordinate loss (10–20 diffusion steps + checkpointing). For first pass, distogram suffices.

## Status of the previous threads

- **Occlusion** (`scripts/run_query_occlusion.py`, `tests/test_query_occlusion.py`, `log/2026-05-05-query-occlusion.md`): code preserved. May resurface as a causal validator for high/low-attribution positions identified by gradients. Not run as a scientific experiment.
- **Gradient attribution input-leaf approach** (`log/2026-05-05-gradient-attribution-design.md`): the `cap.distogram.requires_grad=False` bug at the manually-inserted leaf is now moot — the new framing differentiates wrt internal activations using `retain_grad()`, no manual leaf. The grad-gating workaround (`Boltz2.forward` flips `set_grad_enabled` off in eval) still applies and is required.
- **MSA-side perturbation** (added to agenda this session): companion experiment to query attribution. Per-column conservation knockout + coevolution-pair knockout. Quantifies the MSA-channel contribution; with query attribution decomposes input dependence into query vs MSA. Cross-check the 2026 Porter-adjacent MSA-perturbation paper (lit.md:74–77) before designing.

## Next concrete steps, in order

1. **Locate or generate WT + p40 mutant predicted structures + distograms on the cluster.** If pipeline runs from earlier saved them, no Phase-0 inference needed. If not, one full inference each (~30s on H100 each).
2. **Sanity check the experimental premise**: CA-RMSD on B's residues between WT-pred and mutant-pred. If ≳ 1Å, model isn't actually preserving B for this mutant — pick a different mutant. Also compare WT-pred to 1EMA on B (catches model artefacts).
3. **Pin B**: focused lit check on GFP fluorescence-killing mutations to refine the residue list.
4. **Memory probe**: ~80-line script. Load model with `pairformer_args.activation_checkpointing=True`, flip training gate, forward+backward from a tiny scalar loss on `D[B,B]`, measure peak GPU memory at recycling=1 and recycling=3.
5. **Attribution script proper**: forward mutant with grad enabled + retain_grad on chosen internals + grad-gate workaround, compute scalar loss vs saved WT reference, backward, save per-layer / per-head / per-input gradient maps.
