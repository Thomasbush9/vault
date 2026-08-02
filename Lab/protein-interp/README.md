# protein-interp

Mechanistic interpretability of cofolding diffusion models — primarily Boltz-2, with AF3 / ESMFold as comparators. The core question: cofolding models stay output-invariant under heavy mutation, but is that because the internal representations are also invariant, or because the readout discards mutation-relevant signal? And in either case, where in the model architecture is this localised?

## Status

Active. Rebuilt on the JAX/joltz reimplementation of Boltz-2 (2026-07-30
onward), which makes exact causal interventions possible.

**Where it stands (2026-08-01).** Route decomposition on all 12 assays: the
mutation reaches the trunk output through the pair representation
(necessity 0.840, sufficiency 0.995), *not* through the MSA — which rejects the
project's original MSA-memorisation hypothesis. The trunk represents mutational
effect well (probe rho 0.548 on held-out positions) and the model's own
structural output does not (TM-to-WT 0.214), gap +0.335 in 57/60 assay-splits.
sigma-resolved trajectory analysis localises the loss further: the sampler
ignores the conditioning entirely while the global fold is being decided and
becomes mutation-sensitive only during fine refinement.

Now testing whether this is a **gain** problem — `exp_amplify.py`, scaling only
the mutant-minus-wild-type difference in the trunk state, against a norm-matched
permuted control.

See `publication_plan.md` for what is still missing, `methods_pairformer_interp.md`
for methods, and `log/2026-07-30-jax-pairformer-harness.md` for the chronological
record, including **nine** conclusions withdrawn or corrected — two of them the
same hand-rolled-superposition bug, the second made after the first was written
up, which is why every geometric primitive now self-tests on import.

## Layout

- `agenda.md` — current focus.
- `publication_plan.md` — **what this needs to be publishable**: the framing
  redirection, blocking referee objections, the cross-model plan (OF3/Protenix
  weights are already on disk), and the intervention ladder.
- `methods_pairformer_interp.md` — **technical methods** for the Boltz-2 trunk
  interpretability work: harness, units conventions, cohort construction,
  every experiment with its controls, and the figure index.
- `log/` — dated experiment logs.
- `lit.md` — literature scoping.
- `decisions.md` — running history of pivots, experiment lists, and parked ideas.
- `data.md` — cluster paths, datasets, run IDs.
- `repo.md` — companion GitHub repo.
