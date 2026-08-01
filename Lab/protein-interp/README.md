# protein-interp

Mechanistic interpretability of cofolding diffusion models — primarily Boltz-2, with AF3 / ESMFold as comparators. The core question: cofolding models stay output-invariant under heavy mutation, but is that because the internal representations are also invariant, or because the readout discards mutation-relevant signal? And in either case, where in the model architecture is this localised?

## Status

Active. Rebuilt on the JAX/joltz reimplementation of Boltz-2 (2026-07-30
onward), which makes exact causal interventions possible. Route decomposition
and per-operation attribution replicated on two proteins; ProteinGym stability
probe done on four assays; RSA (pairformer vs structure module) and the
ensemble-spread test running. See `methods_pairformer_interp.md` for methods and
`log/2026-07-30-jax-pairformer-harness.md` for the chronological record,
including four conclusions that were withdrawn or corrected.

## Layout

- `agenda.md` — current focus.
- `methods_pairformer_interp.md` — **technical methods** for the Boltz-2 trunk
  interpretability work: harness, units conventions, cohort construction,
  every experiment with its controls, and the figure index.
- `log/` — dated experiment logs.
- `lit.md` — literature scoping.
- `decisions.md` — running history of pivots, experiment lists, and parked ideas.
- `data.md` — cluster paths, datasets, run IDs.
- `repo.md` — companion GitHub repo.
