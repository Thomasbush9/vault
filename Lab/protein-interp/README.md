# protein-interp

Mechanistic interpretability of cofolding diffusion models — primarily Boltz-2, with AF3 / ESMFold as comparators. The core question: cofolding models stay output-invariant under heavy mutation, but is that because the internal representations are also invariant, or because the readout discards mutation-relevant signal? And in either case, where in the model architecture is this localised?

## Status

Active. PWA-conservation analysis on GFP done (single protein); replicating on more proteins, then probing coevolution / contact-density correlates and a layer-3 intervention.

## Layout

- `agenda.md` — current focus.
- `log/` — dated experiment logs.
- `lit.md` — literature scoping.
- `decisions.md` — running history of pivots, experiment lists, and parked ideas.
- `data.md` — cluster paths, datasets, run IDs.
- `repo.md` — companion GitHub repo.
