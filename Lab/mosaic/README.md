# mosaic

Escalante-bio's `mosaic` is a JAX framework for multi-objective protein design via continuous relaxation: a zoo of structure/sequence models (Boltz-1/2, AF2, OpenFold3, Protenix, ProteinMPNN, ESM2/C, AbLang, stability, trigram, BoltzGen, Proteina-Complexa) reimplemented behind one differentiable interface, so composite losses across multiple predictors can be optimized jointly. This Lab folder tracks my work adapting `mosaic` to run on an HPC for a broader user base, with the workflow assisted by coding agents.

## Status

`active` — initial onboarding. Repo cloned at `~/Documents/ML/mosaic`. Upstream is `escalante-bio/mosaic`. No HPC port or user-facing tooling yet.

## Layout

- `agenda.md` — current focus, next 1–3 things.
- `guide.md` — Claude/me-facing guide to the upstream repo: architecture, conventions, gotchas.
- `log/` — dated experiment logs.
- `lit.md` — literature review (multi-objective design, hallucination, relaxation methods).
- `decisions.md` — design choices + dead ends for the HPC port.
- `data.md` — cluster paths, datasets, run IDs.
- `repo.md` — GitHub repo + setup notes.
