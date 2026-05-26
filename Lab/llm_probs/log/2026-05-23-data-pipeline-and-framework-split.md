# 2026-05-23 — Data pipeline and framework split

Wrapped up `data.py` and clarified what runs on what framework. No model code yet.

## What landed in the repo

### `data.py`

- `DistPriors` dataclass: prior ranges over distribution configs (`types`, `type_weights`, `mu_range`, `sigma_range`, `w_range`, `bimodal_min_separation`). Defaults: `(1, 2, 2)` weighting (uniform underweighted because it has no params), `mu_range=(10, 90)`, `sigma_range=(1, 15)`, `w_range=(0.3, 0.7)`, `bimodal_min_separation=15`.
- `sample_spec(priors, gen)`: draws one config. Bimodal rejection-samples until `|mu1 - mu2| >= bimodal_min_separation`.
- `DistStream(IterableDataset)`: infinite stream for training. Multi-worker safe via `seed + worker_id`. Each item `{type, params, samples}`.
- `DistDataset(Dataset)`: fixed-list sibling for held-out validation sets, same item shape.

Samples come out as raw `np.int64` arrays (not torch tensors) so the model boundary is a single `jnp.asarray(...)`.

### `utils.py`

- Hoisted `SUPPORT = (0, 100)` to module scope. The three samplers no longer take an `rng` kwarg.

## Design choices (full rationale in `decisions.md`)

- **JAX/Flax for the model, torch for data plumbing.** Borrowing torch's `Dataset`/`DataLoader` because JAX has no native one. Dataset returns numpy; conversion to JAX happens at the model boundary.
- **On-the-fly generation via `IterableDataset` + priors.** Sampling is cheap, so no point materializing a fixed training set; sampling distribution configs from priors gives effectively infinite, non-memorisable data.
- **Bimodal min-separation by rejection sampling.** Without this, ~a third of "bimodal" examples are effectively unimodal and the modality label is misleading.
- **Float μ, σ from day one.** Continuous variation gives sharper hidden-state probes than an integer grid. Tokenization of float params is deferred — it's a separate problem.
- **`SUPPORT` is a module constant.** Range is the vocabulary support of the categorical output head, not a per-distribution sampling parameter.

## Open questions (now in `agenda.md`)

- **Tokenizer spec**: vocab (`SUPPORT` integers + `[DIST=...]`, `[MU=...]`, etc. + sample separator + `[START]` + `[EOS]`) and how float μ/σ get encoded. This is the next blocker — model code can't start without it.
- **Collate function for `DataLoader`**: items are dicts containing strings (`type`), dicts of floats (`params`), and numpy arrays (`samples`). Default torch collate won't handle this cleanly. The collate_fn is the natural place to also assemble + tokenize the prompt, so it pairs with the tokenizer decision.
- **`eval.py`**: empty. First metrics planned are TV/KL against the true PMF + a sample-independence measure (autocorrelation across positions, or χ² of consecutive-pair joint vs. product of marginals).

## What I did *not* do

- No tokenizer. Wanted to keep `data.py` format-agnostic so the tokenizer decision can be made independently of the data plumbing.
- No model. Sticking to "data + metrics locked down first" so the first training results are interpretable.
