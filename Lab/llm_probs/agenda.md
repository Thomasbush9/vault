# Agenda

Current focus and the next 1–3 concrete things to do. Keep this short — when something is done, move it to a `log/` entry; when it's a parked idea, move it to `decisions.md`.

## Now

- [ ] **Tokenizer spec.** Vocab = `SUPPORT` integers (0..100) + structural tokens (`[DIST=UNIFORM]`, `[DIST=GAUSSIAN]`, `[DIST=BIMODAL]`, `[MU=...]`, `[SIGMA=...]`, `[MU1=...]`, …, `[W=...]`, `[START]`, sample separator, `[EOS]`). Open sub-question: how to encode float μ/σ — fixed-decimals + per-digit tokens? A learned numeric embedding head? Single-token-per-discretized-bucket? This decision unblocks both the model and the collate_fn.
- [ ] **`collate_fn`** for the `DataLoader`. Pairs with the tokenizer decision — collation is the natural place to assemble `[DIST=...][params][START] s1 ... sN [EOS]`, tokenize, and batch into padded numpy arrays. Output: a dict of numpy arrays ready for `jnp.asarray`.

## Next

- [ ] `eval.py`: TV / KL between empirical model output and the true PMF, plus a sample-independence metric (autocorrelation across positions, or χ² of consecutive-pair joint vs. product of marginals).
- [ ] Tiny causal transformer in JAX/Flax. First sanity run on the uniform sampler (max-entropy baseline) before adding gaussian/bimodal.
