# 2026-05-22 — Samplers and prompt format

First substantive session. Set up the project, defined the prompt format, and wrote the first three samplers.

## What landed in the repo

`utils.py` now has three samplers, all returning `np.ndarray[int]` of length `N`:

- `generate_discrete_uniform(rng=(0, 100), N=512, seed=None)`
- `generate_discrete_gaussian(mu, sigma, rng=(0, 100), N=512, seed=None)` — truncated discrete gaussian, PMF normalized over the integer support.
- `generate_discrete_bimodal_gaussian(mu1, sigma1, mu2, sigma2, w=0.5, rng=(0, 100), N=512, seed=None)` — each component independently truncated and renormalized, then mixed with weight `w`.

Internal helper `_truncated_gaussian_pmf(mu, sigma, support)` factored out so the gaussian and bimodal share the truncation logic.

Also renamed `generate_discrete_guassian → …_gaussian` (typo fix; no callers at the time).

## Design choices (full rationale in `decisions.md`)

- **Prompt format:** `[DIST=GAUSSIAN] [MU=22] [SIGMA=4] [START] s1 s2 … sN`. Conditioning is explicit and tokenized so we can later probe whether it's internally represented.
- **Discrete-integer first, float later.** Float tokenization introduces confounds we don't want in v0.
- **v0 distribution set: uniform + gaussian + bimodal mixture.** Each covers a research question that the others can't: uniform = max-entropy baseline, gaussian = μ/σ probes, bimodal = the *only* one that exercises the modality question.
- **Bimodal truncation:** per-component truncate-then-mix, so `w` stays the effective mixing weight regardless of where the modes sit relative to the support edges.

## Open questions (now in `agenda.md`)

- Separator/EOS token between samples — single delimiter token so position-in-sequence is unambiguous, but the exact token + tokenizer spec is unwritten.
- `data.py` still empty; needs to wrap the samplers and emit prompt-tagged sequences.
- `eval.py` still empty; first metrics will be TV/KL against the true PMF plus a sample-independence measure (autocorrelation across positions, or χ² of consecutive-pair joint vs. product of marginals).

## What I did *not* do

- No model code yet. Wanted the data pipeline and metrics locked down before any training run, so the first model results are interpretable.
- No JAX/flax setup yet beyond `jax` being listed in `pyproject.toml`.
