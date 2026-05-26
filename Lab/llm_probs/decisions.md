# Decisions

Append-only log of design choices and dead ends, so they don't get re-litigated.

## Format

```
### YYYY-MM-DD — <decision title>

**Decided:** what we're doing.
**Why:** the reason.
**Alternatives considered / parked:** what we passed on, briefly.
```

---

### 2026-05-22 — Prompt format for conditioning the model

**Decided:** Sequences look like `[DIST=GAUSSIAN] [MU=22] [SIGMA=4] [START] s1 s2 … sN`.
**Why:** Mirrors the *Bad Dice Players* prompting style, makes the distribution-conditioning the explicit thing the model has to learn, and decouples the "what distribution" decision from the "what samples" decision so we can later probe whether the conditioning is internally represented.
**Alternatives considered / parked:** Implicit conditioning (one model per distribution family) — parked because it eliminates the most interesting probe.

### 2026-05-22 — Discrete-integer tokens first, float tokens later

**Decided:** v0 vocabulary is integers over a fixed support (default `[0, 100]`), tokenized one-integer-per-token. Float-valued samples become variant 2.
**Why:** Collapses the tokenizer question, gives a finite categorical output so KL / TV against the true PMF are well-defined, and keeps the AR-vs-masked comparison apples-to-apples.
**Alternatives considered / parked:** Per-digit float tokenization, IEEE-bit decomposition, bucketed-float — all introduce tokenization confounds we don't want mixed into the first results. Will revisit once the discrete pipeline is end-to-end.

### 2026-05-22 — Starter distribution set: uniform, gaussian, bimodal mixture

**Decided:** v0 ships three samplers in `utils.py`: `generate_discrete_uniform`, `generate_discrete_gaussian`, `generate_discrete_bimodal_gaussian`.
**Why:** Uniform = max-entropy sanity baseline. Gaussian = canonical unimodal, varying μ probes location, varying σ probes scale/entropy. Bimodal mixture is the *only* one that exercises the modality question (Q3) — you cannot probe modality with unimodal data.
**Alternatives considered / parked:** Geometric/truncated-exponential (skew test) — cheap to add later, parked for v0. Dirac/categorical-spike — skipped, mostly tests plumbing not the science.

### 2026-05-22 — Truncated discrete gaussian: per-component truncation for the mixture

**Decided:** For `generate_discrete_bimodal_gaussian`, each component is independently truncated and renormalized over the integer support, *then* mixed with weight `w`.
**Why:** Keeps `w` as the effective mixing weight regardless of where the modes sit relative to the support edges — predictable for the modality experiments.
**Alternatives considered / parked:** Joint truncation (mix two continuous gaussians, then truncate/renormalize together) — shifts the effective mixing weight when a mode is near the boundary. Easy to swap in if we want to study edge effects.

### 2026-05-23 — Framework split: JAX/Flax for the model, torch for data plumbing

**Decided:** The model and training loop will be JAX + Flax. Data loading uses `torch.utils.data.Dataset` / `IterableDataset` / `DataLoader`. The dataset returns plain numpy arrays (no `torch.from_numpy(...)` wrapping); conversion to JAX is a single `jnp.asarray(batch)` call at the model boundary.
**Why:** JAX has no native dataloader, and torch's machinery (multi-worker, batching, pinning) is the standard JAX-ecosystem solution. Returning numpy avoids a useless torch→numpy→jax round-trip and keeps the dataset decoupled from torch tensor semantics.
**Alternatives considered / parked:** Pure JAX with a hand-rolled batcher (more code, no real upside for prototyping). Returning `torch.Tensor` from the dataset (forces a `.numpy()` call before every model step). `tf.data` (extra dependency, ecosystem mismatch).

### 2026-05-23 — On-the-fly data generation via `IterableDataset` + priors

**Decided:** `DistStream(IterableDataset)` generates batches on the fly. Distribution configs are not hardcoded — they're sampled from a `DistPriors` dataclass per item (type weights, `mu_range`, `sigma_range`, `w_range`, `bimodal_min_separation`). The fixed-list `DistDataset(Dataset)` is retained as a sibling for held-out validation sets where the same configs are needed every epoch.
**Why:** Sampling from synthetic distributions is cheap, so there's no reason to materialize a fixed training set — fresh draws each step give effectively infinite data and avoid memorization confounds. The priors layer lets us vary `mu`, `sigma`, `w` continuously, which is what we need to probe whether the model encodes those parameters in its hidden state.
**Alternatives considered / parked:** Pre-materialized training set with a large `N` of pre-sampled sequences (parked — wasteful and locks the parameter sweep at materialization time). Generating samples directly inside a non-Dataset loop (parked — loses multi-worker batching for free).

### 2026-05-23 — Bimodal min-separation enforced by rejection sampling

**Decided:** When sampling a bimodal config from `DistPriors`, reject any draw where `|mu1 - mu2| < bimodal_min_separation` (default `15.0` on a 100-wide support). Resample until the constraint holds.
**Why:** Without this, a non-trivial fraction of "bimodal" samples are effectively unimodal (two modes overlapping inside ~1σ). The model can't learn the modality concept from those examples, and the `BIMODAL` label becomes misleading at evaluation time.
**Alternatives considered / parked:** Analytic constrained sampling (overkill; rejection rate is ~25% with the defaults, so 1–2 attempts on average). No constraint (parked — corrupts the modality probe).

### 2026-05-23 — Continuous (float) μ, σ from day one; tokenization deferred

**Decided:** `DistPriors` samples `mu` and `sigma` as floats (uniform over `mu_range`/`sigma_range`). `w` is also float. No quantization at sampling time.
**Why:** The interesting hidden-state probes (does the model represent `mu`? `sigma`?) are sharper when the conditioning parameters vary continuously rather than over a small integer grid. Tokenization of these float parameters is a separate (later) decision; we'll figure out a "cool way" to tokenize them when we wire up the tokenizer.
**Alternatives considered / parked:** Integer-only μ/σ for v0 simplicity — parked because it caps the resolution of the probe before the model is even built, and the tokenization saving is illusory (we have to design the parameter encoding either way).

### 2026-05-23 — `SUPPORT` is a module-level constant, not a per-call kwarg

**Decided:** `utils.py` exposes `SUPPORT = (0, 100)` at module scope. The three samplers no longer accept an `rng` (range) kwarg — they read `SUPPORT` internally. `data.py` also drops its `rng` kwarg from `DistStream` and `DistDataset`.
**Why:** The range is structurally the **vocabulary support** of the categorical output head, not a per-distribution sampling parameter. It's a project-wide constant (changing it implies changing the tokenizer and the model's output projection in lockstep), so it doesn't belong in a per-call signature where it implies callers can vary it freely. Hoisting also removes a parameter from every sampler call site.
**Alternatives considered / parked:** A `DatasetConfig` dataclass threading `support` + other globals through every entry point — more structure than v0 needs; revisit if/when we add float-token variants that need different support handling.
