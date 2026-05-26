# Data

Pointers to where data and run artifacts live. This file is a lookup, not a story.

## Sampler API (v0)

All three live in `~/Documents/ML/llm_probs/utils.py` and sample over the integer support `SUPPORT = (0, 100)` (module-level constant — see `decisions.md`). Common tail: `N=512, seed=None`.

- `generate_discrete_uniform(N, seed)` → `np.ndarray[int]` of shape `(N,)`.
- `generate_discrete_gaussian(mu, sigma, N, seed)` → truncated-and-renormalized discrete gaussian over `SUPPORT`.
- `generate_discrete_bimodal_gaussian(mu1, sigma1, mu2, sigma2, w, N, seed)` → per-component truncated mixture, see `decisions.md`.

## Dataset API (v0)

In `data.py`:

- `DistPriors` dataclass — prior ranges for distribution configs (`types`, `type_weights`, `mu_range`, `sigma_range`, `w_range`, `bimodal_min_separation`).
- `sample_spec(priors, gen)` — draws one config from priors. Bimodal rejection-samples until `|mu1 - mu2| >= bimodal_min_separation`.
- `DistStream(IterableDataset)` — infinite, multi-worker-safe stream for training. Pair with `DataLoader(stream, batch_size=B, num_workers=N)`.
- `DistDataset(Dataset)` — fixed-list dataset for held-out validation. Same item shape as `DistStream`.

Each item: `{type: str, params: dict, samples: np.ndarray[int64]}`. Conversion to JAX is a single `jnp.asarray(batch["samples"])` at the model boundary.

## Prompt format

```
[DIST=UNIFORM]                                     [START] s1 s2 … sN
[DIST=GAUSSIAN] [MU=22] [SIGMA=4]                  [START] s1 s2 … sN
[DIST=BIMODAL]  [MU1=20] [SIGMA1=3] [MU2=70] [SIGMA2=5] [W=0.5] [START] s1 s2 … sN
```

Separator between samples + EOS handling: TBD (see agenda).

## Cluster paths

- Generated datasets: not yet — will land under `~/tmp-data/tmp_data/llm_probs/` once `data.py` is wired up.
- Outputs / hidden reps: TBD.

## Run IDs

- *(none yet)*
