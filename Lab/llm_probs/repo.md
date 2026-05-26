# Repo

GitHub repo + setup notes for this project.

- **URL:** *(not pushed yet)*
- **Local path:** `~/Documents/ML/llm_probs` (note: not the usual `~/code/<name>` location)

## Setup

```bash
cd ~/Documents/ML/llm_probs
uv sync
```

Python 3.12, managed by `uv`. Framework split:

- **Model + training loop:** `jax`, `flax` (and later `optax`).
- **Data plumbing:** `torch` for `Dataset` / `IterableDataset` / `DataLoader`. The dataset returns numpy arrays; conversion to JAX is a single `jnp.asarray(...)` at the model boundary. See `decisions.md` (2026-05-23).
- **Other:** `numpy`, `matplotlib`, `tqdm`, `ipykernel`, `wandb`.

Full dep list is `pyproject.toml`.

## Entry points

- `utils.py` — three discrete-distribution samplers (uniform, gaussian, bimodal mixture) over an integer support.
- `data.py` — `DistPriors` (prior ranges for distribution configs), `DistStream(IterableDataset)` for on-the-fly training batches, `DistDataset(Dataset)` for fixed held-out validation sets. Each item: `{type, params, samples}`.
- `eval.py` — *(empty)* distributional + sample-independence metrics.
- `main.py` — currently a hello-world placeholder.
