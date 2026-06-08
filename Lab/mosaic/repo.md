# Repo

GitHub repo + setup notes for `mosaic`.

- **URL:** `https://github.com/escalante-bio/mosaic` (via SSH alias `github.com-personal`, ref `origin`)
- **Local path on Mac Mini:** `~/Documents/ML/mosaic`
- **Local path on MBP:** `~/Documents/ML/mosaic`
- **HPC path:** TBD

Upstream is Escalante-bio's; I'm working on `main` for now. License: see repo `LICENSE`.

## Setup (local)

```bash
git clone ssh://git@github.com-personal/escalante-bio/mosaic.git ~/Documents/ML/mosaic
cd ~/Documents/ML/mosaic
uv sync --group jax-cuda   # or jax-cpu / jax-tpu — mutually exclusive groups
```

Notes:
- Python pinned to `==3.12.*`.
- `pyproject.toml` declares `required-environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]`. Several deps (`protenix`, `boltz`, `boltzgen`, `jopenfold3`, `joltzgen`, `jproteina-complexa`) won't resolve on macOS — develop/run on a Linux x86_64 box. macOS is fine for reading code and editing.
- Many deps are git-sourced sibling repos under `escalante-bio/*` and `nboyd/joltz` — see `[tool.uv.sources]` in `pyproject.toml`.

## Entry points

- `examples/example_notebook.py` — main worked example; marimo notebook. Run with `uv run marimo edit examples/example_notebook.py`. The other `examples/*.py` are also marimo notebooks (Protenix, OpenFold3, BoltzGen, Proteina, etc.).
- `src/mosaic/optimizers.py` — `simplex_APGM`, `batched_simplex_APGM`, `gradient_MCMC`.
- `src/mosaic/structure_prediction.py` — uniform `StructurePredictionModel` interface (all model wrappers in `src/mosaic/models/` implement this).
- `src/mosaic/losses/` — all `LossTerm` implementations.

## Tests

```bash
uv run pytest                  # default: excludes @pytest.mark.slow
uv run pytest -m slow          # only slow tests
uv run pytest -m ""            # everything
```

## Lint / typecheck

```bash
uv run ruff check .
uv run ty check
```
