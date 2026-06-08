# mosaic — repo guide

Orientation for working in `~/Documents/ML/mosaic`. For terse, code-level rules see the repo's own `CLAUDE.md`; this file is the longer-form narrative that helps me (and an agent helper) reason about the codebase before touching it.

## What it does, conceptually

`mosaic` solves the hallucination-style protein design problem

$$\min_{s \in A^n} \ell(s)$$

by relaxing $s$ from a discrete sequence (one of 20 amino acids at each of $n$ positions) to a **soft sequence** $P \in \Delta_{20}^n$ — a stack of $n$ probability distributions over the 20 amino acids. Because every modern protein neural net first one-hot-encodes its input, it's trivially extended to a differentiable function on $\mathbb{R}^{n \times 20}$, and we can do continuous optimization over $P$.

The point of the framework is composability of the loss $\ell$. Several different predictors (structure, sequence likelihood, stability) get composed into one $\ell$ using plain arithmetic and the whole thing is JIT'd by JAX into a single forward+backward pass.

## The two load-bearing abstractions

In `src/mosaic/common.py`:

- **`LossTerm`** — an Equinox module with signature `__call__(soft_sequence, *, key) -> (scalar, aux_dict)`. The aux dict is for logging.
- **`LinearCombination`** — a weighted sum of `LossTerm`s. `LossTerm` overloads `+`, `*`, `-`, `__neg__` so you build losses with arithmetic:

```python
loss = 4 * BinderTargetContact() + 0.3 * HelixLoss() + InverseFoldingSequenceRecovery(mpnn, ...)
```

Both are pytrees, so JAX can JIT and differentiate the whole composition. **Everything else in the codebase is either a `LossTerm` subclass or a model wrapper that produces `LossTerm`s.** If you keep these two abstractions in mind, the rest of the layout falls into place.

## Structure prediction: the uniform interface

`src/mosaic/structure_prediction.py` defines `StructurePredictionModel` (abstract) with four methods every backend implements:

| method | purpose |
| :--- | :--- |
| `target_only_features(chains)` | features for predicting an *existing* complex (real sidechain reference atoms available) |
| `binder_features(binder_length, chains)` | features for *design* (sidechain reference atoms stubbed because they're not differentiable for a soft sequence) |
| `predict(features=..., writer=..., key=..., recycling_steps=...)` | run forward, return `StructurePrediction` (`st`, `plddt`, `pae`, `iptm`, `model_output`) |
| `build_loss(loss=..., features=..., recycling_steps=...)` | wrap an inner loss into one that runs the structure model first, then evaluates the loss against `StructureModelOutput` |

**Why two feature paths exist:** AF3-style models (everything except AF2) take a reference-atomic-positions channel for the binder. Each AA has a different atom count, which breaks differentiability and JIT. `binder_features` stubs the reference channel with UNK/G, so predictions made during *design* don't carry sidechains. After design, you can repredict with `target_only_features` to get sidechains back. Don't try to merge these paths.

`StructureModelOutput` (`src/mosaic/losses/structure_prediction.py`) is the **model-agnostic** view: distogram logits+bins, plddt, pae, atom37 coords+mask, backbone coords, full_sequence, asym_id, residue_idx. A new loss term written against `StructureModelOutput` works across **all** structure backends — that's the payoff of the abstraction.

### JIT pruning is load-bearing

Under JIT, JAX strips computation whose output is unused. If your inner loss only reads `distogram_logits`, the structure module and confidence head get pruned. **Trunk-only losses are much faster than confidence losses.** When choosing or designing a loss for an expensive iteration loop, prefer trunk-only.

## File map

```
src/mosaic/
├── common.py                 # LossTerm, LinearCombination, TOKENS
├── structure_prediction.py   # uniform StructurePredictionModel interface
├── optimizers.py             # simplex_APGM, batched_simplex_APGM, gradient_MCMC
├── models/                   # one wrapper per structure backend
│   ├── af2.py
│   ├── boltz1.py
│   ├── boltz2.py
│   ├── of3.py
│   ├── protenix.py
│   ├── boltzgen.py
│   └── proteina.py
├── losses/                   # all LossTerm implementations
│   ├── structure_prediction.py  # model-agnostic: BinderTargetContact, WithinBinderContact,
│   │                             # DistogramRadiusOfGyration, HelixLoss, PLDDTLoss, ...
│   ├── transformations.py       # ClippedLoss, SoftClip, NoCys, SetPositions,
│   │                             # FixedPositionsPenalty, ClippedGradient, NormedGradient
│   ├── protein_mpnn.py          # InverseFoldingSequenceRecovery, FixedStructureInverseFoldingLL,
│   │                             # ProteinMPNNLoss
│   ├── esm.py, esmc.py          # pseudo-likelihoods
│   ├── ablang.py, ablang2.py
│   ├── stability.py             # delta-G regressor on megascale
│   ├── trigram.py
│   └── boltz.py, boltz2.py, of3.py, protenix.py, proteina.py  # backend-specific losses
├── proteinmpnn/              # JAX port of ProteinMPNN + bundled weights (vanilla, soluble, AbMPNN)
├── alphafold/                # vendored DeepMind AF2 (Apache-licensed, see src/mosaic/alphafold/LICENSE)
├── stability_model/train.py  # tiny regression head on ESM embeddings, megascale ΔG dataset
└── data/trigram_seg.pkl      # n-gram weights

examples/                     # marimo notebooks (not plain scripts), one per workflow
tests/                        # pytest; mark expensive tests @pytest.mark.slow (excluded by default)
```

## Conventions that bite if you ignore them

- **`TOKENS = "ARNDCQEGHILKMFPSTWYV"`** is the canonical 20-AA column order. Every `[N, 20]` soft-sequence array assumes this. Model wrappers translate to/from their native alphabets internally — when adding a new model, audit conversions carefully.
- **No torch in the hot path.** Pretrained weights are loaded via torch (`esm.pretrained.esm2_t33_650M_UR50D()`, `ablang.pretrained("heavy")`, etc.) and **immediately** converted to Equinox using `esm2quinox.from_torch`, `esmj.from_torch`, `jablang.from_torch`. The JAX module is what gets JIT'd. Don't call into torch inside a `LossTerm.__call__`.
- **First JIT call is slow** (sometimes minutes for big structure models). This is why every example is a marimo notebook — they hold state. Don't write CLI scripts that re-JIT each run; persist a kernel/process.
- **Wrap aggressive losses in `ClippedLoss(..., 2, 100)` or `SoftClip`.** Raw PLLs and inverse-folding likelihoods optimize to homopolymers. This is the most common foot-gun, called out throughout the README.
- **Loss terms are pytrees.** Hyperparameters that need to be JAX-traced should be `eqx.field(converter=jnp.array)` (see `SoftClip`). Plain Python floats become static under `eqx.filter_jit` and changing them recompiles.
- **`_eval_loss_and_grad` projects gradients to the simplex tangent space** (`g - g.mean(-1, keepdims=True)`). New optimizers operating on soft sequences should preserve this or document why not.
- **`simplex_APGM` step-size heuristic:** start at `0.1 * sqrt(binder_length)`, `scale=1.0` for an initial soft solution, ramp `scale` up to drive sparsity / one-hot. `logspace=True` switches to entropic mirror descent.

## HPC-relevance hot spots

Things to look at first when adapting for HPC + multi-user use:

- `pyproject.toml` `required-environments` is already pinned to Linux x86_64 — good. But many deps are git-sourced, so the install must reach GitHub from build nodes (or pre-mirror them).
- GPU JAX install: `uv sync --group jax-cuda` is the canonical path. CUDA driver/runtime versions need to match `nvidia-cublas-cu12 >= 12.9.1.4` and `nvidia-cusolver-cu12 == 11.7.3.90` (pinned in `pyproject.toml`).
- First-JIT cost is a non-trivial HPC issue: each user re-paying minutes per run is bad. Consider a persistent worker model (Slurm + long-running session) or AOT compilation + cache sharing (`JAX_COMPILATION_CACHE_DIR`).
- Marimo-as-entry-point doesn't map cleanly to batch HPC. Either (a) wrap the design loops into idempotent CLI/Python entry points, or (b) run marimo over a forwarded port — pick deliberately, and write that decision into `decisions.md`.
- Weights for several models (ESM2-650M, ESMC-300M, ProteinMPNN variants, Protenix family, Boltz, OpenFold3, etc.) need to live somewhere accessible to compute nodes. Cache strategy + paths belong in `data.md` once decided.
- Multi-user GPU sharing: `simplex_APGM` is single-GPU; `batched_simplex_APGM` vmaps over batch — useful for small targets on large GPUs, but doesn't shard across devices. Multi-device design is a green field.
