# 2026-05-26 — Drop CUDA base image, fix ESMFold prefetch

Two changes to `containers/protforge-gpu.def` ahead of the next rebuild on
Kempner. Both are uncommitted in the working tree.

## ESMFold prefetch — `from_pretrained` → `snapshot_download`

Smoke `[6/7]` on the May-16 SIF failed with `OSError: no pytorch_model.bin
... in models--facebook--esmfold_v1/snapshots/<sha>/`. The snapshot dir held
only tokenizer + config files. Root cause: the old `%post` used
`EsmForProteinFolding.from_pretrained(..., use_safetensors=True)` at build
time. `facebook/esmfold_v1` doesn't ship a `model.safetensors`, so the
safetensors branch never materialised the `.bin` into the snapshot symlink
farm; at runtime under `HF_HUB_OFFLINE=1` that triggers transformers'
auto-conversion path which dies without network access.

Replaced with `huggingface_hub.snapshot_download(repo_id='facebook/esmfold_v1',
allow_patterns=['pytorch_model.bin','*.json','*.txt','*.model'])`. This
writes the `.bin` + configs as real symlinks under
`hub/models--facebook--esmfold_v1/snapshots/<sha>/`, with no Python-object
instantiation. `run_esmfold.py` then loads via the repo id with `HF_HOME`
pointed at `/opt/weights/hf` and `HF_HUB_OFFLINE=1` — cache hit, no network.

`allow_patterns` deliberately excludes `*.safetensors` so the image stays
single-weight-file even if upstream adds a safetensors mirror later;
`EsmForProteinFolding` loads `.bin` fine.

## Base image — `nvidia/cuda:12.4.1-runtime-ubuntu22.04` → `ubuntu:22.04`

Asked: do we still need the CUDA base if Kempner has `cuda`/`cudnn` modules?

Answer: we never used those modules in container mode anyway. Host modules
are only loaded in the bash/conda fallback path (`workflow/rules/*.smk` else
branches). In container mode they're inaccessible by default and not bound.

What *was* doing the work in the CUDA base layer:

- `libcudart`, `libcublas`, `libcurand`, ... — duplicated, PyTorch's `cu124`
  wheel bundles its own copies under `torch/lib/`.
- `libcudnn` — same: torch wheel ships it.
- Driver libs (`libnvidia-*.so`, `libcuda.so.1`) — never in the base image
  anyway; `singularity exec --nv` injects them from the host driver.
- nvcc / ptxas for JIT compilation — Triton (used by Boltz) bundles its own
  ptxas under `triton/third_party/cuda/bin/`. We never call nvcc at runtime.

So the CUDA base layer was ~2 GB of dead weight. Dropped it. The image now
goes from `ubuntu:22.04` straight to apt-installs and pip wheels. PyTorch's
`cu124` wheels still pin the CUDA version implicitly via what they bundle.

Verified all four rules (`msa`, `boltz`, `esm`, `esmfold`) gate their
`module load cuda/cudnn ...` calls behind the `else` branch, so the change
has no effect on container-mode execution paths.

## What to expect on rebuild

- SIF size drops from ~15 GB → ~12 GB.
- Smoke step 2 (`torch=2.6.0+cu124, cuda=True, device=...H100`) still passes;
  the CUDA libs come from the torch wheel, the device check goes through the
  injected driver. If it doesn't, the diagnosis is `--nv` not being passed
  by the smoke wrapper (unlikely — `smoke.sh` uses `singularity exec --nv`).
- Smoke step 3 passes (the May-19 `.singularityignore` finally applies).
- Smoke step 6 passes (snapshot now contains `pytorch_model.bin`).

If step 2 regresses, the fallback is to add `nvidia-cuda-toolkit` from apt,
or re-pin to a slim CUDA base — but neither should be necessary based on
how PyTorch wheels are packaged since 2.0.

## Items deferred at user request

- Supply-chain pins (CUDA digest, mmseqs versioned + sha256, pip pins, HF
  revision). Brief #2 — re-open later.
- Stage-1 E2E test scripting (input → MSA → Boltz → ESMFold → ESM). Brief
  #8 — agreed as the right target; not blocking the next rebuild.

## Next user action

Rebuild on the Kempner H100 node (`PROTFORGE_ROOT` set to the parent of the
repo), then re-run `containers/test/smoke.sh`. If all 7 steps green, the
container is unblocked for the streamlit-launches-snakemake-launches-SIF
workflow.

## Update: HF rate limit on the first rebuild

First rebuild attempt failed at the ESM-C prefetch with
`huggingface_hub.errors.LocalEntryNotFoundError` and a "rate limit your IP
(140.247.111.5)" message. ESMFold (`facebook/esmfold_v1`, one ~8 GB .bin)
came down fine; ESM-C (`EvolutionaryScale/esmc-600m-2024-12`) gets many
small files via the esm SDK, which trips HF's per-IP anonymous limit much
faster on Kempner's shared egress IPs.

Fix: bind a `--hf-token` (or `$HF_TOKEN`) into the build at
`/run/secrets/hf_token:ro` via `containers/build.sh`. `%post` reads it with
shell-trace suppressed and `unset`s it before the build context is staged,
so the token never lands in either the build log or the image. Runtime
`%environment` never sets HF_TOKEN.

User needs a read-only token from
<https://huggingface.co/settings/tokens> before retrying:

```bash
export HF_TOKEN=hf_...
bash containers/build.sh    # picks up env automatically
# or: bash containers/build.sh --hf-token hf_...
```
