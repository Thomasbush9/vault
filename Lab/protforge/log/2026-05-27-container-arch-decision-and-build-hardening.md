# 2026-05-27 — Container architecture decision + build hardening

## TL;DR

- Locked the container architecture: **compute-only SIF**; orchestrator (webapp + Snakemake) on host. Rejected the all-in-one design where the SIF would host the webapp + submit `sbatch` jobs from in-container. See `decisions.md` (2026-05-27 entry) for the full write-up.
- Companion hardening landed in the same session: build manifest, offline-by-default HF flags, `PROTFORGE_CONTAINER` marker, HF revision pinning in `scripts/download_models.py`.
- Reverted the webapp-role additions to the def file (`%post` streamlit/snakemake pip line + Streamlit `%runscript`) since they only make sense under the rejected architecture.

## Context

Going into the session, the user wanted to extend the model-mount work (already landed: ESM-C + ESMFold weights moved out of `%post` into a host-mounted HF cache, no more HF token in builds). Open question was "what else to harden, and what does the webapp side look like in container mode."

Initial framing drifted toward an all-in-one container: webapp inside, Snakemake inside, single artefact. User pushed back twice to clarify that the webapp UI should stay invariant and that the container's purpose is to "crystallize the backend" so users don't manage envs by hand.

The breaking insight came when laying out compute-side details: Snakemake's slurm executor calls `sbatch` directly, which inside a Singularity container requires binding the host's slurm tooling (sbatch binary, libslurm, /etc/slurm, Munge socket). User raised the practical concern, pointed at <https://learningpatterns.me/posts/2018-04-05-gatk-singularity-docker-job-array/>. The article confirms the unambiguous community pattern: orchestrator on host, container per-task.

## Architecture comparison (for future re-litigation)

| | A: All-in-one | B (chosen): Host + container compute | C: Hybrid (container webapp, host snakemake via ssh) |
|---|---|---|---|
| `sbatch` invoker | In-container, needs bound slurm tooling | Host (native) | Host (via ssh from container) |
| Single artefact | Yes | Mostly — heavy stack still in SIF, ~50 MB host venv | No |
| HPC community pattern? | No | Yes (GATK / Snakemake / Nextflow all do this) | No |
| Brittle to slurm site config | Yes | No | Partially |
| Webapp UI changes | No | No | No |

Decision: B. The cost (one-time host venv install, ~50 MB) is trivial; the benefit (no slurm-tooling bind list, no nested `singularity exec`, robust to Kempner slurm upgrades) is large.

## Concrete changes

### `containers/protforge-gpu.def`

- **Top-of-file comment.** Rewrote to "compute-only artefact"; explains orchestrator-on-host and links to the decision note.
- **`%post`.** Did NOT add streamlit/snakemake/huggingface_hub — left the existing science stack (torch+cu124, boltz, esm SDK, transformers, colabfold, mmseqs) intact. Added a comment block explaining why orchestration packages are on the host.
- **`%environment`.** Added `PROTFORGE_CONTAINER=1` marker, `HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}`, `TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}`. `${VAR:-1}` form lets `download_models.py` override.
- **`%runscript`.** Reverted to passthrough (`exec "$@"`). Compute path uses `singularity exec`, which bypasses `%runscript` anyway.
- **Build manifest.** Writes `/opt/protforge/container-build-manifest.txt` at end of `%post` with timestamp, mmseqs/boltz/python versions, full `pip freeze`. `|| echo unknown` on each capture so a single missing tool doesn't fail the build under `set -e`.
- **`%labels` + `%help`.** Updated to describe the compute-only role. Version bumped to 0.2.0 (manifest + offline defaults are meaningful changes vs 0.1.0).

### `scripts/download_models.py`

Pinning + provenance:

- New `--esmfold-revision` and `--esmc-revision` CLI flags. Default `main`.
- `_resolve_revision()` resolves `main` (or any branch name) to the current commit SHA via `HfApi.repo_info()` *before* `snapshot_download`. This:
  - Locks the snapshot in time even if upstream pushes mid-download.
  - Logs the SHA so the user can paste it into a pinned config later.
- Defensively pops `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` in case the script is invoked from inside the SIF (where they default to 1, which would fail).
- End-of-run prints "Resolved revisions (pin these to lock the cache contents)" so the SHAs are easy to copy.

## What's still on the list

In priority order:

1. **`config.template.yaml` defaults aligned with in-container fixed paths.** So `cache_dir: /models/hf`, `mmseq2_db: /data/colabfold_db`, etc. are the defaults — the webapp shows them pre-filled per [[2026-05-27 decision]] but no UI change needed.
2. **`webapp/connect.sh` doesn't need container-mode changes** under architecture B. It continues to ssh into a login node, port-forward, and run host `streamlit run webapp/app.py`. Maybe document the one-time `pip install -r requirements-host.txt` setup more visibly.
3. **Lean vs fat def split.** Lean (current) for clusters with shared colabfold-conda + DBs. Fat keeps the `wget mmseqs` + `pip install colabfold@git` for portable single-shot deployments. Deferred until the lean SIF is rebuilt and smoke-tested with today's changes.
4. **Strip build-only apt packages** (`build-essential`, `cmake`) from `%post` final layer. Modest size win, modest attack-surface win.
5. **Pin base image by digest** (`ubuntu:22.04@sha256:...`). Last, once the def is stable.

## Why architecture B works for the user's stated goal

User's framing: "we are only using containers to crystallize the current version, so that it will be easier for users to launch jobs etc." Under B:

- The crystallized backend (envs + science stack) IS the SIF. That's the durable, hard-to-set-up part.
- The orchestration layer (snakemake + streamlit) is the lightweight, easy-to-install part. `pip install -r requirements-host.txt` once and forget.
- Users still get a single command to launch (`bash webapp/connect.sh`), still get the webapp UI unchanged, still get reliable per-stage execution via the SIF.

The bit that's NOT crystallized — the host venv — is also the bit that benefits most from being native (sbatch / munge / login node binaries). Net win.
