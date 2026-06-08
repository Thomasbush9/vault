# Decisions

Design choices that shouldn't get re-litigated. New entries go on top.

## 2026-05-28 — Container is mandatory: conda/module-load fallback dropped

**Decision.** The Singularity SIF becomes the sole execution contract. Every rule's `if [ -n "{params.container_cmd}" ]; then singularity exec ...; else module load + mamba activate + ...; fi` collapses to a single `singularity exec` line. `containers.gpu` (or per-stage override) becomes required config; an empty value is a hard error, not a silent fallback.

**Why.** Maintaining two execution paths means two test surfaces, two environments to keep in sync, and two failure modes per rule. The conda path was a transitional safety net while the SIF was being hardened (see [[container-audit]] H1–H5, closed 2026-05-19). The image now passes smoke, the audit's high items are done, and the diagram (Container Plan for ProtForge, 2026-05-28) commits to a single execution path.

**Implication for collaborators.** A future user on a non-Kempner cluster gets the SIF (build or pull), not a parallel conda recipe. The right portability story is "the image runs anywhere Singularity/Apptainer + `--nv` work," not "two ways to run the same pipeline."

**If we ever revisit.** Reintroducing a fallback is straightforward (the `else` branch pattern is well-understood) — but it should be motivated by a real second-path user, not pre-emptive optionality.

## 2026-05-28 — Manual `container_cmd()` over Snakemake's native `container:` directive (M7)

**Decision.** Keep the host-side helper `Snakefile:container_cmd(stage, extra_env="")` that returns a fully-rendered `singularity exec --nv --cleanenv -B ... <sif>` string, interpolated into each rule's shell block. Do **not** migrate to rule-level `container: CONTAINERS["gpu"]` + global `--singularity-args` in the profile.

**Why.**
1. **Per-rule `--env` injection.** 4 of 5 GPU rules inject stage-specific env vars before the SIF path: `esm.smk` sets `TORCH_HOME` + `HF_HOME` from `config.esm.cache_dir`; `esmfold.smk` mirrors that with `HF_HOME`; `boltz.smk:134` injects `--env TRITON_CACHE_DIR=$TRITON_CACHE_DIR` where the value is a runtime-bash export (Snakemake doesn't see it). The native `container:` directive feeds `--singularity-args` globally — per-rule env requires either Snakemake's `envvars:` (host-env forwarding, awkward for the Triton case) or inline `KEY=val python ...` patterns inside the in-container command.
2. **Snakefile already encapsulates the policy.** `container_cmd()` owns `--cleanenv`, `-B ${SLURM_TMPDIR:-/tmp}:/tmp`, bind-path parsing (`_parse_bind` validates `ro`/`rw`), runtime selection (singularity vs apptainer), and the `containers.<stage>` → `containers.gpu` fallback chain. All four are exactly the kind of project-specific logic that doesn't fit Snakemake's generic directive.
3. **What we lose is small.** `snakemake --report` no longer shows container metadata per-rule; `--containerize` (auto-generate `Dockerfile` from `conda:` directives) doesn't apply because we don't use `conda:`. Neither is in our current workflow.

**What this PR cleans up.** Each rule's body collapses from the if/else dispatch to a single `{params.container_cmd} python /opt/protforge/slurm_scripts/run_X.py ...` line. The "manual" cost is just the `container_cmd(...)` call in `params:` — same boilerplate as `resources:`.

**If we ever revisit.** Migration to native `container:` is reversible — it would mean moving `TORCH_HOME`/`HF_HOME` to `envvars:` and rewriting the Triton cache injection as an inline `TRITON_CACHE_DIR=$SCRATCH/triton python ...`. Trigger to reconsider: if Snakemake's container support gains per-rule env injection, or if we end up needing `--report` container metadata for compliance/audit.

## 2026-05-28 — ES (PDAnalysis) folds into `protforge-gpu.sif` later, without MPI

**Decision (forward commitment, not this PR).** A future change will merge PDAnalysis into `protforge-gpu.sif` and drop the separate `protforge-es.sif`. PDAnalysis will run **without MPI** — single-node parallelism (Python multiprocessing or Snakemake-level chunking), not `mpi4py` + `srun --mpi=pmix`.

**Why folding in.** The diagram (Container Plan for ProtForge) shows one container. Two SIFs adds maintenance overhead (two rebuilds, two manifests, two version pins) for a stage that's CPU-only and already small relative to the GPU image. Once MPI is out of the picture, there's nothing keeping ES separate.

**Why dropping MPI.** MPI-in-container is brittle: the in-container OpenMPI must match the host's PMIx ABI exactly, which differs per cluster and per slurm version. The original justification for MPI was multi-node parallelism over many CIF files; in practice, Snakemake's per-chunk job array already gives us that parallelism — each chunk processes its files serially (or with Python multiprocessing inside the chunk) and Snakemake schedules N chunks in parallel. The MPI layer adds a portability tax without buying parallelism we don't already have.

**Implication for the current container push.** Out of scope for PR-1 and PR-2 (container close-out). ES SIF stays as-is; `containers.es` override remains supported. The merge lands in a follow-up that touches `protforge-gpu.def`'s `%post` (add `PDAnalysis` to the pip install) and `workflow/rules/es.smk` (drop the `srun --mpi=pmix` invocation; switch to a `python -m PDAnalysis ...` wrapper).

**If we ever revisit.** Reasons to keep MPI would be: real measured speedup on a multi-node ES run that Snakemake-level parallelism can't match, OR a downstream PDAnalysis API that requires MPI primitives. Neither is true today.

## 2026-05-27 — Container architecture: orchestrator on host, compute in container

**Decision.** The ProtForge Singularity SIF (`containers/protforge-gpu.def`) is a **compute-only** artefact. Each Snakemake-submitted SLURM stage invokes `singularity exec --nv ... protforge.sif python <stage_script>` on a GPU compute node. The orchestrator — Streamlit webapp + Snakemake + slurm executor plugin — lives on the host (login node) in a Python venv created from `requirements-host.txt`.

**Rejected.** An all-in-one design where the SIF would host the webapp + Snakemake AND serve as the per-stage compute artefact. Under that design, `singularity run protforge.sif` would launch Streamlit and in-container Snakemake would call `sbatch` to submit jobs that themselves `singularity exec` the same SIF.

**Why rejected.**
1. **`sbatch` from inside a Singularity container is brittle.** Submitting jobs from in-container requires binding the host's slurm tooling: the Munge auth socket (`/var/run/munge/munge.socket.2` — permissions-sensitive), `/etc/slurm/*`, libslurm (`/usr/lib/x86_64-linux-gnu/slurm-wlm/*` or wherever the site puts it), and the binaries (`sbatch`, `squeue`, `scancel`). Paths differ per site and per slurm version; locks the SIF to one cluster's exact install.
2. **Nested `singularity exec` is not first-class.** Even if the bound `sbatch` worked, the submitted jobs would do `singularity exec` from inside a running container — Singularity supports this only patchily.
3. **The community pattern is unambiguous.** GATK, Snakemake, and Nextflow on HPC all run the orchestrator on the host and use the container only as the per-task compute artefact. See e.g. <https://learningpatterns.me/posts/2018-04-05-gatk-singularity-docker-job-array/> (Grid Engine, same principle).
4. **The "single artefact" benefit is mostly preserved anyway.** The heavy ~10 GB stack (torch+cu124, boltz, esm, transformers, mountpoints) is in the SIF. The host venv is ~50 MB of `streamlit` + `snakemake` + `pyyaml`. One-time install via `pip install -r requirements-host.txt`.

**Concrete contract.**
- **Host (login node):** `python -m venv .venv && source .venv/bin/activate && pip install -r requirements-host.txt && streamlit run webapp/app.py`. Wired by `webapp/connect.sh`.
- **Container (compute node):** `singularity exec --nv -B <DBs/cache/workdir> protforge.sif python /opt/protforge/slurm_scripts/run_X.py ...`. Emitted by `Snakefile:container_cmd()`. Stage rules in `workflow/rules/*.smk` already implement the `if [ -n "{params.container_cmd}" ]` branch — no Snakefile changes needed.
- **SIF does NOT bake** `streamlit`, `snakemake`, or `snakemake-executor-plugin-slurm`. `%runscript` is a passthrough (`exec "$@"`).

**Companion changes that landed alongside this decision (still in `protforge-gpu.def` and `scripts/download_models.py`):**
- Build manifest at `/opt/protforge/container-build-manifest.txt` (tool versions + `pip freeze`).
- `PROTFORGE_CONTAINER=1` env marker in `%environment`.
- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` defaults in `%environment`. Overridable per-invocation via `--env HF_HUB_OFFLINE=0`.
- HF revision pinning in `scripts/download_models.py` — `--esmfold-revision` / `--esmc-revision` CLI flags; default `main` resolves to the current commit SHA at download time and logs it so users can pin later.

**If we ever revisit:** the path back to all-in-one would be a thin shim that does `ssh login-node 'sbatch ...'` from inside the container (avoiding the slurm-tooling bind list) — but that adds an ssh round-trip per job and depends on key setup. Not worth it unless single-artefact distribution becomes a hard requirement we can't get any other way.

## 2026-05-04 — Calibration design choices (h100 sweep v1)

- **Option A (single DAG: MSA+Boltz+ESM+ESMFold) over Option B (two-phase MSA-then-rest)**. Simpler for one GPU type. If we sweep multiple GPU types in the future and MSA cost becomes a problem, switch to B (MSA once, then per-GPU calibrations against the same YAML tree). For now, A.

- **`max_files_per_job: 1` for calibration**. Production runs 25 seqs/chunk, but Snakemake's `benchmark:` directive aggregates per-job. We need one TSV row per length value to fit `runtime ~ L²`. Tradeoff: per-job startup overhead is no longer amortized → ESM wall times calibrate slightly high (negligible for Boltz). For sizing safety margins, that's the safe direction. See `calibration.md`.

- **Calibration-generous Boltz/ESMFold resource ceilings (80 GB, 4 / 2 hours)** vs rule defaults (16 GB / 60 min). SLURM allocates what the job uses, not what we ask, so over-asking is free; under-asking kills long-tail data. Calibration is for measuring, not for production sizing.

- **`SLURM_ACCOUNT` defaulted to `kempner_bsabatini_lab` in `calibrate.sh`** (not probed via `sacctmgr`). Kempner partitions need the `kempner_*_lab` account; `sacctmgr` returns the lab's plain `bsabatini_lab` which works on default partitions but rejects on `kempner_h100`. Hardcoding the right one makes calibrate.sh just work.

- **`calibrate.sh` is fully self-contained** — writes its own `config.yaml` from hardcoded shared Kempner paths (template) plus `$USER`-derived user paths. Does NOT merge from repo `config.yaml`, because the webapp owns that file and the cluster checkout may not have it.

- **ESMFold env default = shared lab env** at `/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/envs/esmfold` (matching `webapp/app.py:1383`), not the user-specific path in `config.template.yaml`. The template is misleading on this; most users don't set up their own ESMFold env. ESM keeps the user-specific default since users do typically have their own ESM env.

- **Lab notes go in vault, not repo**. The 4-stage pipeline calibration is a recurring lab activity with experimental logs accumulating over time — those belong in `Lab/protforge/log/`. Durable reference (workflow, troubleshooting) goes in `Lab/protforge/calibration.md`. The repo keeps only the code and a one-line CLAUDE.md pointer.
