# 2026-05-14 — Container migration: design, branch, scaffolding

Today's session was design + scaffolding for moving ProtForge off shared-conda-envs and onto a single Singularity image. Branch `containers` created from `webapp`; `webapp` keeps working untouched on the cluster.

## 1. The design call

Several converging discussions today:

### 1a. Why containerize at all?

Current Kempner setup couples *tools* and *data* in the same shared path
(`/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/boltz/`). Every time someone bumps the shared env, calibrations drift, and yesterday's `colabfold_search: command not found` regression (see 2026-05-13 in-progress) showed the lack of an authoritative install. The container freezes the tool stack — same versions for everyone, calibrations stay valid.

### 1b. Architectural choices (all settled today)

- **One fat image**, not per-stage. Doc spec (`docs/CONTAINERS.md`) proposed 4–5 separate `.sif` files. For a single-lab project this is over-engineered. One `protforge-gpu.sif` (~15 GB) holds boltz, colabfold_search, mmseqs2, esm SDK, transformers. PDAnalysis (ES, MPI) deferred to a separate image.
- **Tools baked, data bind-mounted.** The image installs *tools*; data (MSA DBs ~700 GB, Boltz checkpoint ~5 GB) is bind-mounted from `/n/holylfs06`. This is the AlphaFold/ColabFold/Nextflow convention.
- **Default model weights baked**, big ones bind-mountable. ESM-C 600M and ESMFold (`facebook/esmfold_v1`) ship inside the image (~7 GB total). Larger ESM-2 variants (3B, 15B) deferred — can be bind-mounted via `TORCH_HOME` / `HF_HOME` overrides.
- **SLURM and non-SLURM use the same image.** Difference is purely the Snakemake profile (`profiles/slurm/` vs a future `profiles/local/`). The container itself doesn't know or care about SLURM.
- **Single `PROTFORGE_HOME` env var** for cache root, with per-tool overrides (`HF_HOME`, `TORCH_HOME`). Mounted at `/data/protforge` by convention. Per-tool overrides remain for users with split filesystems.
- **Soft migration.** Dual-mode `if [ -n "$container_cmd" ]; then …; else module load …; fi` already exists in every rule. Container path becomes default once validated; legacy path stays as off-ramp.

### 1c. The rebuild-redownload problem

User worry: rebuilding the image shouldn't redownload ESMFold every time. The fix is Docker-style layer ordering inside `%post`, so iterating on the codebase only touches the last layer. Singularity `%post` is monolithic (no automatic layer cache), so for fast iteration we use a writable sandbox:

```bash
# One-time:
singularity build --sandbox /tmp/pfsandbox containers/protforge-gpu.def
# Iterate on code, then convert when ready:
singularity build protforge-gpu.sif /tmp/pfsandbox
```

For now this is fine. If iteration becomes painful, move to a GHCR-published image with Docker BuildKit (real layer caching). Deferred.

### 1d. What's deferred (explicit non-goals for this branch)

- ES/PDAnalysis MPI image.
- `profiles/local/` (non-SLURM execution). Rules already mostly work locally; the local profile is a thin add later.
- GHCR-published image + GitHub Actions CI build. Will happen once the def file is validated on the cluster.
- Mac dev builds. Skipped entirely — we don't need cross-arch builds.

## 2. What landed in the repo (branch `containers`, uncommitted)

```
containers/
├── protforge-gpu.def     # Singularity def, CUDA 12.4 runtime base
├── build.sh              # build / pull wrapper
├── README.md             # short usage doc
└── test/
    ├── smoke.fasta       # 2 short sequences (~50aa)
    └── smoke.sh          # 5-step smoke (GPU → torch → tools → weights → ESMFold)
```

### Layer order in `protforge-gpu.def` `%post`

1. Apt system deps (changes ~never).
2. mmseqs2 binary (changes when bumping).
3. Python tools: pytorch (cu124) → boltz → esm SDK → transformers → colabfold (git install) → runtime deps.
4. **Model weights** — `esmc_600m` + `facebook/esmfold_v1` cached under `/opt/weights/hf`. This is the layer you don't want re-running on code edits; sandbox workflow above sidesteps that.
5. `%files` copies the codebase to `/opt/protforge` last.

### `%environment` defaults

- `PROTFORGE_HOME=/data/protforge` (host bind-mount convention).
- `HF_HOME=/opt/weights/hf` (overridable to point at host cache).
- `TORCH_HOME=/opt/weights/torch`.
- `PYTHONPATH=/opt/protforge:…`.

### Bind-mount convention (host → container)

| Host path | Container path |
|---|---|
| `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/colabfold/databases` | `/data/colabfold_db` |
| `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db` | `/data/boltz_db` |
| `$PWD` (working dir) | `$PWD` |

Decoupling host from container paths means moving the DBs on the host won't break the rules.

## 3. Cluster instructions (for tomorrow)

### Per Kempner handbook ([link](https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/development_and_runtime_envs/containerization.html))

The handbook documents `singularity pull docker://...` from an interactive compute node as the canonical path. It does NOT document local builds from def files. **Always run builds/pulls from an interactive allocation, not a login node.**

### Step-by-step

1. **On the cluster, `git pull` ProtForge on `containers` branch:**
   ```bash
   cd ~/code/ProtForge   # or wherever
   git fetch
   git checkout containers
   git pull
   ```

2. **Get an interactive allocation** (handbook example):
   ```bash
   salloc --partition=test --account=<your_account> \
          --nodes=1 --ntasks-per-node=4 --mem-per-cpu=3200M --time=4:00:00
   ```

3. **Try the local build first** (`--fakeroot`):
   ```bash
   bash containers/build.sh
   # → writes ~/sifs/protforge-gpu.sif (allow ~20–30 min for first build)
   ```

   If this fails with a "fakeroot not allowed" error, fall back to publishing the image to GHCR via GitHub Actions and pulling:
   ```bash
   bash containers/build.sh --from-docker docker://ghcr.io/<owner>/protforge-gpu:latest
   ```

4. **Smoke-test on a GPU node:**
   ```bash
   salloc -p kempner_h100 --gres=gpu:1 -t 30 --mem=32G
   bash containers/test/smoke.sh
   ```
   This validates: GPU visible, PyTorch+CUDA, tools importable, baked weights load, ESMFold folds a 49aa peptide end-to-end. Does NOT exercise MSA/Boltz (those need DB bind-mounts).

5. **First real run (Boltz only, container mode):**
   Point a test session config at the SIF + bind paths:
   ```yaml
   containers:
     boltz: ~/sifs/protforge-gpu.sif
     bind_paths: "/n/holylfs06,/n/home06"
   ```
   Then `snakemake --profile profiles/slurm/ -n` to inspect the DAG, followed by a real run on a small dataset. Existing dual-mode `if/else` in `boltz.smk:154` dispatches automatically.

### Sharp edges to watch for during the first build

1. **`pip install "colabfold @ git+..."`** may pull JAX/AlphaFold deps even without explicit extras. If it does, image grows and there are install errors. Fallback: install only what's needed for `colabfold_search` (= mmseqs2 wrapper).
2. **`pip install esm`** must resolve to Evolutionary Scale's new SDK (provides `esm.models.esmc.ESMC`), not the legacy `fair-esm` package. Verify with `python -c "from esm.models.esmc import ESMC"` inside the sandbox before final build.
3. **`--fakeroot` may not be permitted.** If so, see step 3 fallback (GHCR pull). Confirm with: `singularity --version` then `singularity build --fakeroot --help`.

## 4. Open questions for next session

- **Does fakeroot work on Kempner?** Only way to know is to try. If not, jump to the GHCR + GitHub Actions path (deferred but ready when needed).
- **Where do the bind paths live in config schema?** Current `containers.bind_paths: "comma,separated"` works for now. May want to refactor to a list of `host:container` pairs to support the `/data/*` indirection cleanly (currently the host path *is* the in-container path, which is brittle when paths move).
- **Will the existing rules need any tweaks to actually use `/data/colabfold_db` and `/data/boltz_db`?** They currently take the host path from config (`msa.mmseq2_db`, `boltz.cache_dir`). Two options:
  - (a) Set those config values to `/data/colabfold_db` / `/data/boltz_db` when running container mode (cleanest, no rule changes).
  - (b) Add a "container path remap" layer (more flexible, more code).
  Recommend (a) for now.

## 5. Pending commits

Nothing committed yet on `containers` branch. Files staged but not added:

- `containers/protforge-gpu.def`
- `containers/build.sh`
- `containers/README.md`
- `containers/test/smoke.fasta`
- `containers/test/smoke.sh`

Waiting until first build succeeds on cluster before committing. If you want to commit the scaffolding now so the cluster has something to `git pull`, that's also fine — the dual-mode wiring on this branch is unchanged from `webapp`, so nothing in the actual pipeline behavior shifts.

## 6. Provenance

- Branch: `containers` (off `webapp`, off `main`).
- Kempner containerization handbook: `handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/development_and_runtime_envs/containerization.html`.
- Earlier dual-mode plumbing already in repo: `Snakefile:127-136`, `workflow/rules/{msa,boltz,esm,esmfold}.smk` (each has the `if [ -n "$container_cmd" ]` branch).
- Prior spec for what containers should look like: `docs/CONTAINERS.md` (predates this redesign — single-image plan supersedes its per-stage proposal).
