# 2026-05-15 — `containers/build.sh` bug fixes

Quick patch session in response to file-creation errors reported during the
first cluster build attempt (branch `containers`). I'm mobile / off-cluster, so
this is a code review against the latest commit on `containers` (`e9336f4`,
"updated container for home directory"), with two fixes applied.

## What was wrong

### 1. `singularity build` without `--force`

`containers/build.sh:107` (pre-patch) invoked:

```
singularity build --fakeroot "$OUT" "$DEF_FILE"
```

If `$OUT` already exists — which it always does on the second attempt after a
failed first build, or when iterating on the def file — singularity aborts with
`FATAL: While performing build: image file already exists`. This matches the
"errors related to the creation of the files" symptom and is the most likely
root cause.

The pull branch (`--from-docker`) already passed `--force`. Asymmetry.

**Fix.** Add `--force` to the build invocation. Same semantics as the pull
branch: overwrite a pre-existing SIF at `$OUT` rather than erroring out.

### 2. `SINGULARITY_TMPDIR` / `CACHEDIR` only auto-set when `PROTFORGE_ROOT` is set

Pre-patch logic at `build.sh:82`:

```bash
if [[ -n "${PROTFORGE_ROOT:-}" ]]; then
    # default SINGULARITY_CACHEDIR / TMPDIR under $PROTFORGE_ROOT
fi
```

But `containers/README.md` and the script's own help text both document
`PROTFORGE_SIF_DIR` as a valid standalone export — so users following the
README who set only `PROTFORGE_SIF_DIR` get TMPDIR = `/tmp` on the compute
node. On Kempner compute nodes `/tmp` is small (often <10 GB) and sometimes
mounted `noexec`, so a 15 GB fakeroot build either fails partway with `no
space left on device` or chokes earlier with overlay-creation errors. Same
class of "can't create files" symptom.

**Fix.** Derive a base for the cache/tmp dirs from whichever of
`PROTFORGE_ROOT` / `PROTFORGE_SIF_DIR` is set. If only `SIF_DIR` is given, use
its parent (i.e. the equivalent of `PROTFORGE_ROOT` one level up).

```bash
SING_BASE=""
if [[ -n "${PROTFORGE_ROOT:-}" ]]; then
    SING_BASE="${PROTFORGE_ROOT%/}"
elif [[ -n "${PROTFORGE_SIF_DIR:-}" ]]; then
    SING_BASE="$(dirname "${PROTFORGE_SIF_DIR%/}")"
fi
```

This is conservative: if the user explicitly exports `SINGULARITY_CACHEDIR`
or `SINGULARITY_TMPDIR`, those win. If neither base var is set, no defaults
are applied (system behavior unchanged).

## What I did NOT change

- `%files . /opt/protforge` still copies the whole repo (including `.git/`).
  Inelegant but not an error. Leave for a separate clean-up pass; a
  `.singularityignore` or explicit subdir copies would be the fix.
- Smoke test's redundant bind-mount of `WORK` (subdir of `SCRIPT_DIR` which
  is also bound). Harmless, deferred.
- Writability validation for `PROTFORGE_ROOT`. The `mkdir -p` error is
  already reasonably clear when permissions are wrong.

## Files touched

- `containers/build.sh` (2 hunks, both small):
  - `--force` on the build command (~line 117).
  - `SING_BASE` derivation for cache/tmp defaults (~lines 82–101).

Not yet committed — leaving it on the working tree so the user on the cluster
can review the diff before merging, but the changes are minimal and safe.

## Provenance

- Branch: `containers`.
- Previous build session: `2026-05-14-containerization-design-and-scaffold.md`.
- Trigger: user reported file-creation errors mid-build, can't access cluster
  right now to share the exact error.
- Code review only — no cluster execution this session.

## Update — actual error from cluster + 3rd fix

User pasted the real error later in the same session:

```
/usr/bin/cp: cannot copy a directory, '.', into itself,
 '/n/holylfs06/.../tbush/ProtForge/sing_tmp/build-temp-.../rootfs/opt/protforge'
salloc: Job 12975775 has exceeded its time limit and its allocation has been revoked.
```

### Diagnosis

`%files . /opt/protforge` in the def file does `cp -r <build context> rootfs/opt/protforge`.
The build context is `REPO_ROOT` (where `singularity build` runs, set by `build.sh:135` —
`cd "$REPO_ROOT"` before invoke). The build's `rootfs/` lives under `SINGULARITY_TMPDIR`.

The user had set `PROTFORGE_ROOT=/n/holylfs06/.../tbush/ProtForge` — i.e. the repo path
itself. `build.sh` then derived `SINGULARITY_TMPDIR=$PROTFORGE_ROOT/sing_tmp`, which is
**inside** REPO_ROOT. So when `%files .` ran, `cp` walked the source tree, reached its own
destination directory under `sing_tmp/build-temp.../rootfs/opt/protforge`, and bailed.
The `salloc` timeout is incidental — the 4 h allocation expired before the user could
recover (and given how cp -r recursion is supposed to be detected up-front, the build
probably died fast and the rest of the time was lost to debugging interactively).

The fix is conceptual: `PROTFORGE_ROOT` is meant to be the *workspace parent*, with the
repo as a sibling of `sifs/`, `sing_cache/`, `sing_tmp/`. The README led the user the
wrong way.

```
$PROTFORGE_ROOT/                  <- workspace parent (e.g. .../tbush/)
├── ProtForge/                    <- the repo (REPO_ROOT)
├── sifs/
├── sing_cache/
└── sing_tmp/
```

User confirmed they've already restructured to this layout on disk — `ls .../tbush/`
shows `ProtForge sifs sing_cache` as siblings, which is correct.

### Three patches in this session

1. **`containers/build.sh`** — added `--force` to `singularity build` (avoid "image
   already exists" on rebuilds; matches the pull branch).
2. **`containers/build.sh`** — `SING_BASE` derivation: cache/tmp defaults now also
   fire when only `PROTFORGE_SIF_DIR` is set (uses its parent).
3. **`containers/build.sh`** — guard at line ~114 that aborts the build if
   `SINGULARITY_TMPDIR` or `dirname $OUT` lands inside `REPO_ROOT`, with a clear
   error message pointing at the README. Catches the misconfiguration up-front
   instead of letting `cp` discover it 10 minutes into `%post`.
4. **`containers/README.md`** — example block now sets `PROTFORGE_ROOT` to the
   parent dir (e.g. `.../tbush`), not the repo path. Added a prominent "IMPORTANT"
   callout above the snippet explaining the layout constraint.

All four hunks live on the working tree (`containers` branch). Not committed yet —
user is mobile, so a single commit message after they sight-check on the cluster.

### Still deferred

- Slim `%files . /opt/protforge` to explicit subdirs (drops `.git/` and any local
  outputs from inside the image). Bigger change, needs to enumerate which dirs the
  in-container scripts import from; doing it now would block the user on a follow-up
  question. Guard above prevents the most likely failure mode in the meantime.

## Update — ESMFold double-download fix

User asked to slim the ESMFold prefetch. Patched `containers/protforge-gpu.def`
to pass `use_safetensors=True` to `EsmForProteinFolding.from_pretrained`:

```python
EsmForProteinFolding.from_pretrained(
    'facebook/esmfold_v1',
    low_cpu_mem_usage=True,
    use_safetensors=True,   # <- new
)
```

`facebook/esmfold_v1` ships both `pytorch_model.bin` (8.44 GB) and
`model.safetensors` (8.44 GB). Without the flag, the transformers ≥4.40 HF-hub
resolver pulls both. `use_safetensors=True` pins it to safetensors only —
expected savings ~8.4 GB of download + image size, ~60–80 s wall time on the
build at 117 MB/s.

If the HF revision ever drops safetensors (unlikely), this call raises
`OSError: facebook/esmfold_v1 does not appear to have a file named model.safetensors`,
which is loud enough to debug. No fallback added.

## Update — `PROTFORGE_ROOT` layout, container/ consolidation

User wants to consolidate everything under a single `container/` workspace
dir under `tbush/` (i.e. `tbush/container/{ProtForge,sifs,sing_cache,sing_tmp}`)
rather than the loose layout at `tbush/` level. Recommended `mv` over re-clone
(faster, preserves local config + any uncommitted state). New `PROTFORGE_ROOT`
becomes `.../tbush/container`. The repo guard added earlier in this log will
reject the build if `PROTFORGE_ROOT` is ever set to the repo path again
(would be `.../tbush/container/ProtForge`), so the layout is now self-policing.
