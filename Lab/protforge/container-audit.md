# Container audit — open punch list

Source: subagent audit on 2026-05-16 against `containers/protforge-gpu.def`,
`containers/build.sh`, `Snakefile`, `workflow/rules/*.smk`,
`profiles/slurm/config.yaml`, `config.template.yaml`. Branch `containers`,
first successful build today. Items here are what the audit flagged that we
did **not** fix today.

Items resolved today are in `log/2026-05-15-build-sh-bugfixes.md` (build.sh
`--force` + TMPDIR derivation + REPO_ROOT guard, README layout fix, def-file
`use_safetensors=True`). Cross-check that log before re-opening.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` deferred /
won't fix (with rationale).

---

## HIGH — fix before sharing the image across users

- [x] **H1. Host env leaks into the container.**
  *Done 2026-05-16* — `container_cmd()` now emits `--cleanenv` +
  `--env TMPDIR=/tmp`; `smoke.sh` matches the production flags and adds
  step [6/6] which sets `PYTHONPATH=/host/leaky/smoke` before invoking and
  asserts it does NOT appear inside the container.
  *Watch:* rules that `export FOO=...` on the host side BEFORE the container
  call will no longer leak `FOO` into the container. Pre-existing rules
  already use the correct pattern (`{container_cmd} --env FOO=$FOO ...` —
  see `boltz.smk:156`). Re-grep `workflow/rules/*.smk` for `export ` before
  the next real run to catch any that don't.

- [ ] **H2. `%files . /opt/protforge` drags in `.git/` and any local files.**
  `containers/protforge-gpu.def:20-23`. Real reproducibility + secrets risk
  (deleted credentials in git history, local `config.yaml`).
  *Fix:* explicit allow-list (`workflow/`, `slurm_scripts/`, `utils/`,
  `scripts/`, `Snakefile`, `config.template.yaml`) or stage via
  `git archive HEAD | tar -x -C $stage` before build.

- [~] **H3. `bind_paths` default mounts whole filesystems read-write.**
  *Partial 2026-05-16* — `container_cmd()` parser now accepts
  `host:container:ro` and `host:container` entries (backward-compatible
  with bare `host`). Users can opt into `:ro` for DBs in their config.
  *Still TODO:* drop the `/n/home06` default in `Snakefile:128` (currently
  blanket-binds every user's home for compat) and update the
  `config.template.yaml:143` example to show the `:ro`-tagged DB paths.
  Defer until after first real-config run validates the parser doesn't
  break existing setups.

- [ ] **H4. Schema mismatch — `containers:` block lists 5 per-stage SIFs.**
  `config.template.yaml:137-142`. We ship ONE fat SIF. Users could set
  `containers.boltz` and silently get legacy `module load` for ESM/MSA.
  *Fix:* collapse to `containers.gpu:` (+ future `containers.cpu_mpi:` for
  the ES image). Rewrite `container_cmd(stage)` to dispatch by capability
  (`gpu`/`mpi`), not stage name. We flagged this earlier in the design log
  too — still open.

- [x] **H5. `/tmp` inside the container is the host's tiny tmpfs.**
  *Done 2026-05-16* — `container_cmd()` appends
  `-B "${SLURM_TMPDIR:-/tmp}":/tmp --env TMPDIR=/tmp`. The
  `${SLURM_TMPDIR:-/tmp}` expansion is resolved by the rule's bash at
  runtime (Snakemake interpolates the helper output as a string).
  *Watch:* if NCCL complains about /dev/shm, add a parallel bind:
  `-B "${SLURM_TMPDIR}":/dev/shm`. Not added pre-emptively because most
  single-GPU rules don't hit /dev/shm.

## MEDIUM — correctness / performance

- [ ] **M6. No version pins on Python packages.**
  `protforge-gpu.def:62, 68, 71-78`. CUDA tag is pinned; boltz / esm /
  transformers / colabfold-from-git aren't. Rebuild in 3 mo → different
  models, different results.
  *Fix:* `requirements-container.txt` seeded from `pip freeze` of the first
  good build; `pip install --no-deps -r ... && pip check`. Bump deliberately.

- [ ] **M7. Manual `if container_cmd then ... else module load` in all rules**
  vs Snakemake's native `container:` directive. Tradeoff: manual keeps the
  legacy fallback and per-line `--env` injection (e.g. `TRITON_CACHE_DIR` in
  `boltz.smk:156`); but it loses `--use-singularity` apparatus, breaks
  `--containerize`, makes `snakemake --report` lose container metadata,
  and forces duplicated boilerplate.
  *Decision needed:* if containers become default within ~3 mo, migrate to
  rule-level `container:` + `--singularity-args` in profile. Otherwise keep
  manual. Discuss before acting.

- [ ] **M8. `--nv` is unconditional**, even when a stage has `gpus: 0`.
  Container runs on a node with no NVIDIA driver → libnvidia stub warning
  or failure. `msa.smk:87` + friends.
  *Fix:* branch `--nv` on `stage_uses_gpu(...)` in `container_cmd()`,
  not on stage name.

- [ ] **M9. `chmod -R a+rX /opt/weights`** is set; `/opt/protforge` and
  `/opt/mmseqs` aren't normalized. `protforge-gpu.def:108`.
  *Fix:* append both paths to the chmod line for shared-image use.

- [ ] **M10. `pip`/`setuptools` upgrade not pinned.** `protforge-gpu.def:53`.
  Subtle wheel-resolution drift across rebuilds.
  *Fix:* pin pip + setuptools + wheel versions.

- [ ] **M11. No `tmpdir` resource in the SLURM profile.**
  `profiles/slurm/config.yaml`. Snakemake's SLURM executor honors
  `default-resources.tmpdir` and lets every rule auto-use `$SLURM_TMPDIR`,
  killing the per-rule `export TMPDIR=...` dance in `boltz.smk:149`.
  *Fix:* set `default-resources.tmpdir: "$TMPDIR"` (or a workspace path).

- [x] **M12. Smoke test doesn't verify env-isolation invariant.**
  *Done 2026-05-16* — added step `[6/6] Host env isolation (--cleanenv
  regression)` to `containers/test/smoke.sh`. Sets `PYTHONPATH=/host/leaky/smoke`
  before invoking and asserts both `sys.path` and the in-container
  `PYTHONPATH` env var don't contain it. Regression guard for H1.

## LOW — polish

- [ ] **L13. `%labels` has no git SHA / build date.**
  `protforge-gpu.def:134-137`. Self-documenting `singularity inspect` is a
  cheap win.
  *Fix:* write `git rev-parse HEAD` into `/opt/PROTFORGE_GIT_SHA` in
  `%post`, add `Build_Date $(date -u +%FT%TZ)` to `%labels` via build arg.

- [x] **L14. `build.sh` silently falls back to `apptainer`** but still uses
  `SINGULARITY_*` env vars. `containers/build.sh:103-112`.
  *Done 2026-05-16* — `build.sh` now exports `APPTAINER_*` alongside
  `SINGULARITY_*`, prints which runtime + version it picked. `Snakefile`
  `container_cmd()` now reads `containers.runtime` (default `auto`) from
  config. `config.template.yaml` documents the new field. Working tree,
  uncommitted.

- [ ] **L15. README's `--from-docker` is dead code** until GHCR/CI lands.
  `containers/README.md:81-83, 141`.
  *Fix:* document the manual `docker build && docker push ghcr.io/...`
  recipe, or note `--from-docker` as deferred.

---

## Suggested grouping for follow-up PRs

1. **container_cmd refactor:** H1 + H3 + H5 (single touch surface — add
   `--cleanenv`, `:ro` on DBs, bind `/tmp` from SLURM_TMPDIR).
2. **Schema + migration call:** H4 + M7 (decide whether to migrate to
   native `container:` directive; collapse `containers:` block either way).
3. **Reproducibility:** H2 + M6 + M10 (allow-list `%files`, pin pip deps,
   pin pip itself).
4. Remaining mediums and lows as time permits.
