# Decisions

Append-only log of design choices and dead ends, so they don't get re-litigated.

### pre-2026-07-10 — Receiver gets per-cell embeddings, message acts as a query

**Decided:** the receiver vision trunk returns one embedding per grid cell
(`[B, 16, D]`, adaptive-pooled to the 4x4 label grid); the game receiver projects
the message to a query vector and scores it against cells by dot product.
**Why:** with a single global scene vector the receiver could ignore the message
entirely and the game collapsed. With the pointing/query design the message is
structurally required — without it there is no ranking over cells.
**Alternatives parked:** global scene embedding + MLP over [message; scene].

### 2026-07-10 — Seed and checkpoint everything; judge configs by success rate

**Decided:** all training runs go through `seed_sweep.py`-style scripts with
explicit seeds and saved checkpoints/histories; single-run comparisons are not
evidence.
**Why:** the 5-seed sweep showed the game is bimodal (0.81–0.91 vs ~0.29 stuck at
the random-occupied-cell baseline); day-to-day differences (0.85 vs 0.67) were
pure seed lottery, not code changes. Fate is decided in the first ~20 epochs.
**Alternatives parked:** longer training as a fix — a 200-epoch run showed stuck
runs stay stuck; the collapse is a pooling equilibrium (sender emits ONE message
for every input; receiver plays "point at an occupied cell"), not slowness.

### 2026-07-11 — Fixed vision pair for channel experiments

**Decided:** basin/intervention experiments reuse seed_1's pretrained vision
checkpoints for every run and vary only the game (channel) seed.
**Why:** isolates channel-initialization variance from vision-pretraining
variance; vision quality was ~0.99 for every sweep seed so it is not the
differentiator. Also ~3x faster per run.

### 2026-07-11 — Heavy artifacts out of git

**Decided:** `runs/` and `figures/` moved to
`.../language/data/language_emergence/` and symlinked into the repo; gitignored.
**Why:** 168MB of checkpoints had been committed; artifacts don't belong in the
code repo. NOTE: commit `5ca239f` (unpushed at the time of the fix, parent of the
removal commit) still carries the blobs in history — if push size ever matters,
rewrite before pushing.
