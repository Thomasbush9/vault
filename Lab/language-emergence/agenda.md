# Agenda

Current focus and the next 1–3 concrete things to do. Keep this short — when something is done, move it to a `log/` entry; when it's a parked idea, move it to `decisions.md`.

## Now

- [ ] Basin experiments running (2026-07-11): baseline vs temperature-annealed vs
      asymmetric-lr, 20 game seeds each, fixed vision. Fill in results in the
      2026-07-11 log when done.
- [ ] Striking early signal: baseline w/ fixed seed_1 vision started 5/5 — if the
      arms all saturate, the follow-up is a vision-seed sweep (fix channel seed,
      vary vision seed) to locate the lottery.

## Next

- [ ] Protocol analysis across successful runs: topographic similarity,
      per-position symbol/attribute specialization (seed 1 was color-first,
      seed 3 shape-first).
- [ ] If collapse persists in some arm: warm-start experiment (few supervised
      epochs fixing a (shape,color)→symbols protocol, then release).
