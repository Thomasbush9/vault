# Agenda

Current focus and the next 1–3 concrete things to do. Keep this short — when something is done, move it to a `log/` entry; when it's a parked idea, move it to `decisions.md`.

## Now

- [ ] **Dual-sender: separate receiver query head per speaker** (top priority —
      the proper "receiver knows the speaker" test). In `dual_sender_experiment.py`
      the with_id token only *adds* an embedding to the message before ONE shared
      query matrix; it did not rescue shape (0/10). Give the receiver an
      independent query head per speaker id (two decoding subspaces) and re-run
      with_id vs no_id, 10 seeds. Question: does shape survive when it isn't
      crowded out of a shared query, or does colour still dominate the shared
      vision/optimiser? Reuse the fixed-vision setup and the same metrics
      (per-sender acc, distinct msgs, NMI, Jaccard).
- [ ] Then curriculum / lr-balancing: let shape train alone a few epochs (or slow
      colour's lr) so it gets a foothold before colour captures the receiver.

## Recently done

- 2026-07-11 basin experiments — annealed 90% > baseline 80% > asym_lr 30%
  (fixed vision). See log.
- 2026-07-12 dual-sender — colour monopolises a shared receiver, shape collapses
  0/20, id token does not help. See log.
- Parked: vision-seed sweep (fix channel seed, vary vision-pretrain seed) to
  locate how much of the lottery lives in vision init.

## Next

- [ ] Protocol analysis across successful runs: topographic similarity,
      per-position symbol/attribute specialization (seed 1 was color-first,
      seed 3 shape-first).
- [ ] If collapse persists in some arm: warm-start experiment (few supervised
      epochs fixing a (shape,color)→symbols protocol, then release).
