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

- 2026-07-14 channel-width sweep (vocab 3/4/5/7/10 on d2, +k7 on d1) —
  width is the wrong lever: k=7 ≈ k=10 (inventory equilibrium ~8–15 msgs is
  dynamics-set, not capacity-set), capacity binds only below k≈5 (k=4, k=3:
  0/10), and the inventory-vs-accuracy law holds across all widths (r=0.98,
  40 runs). d1 control: same channel relaxes to colour-only without hard
  negatives (shape NMI 0.19 vs 0.42). New `channel_game.py` agnostic runner.
  See log.
- 2026-07-13 d2 retraining suite (channel / full / REINFORCE, 25 runs) —
  compositional usage appears (all successes above the 0.40 colour-only
  ceiling) but acc on d2 is a near-deterministic function of surviving
  message-inventory size (Spearman 0.98 across all arms); REINFORCE+entropy
  0.1 collapses hardest (1/5), GS + d2-vision keeps most messages (best
  0.840, 21 msgs). Inventory preservation is THE lever. See log.
- 2026-07-13 difficulty ladder eval — zero-shot transfer of 9 checkpoints to
  d1–d4: graceful to d2 (0.94→0.76 best), cliff at d3 size variation (0.38);
  failure is mostly receiver-side vision, colour-first protocols cost ~0.2 acc
  under same-colour distractors. See log.
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
