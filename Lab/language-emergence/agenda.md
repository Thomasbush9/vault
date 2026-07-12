# Agenda

Current focus and the next 1–3 concrete things to do. Keep this short — when something is done, move it to a `log/` entry; when it's a parked idea, move it to `decisions.md`.

## Now

- [x] Basin experiments (2026-07-11): baseline 80% / annealed 90% / asym_lr 30%,
      fixed vision. Results in the 2026-07-11 log. Findings: (a) fixing vision is
      the big lever (80% vs 40% full-pipeline); (b) annealing helps modestly;
      (c) asymmetric lr hurts — accelerates the pooling collapse.
- [ ] **Vision-seed sweep** (the follow-up the results point to): fix the channel
      seed, vary the vision-pretraining seed, ~20 seeds. If success tracks the
      vision seed, the lottery is mostly in vision init and we can screen vision
      checkpoints cheaply before ever training the channel.

- [x] Dual-sender (2026-07-12): shape- vs colour-specialist senders share one
      receiver, with/without speaker-id token, 10 seeds each. Result: colour
      monopolises the shared receiver, shape collapses 0/20 (vs 2/5 solo); the
      id token does not help. See the 2026-07-12 log.

## Next

- [ ] **Dual-sender follow-ups** (the results point here): (a) separate receiver
      query head per speaker — does shape survive with its own decoding subspace?
      (b) curriculum / lr-balancing so shape gets a foothold before colour
      captures the receiver. Current id-token only shifts a shared query.
- [ ] Protocol analysis across successful runs: topographic similarity,
      per-position symbol/attribute specialization (seed 1 was color-first,
      seed 3 shape-first).
- [ ] If collapse persists in some arm: warm-start experiment (few supervised
      epochs fixing a (shape,color)→symbols protocol, then release).
