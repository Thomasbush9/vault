# 2026-07-13 — Channel retrained on d2_hardneg: does compositionality emerge?

## Question

grid_iid lets colour-heavy codes win (single-attribute oracle 0.83). On
d2_hardneg every scene has a same-shape AND a same-colour distractor, so the
colour-only ceiling drops to **0.397** while shape+colour still reaches 1.0.
If hard-negative pressure forces compositionality, channels retrained on d2
should encode both attributes — and ideally approach the 1.0 ceiling.

## Setup

`experiments/d2_channel.py`. Vision NOT touched: seed_1's grid_iid-pretrained
trunks, frozen. Channel re-seeded and trained from scratch on d2_hardneg
train (8000 ex), 15 seeds x 250 epochs, standard config (vocab 10, len 2,
GS τ=1.0, Adam 1e-3). Per-epoch: val acc, distinct messages, NMI(msg;
shape/colour). Success = final acc (mean last 3) > 0.5. Outputs:
`runs/d2_channel/` (all 15 game checkpoints kept; evaluable via
`experiments/evaluate.py d2_channel/seedN`).

Control (same day, `runs/d2_full/`): full pipeline with BOTH vision modules
pretrained on d2 per seed, 5 seeds — isolates what d2-native vision buys.

## Results

**11/15 success (73%)**; mean successful acc 0.683, best 0.777 (seed 11).
Mean successful NMI shape/colour 0.44/0.83.

| basin | seeds | acc | code |
|---|---|---|---|
| compositional | 11 | 0.50–0.78 | both attrs (NMI sh 0.21–0.68, co 0.67–0.94) |
| colour-only | 1 (seed 6) | 0.44 ≈ colour ceiling | NMI 0.03/0.94, 6 msgs |
| partial/weak | 1 (seed 0) | 0.47 | NMI 0.33/0.51, 5 msgs |
| full collapse | 2 (seeds 4, 10) | 0.12 = floor | 1 message |

1. **Yes, both attributes enter the code.** Every success is far above the
   colour-only ceiling (0.397) — impossible without shape. The old grid_iid
   stuck basin splits in two here: full collapse (floor) and a new
   colour-only basin whose value (~0.44) now reflects colour's worth.
2. **But no more shape than the best grid-trained protocols.** On the
   NMI-colour x NMI-shape plane the d2-trained seeds overlap the grid-trained
   references evaluated on the same split (annealed_seed5 0.47/1.00, seed_3
   0.74/0.26). Colour still dominates 10/11 successes. Hard negatives set a
   floor on shape usage; they do not shift the ceiling.
3. **Protocols erode with long training**: successful seeds peak 0.80–0.84
   mid-run (best 0.844 @ep123), then decay — mean drop 0.087 acc and 0.11
   shape-NMI from peak to epoch 250. 250 epochs is past the optimum; colour
   slowly crowds shape back out even under hard-negative pressure. Early
   stopping (or shape-protective regularization) would keep ~0.8.
4. **The binding constraint is message-inventory collapse.** Successful runs
   use only 9–15 distinct messages for 36 (shape,colour) combos — the code is
   lossy by construction, capping d2 acc around 0.78 despite the 1.0 ceiling.
   Best-seed protocol is entangled, not slot-structured: each position carries
   BOTH attributes at NMI 0.44–0.56 (no colour-slot/shape-slot separation).
5. **Ladder eval (agnostic CLI): d2-trained ≈ grid-trained everywhere.**
   seed11: grid 0.922 / d1 0.847 / d2 0.774 / d3 0.280 / d4 0.203 — vs
   annealed_seed5 (grid-trained): 0.940 / 0.840 / 0.755 / 0.375 / 0.248.
   Training distribution barely matters for this channel; d2-native training
   buys ~+0.02 on d2 and transfers backward to grid_iid essentially intact.

## Implications / next

- Hard negatives are enough to *guarantee* shape usage but not to produce a
  clean compositional code or to reach the ceiling. The bottlenecks are now
  (a) message-inventory collapse (~15/100 used), (b) colour dominance
  drift, (c) entangled positions.
- Candidate levers: temperature annealing on d2 (basin recipe transplanted),
  length-3 / larger-vocab channel, early stopping at the acc peak,
  entropy bonus on the sender to keep the inventory from collapsing.
- d2_full control (d2-pretrained vision) running — completes the picture on
  whether vision or channel is limiting on d2.

## Addendum: controls in flight (evening)

- **d2_full** DONE (vision pretrained on d2 per seed, 5 seeds,
  `runs/d2_full/`): pretexts near-perfect on d2 (sender loss ~0.000, receiver
  occupied-cell acc 0.983–0.994) — d2 vision is easy. Game: 4/5 success,
  finals 0.724 / 0.769 / 0.790 / **0.840** (+1 collapse), mean successful
  0.781, mean NMI sh/co 0.56/0.93. vs d2_channel (frozen grid_iid vision:
  best 0.777, mean successful 0.683): same regime, same collapse risk, same
  colour-dominant entangled codes — but a modest upward shift, and the best
  run (seed 2: 0.840, 21 msgs, NMI shape 0.70, no erosion) beats every
  frozen-vision run. d2-native vision doesn't change the story but seems to
  raise the ceiling slightly; consistent with inventory size driving acc
  (21 msgs → 0.84).
- **d2_reinforce** DONE (REINFORCE + sender-entropy bonus 0.1, frozen vision,
  5 seeds, `runs/d2_reinforce/`): **1/5 success** (0.549); stuck runs 0.217 /
  0.229 (2-message colour codes) and 0.441 / 0.448 (colour-only). REINFORCE
  is strictly worse here: sender entropy collapses to ~0 by epoch 25 despite
  the 0.1 bonus, inventories stay at 2–9 messages, and one seed regressed
  from 0.49 to 0.22 mid-training. The entropy bonus at this coefficient does
  NOT prevent inventory collapse — it would need a larger coeff and/or
  annealing.

### Three-way comparison (figure: `figures/d2_comparison.png`)

| arm | success | mean successful acc | best | final msgs |
|---|---|---|---|---|
| GS, frozen grid vision (15) | 11/15 | 0.683 | 0.777 | 5–15 |
| GS, d2-trained vision (5) | 4/5 | 0.781 | 0.840 | 12–21 |
| REINFORCE, frozen vision (5) | 1/5 | 0.549 | 0.549 | 2–9 |

**The single strongest finding: final accuracy on d2 is a near-deterministic
function of how many distinct messages survive training — Spearman r = 0.98
across all 25 runs (0.97 excluding full collapses), with all three regimes
on the same curve.** Estimator and vision matter only insofar as they shift
where on the inventory curve a run lands (REINFORCE collapses hardest, GS +
d2-vision keeps the most messages alive). Message-inventory preservation is
THE lever for d2; the ceiling (1.0) needs ~36 distinguishable messages and
the best run keeps 21.
- Infrastructure landed with this line of work: `lang_emergence/evaluation.py`
  + `experiments/evaluate.py` (agnostic checkpoint x dataset eval CLI, oracle
  ceilings as standard baselines), REINFORCE game variant in the library,
  `game_type` recorded in run records so mixed-estimator checkpoints all load
  through the same CLI.
