# 2026-07-10 — Seed sweep + basin diagnosis

## Context

Yesterday's notebook run reached 0.85 val acc on the pointing game; this
morning's re-run (same code) plateaued at 0.67. Question: did something change?

## What we did

1. Audited the diff: only real change was the game-loss wiring in `egg_start.py`
   (a deleted global `game_loss` cell; fixed the stale `game_rnn` cell to use its
   local `_game_loss`). Vision pretraining was perfect in both runs — not the cause.
2. A single re-run escaped its 0.66 plateau at epoch 89 and was still climbing
   (0.79) when the 100-epoch cap hit → suspected seed lottery, not regression.
3. Built `seed_sweep.py`: fully seeded pipeline, 5 seeds x 200 game epochs,
   checkpoints + histories + plots saved.
4. Diagnosed the two basins by loading the saved games and probing the messages.

## Results

Sweep (200 epochs, config: vocab 10, max_len 2, hidden 128, temp 1.0, Adam 1e-3):

| seed | final acc | messages used | NMI(msg,shape) | NMI(msg,color) |
|---|---|---|---|---|
| 0 | 0.281 | 1 | 0.00 | 0.00 |
| 1 | 0.909 | 13 | 0.37 | **1.00** |
| 2 | 0.311 | 1 | 0.00 | 0.00 |
| 3 | 0.808 | 7 | **0.74** | 0.26 |
| 4 | 0.281 | 1 | 0.00 | 0.00 |

- **Bimodal.** Solved runs converge by epoch ~10–20; stuck runs sit exactly on the
  random-occupied-cell baseline (0.292) for all 200 epochs.
- **Stuck = pooling equilibrium.** The sender emits literally one message for
  every input; shuffling messages across examples doesn't change stuck-run
  accuracy (0.281 → 0.281) but crashes solved runs (0.909 → 0.259).
- **Mechanism:** receiver's best response to initial noise is "point at an
  occupied cell"; once its output ignores message content, the gradient to the
  sender vanishes and the sender's symbol distribution collapses. Self-sealing.
- **Solved protocols differ:** seed 1 encodes color perfectly + partial shape;
  seed 3 is shape-dominant. Interesting for later compositionality analysis.
- There is also an intermediate ~0.66 basin (one unseeded run escaped it late);
  likely a one-attribute protocol.

## Artifacts

- `runs/seed_sweep/` (checkpoints, histories, summary.json)
- `figures/seed_sweep.png/pdf`

## Next

- Interventions to beat the lottery: temperature annealing, asymmetric lr,
  fixed vision, ~20 seeds per arm → `basin_experiments.py` (2026-07-11 log).
