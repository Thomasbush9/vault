# Literature

Scoped lit review. Group by sub-topic. For each entry: 1-line takeaway + 1-line "why it matters for us" + link.

## Toolkit

1. **Kharitonov et al. (2019, EMNLP demo).** EGG: a toolkit for multi-agent
   emergent-communication games with pluggable Gumbel-Softmax / REINFORCE
   channels. *Why it matters:* our game is built on `egg.core`
   (`RnnSenderGS`, `RnnReceiverGS`, `SenderReceiverRnnGS`).
   <https://github.com/facebookresearch/EGG>

## Referential games & emergence

1. **Lazaridou et al. (2017, ICLR).** Two agents playing referential games over
   images develop communication; protocol structure depends heavily on game
   design. *Why it matters:* our pointing game is a variant of this setup with
   spatial grounding (point at the cell) instead of image choice.
   <https://arxiv.org/abs/1612.07182>
2. **Havrylov & Titov (2017, NeurIPS).** Sequences of discrete symbols learned
   end-to-end with Gumbel-Softmax relaxation; GS trains faster and more stably
   than REINFORCE. *Why it matters:* justifies the GS channel; also documents
   its sensitivity to temperature — our annealing arm probes exactly this.
   <https://arxiv.org/abs/1705.11192>
3. **Kottur et al. (2017, EMNLP).** "Natural language does not emerge naturally"
   — without pressure, agents find degenerate/non-compositional codes.
   *Why it matters:* our pooling-equilibrium collapse (one message for all
   inputs) is the extreme version of this; interventions = adding the missing
   pressure.
   <https://arxiv.org/abs/1706.08502>

## To read

- Chaabouni et al. (2019/2020) on anti-efficient encoding and compositionality
  metrics (topographic similarity) — relevant once we analyze successful
  protocols across seeds.
- Rita et al. (2022) on population-level training as a collapse fix.
