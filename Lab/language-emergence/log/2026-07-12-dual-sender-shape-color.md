# 2026-07-12 — Dual-sender: can one receiver serve a shape- and a colour-speaker?

## Question

Force perceptual specialisation at the sender (one can only see shape, one only
colour) and give both a *single shared receiver*. Does the receiver learn both
"languages" at once, and does it need to be told who is speaking?

## Setup

`dual_sender_experiment.py`. Vision held FIXED (the regime that gave high basin
success): specialists pretrained once, receiver reuses seed_1's full-colour
vision. Only the channel is re-seeded.

- **shape-sender** input = binary silhouette (pure shape). Purity probe:
  colour decodable at 0.195 (chance 0.167). Shape classification 1.000.
- **colour-sender** input = solid swatch of the object's mean colour (pure
  colour). Purity probe: shape decodable at 0.166 = chance. Colour 1.000.
  (Pixel-shuffle was tried and rejected — it leaked shape at 0.459.)

Each step routes to one sender with p=0.5; both senders + the shared receiver
update. Two conditions:

- **with_id** — receiver gets a learned embedding of the speaker id (via
  `aux_input`), *added to the message vector* before a single shared query.
- **no_id** — receiver gets nothing; must partition the code to disambiguate.

Solo baselines (each specialist alone with the receiver) bound each language.
40 epochs; success = per-sender acc (mean last 3) > 0.5; 10 seeds/mixed arm,
5/solo arm.

## Results

| arm | success (both>0.5) | shape acc | colour acc | mean Jaccard |
|---|---|---|---|---|
| with_id | **0/10** | 0.299 | 0.580 | 0.19 |
| no_id | **0/10** | 0.298 | 0.605 | 0.03 |
| shape_solo | 2/5 | 0.456 (bimodal) | — | — |
| colour_solo | 5/5 | 0.650 | — | — |

Per-sender, per-run: **shape collapses to 1 message, NMI 0.00 in all 20 mixed
runs.** Colour uses 2–4 messages, NMI(colour) 0.5–0.85, and clears 0.5 in 18/20
mixed runs (both colour failures were with_id seeds 6–7, where *both* senders
collapsed).

### Findings

1. **Colour monopolises the shared receiver; shape is annihilated.** Shape
   succeeds 2/5 *alone*, but 0/20 when it shares a receiver with a colour-
   speaker. Colour converges fast and reliably (5/5 solo), captures the
   receiver's single query projection early, and the shape gradient vanishes —
   the same self-sealing pooling collapse as before, now *induced by cross-
   sender competition* rather than a bad seed. Sharing a listener with an easy,
   reliable speaker is strictly worse for the hard speaker than going it alone.
2. **The identity token does NOT rescue shape** (0/10 in both; with_id was if
   anything slightly worse — colour also collapsed in 2 seeds). So the
   bottleneck is not "receiver can't tell who's talking." Shape never produces
   an informative message in the first place, so disambiguation is moot.
3. **Single-attribute ceiling ≈ 0.65–0.73**, not 0.9. A pure colour/shape
   message can only uniquely identify the target when no other object in the
   scene shares that attribute: with ~3.5 objects and 6 values,
   (5/6)^~2.5 ≈ 0.65. Colour-solo tops out right there. (The 0.909 of the old
   full-colour seed-1 run must have leaked a second attribute — NMI(shape) was
   0.37 there.)

### Caveats

- **The Jaccard "partition" result is confounded and should not be read as
  clean code separation.** Shape collapses to a single degenerate message, so
  low overlap mostly means "shape's one message isn't in colour's set." The
  partition hypothesis can't be tested until shape actually learns a code.
- **The id token is a weak form of conditioning.** It adds an embedding to the
  message but the receiver still has ONE shared query matrix — it can shift the
  query, not give each speaker an independent decoding subspace. A separate
  query head per speaker is the honest "receiver knows the speaker" design and
  is the obvious next test.

## Artifacts

- `runs/dual_sender/summary.json`, per-run JSON in `runs/dual_sender/runs/`,
  saved receivers for successful runs (none for mixed), specialist vision in
  `runs/dual_sender/vision/`.
- `figures/dual_sender_accuracy.*`, `figures/dual_sender_curves.*`.

## Next

- **Separate receiver query head per speaker** (proper with_id) — does shape
  survive when it has its own decoding subspace? If yes, the collapse is pure
  representational crowding in a shared query.
- **Convergence-balancing / curriculum**: let shape train alone for a few
  epochs before releasing colour, or slow colour's lr — test whether shape can
  get a foothold before colour captures the receiver.
- Either way the headline stands: naive receiver-sharing yields *dominance, not
  division of labour* (cf. Kottur "does not emerge naturally"; population-EC).
