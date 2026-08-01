# 2026-07-30 — Pivot to JAX/joltz, and a circuit-level framing of the MSA bias

Restart after the May pause. Two things changed the plan: the fold-switch
predictions on the cluster are **gone** (`/n/holylfs06/.../protein_rsa/` no
longer exists — setup scripts survive in the repo, the scored CSVs survive on
the Mac), and `mosaic_setup` now has a **JAX/equinox reimplementation of
Boltz-2** that makes real mechanistic interpretability tractable in a way the
PyTorch hook stack never was.

## Why the old result was weak, stated precisely

"The MSA of a mutant is a subset of the WT MSA, therefore the internal
representations are similar" is a statement about the *input*, not about the
model. It explains nothing about the architecture: any model would be invariant
to a perturbation that barely changes its input. It also can't distinguish the
two things we actually care about — whether the model *discards* mutation
information it received, or never *received* it in a form that could compete
with the MSA.

The May PWA-conservation work (`2026-05-04`) was closer: it showed MSA column
attention is nearly query-blind (JS ≈ 0.004 between WT and a 40 %-mutant query,
vs ≈ 0.7 for random distributions) and tracks *variable* rather than conserved
columns. But it was correlational, N = 1 protein, and had no causal handle.

## The architectural claim worth testing

Reading joltz's Boltz-2 trunk, query-sequence identity reaches the pair
representation `z` by exactly **four separable routes**, plus one into the
single representation `s`:

| route | path | diluted by MSA depth? |
|---|---|---|
| `z_direct` | `s_inputs → z_init_1/z_init_2` outer sum | no |
| `s_direct` | `s_inputs → s_init` | no |
| `msa_bcast` | `s_inputs → msa_module.s_proj`, added to **every** MSA row | no |
| `msa_query` | MSA **row 0** (the query) → OuterProductMean | **yes, ~1/S** |
| `msa_prior` | MSA rows 1..S → OuterProductMean | — this is the "cheat sheet" |

`OuterProductMean` computes `einsum("bsic,bsjd->bijcd", a, b) / num_unmasked_rows`
(joltz `__init__.py:308`). So the query, as one row among S, enters `z` at
roughly 1/S the weight of the alignment as a whole. With S ≈ 700 for GFP that
is a ~700-fold gain difference — *if* `msa_query` were the only query route.
It isn't: `msa_bcast` and `z_direct` are undiluted. **Whether the net query
influence decays with S is therefore an empirical question, not a corollary**,
and it is one we can now measure directly. That is the whole experiment.

This is a genuine circuit claim: it names specific components, predicts a
specific scaling, and is falsifiable.

## Why JAX/joltz instead of the PyTorch stack

`mosaic_setup/mosaic` (fork of escalante-bio/mosaic) contains `joltz`, a
from-scratch equinox reimplementation of Boltz-1/2. Torch is used exactly once,
at checkpoint load. Consequences:

- Per-layer capture is free: `Pairformer2` stores a *stacked* parameter pytree
  plus a `static` template and runs `jax.lax.scan`, discarding `ys`. Re-running
  that scan with `ys` populated gives all 64 layers' activations **without
  patching the library**.
- Path patching is `eqx.tree_at` on an `InitialEmbedding` pytree — no hooks, no
  monkeypatching, no `torch.compile` unwrapping, no fighting activation
  checkpointing (all of which the May stack had to do; see `extractor_boltz.py`).
- Gradients through the whole trunk come for free.
- AF2 (monomer + multimer, weights on disk) and OpenFold3 / Protenix as AF3
  stand-ins are in the same library, so the "then AF2/3" half of the project is
  a port, not a rewrite.

## Infrastructure built today

Work area: `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/`
(`harness/`, `data/`, `runs/`, `logs/`, `figures/`).

- `harness/pi_core.py` — model/feature loading (offline, no ColabFold server),
  instrumented `Pairformer2` and `MSAModule2` scans, `iteration` / `run_trunk`,
  distogram **logit lens** (apply the distogram head to intermediate `z` to see
  what contact map the trunk would emit if it stopped at layer L).
- `harness/pi_paths.py` — the five-route decomposition, `patch()`, MSA depth
  truncation.
- `harness/build_dataset.py` — the "physics vs memory" cohort.
- `harness/run.sbatch` — GPU wrapper via `mosaic-exec.sh` (kempner_h100).

**Validation is done and it matters.** `harness/test_equivalence.py` shows
`pi_core.iteration` is *bit-identical* to joltz's own `trunk_iteration`
(relative error exactly 0.0 on `s`, `z`, distogram and contact map, with
capture both on and off). Dropout is confirmed off under `deterministic=True`
(different keys give identical output), and `subsample_msa=False` is set so
MSA depth is an exactly controlled quantity rather than a random 1024-row draw.

One trap found and removed: an earlier `trunk_capture` ran some recycles
through joltz's jitted `trunk_iteration` and only the last through the capture
path. That produced a ~6e-4 relative drift that looked exactly like float32
fusion noise but was two different computations. There is now **one** definition
of `iteration`, and a comment saying why a second one is a trap.

## The cohort

`data/gfp_physics` — GFP (238 aa, MSA depth 731, from the ProtForge run at
`test_protforge/outputs/sequences/34073`).

- `core_{1,2,4,8,16,32}` — buried hydrophobic → charged (C70E, F71K, V112R,
  F83K, A87K, V68R, F84R, M88R, …). Burial = CA neighbour count within 10 Å;
  sites taken most-buried-first. These should be structurally catastrophic on
  physical grounds.
- `surface_{1,2,4,8,16,32}` — exposed non-hydrophobic → polar, count-matched control.
- Substitutions are a deterministic function of the site, so the series is
  properly **nested**: the n=4 mutant is the n=32 mutant restricted to its first
  four sites. Otherwise dose-response confounds "more mutations" with
  "different substitutions".
- **Every mutant carries the WT alignment verbatim** (row 0 rewritten; grafting
  per the ProtForge MSA-bench finding that this is quality-neutral and ~100×
  cheaper). So the alignment says "GFP barrel" while the sequence says "this
  cannot fold". The arbitration between those is the object of study.

## Experiments running

1. **`exp_paths.py`** — for each mutant, patch each of the five routes from WT
   and measure, in Ångström of mean |ΔE[d]|:
   - *necessity*: run mutant with route r restored to WT — how much of the
     mutation's effect does r carry?
   - *sufficiency*: run WT with route r injected from the mutant — is r alone
     enough to move the prediction?
   Endpoint sanity checks (patch none ⇒ mutant, patch all ⇒ WT) run first and
   are treated as blocking.
2. **`exp_depth.py`** — mutation sensitivity vs MSA depth S ∈ {1…731}. S=1 is
   the single-sequence control. Tests the 1/S prediction directly.

## Known caveats to carry forward

- Depth truncation takes rows in file order; a ColabFold a3m is roughly sorted
  by similarity, so depth and diversity move together. `--shuffle-rows` exists
  to break that; both should be reported.
- Distogram-only. The diffusion module and the actual 3-D output are downstream
  of everything measured here; "the trunk decided X" is not yet "the structure
  is X". Scoring predicted structures is a separate step.
- N = 1 protein again. This is deliberate for harness bring-up, and the fold-
  switch cohort (contrastive, two ground truths) is the right generalisation
  target — but it needs regenerating since the cluster copy was deleted.

## Next

- Read out the two experiments; decide whether the story is "query never
  written" or "query written then erased" from the per-layer logit-lens curves.
- Regenerate the fold-switch cohort (scripts survive; Zenodo MSAs need
  re-downloading) — it is the cohort where "memorised vs not" has ground truth.
- Score predicted structures for the core mutants with the mosaic stability
  head / ProteinMPNN to quantify "how energetically implausible is what the
  model predicted" — that is the "physics vs memory" claim made numerical.

---

## Results (2026-07-31)

GFP, N=238, MSA depth S=731, 3 recycles, distogram-level metrics. Distances are
mean |ΔE[d]| over off-diagonal residue pairs, in Ångström.

### 1. The five routes are an exact decomposition

Both endpoint checks close to **0.00e+00 Å**: patching no routes reproduces the
mutant bit-exactly, patching all five reproduces the wild type bit-exactly. So
the routes are not an approximate attribution scheme — they exhaust the
difference between the two runs, and the per-route numbers below are a genuine
causal decomposition rather than a saliency heuristic.

### 2. Almost all surviving mutation signal travels the *direct* path, not the MSA

`gfp_core_32` (32 buried hydrophobics → charged), total effect **0.305 Å**:

| route | necessity | sufficiency |
|---|---:|---:|
| `z_direct` (s_inputs → z_init) | **0.879** | **0.720** |
| `msa_bcast` (s_inputs → every MSA row) | 0.276 | 0.121 |
| `msa_query` (query as MSA row 0) | 0.023 | 0.005 |
| `s_direct` (s_inputs → s_init) | 0.000 | 0.000 |
| `msa_prior` (MSA rows 1..S) | 0.000 | 0.000 |

Replicated across the cohort (necessity / sufficiency):

| mutant | total | `z_direct` | `msa_bcast` | `msa_query` | `s_direct` | `msa_prior` |
|---|---:|---|---|---|---|---|
| `core_32` | 0.305 Å | 0.879 / 0.720 | 0.276 / 0.121 | 0.023 / 0.005 | 0 / 0 | 0 / 0 |
| `core_08` | 0.247 Å | 0.934 / 0.783 | 0.210 / 0.063 | −0.005 / 0.002 | 0 / 0 | 0 / 0 |
| `surface_32` | 0.103 Å | 0.717 / 0.689 | 0.306 / 0.318 | −0.022 / 0.155 | 0 / 0 | 0 / 0 |

Necessities sum to >1, so the routes are redundant rather than additive —
`z_direct` and `msa_bcast` both carry overlapping copies of the same
`s_inputs` signal, which is expected given they share a source.

Two of these are controls that came out exactly right, which is what makes the
rest believable:

- **`msa_prior` = 0.000 by construction** — every cohort member shares an
  identical alignment body, so patching those rows must do nothing. It does
  nothing, to machine precision.
- **`s_direct` = 0.000 architecturally** — in Boltz-2's `PairformerLayer`, `z`
  is updated only by tri-mul / tri-attention / `transition_z`, all functions of
  `z` alone; `s` reads `z` as attention bias but never writes back. So the
  distogram is *provably* independent of `s_init`, and the measurement agrees.
  (`s_direct` is not irrelevant to the model — the structure module consumes
  `s_trunk` — it is irrelevant to the distogram.)

**The headline:** `msa_query` carries ~2 %. The query's presence *inside* its own
alignment is causally almost irrelevant, exactly as OuterProductMean's `1/S`
normalisation predicts (1/731 ≈ 0.1 %). What mutation signal survives arrives
through the undiluted `s_inputs → z_init` path.

This kills the naive version of our own hypothesis. The MSA does **not**
suppress mutations by drowning the query row in the alignment average — that
route was never carrying anything to begin with. Something else is doing the
suppressing.

### 3. What actually suppresses it: a fixed band of Pairformer layers

Per-layer logit lens across the 64 Pairformer blocks, ΔE[d] vs wild type:

| mutant | layer 0 | peak | final | final/peak |
|---|---:|---:|---:|---:|
| `core_32` | 0.357 | **0.833 @ L34** | 0.306 | 0.368 |
| `core_08` | 0.197 | **0.504 @ L34** | 0.245 | 0.486 |
| `surface_32` | 0.271 | **0.453 @ L34** | 0.103 | 0.227 |

The trace is the same shape in every case: divergence **rises monotonically
through layers 0–34** (the Pairformer amplifies the mutation more than
two-fold), then **collapses over layers ~37–45** (steepest single-layer drop at
L38→L39), then flattens with a slight recovery.

So the answer to "never written or written then erased" is **written then
erased**. And the erasure is localised to the same narrow band of layers
regardless of how many mutations there are or whether they are in the core or on
the surface — it is a property of the network, not of the input.

**Important refinement — it is not erasure, it is rotation out of the readout.**
The representation-space trace (mean |Δ‖z‖|) does *not* follow the
structure-space trace. It rises to ~L35, dips around L50, and then climbs to its
**maximum at the final layers (L62–63)** — precisely where the distogram
divergence falls to its **minimum**. So at the output of the trunk the mutant's
pair representation is *further* from the wild type's than at any earlier layer,
while the contact map read off that representation is closest to wild type.

The mutation information is therefore still present in `z`; it has been moved
into directions the distogram head does not read. That is a materially
different claim from "the signal is destroyed", and a much more tractable one:
a linear probe on final-layer `z` should recover the mutation even though the
distogram cannot. Worth testing early — it is cheap and it decides between
"information destroyed" and "information retained but unread", which are
different papers.

There is real selectivity in *how much* survives: the surface control is
suppressed hardest (retains 23 %), core_32 retains 37 %. That is the right sign
— surface mutations genuinely should not move the backbone — but the absolute
scale is the story: **32 buried charge substitutions end up moving the predicted
distogram by 0.3 Å and flipping 0.6 % of contacts.**

### 4. MSA depth matters, but not as 1/S

`core_32` sensitivity vs depth (rows taken in file order):

| S | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 731 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ΔE[d] (Å) | 1.355 | 0.901 | 0.892 | 0.956 | 0.778 | 1.161 | 0.453 | 0.378 | 0.396 | 0.401 | 0.305 |

Single-sequence is **4.4× more mutation-sensitive** than full-depth. But the
curve is emphatically not `1/S` — it is roughly flat from S=2 to S=32, jumps
*up* at S=32, then steps down at S=64 and plateaus. Depth per se is not the
variable; **which homologs are present is.** That is consistent with §2: the
query's dilution among rows is not the mechanism, so row *count* shouldn't be
the controlling parameter, and it isn't.

Through all of this `|z|` after the MSA module is ~3.3× `|z|` after `z_init`
(≈265–300 vs ≈82–86), and the OPM write grows across the four MSA blocks
(10.3 → 18.2 → 21.4 → 43.2). The MSA writes a large, query-independent signal
into `z`; the query writes a small one; the mid-stack then renormalises the
small one away.

### 5. Structure level — resolved: the TM numbers were sampler noise, not signal

`exp_predict.py` initially reported TM ≈ 0.11–0.53 against the wild-type
prediction for every mutant including n=1, with no dose-response, while pLDDT
showed a clean one. `harness/check_predict.py` settles which half to believe:

| check | result | verdict |
|---|---|---|
| CA–CA spacing, all 13 structures | 3.80 Å (IQR 3.79–3.81) | correct — extraction is fine |
| radius of gyration, all 13 | 17.1–17.5 Å | correct, and matches the reference (17.52 Å) |
| **JAX WT vs independent PyTorch Boltz-2 WT** | **TM 0.697, RMSD 5.50 Å** | **this is the problem** |
| WT vs shuffled WT (floor) | TM 0.109 | — |

The same model, same sequence, same alignment, run through two entirely
separate code paths, agrees with itself at only **TM ≈ 0.70**. That is the
run-to-run reproducibility floor of a **single** diffusion sample. Every
mutant-vs-WT TM we measured (0.11–0.53) sits *at or below* that floor, so those
numbers carry no information about the mutations — they are sampler variance.
Boltz's own default is 25 diffusion samples ranked by confidence, and we used 1.

What does survive, and is worth keeping:

- **Every predicted structure is a compact, well-formed, GFP-sized globule** —
  Rg 17.1–17.5 Å against the wild type's 17.46 Å — *including* `core_32`, whose
  32 buried hydrophobics are all charged. The model does not produce anything
  resembling an unfolded or expanded chain for a sequence that cannot fold.
- **pLDDT falls monotonically with core mutation load** (0.948 → 0.941 → 0.929
  → 0.892 → 0.830 for n = 0,1,2,4,8) and **barely moves for surface mutations**
  (0.948 → 0.932 at n=32). That is a single array with no geometry or
  superposition in it, so it is not affected by the sampling issue. The model
  *is* registering the core mutations, in its confidence, while still emitting
  the barrel.

**Fix before re-running:** either raise `--diffusion-samples` to 25 and rank by
confidence (matching Boltz's default), or drop coordinate-level comparison
entirely and stay with the distogram, which is deterministic and is where the
mechanism lives anyway. The honest headline is the pLDDT + Rg pair, not TM.

## Figures

`prot_interp_files/figures/` — `paths_gfp.png` (route decomposition, 3 mutants),
`layers_gfp.png` (per-layer divergence, structure- and representation-space),
`depth_gfp_core32.png` (sensitivity vs MSA depth with the 1/S reference).
Regenerate with `harness/make_figures.py --runs runs --figures figures`.

## Reading

The story we started with — "the MSA drowns the mutant query in the alignment
average" — is wrong in its specific mechanism, and we can now say so with a
causal measurement rather than a correlation. The query row contributes ~2 %;
removing the dilution would change almost nothing.

What we have instead is more interesting and more localised: the query's
mutation signal enters `z` through a single direct projection, is *amplified*
by the first half of the Pairformer, and is then **actively attenuated by a
fixed band around layers 37–45** — the same layers, at the same place, whatever
the mutation is. That band is the object to study next. It is also exactly the
kind of thing that is hard to see in PyTorch and easy to see here.

## Next

1. **Fix and re-run the structure-level readout** (§5). Without it we have a
   trunk result and no behavioural anchor.
2. **Characterise the L37–45 band.** Which sublayer does it (tri-mul out/in,
   tri-attention start/end, transition)? Per-sublayer capture is a small
   extension of `pairformer_capture`. Then ablate: zero the suppressing
   sublayer's contribution in that band and see whether mutation sensitivity is
   restored — that turns "correlated with layers 37–45" into a causal claim.
3. **Generalise off N=1.** Same three experiments on the fold-switch cohort and
   on a handful of proteins with varied MSA depth.
4. **Depth vs diversity.** Rerun the sweep with `--shuffle-rows` and several
   seeds; the non-monotonicity in §4 says composition matters more than count,
   and that deserves its own measurement.

---

## Addendum (2026-07-31, later) — two follow-ups, and a correction to my own reading

### 6. The suppressor is `transition_z`, not the geometric operations

`exp_sublayers.py` splits each Pairformer layer into its five sequential writes
into `z` and measures each one's contribution to the divergence. Net change
contributed by each op, in Å of mean |ΔE[d]| (negative = reduces divergence):

| op | `core_32` L37–45 | `core_32` all 64 | `surface_32` L37–45 | `surface_32` all 64 |
|---|---:|---:|---:|---:|
| **`transition_z`** | **−0.309** | **−0.691** | **−0.193** | **−0.341** |
| `tri_mul_in` | −0.084 | +0.306 | −0.042 | +0.118 |
| `tri_mul_out` | −0.073 | +0.019 | −0.042 | +0.025 |
| `tri_att_end` | −0.061 | +0.081 | −0.032 | −0.012 |
| `tri_att_start` | +0.054 | +0.213 | +0.036 | +0.121 |

Unambiguous and replicated across both mutants: **the triangle operations
(multiplication and attention) *amplify* the divergence; the per-pair
feed-forward `transition_z` is the only thing reducing it**, and it accounts for
essentially all of the net reduction. It is not the geometric-reasoning
machinery that removes the mutation — it is the channel-mixing MLP.

### 7. Correction: it is **not** "rotated out of the readout", and probably not erased either

I proposed that the mutation signal survives in `z` but is moved into
directions the distogram head cannot see. That is now **refuted**. The head is
`W @ (z + zᵀ)` with `W` of shape (64, 128) and full row-rank 64, so it is blind
to the antisymmetric part of `z` and to half the symmetric channels — all
exactly computable. `exp_subspace.py` decomposes Δz per layer:

| | `core_32` L0 | L34 | L63 | `surface_32` L0 | L63 |
|---|---:|---:|---:|---:|---:|
| ‖Δz‖ | 3326 | — | **7010** | 2086 | **3177** |
| readable fraction | 0.277 | 0.291 | 0.297 | 0.280 | 0.261 |
| ‖readable part‖ | 1748 | — | **3817** | 1102 | **1626** |

The readable fraction is **flat** (0.25–0.30) across all 64 layers, and the
readable component's magnitude **more than doubles**. Nothing is being rotated
anywhere, and the mutation's footprint in the readout-visible subspace is
*larger* at the end of the trunk than at the beginning.

So we have ‖Δz‖ rising monotonically, its readable part rising with it, and
mean |ΔE[d]| falling by 2.7×. Those are only compatible if the discrepancy is
in the **readout nonlinearity**: `E[d] = Σ_b softmax(logits)_b · centre_b`. As
the trunk sharpens the distogram, a fixed perturbation in logit space moves E[d]
less. A falling |ΔE[d]| would then be a *measurement artefact of working in
Ångström against a sharpening distribution*, not suppression.

**This puts §3 and §6 in question.** Both measured divergence in E[d]. If the
saturation explanation holds, "a band of layers that suppresses the mutation"
and "`transition_z` is the suppressor" may both be restatements of "the
distogram gets sharper, and `transition_z` is what sharpens it" — which is a
much less interesting claim, and arguably just describes a confident model.

`exp_kl.py` is running and is the arbiter: symmetric KL between the mutant and
wild-type distogram distributions per layer is invariant to peakedness in the
way E[d] is not, and it reports the wild-type entropy per layer alongside.

- **KL falls across L37–45** ⇒ suppression is real, §3/§6 stand (with E[d]
  exaggerating the magnitude).
- **KL flat or rising while E[d] falls** ⇒ there is no suppression band; the
  finding is readout saturation and the whole §3 reading must be withdrawn.

Either way the §2 route decomposition is unaffected — necessity/sufficiency are
ratios of E[d] distances measured at the *same* layer (the output), so
saturation cancels.

**Lesson to carry:** every per-layer claim in this project must be made in a
scale-free measure. E[d] in Ångström is intuitive and directly comparable to
structural intuition, which is exactly why it was seductive, but it is not a
valid yardstick for comparing *across* depth when the distribution's sharpness
is itself changing with depth.

---

## 8. The arbiter: the band is real but was overstated; the *terminal* fall is an artefact

`exp_kl.py` measures the mutant-vs-wild-type distogram difference in a
peakedness-invariant way (symmetric KL over the 64 bins) alongside the same
quantity in Ångström, with the wild-type distogram entropy per layer.

First correction to my first correction: I initially read this as "no
suppression anywhere, §3 and §6 withdrawn". That was itself too strong. The
numbers separate into two distinct events that E[d] had merged:

### Event 1 — the L37–45 band is real, but ~1.7× smaller than it looked

| | `core_32` | `surface_32` |
|---|---:|---:|
| KL, L34 → min in band | 0.357 → 0.256 (L41) | 0.116 → 0.065 (L42) |
| **drop in KL** | **1.40×** | **1.78×** |
| drop in E[d] over the same band | 2.38× | 2.69× |
| entropy across the band (L34→L45) | 1.62 → ~1.68 nats (flat) | (same model) |

Entropy is essentially *flat* across L37–45, so saturation cannot explain this
one — **there is a genuine, transient reduction in the mutation's distributional
footprint there.** §3 and §6 survive as findings, with their magnitude cut
roughly in half: the band suppresses ~1.4–1.8×, not the ~2.5–2.7× that E[d]
implied.

### Event 2 — the fall to the trunk output *is* pure saturation

| L55 → L63 | `core_32` | `surface_32` |
|---|---:|---:|
| symmetric KL | 0.484 → **0.913** (↑1.89×) | 0.120 → **0.215** (↑1.80×) |
| mean \|ΔE[d]\| | 0.521 → 0.307 (↓0.59×) | 0.235 → 0.100 (↓0.43×) |
| WT distogram entropy | **2.156 → 0.805 nats** | (same model) |

Over the last eight layers the two measures move in **opposite directions**
while the distogram entropy collapses by 2.7×. Here E[d] is unambiguously
lying: the mutation's effect on the predicted distribution nearly doubles while
its effect on the expected distance almost halves.

Net across the whole stack, symmetric KL grows **5.6×** (0.162 → 0.913) and is
maximal at the final layer for both mutants (final/peak = 1.000), as do ‖Δz‖
(2.1×), the readable part of ‖Δz‖ (2.2×), and |Δlogit| (2.4×). **The trunk ends
up carrying more mutation signal than it started with, not less.**

### What this means

The original puzzle is resolved differently from anything I proposed. There is
a modest, real suppression event mid-stack, but it is transient and more than
undone later. The reason the *predicted structure* barely changes is not that
the signal was removed — it is that **Boltz-2 expresses a destabilising mutation
as a change in the shape of its predicted distance distribution rather than a
shift in its mean.** A large KL with a small ΔE[d] is what "same mode, more
spread" looks like.

That is independently corroborated by the confidence head: pLDDT falls
monotonically with buried-core mutation load (0.948 → 0.830 for n = 0…8) while
barely moving for count-matched surface mutations (0.932 at n = 32). The model
does register that the sequence cannot fold; it routes that into **uncertainty**
rather than **geometry**, and the geometry stays the memorised wild-type barrel.
Four independent measurements agree (‖Δz‖, KL, |Δlogit|, pLDDT).

### What survives, precisely

- **§2 (routes) unaffected** — necessity/sufficiency are ratios taken at the
  same layer, so sharpening cancels. `z_direct` ~0.72–0.93, `msa_query` ~0.02.
- **§3 (band) survives at reduced magnitude** (1.4–1.8× in KL, entropy flat
  there), but its *framing* changes: it is a transient dip, not a terminal
  erasure, and it is fully recovered by the output.
- **§6 (`transition_z`) is now ambiguous and needs re-running in KL units.**
  It was measured in E[d]. Its large negative contribution could be the band
  dip (real) or the terminal sharpening (artefact) or both — and note
  `transition_z` is exactly the kind of op that would sharpen a distribution.
  **Do not cite §6 until it is repeated in KL.**
- **§4 (depth) and §5 (structure)** unaffected; both are output-only.

### Immediate next steps

1. **Re-run `exp_sublayers.py` in KL units.** One-line change to its reduce_fn.
   This decides whether `transition_z` is a suppressor or a sharpener — the
   single most load-bearing open question now.
2. Decompose the final-layer distogram change into **mean-shift** vs
   **width/shape** components per residue pair, and test whether the shape
   component concentrates on mutated positions.
3. Treat the **confidence head as central**, not peripheral: if mutation
   information is routed to uncertainty, pLDDT/PAE is where it lives.

**Methodological lesson, which cost two wrong readings in one session:** a
quantity in physical units (Å) is not a valid yardstick across depth when the
distribution generating it changes sharpness with depth. Use a divergence,
report entropy alongside, and check whether the two disagree *before* believing
either.

---

## 9. §6 resolved: `transition_z` is a genuine suppressor, and the Pairformer is a competition

`exp_sublayers.py` re-run with symmetric KL instead of E[d] (`runs/sublayers_kl_gfp.json`).
Net change in KL contributed by each op:

| op | `core_32` L37–45 | `core_32` all 64 | `surface_32` L37–45 | `surface_32` all 64 |
|---|---:|---:|---:|---:|
| **`transition_z`** | **−0.175** | **−0.359** | **−0.062** | **−0.093** |
| `tri_att_end` | +0.019 | +0.162 | −0.000 | +0.019 |
| `tri_mul_in` | +0.025 | +0.370 | +0.006 | +0.072 |
| `tri_mul_out` | +0.036 | +0.345 | +0.008 | +0.088 |
| `tri_att_start` | +0.045 | +0.227 | +0.012 | +0.061 |

The arithmetic closes exactly, which is the built-in check: for `core_32`,
entering KL 0.1637 + (triangle ops +1.1045) + (`transition_z` −0.3594) = 0.9088
= leaving KL. Same for the surface control (0.0684 + 0.2404 − 0.0930 = 0.2159).

**§6 survives the change of units.** KL is invariant to how peaked the
distogram is, so this cannot be a sharpening artefact: `transition_z` genuinely
removes mutation-induced divergence, and it is the *only* op that does.

### The mechanism, stated properly

The Pairformer is a **competition between two opposing populations of operations**:

- the four **triangle operations** (multiplication outgoing/incoming, attention
  starting/ending node) — the geometric-reasoning machinery — **amplify** the
  mutation's distributional footprint, +1.10 in KL over 64 layers;
- the per-pair feed-forward **`transition_z`** **attenuates** it, −0.36.

Amplification wins roughly 3:1, which is why the net effect across the trunk is
a 5.6× growth in KL. But the ratio is not constant with depth: in the L37–45
band `transition_z` (−0.175) outweighs the triangle ops (+0.124), and that
local reversal *is* the transient dip seen in §3 and §8(a). So the "suppression
band" is real, and it is now identified with a specific component — it is the
stretch of the stack where the channel-mixing MLP briefly out-competes the
geometric operations.

Note this is the opposite of the naive expectation. One would guess the
*geometric* machinery enforces the memorised fold and the MLP passes
information through. It is the other way round: triangle attention and
multiplication carry the mutation forward, and the per-pair MLP is what pushes
the representation back toward the wild-type distribution.

### Status of every claim in this log

| § | claim | status |
|---|---|---|
| 2 | routes: `z_direct` ~0.72–0.93, `msa_query` ~0.02, exact decomposition | **holds** |
| 3 | L37–45 suppression band | **holds**, magnitude ~1.7× smaller than first reported |
| 4 | depth sweep, not 1/S | holds (output-only); repeat in KL for unit consistency |
| 5 | structure level | TM is sampler-noise-limited; pLDDT + Rg are the usable readouts |
| 6 | `transition_z` is the suppressor | **holds** — confirmed in KL (§9) |
| 7 | "rotated out of the readout" | **refuted** by §8 |
| 8 | terminal fall in E[d] is saturation | **holds** |

### Next

1. **Ablate `transition_z` in L37–45** (scale its contribution to zero via
   `eqx.tree_at` on the stacked parameters) and confirm the band dip disappears.
   That converts §9 from an attribution to a causal claim, and it is now a
   ~20-line experiment.
2. Re-express §4 in KL for unit consistency.
3. Decompose the final distogram change into mean-shift vs width components per
   pair; test whether the width component localises on mutated positions.
4. Generalise off N=1 (fold-switch cohort), then port to AF2.

---

## 10. Causal test: ablating `transition_z` in L37–45 abolishes the dip — but does not "free" the signal

`exp_ablate.py` zeroes `transition_z`'s output projection (`fc3.weight`, and
bias if present) on a chosen slice of the stacked Pairformer parameters, which
deletes the `z += transition_z(z)` write exactly. Untouched layers are
bit-identical. `gfp_core_32` vs WT, symmetric KL:

| condition | KL change across L36→L45 | final KL | vs intact |
|---|---:|---:|---:|
| **intact** | **−0.0418** | 0.926 | — |
| **ablate `transition_z`, L37–45** | **+0.0625** | 0.544 | ×0.587 |
| ablate `transition_z`, control band L10–18 | −0.0125 | 0.945 | ×1.021 |
| ablate `transition_z`, all 64 layers | +0.0073 | 0.238 | ×0.257 |

**The targeted claim is confirmed.** Deleting `transition_z` in L37–45 does not
merely shrink the dip — it **flips its sign**, from −0.042 (suppressing) to
+0.063 (amplifying, like every other stretch of the stack). A width-matched
control ablation of nine layers elsewhere (L10–18, where `transition_z`'s
attributed contribution is small) leaves the band still negative and final KL
essentially unchanged (×1.02). So this is specific to those layers, not a
generic consequence of removing nine MLPs. §3/§6/§9 now rest on an intervention,
not an attribution.

**The unexpected part, which matters.** I predicted that removing a suppressor
would *raise* final KL. It does the opposite: band ablation drops final KL to
0.59× intact, and ablating all 64 layers drops it to 0.26×. `transition_z` is
therefore not a dedicated "mutation eraser" that the rest of the trunk works
around — it is load-bearing for the trunk's normal operation, and the growth of
the mutation signal across the stack *depends on it* even though it locally
subtracts.

Two consequences:

1. **Do not read this as "delete `transition_z` and the model becomes
   mutation-sensitive."** It becomes *less* differentiated overall. The
   all-layers condition in particular is a large off-distribution intervention
   and its final-KL number should not be interpreted mechanistically.
2. The honest statement is narrow and holds: *within layers 37–45,
   `transition_z` is causally responsible for the local reduction in the
   mutation's distributional footprint.* Whether that local reduction matters
   for the final prediction is a separate question this experiment does not
   answer — the ablation perturbs too much downstream to isolate it.

A cleaner follow-up would be a **scaling** intervention rather than deletion
(multiply `transition_z`'s contribution by α ∈ {0, 0.5, 1, 1.5} in the band and
watch KL and pLDDT move monotonically), which stays much closer to the
distribution the model was trained on.

---

## 11. Correction to §8: "the trunk amplifies the mutation" was an un-normalised number

Prompted by the obvious question — if the signal is amplified through the trunk,
why is the predicted structure unaffected? The answer is that it is not really
amplified. `exp_relative.py` adds the denominators §8 lacked, with a
**scrambled-sequence control** (same amino-acid composition, 5 % identity to WT,
same wild-type alignment) as the "how big is big" scale: it is the maximal
query-side perturbation available without touching the MSA.

| | `core_32` | `surface_32` | scramble |
|---|---:|---:|---:|
| raw KL, L0 → L63 | 0.163 → 0.927 (×5.7) | 0.068 → 0.222 (×3.3) | 1.459 → **6.316** |
| ‖z_wt‖, L0 → L63 | 285.5 → 282.8 (**×0.99, flat**) | — | — |
| relative ‖Δz‖/‖z‖, L0 → L63 | 0.150 → 0.334 | 0.084 → 0.133 | — |
| **KL as fraction of scramble**, L0 → L34 → L63 | **0.112 → 0.133 → 0.147** (×1.31) | **0.047 → 0.043 → 0.035** (×0.75) | 1.0 |

Two things follow.

**The raw ×5.7 growth is mostly not about the mutation.** The scramble control
grows ×4.3 over the same layers. Measured as a fraction of that, the core
mutant's footprint grows only ×1.31 across 64 layers, and the surface control's
*shrinks* (×0.75). Relative to the scale of query-driven variation the mutation
occupies a roughly **constant** share of the pair representation through the
trunk. `rel_dz` is flat at 0.15–0.20 for the first ~57 layers and only rises to
0.33 in the last handful.

**And the share is small.** Replacing 32 buried hydrophobics with charged
residues moves the pair representation about **15 %** as much as destroying the
sequence entirely. The surface control: 3.5 %.

So §8's "the trunk accumulates the mutation signal" is withdrawn in its strong
form. What is true: the trunk does not *suppress* it either (the §3 band is a
local ~1.4–1.8× dip that is recovered). The mutation's share is approximately
**conserved** through the Pairformer.

### This answers the structure question

The predicted structure barely moves because the mutation never becomes a large
perturbation at any point in the computation:

1. It enters small — ~11 % of a scrambled query at layer 0 — because the MSA
   write into `z` is ~3.3× the direct query write (§1), and the query's own MSA
   row contributes ~2 % (§2).
2. It stays small — the share is roughly conserved across all 64 layers.
3. What growth there is goes disproportionately into distribution *shape*, not
   *mean*: at the output, KL is 0.93 while mean |ΔE[d]| is 0.31 Å and only
   0.59 % of contacts flip. The structure module reads geometry, and the mean
   geometry is what barely changed.
4. Even the *maximal* query perturbation does not break the fold: the scramble
   control still yields a folded prediction, consistent with the published
   TM ≈ 0.33 for random-query-plus-wild-type-MSA runs. The MSA alone determines
   the fold to first order; the query modulates confidence.

**Methodological note, now the third instance of the same error in one session:**
first Ångström-vs-KL (units), then absolute-vs-relative (denominator). Both are
"a number that means nothing without something to divide by". Every future claim
of the form "X grows/shrinks through the stack" must ship with (a) a scale-free
measure and (b) a control that says how large the effect could possibly be.

### Caveat on run settings, and the real remaining gap

All of the above is at **3 recycles, distogram level**. Prior PyTorch-Boltz runs
at **15 recycles / 25 diffusion samples** showed TM staying very high for point
mutations and even ~20 % mutation. Both settings push *toward* invariance —
more recycling gives the MSA-derived prior more opportunity to reassert, and
ranking 25 samples by confidence selects the most canonical one. So real-world
invariance is if anything *stronger* than these trunk numbers imply, and the
trunk-level result understates it.

**The untested link in the causal chain is the diffusion module.** Everything
here is the distogram — a readout head. The structure is produced by
`AtomDiffusion` conditioned on `(s_trunk, z, s_inputs)`, and we have never
measured how much of the pair-representation difference survives into
coordinates. That is the next experiment: propagate the same mutant/WT pair
through `boltz2_forward_from_trunk` with matched noise and multiple samples, and
measure coordinate divergence against the trunk divergence that produced it.

---

## 12. The circuit as a matrix over layers

`exp_matrix.py` + `fig_matrix.py`. Divergence resolved per residue and per
sequence separation, in symmetric KL, using pair sampling **stratified by
residue** (80 partners each) so per-residue rows are unbiased. Figures:
`figures/matrix_gfp_{core_32,surface_32,scramble}.png`, four stacked panels on
a shared layer axis.

### What the panels say

**Residue × layer.** The divergence is not diffuse — it lives in a handful of
sharp horizontal bands. The strongest sit at residues ~65–90, exactly where the
buried-core substitutions are (C70E, F71K, F83K, F84R, A87K, M88R, V68R). Those
bands are present from **layer 0** and intensify monotonically. A second set of
bands (~110, ~150, ~200) is faint early and lights up only after ~L45 — these
are sequence-distant residues that are *spatially* adjacent in the barrel.

**Separation × layer.** Early layers spread the signal fairly evenly across
separations. Late layers concentrate it at **long range (48–95, ≥96)** and at
short range (3–5), with the 1–2 bin weakest throughout. So the Pairformer
converts a local sequence change into a long-range pair-structure change — which
is what the triangle operations are for.

**Operation × layer.** `transition_z` is a persistent blue (divergence-reducing)
band through most of the stack, densest in L37–45 as §9 found. Then at **L60–63
it flips strongly red** — the terminal sharpening region from §8(b), now visible
as a per-layer event rather than inferred from an entropy curve.

**Enrichment at mutated residues.** 3.41× at L0, peak 3.72× at L21, decaying to
**2.28×** by L63 (surface control: 5.20 → 2.05). So the mutation enters sharply
localised on the substituted positions and **delocalises with depth** — but
never fully; it remains >2× enriched at the trunk output.

### The circuit, end to end

1. **Entry (L0).** Query identity arrives almost entirely via `s_inputs → z_init`
   (§2, ~90 % of the causal effect); the query's own MSA row contributes ~2 %.
   The footprint is ~11 % of a scrambled query (§11) and 3.4× concentrated on
   the mutated residues.
2. **Propagation (L0–45).** Triangle multiplication and attention spread the
   signal from the mutated positions outward and from short to long sequence
   separation. `transition_z` trims it continuously; the net over these layers
   is amplification, but the mutation's *share* stays roughly constant (§11).
3. **Local reversal (L37–45).** `transition_z` briefly out-competes the triangle
   ops, producing the 1.4–1.8× dip. Causally confirmed by ablation (§10).
4. **Terminal sharpening (L56–63).** Distogram entropy collapses 2.16 → 0.81
   nats; `transition_z`'s per-layer contribution flips sign; enrichment falls
   to 2.3× as the signal delocalises. Divergence in KL rises steeply while
   divergence in Å falls — the artefact that misled §3.
5. **Output.** Large distributional change (KL 0.93), small mean-geometry change
   (0.31 Å, 0.59 % of contacts flipped). Expressed as uncertainty, not geometry.

### Structure-level, for the record (not pursued further)

`exp_diffusion.py` sampled K=8 structures per condition from a fixed trunk
state. The usable result is the confidence ordering, which is superposition-free
and clean: **WT 0.949, surface_32 0.930, core_32 0.829, scramble 0.279.**

The TM numbers from that run are **not usable** and should not be quoted. The
measured noise floor — mean pairwise TM among 8 samples of the *same* wild-type
trunk state — is 0.409 ± 0.313, which is both implausibly low and implausibly
variable for a model reporting pLDDT 0.949. Combined with the earlier finding
that our JAX wild-type prediction scores only TM 0.697 against the PyTorch
Boltz-2 prediction of the same sequence despite both having correct geometry
(CA–CA 3.80 Å, Rg 17.5 Å), the likeliest explanation is a defect in
`geom.tm_score`'s iterative superposition, not in the model. **Fix or replace it
with `tmtools` before any structure-level claim.** Focus stays on the distogram.

---

## 13. Propagation test: no contact-graph spreading, but a clean spatial *magnitude* effect

The matrix figure showed bands at residues ~110/~150/~200 appearing only after
~L45 — sequence-distant but plausibly space-adjacent to the mutated core. The
obvious hypothesis was propagation through the contact graph: onset depth should
track 3-D distance from the nearest mutated residue. `analyze_onset.py` defines
onset as the divergence-weighted mean layer (where a residue's divergence
actually accumulated) and tests it. **The hypothesis is refuted**, and the
reversed version is much weaker than it first looked.

| `core_32` | value |
|---|---:|
| ρ(onset, 3-D distance) raw | **−0.503** |
| ρ(onset, sequence distance) | −0.333 |
| partial ρ(onset, 3-D \| sequence) | −0.398 |
| **ρ(onset, total divergence)** | **+0.603** ← the confound |
| ρ(total divergence, 3-D distance) | −0.564 |
| **partial ρ(onset, 3-D \| total divergence)** | **−0.258** |
| ρ(onset, 3-D) using layers 0–55 only | **−0.155** |

The sign is *negative*: residues far from the mutation accumulate divergence
**earlier**, not later — the opposite of propagation. But most of that is an
artefact. Residues near the mutation diverge more, and larger divergences
accumulate later (ρ = +0.603), so controlling for magnitude cuts the spatial
correlation from −0.503 to −0.258; dropping the terminal sharpening layers cuts
it further to −0.155. Whatever timing structure exists is weak and lives almost
entirely in the last eight layers.

**What is robust is the magnitude, not the timing.**

| | ρ(total divergence, 3-D distance) |
|---|---:|
| `core_32` (buried) | **−0.564** |
| `surface_32` (exposed) | **+0.032** |

For buried mutations the model's response is **spatially localised in 3-D
around the mutation site** — divergence falls off cleanly with distance, and
this is a genuine structural statement, not a sequence-proximity artefact. For
count-matched surface mutations there is **no distance dependence whatsoever**.
That contrast is clean, needs no timing story, and is exactly the kind of thing
the model *should* do: a buried substitution perturbs its packing neighbourhood;
an exposed one perturbs nothing in particular.

Caveat on the shells: with 32 mutations spread through the core, 146 of the 206
non-mutated residues are within 8 Å of one, so the outer shells are thin
(n = 8 and n = 6). The whole-set Spearman is the number to quote, not the shell
means.

**Fourth instance of the same methodological pattern** (Å-vs-KL, absolute-vs-
relative, raw-vs-partial, now timing-vs-magnitude). Every spatial or temporal
claim in this project needs the corresponding nuisance variable partialled out
before it is believed. The controls are now built into `analyze_onset.py`.

## 14. Dose series — the localisation profile has a shape that is dose-independent

`runs/matrix_dose.json`, `figures/dose_enrichment.png`. Enrichment of KL on the
mutated residues, for the nested core series and the surface control:

| n mutated | L0 | peak (layer) | final L63 | final/L0 |
|---:|---:|---:|---:|---:|
| 1 | 53.1 | 57.5 (L10) | 8.4 | 0.16 |
| 2 | 32.0 | 45.1 (L34) | 11.0 | 0.34 |
| 4 | 13.3 | 18.5 (L43) | 6.6 | 0.49 |
| 8 | 6.6 | 9.9 (L42) | 6.0 | 0.91 |
| 16 | 5.0 | 6.3 (L42) | 3.6 | 0.72 |
| 32 | 3.4 | 3.7 (L21) | 2.3 | 0.67 |
| surface 32 | 5.2 | 5.3 (L36) | 2.1 | 0.40 |

The absolute enrichment falls with n, but that is mostly a denominator effect —
with 32 mutations most of the protein is near one, so the "non-mutated" baseline
is contaminated. **The comparable quantity is the shape**, and normalised to
layer 0 it is strikingly consistent:

- **Every core curve rises to a peak at ~L43** (1.2–1.5× its layer-0 value)
  **then collapses over the last 20 layers**, converging to 0.35–0.9. A single
  point mutation and 32 of them trace the same profile.
- The inflection at **L43–45** is the same boundary identified independently in
  §9/§10 (end of the `transition_z`-dominated band) and §8 (start of the
  terminal sharpening).
- **The surface control never rises above 1.0.** It sits at or below its
  layer-0 localisation for the whole stack and falls earliest and furthest
  (0.40).

So: through the mid-stack the model **progressively focuses** on buried
mutations — and does not do this for exposed ones — then delocalises sharply in
the last 20 layers regardless. That focusing/defocusing profile is a property of
the network (same shape at every dose), and the discrimination between buried
and exposed is real.

Combined with §13, the spatial picture is: the response to a buried mutation is
localised in 3-D around the site (ρ = −0.56 of divergence with distance, vs
+0.03 for surface), the model sharpens that focus until ~L43, then spreads it.

---

## 15. tmtools — a large correction, and the result it unlocks

`geom.tm_score` was hand-rolled and **badly wrong**. Against tmtools on
identical coordinates:

| comparison | old hand-rolled | **tmtools** |
|---|---:|---:|
| JAX WT vs independent PyTorch Boltz-2 WT | 0.697 | **0.978** (RMSD 1.10 Å) |
| core n=1 vs WT | 0.218 | **0.977** |
| core n=32 vs WT | 0.353 | **0.935** |
| shuffled-residue floor | — | 0.155 |
| within-WT, 3 diffusion samples | 0.409 ± 0.313 | **0.980** |

**§5's "sampler-noise-limited" conclusion is withdrawn.** There was never a
sampling problem; the diffusion is highly reproducible (within-WT TM 0.980).
`geom.py` is rewritten around tmtools with this failure recorded in its
docstring.

## 16. Structure vs confidence vs internal state, measured together

`exp_bench.py` + `score_bench.py`, GFP, all on the same variants and settings.
Adds a **random**-mutation series at 5/10/20/40/70 % — the levels used in the
adversarial-mutation literature — because the targeted core series is only
13.4 % mutated and so was never comparable to a random-mutation rate.

| variant | % mut | TM | pLDDT | KL |
|---|---:|---:|---:|---:|
| WT | 0 | 0.9865 | 0.950 | 0 |
| surface ×1 | 0.4 | 0.9851 | 0.950 | **0.0003** |
| core ×1 | 0.4 | 0.9828 | 0.940 | 0.0350 |
| core ×8 | 3.4 | 0.9541 | 0.833 | 0.586 |
| core ×32 | 13.4 | 0.9298 | 0.831 | 0.926 |
| surface ×32 | 13.4 | 0.9702 | 0.934 | 0.221 |
| random 20 % | 20.2 | 0.9442 | 0.885 | 0.732 |
| random 40 % | 39.9 | 0.8975 | 0.802 | 1.776 |
| random 70 % | 70.2 | 0.7342 | 0.444 | 3.724 |
| scramble | 95 | 0.3221 | 0.279 | 6.318 |

### The headline

**At the paper's own comparison — 20 % vs 40 % random mutation:**

| measure | 20 % → 40 % | change |
|---|---|---:|
| TM-score | 0.944 → 0.898 | **+4.9 %** |
| pLDDT | 0.885 → 0.802 | +9.3 % |
| **internal KL** | 0.732 → 1.776 | **+142.7 %** |

Doubling the mutation load moves the predicted structure by 5 % and the internal
pair representation by 143 %. This reproduces the literature result (TM barely
separates 20 from 40) *and* explains it: the information is there, the geometry
readout does not express it.

Dynamic range across the random series, relative to each measure's own noise:

| measure | span | span / noise | Spearman with % mut |
|---|---:|---:|---:|
| TM | 0.239 | 12× | −0.900 |
| pLDDT | 0.483 | 522× | −0.900 |
| **internal KL** | 3.440 | **3440×** | **+0.900** |

All three rank variants equally well (|ρ| = 0.9); they differ by ~300× in
resolution. That is the quantitative basis for using internal state as a
mutation-effect regressor rather than the model's own outputs.

### Targeting matters — the objection was right

- **core ×32 (13.4 % mutated) is more damaging than random 20 %** on all three
  measures (TM 0.930 vs 0.944, pLDDT 0.831 vs 0.885, KL 0.926 vs 0.732). So the
  earlier core-series dose-response could not be compared to a random-mutation
  rate, and any claim that did so is void.
- At **identical count** (32), core vs surface: TM deviation ×3.5, pLDDT
  deviation ×7.6, KL ×4.2. Burial, not count, drives the response.
- **A single surface mutation is invisible**: KL = 0.0003, pLDDT unchanged at
  0.950, TM inside the noise band. A single core mutation is ~100× larger in KL
  (0.035) while still invisible in TM. The internal state resolves single
  substitutions that the output cannot.

## 17. Replication on a second protein, and the head question

**DIO3** (237 aa, unrelated fold, 3411-deep MSA). Route decomposition, endpoint
checks again exact (0.00e+00 both directions):

| route | GFP core ×32 | DIO3 core ×32 |
|---|---:|---:|
| `z_direct` | 0.879 | 0.830 |
| `msa_query` | 0.023 | **0.020** |
| `msa_prior`, `s_direct` | 0.000 | 0.000 |

The central claim — the query's own MSA row carries ~2 %, the direct projection
carries the rest — is not a GFP artefact.

**Heads (GFP, `exp_heads.py`).** Ablating all of `tri_att_start` halves the
final KL (×0.485); `tri_att_end` is weaker (×0.731). Per individual head the
effects are small and even (×0.84 to ×1.10; one head *raises* KL). Individual
head effects sum to far less than the whole-module effect. **The spreading is
distributed and redundant — there is no "mutation head".** A useful negative:
do not go looking for one.

## Next

- 8-protein panel MSAs are done (`data/panel/colabfold_output/`): ubiquitin,
  lysozyme, myoglobin, cheY, thioredoxin, barnase, snase, acylphos — spanning
  all-α, all-β, α+β, α/β. With GFP and DIO3 that is the 10-protein set.
- **ProteinGym regression** is now justified by §16: predict DMS fitness from
  internal KL vs pLDDT vs TM on the same variants. Zenodo is reachable from the
  login node.

---

## 18. ProteinGym probe — design, and two methodology bugs caught before the data

Running: four **Tsuboyama 2023** folding-stability assays (RCRO_LAMBD 63 aa,
RS15_GEOSE 63 aa, NKX31_HUMAN 61 aa, PSAE_PICP2 68 aa), 400 single mutants each,
MSAs built through the ProtForge container (827–14,243 rows). Tsuboyama is
chosen deliberately: it measures ΔG of folding, which is the quantity this
project is actually about, rather than a functional proxy.

Per variant, per Pairformer layer, four feature families: global KL, KL at pairs
touching the mutated site, ‖Δz‖ at that site, ‖Δs‖ at that site. 256 features.

**Split is grouped by residue position, not random.** A random split puts other
substitutions at the same site in both train and test, and position identity
alone carries much of the stability signal (buried sites are intolerant to
everything), so a random split flatters every model and hides whether the
features generalise to unseen sites. A position-only predictor is reported as
the baseline that a random split would effectively be leaking.

A synthetic check with a **planted** signal (informative feature at `kl_site`
layer 40, plus a position effect, plus 254 noise features) caught two bugs that
would have produced wrong headline numbers:

1. **Multi-feature models lost to the single best feature** (ridge ρ 0.541, MLP
   0.186, vs 0.843 for the planted feature) — pure overfitting of 256 features
   to ~284 rows with a fixed λ. Fixed by tuning λ, feature count k, and MLP
   width on an **inner position-grouped validation split of the training rows**.
   Now ridge 0.844, MLP 0.765 — and the MLP correctly does *not* beat ridge on
   data whose true relationship is linear, which is the right behaviour.
2. **The "best layer" was selected on the test set** — argmax over 256
   candidates evaluated on test. Fixed: the layer is chosen on train, its score
   reported on test. The effect is visible in the synthetic control: the two
   pure-noise blocks dropped from an inflated 0.218/0.239 to 0.058/0.120, while
   the planted signal stayed at 0.843.

Both are the same failure mode as §8/§11/§13 — a number without the control that
makes it meaningful. The difference is that this time the check ran *before* the
real data, not after a conclusion had been drawn.

**Known inefficiency for future scaling:** ~20 s/variant, dominated by Boltz
re-parsing the a3m in `process_inputs` on every variant (RCRO at 827 rows runs
~2× faster than NKX31 at 14,243). Truncating the grafted alignment to ~2048 rows
once, or featurising the wild type once and patching only the sequence-dependent
tensors, would cut this substantially. Not worth restarting the current run for.

---

## 19. ProteinGym results — internal state predicts folding stability

Four Tsuboyama 2023 stability assays, 400 single mutants each, **5 independent
position-grouped splits** (no residue in both train and test). Spearman on
held-out positions, mean ± sd across splits.

| assay | best single feature | linear (all layers) | small MLP | `ds_site` (single rep) | position-only |
|---|---:|---:|---:|---:|---:|
| NKX31 | 0.434 ± 0.093 | **0.488 ± 0.099** | 0.412 ± 0.117 | 0.250 | 0.068 |
| PSAE | 0.412 ± 0.091 | **0.444 ± 0.068** | 0.444 ± 0.087 | 0.234 | −0.104 |
| RCRO | 0.631 ± 0.063 | 0.634 ± 0.055 | 0.612 ± 0.047 | 0.303 | 0.230 |
| RS15 | 0.402 ± 0.127 | **0.618 ± 0.091** | 0.392 ± 0.147 | 0.311 | −0.077 |
| **all** | 0.470 ± 0.134 | **0.546 ± 0.114** | 0.465 ± 0.137 | 0.275 ± 0.103 | 0.029 ± 0.202 |

### Findings

1. **The trunk's internal state predicts measured ΔG at ρ ≈ 0.55** on unseen
   positions, against a position-only baseline of 0.03. The signal is about the
   substitution, not the site.
2. **Combining layers helps**: linear beats the best single feature in
   **14/20** assay-splits, mean gain **+0.076**. On RS15 it is +0.22.
3. **Nonlinearity does not.** The MLP beats linear in only **4/20**, mean gain
   **−0.081**. Tuned identically, on the same inner validation splits. The
   relationship between internal divergence and stability is essentially
   linear-in-features once several layers are combined.
4. **Layer choice is not the story.** The best single layer has median L58 but
   ranges L38–L63 across assays and splits, and is frequently no better than the
   final layer. Multi-layer *combination* is what adds, not picking one layer.
5. **The single representation is a bystander.** `ds_site` reaches only 0.275
   against ~0.55 for pair-representation features. Consistent with the
   architecture: in Boltz-2's Pairformer `s` never writes back into `z`, so the
   mutation-effect signal lives in the pair track.

### Correction to what I reported after the first assay

From RCRO alone I concluded "combining layers doesn't help, the signal is
one-dimensional". That was **wrong** — RCRO is the one assay of four where
linear ≈ best-single (0.634 vs 0.631). On the other three, and across 20
assay-splits, linear clearly wins. The MLP-doesn't-help half of that claim did
survive. Reporting from n=1 assay was the error, not the analysis.

### Practical read

For predicting mutational effect from Boltz-2, use a **linear readout over
per-layer pair-representation divergences** — not a single layer, not the final
layer alone, and not a nonlinear probe. ρ ≈ 0.55 without any training on
structures and without running diffusion.

---

## 20. Internal state vs the model's own outputs — like-for-like, and it is decisive

`compare_internal_output.py` on the gym2 runs, which capture per-layer internal
features, the predicted structure and pLDDT for the **same** 250 variants, so
this is the only fully matched comparison available. Spearman on held-out
positions, mean ± sd over 5 position-grouped splits, identical rows for every
predictor.

| assay | internal (linear) | TM to WT | pLDDT at site | pLDDT mean | position baseline |
|---|---:|---:|---:|---:|---:|
| RCRO | **+0.634 ± 0.044** | +0.280 ± 0.127 | +0.074 | +0.104 | +0.191 |
| NKX31 | **+0.490 ± 0.117** | +0.143 ± 0.153 | +0.085 | +0.174 | +0.127 |

**The trunk carries 2.3–3.4× more mutation-effect information than the predicted
structure**, and pLDDT is at or below the position baseline — for single
substitutions the model's own confidence is close to useless, which is a
sharper statement than the GFP cohort supported (there the mutations were 32 at
a time).

An earlier, non-comparable version of this claim used the probe (400 variants,
held-out) against the ensemble run (different 120-variant draw, scored
in-sample). Those differ in both the mutant set and the protocol; only 44
mutants overlapped. The table above supersedes it.

## 21. Ensemble-spread hypothesis — refuted

`exp_ensemble.py` / `analyze_ensemble.py`, K = 6 structures per variant from the
same deterministic trunk state, β = 1.

| | RCRO | RS15 |
|---|---:|---:|
| WT ensemble spread (mean pairwise TM) | 0.9900 | 0.9797 |
| variant ensemble spread | 0.9867 | 0.9789 |
| ρ(ensemble spread, ΔG) raw / partial | +0.161 / +0.153 | +0.205 / +0.244 |
| **ρ(TM to WT, ΔG) raw / partial** | **+0.435 / +0.427** | **+0.489 / +0.508** |
| ρ(pLDDT, ΔG) | +0.037 / +0.031 | +0.216 / +0.242 |

I predicted the mutation would show up as a *widened* ensemble rather than a
shifted mean, by analogy with the distogram result where KL moved 143 % and mean
E[d] moved 5 %. **That analogy does not carry to coordinates.** The mean
structure's displacement from wild type predicts stability 2–3× better than the
ensemble's width does. The shape-not-location pattern is real at the distogram
and false at the structure module.

Ensembles are also extremely tight: WT pairwise TM 0.98–0.99 across six noise
draws, variants barely lower. There is almost no ensemble to carry information
at stock settings — which is the confound Boltz-sample's β was added to test.

## 22. RSA — the geometry match is in the trunk, not downstream

`analyze_rsa.py`, 110 variants, ~6000 variant pairs, partialled for
same-position, BLOSUM62 distance and score magnitude.

| | NKX31 | RCRO |
|---|---:|---:|
| RSA(Pairformer best layer, EXP) | **+0.169** (L3) | **+0.199** (L63) |
| RSA(Pairformer final layer, EXP) | +0.156 | +0.199 |
| RSA(distogram, EXP) | −0.004 | +0.093 |
| RSA(predicted structure, EXP) | +0.006 | +0.029 |
| RSA(pLDDT at site, EXP) | +0.060 | +0.047 |
| RSA(Pairformer, STRUCT) | −0.001 | −0.086 |
| mean variant-to-variant TM | 0.9713 | 0.9812 |

The Pairformer's notion of "these two mutations are alike" matches the
experiment's weakly but consistently (≈ 0.17–0.20 after controls). The
distogram's and the structure's match it **not at all**, and the Pairformer's
variant geometry has essentially **no** relationship to the structure's
(RSA(PF, STRUCT) ≈ 0). All 110 mutants of a 61-residue protein predict to within
TM 0.97 of each other.

Note these are second-order (pairwise-similarity) quantities after three
partials, so they are not on the same scale as the first-order predictions in
§20 and should not be compared to them numerically.

## 23. Where the information is lost — it is the sampler, not the conditioning

The diffusion conditioning tensors differ substantially between mutant and wild
type. Relative difference ‖Δ‖/‖WT‖, averaged over 120 variants:

| tensor | RCRO | RS15 |
|---|---:|---:|
| `q` | **0.285** | **0.286** |
| `c` | 0.089 | 0.083 |
| `token_trans_bias` | 0.046 | 0.044 |

So the chain is: the Pairformer carries the mutation (§20, ρ ≈ 0.49–0.63) → the
diffusion conditioning still carries it (`q` differs by ~28 %) → the sampled
structure does not express it (TM 0.98 between all variants, ρ ≈ 0.14–0.28).

**The loss is downstream of the conditioning projection, in the diffusion
sampler itself.** That answers the question posed in the plan — "conditioning
discards" vs "sampler contracts" — in favour of the sampler.

*Calibration caveat:* 28 % is a relative magnitude with no reference scale yet.
The scrambled-sequence control needs running through the same measurement to say
whether 28 % is large or small relative to a maximal query perturbation.

This is exactly the stage Boltz-sample's β operates on, which is why the β ∈
{1.5, 2.0} runs are the right next read: if mutation information appears in the
output only once the sampler is widened, the sampler's default dispersion is the
mechanism.

### 20b. Third assay, and the pooled statistic

RS15 added. Spearman on held-out positions, 3 assays × 5 position-grouped
splits = 15 assay-splits, identical rows for every predictor:

| predictor | pooled mean ± sd |
|---|---:|
| **internal (linear over per-layer pair features)** | **+0.506 ± 0.150** |
| TM to wild type | +0.233 ± 0.139 |
| pLDDT mean | +0.174 ± 0.176 |
| pLDDT at mutated residue | +0.079 ± 0.184 |
| position-only baseline | +0.045 ± 0.226 |

- internal > TM-to-WT in **14/15** assay-splits, mean gain **+0.273**,
  95 % paired bootstrap CI **[+0.192, +0.353]**.
- Per assay the ratio is **2.3× (RCRO), 3.4× (NKX31), 1.4× (RS15)** — consistent
  in direction, variable in size. RS15 is the weakest case and should be quoted
  whenever the headline is.
- pLDDT *at the mutated residue* is the worst predictor of all (+0.079), below
  the chain mean (+0.174) and barely above the position baseline. For single
  substitutions the confidence head carries almost nothing, and localising it to
  the mutated site makes it worse rather than better.

RSA replicates in direction on the third assay too (Pairformer best +0.094,
distogram −0.023, structure −0.046) though weaker than RCRO/NKX31. Pairformer >
downstream in **3/3** assays; mean variant-to-variant TM 0.971–0.986 in all
three.

### 20c / 22b. All four assays

**Internal vs output**, 4 assays × 5 position-grouped splits = 20 assay-splits,
identical rows for every predictor:

| predictor | pooled mean ± sd |
|---|---:|
| **internal (linear, per-layer pair features)** | **+0.480 ± 0.165** |
| TM to wild type | +0.191 ± 0.141 |
| pLDDT mean | +0.178 ± 0.168 |
| **pLDDT at mutated residue** | **−0.006 ± 0.237** |
| position-only baseline | +0.019 ± 0.212 |

internal > TM-to-WT in **19/20** assay-splits; gap **+0.289**, 95 % paired
bootstrap CI **[+0.212, +0.366]**. Per-assay ratio 1.4× (RS15), 2.3× (RCRO),
3.4× (NKX31), 6.0× (PSAE).

**pLDDT at the mutated residue pools to −0.006** — indistinguishable from zero,
and *worse* than the chain mean (+0.178). PSAE drives this hardest (−0.261).
Localising the confidence head to the substituted residue does not help; it
destroys what little signal the chain average had. For single substitutions the
confidence head is not a usable readout of stability.

**RSA**, same four assays (110 variants, ~6000 pairs, partialled for
same-position / BLOSUM62 / score magnitude):

| assay | PF best | PF final | distogram | structure | pLDDT | var-var TM |
|---|---:|---:|---:|---:|---:|---:|
| NKX31 | +0.169 | +0.156 | −0.004 | +0.006 | +0.060 | 0.9713 |
| PSAE | +0.091 | +0.084 | +0.024 | +0.012 | +0.060 | 0.9111 |
| RCRO | +0.199 | +0.199 | +0.093 | +0.029 | +0.047 | 0.9812 |
| RS15 | +0.094 | +0.079 | −0.023 | −0.046 | −0.012 | 0.9856 |
| **mean** | **+0.138** | +0.130 | +0.023 | **+0.000** | +0.039 | 0.962 |

Pairformer > distogram and Pairformer > structure in **4/4**. The structure's
variant geometry matches the experiment's at a pooled **+0.000**.

Magnitudes are modest (PF ≈ 0.09–0.20) and these are second-order quantities
after three partials, so they are not comparable to the first-order numbers
above. The direction is what replicates, not the size.

---

## 24. β scaling — an invalid test, caught by its own diagnostic

The β sweep launched to test "does widening the sampler let the trunk's
information through" returned **no effect** at β ∈ {1.0, 1.5, 2.0}: WT ensemble
spread 0.9900 → 0.9908 → 0.9911 (marginally *tighter*), ρ(TM to WT, ΔG) 0.435 →
0.435 → 0.444, ρ(spread, ΔG) 0.161 → 0.132 → 0.159.

**This is not a null result.** The conditioning difference ‖Δq‖/‖q_wt‖ came out
as 0.2847031 / 0.2847031 / 0.2847034 — identical to seven decimal places across
a 2× scale. An intervention that leaves the conditioning bit-identical never
reached the sampler.

Cause: `PairwiseConditioning.__call__` opens with
`nn.LayerNorm(concat([z_trunk, token_rel_pos_feats]))`, which absorbs a global
scale on `z_trunk`. Scaling there is very nearly inert by construction.

Relaunched scaling the three **pair-derived attention biases**
(`atom_enc_bias`, `atom_dec_bias`, `token_trans_bias`) that
`DiffusionConditioning` returns — added to attention logits inside the diffusion
transformer, downstream of that LayerNorm, and the quantity that actually sets
how strongly pairwise information competes with noise. Implemented as a
`ScaledConditioning` wrapper via `eqx.tree_at` so the library's forward runs
unchanged. Method and the required pre-check are in
`methods_pairformer_interp.md` §4.13.

**Standing diagnostic:** before believing any β result, confirm the conditioning
tensors differ across β. If ‖Δq‖/‖q‖ is unchanged, the knob is not connected.

This also matters for reading Boltz-sample: if their scalar is applied to
`z_trunk` rather than downstream of `PairwiseConditioning`'s LayerNorm, the same
absorption applies, and where exactly they insert it is worth checking in the
full text (bioRxiv 403s the PDF from the cluster).

## 25. Expanding N

Eight further Tsuboyama stability assays selected (CBPA2, CSN4, CATR, ILF3,
NUSA, PKN1, ARGR, HECD1; 69–72 aa, 1244–1357 single mutants each), one per
source protein so folds are not duplicated. With the existing four that is
**12 assays**. MSAs generating via the ProtForge container. This is the fix for
the largest weakness in §20–22: effect sizes span 1.4×–6.0× across four assays
and four is not enough to say whether that is fold-dependent or noise.

---

## 26. N = 12 — the main result holds and tightens

Twelve Tsuboyama folding-stability assays, 250 variants each, Spearman on
held-out **positions**, 5 position-grouped splits per assay = **60 assay-splits**.
Identical rows for every predictor.

| predictor | pooled mean ± sd |
|---|---:|
| **internal (linear over per-layer pair features)** | **+0.548 ± 0.169** |
| pLDDT mean | +0.244 ± 0.200 |
| TM to wild type | +0.214 ± 0.173 |
| pLDDT at mutated residue | +0.037 ± 0.227 |
| position-only baseline | +0.069 ± 0.186 |

internal > TM-to-WT in **57/60** assay-splits; gap **+0.335**, 95 % paired
bootstrap CI **[+0.288, +0.380]**. Per-assay internal ρ ranges **0.392–0.704**
(mean 0.548) — far tighter across twelve proteins than the four-assay sample
suggested, which was the main worry.

Two changes from the N=4 numbers worth noting:

- **pLDDT mean (+0.244) now slightly beats TM-to-WT (+0.214).** The model's
  confidence is a marginally better stability readout than its own geometry.
  Both remain less than half the internal state.
- **Stop quoting the ratio.** It ranges 1.3× (NUSA) to 28.3× (PKN1), but PKN1's
  TM correlation is 0.017, so the ratio is a near-zero denominator artefact. The
  **gap** (+0.335, CI [+0.288, +0.380]) is the stable statistic and is what
  should be reported.

## 27. β on the bias path — the knob works, and the hypothesis is refuted

RCRO, 120 variants × K=6 samples, β applied to the pair-derived attention biases.

| β | WT ensemble spread | ρ(spread, ΔG) | ρ(TM to WT, ΔG) | ρ(pLDDT, ΔG) | WT pLDDT |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.9900 | +0.161 | **+0.435** | +0.037 | 0.858 |
| 1.5 | 0.8031 | +0.208 | +0.294 | +0.139 | 0.798 |
| 2.0 | 0.6139 | +0.156 | +0.283 | +0.208 | 0.736 |
| 3.0 | 0.3032 | −0.139 | +0.018 | **+0.299** | 0.569 |

**The intervention is unambiguously connected**: wild-type ensemble spread goes
from 0.990 to 0.303 in mean pairwise TM — the sampler is enormously widened.

**But widening does not liberate the trunk's information.** ρ(TM to WT, ΔG)
*degrades monotonically*, 0.435 → 0.018. The structural stability signal is
destroyed, not released. Ensemble spread does not improve either (0.161 → 0.208
→ 0.156 → −0.139).

So the hypothesis that **the sampler's default narrowness is what hides the
mutation** is **refuted** in the range tested. Widening the pairwise bias just
adds noise to the geometry.

One genuinely interesting counter-trend: **ρ(pLDDT, ΔG) rises monotonically with
β** (0.037 → 0.299). As the sampler widens, the confidence head becomes a
*better* stability readout — presumably because destabilising variants generate
more disagreement among samples, which pLDDT registers. That is a usable
observation in its own right, and it is the one thing β bought.

*Caveat:* β = 1.5 already drops WT pLDDT from 0.858 to 0.798 and ensemble spread
to 0.80, so the model is off-distribution quickly. A gentler grid (1.05, 1.1,
1.2) could in principle show a peak before the degradation; the trend from 1.0
is monotone down, so it is unlikely but untested.

## 28. Correction to the β diagnostic in the methods

The check prescribed in `methods_pairformer_interp.md` §4.13 — "confirm
‖Δq‖/‖q_wt‖ changes across β" — is **wrong, and I wrote it into the pushed
methods document**. ‖Δq‖/‖q_wt‖ is a relative difference, so a global scale
cancels exactly; it read 0.284703 for every β while the intervention was
working. It also reads the unscaled conditioning, since `cond` is computed from
the unwrapped model.

Replaced with scale-sensitive checks (wild-type pLDDT, wild-type ensemble
spread), both of which move hugely and monotonically. General rule now recorded:
*a diagnostic must be sensitive to the quantity the intervention changes* — and
that has to be verified before the diagnostic is trusted, not after.

---

## 29. Denoising trajectory — inconclusive, and why

`exp_trajectory.py`. The `AtomDiffusion2.sample` scan discards `ys` exactly like
the Pairformer, so the trajectory is captured the same way (re-run the scan with
`ys` populated, dynamics bit-identical). 200 steps, CA only, 24 variants per
assay chosen as the 12 most and 12 least destabilising.

**The raw divergence curve is uninformative.** Mean mutant-vs-WT CA RMSD tracks
the noise schedule and nothing else: 2605 Å at σ=2560 down to 8.75 Å at
σ=0.002 (RCRO); 2787 → 8.16 Å (RS15). That is the σ envelope, not the mutation.

**The cross-run comparison is not properly paired.** A substitution changes the
side chain, so the all-atom count differs between wild type and mutant (492 vs
491 for RCRO). `shape = (*atom_mask.shape, 3)` therefore differs, so
`jax.random.normal(shape=shape, key=k)` draws a **different noise realisation**
even from an identical key. The two trajectories are independent samples, not a
paired comparison. Corroborating sign: final mutant-vs-WT RMSD is 8.2–8.7 Å,
whereas the ensemble runs put variant-vs-WT at TM ≈ 0.985 (~1 Å) — the gap is
the unpaired noise.

**The rank statistic across variants does not replicate.** Spearman(divergence,
ΔG) at fixed step, which divides out the common σ:

| | step 0 (σ 2560) | step 120 (σ 14) | final (σ 0.002) | peak |
|---|---:|---:|---:|---:|
| RCRO | +0.479 | +0.321 | **−0.538** | +0.608 (step 23) |
| RS15 | +0.136 | +0.281 | +0.043 | +0.437 (step 34) |

The two assays disagree in magnitude and, at the final step, in **sign**. With
n = 24 the standard error on a Spearman is ≈ 0.21, so almost none of this is
distinguishable from zero. The within-run convergence curves (each trajectory vs
its own endpoint, which needs no cross-run pairing) disagree just as badly:
ρ(convergence, ΔG) at step 160 is +0.483 for RCRO and +0.250 for RS15, and both
collapse to ≈ 0 at the final step.

**Confound checked and cleared, which makes it worse not better.** Substitutions
to larger residues change the atom count *and* tend to destabilise, so atom count
could have driven everything. It does not: ρ(Δatom count, ΔG) = +0.091 (RCRO) /
+0.606 (RS15), and partialling it out leaves the step-0 correlation essentially
unchanged (+0.479 → +0.490 for RCRO). So the RCRO step-0 correlation at σ=2560
is not an atom-count artefact — but it is also not interpretable, because at that
noise level there is nothing for it to mean. That points at n=24 noise rather
than signal.

**Verdict: no conclusion drawn.** The experiment needs redesigning before it can
answer "never grown" vs "grown then lost":

1. **Pair the noise.** The comparison must use the same realisation. Either draw
   noise once at a fixed maximum size and index it consistently, or — simpler
   and assumption-free — add a **within-variant noise floor**: run each variant
   twice with different keys and report mutant-vs-WT divergence *relative to*
   same-variant-different-key divergence. Any claim needs that denominator.
2. **n = 24 is far too small** for a rank statistic. 100+ variants per assay.
3. **Report at matched σ**, not matched step index, if sampling-step counts ever
   differ between runs.

Cost of the corrected version is roughly 5× what was just spent, which is
affordable but should not be spent until the pairing is fixed — an unpaired
experiment at n=100 is still unpaired.

---

## 2026-08-01 — trajectory, run twice; and a second superposition bug

### The bug first, because it invalidated a whole run

The corrected trajectory design (within-sequence noise floor, n=100) ran and
returned a clean, plausible answer: divergence *below* the WT key-to-key floor
at essentially every step, ratio 0.5–0.9, ρ(divergence, ΔG) never above 0.3.
Every number was wrong.

`exp_trajectory.py` had defined its **own** `kabsch_rmsd` instead of importing
one, with the rotation transposed — `R = U diag Vᵀ` where Kabsch gives
`R = V diag Uᵀ` — so it applied the inverse rotation. Tested on an input with a
known answer it scored two structures **identical up to a rigid motion at
12.68 Å instead of 0**. Both numerator and denominator were wrong in the same
direction, which is precisely why the ratio looked stable.

Caught by comparing `kabsch_rmsd` against `tmtools` on the GFP cohort, where
they should agree when TM-align aligns all residues, and they did not.

**This is the same mistake as the hand-rolled TM-score, made again after that
one was documented in this very file.** The corrected floor is 0.3–0.7 Å where
the broken one said 13.4 Å.

Fix is structural, not local: `geom.py` is the single copy of every
superposition primitive, `geom.self_test()` runs **on import** and asserts a
rigid motion gives exactly 0, a reflection does not, a pure translation gives 0.
`exp_trajectory.py` now imports it and also saves endpoint coordinates so
re-analysis never needs the GPU again. Invalid files kept under
`runs/_invalid/` with a note rather than deleted.

**Generalisable lesson.** Both times the broken function returned numbers in the
right units, with the right sign, ordered the way intuition expected. Nothing
about the output said "wrong". Writing the lesson down did not prevent the
repeat; only an executable check did.

### The actual result (`runs/traj3_*.npz`, 3 assays × 100 variants)

| assay | ratio σ>10 Å | ratio σ<1 Å | endpoint ρ | p |
|---|---|---|---|---|
| RCRO_LAMBD | 0.68 | **1.03** | **−0.380** | 1e−4 |
| RS15_GEOSE | 0.63 | 0.86 | −0.060 | 0.55 |
| NKX31_HUMAN | 0.62 | 0.53 | −0.003 | 0.98 |

**Robust across all three:** during global fold determination (σ 2560 → ~1 Å,
80 % of the schedule) the ratio is flat at ~0.6. The mutant's path is *closer*
to the WT's than two WT runs are to each other. The conditioning difference does
not steer the trajectory while the fold is being chosen.

**Varies by protein:** below σ ≈ 1 Å the curves separate (1.08 / 0.89 / 0.52).
A mutation-specific response exists but is confined to fine refinement, after
the fold is committed.

**Limits on the RCRO number, which must travel with it:** (1) ρ = −0.380 is
in-sample over 100 variants, not held-out, so it is *not* comparable to the
probe's 0.548; (2) endpoint divergence is essentially "structure differs from
WT", i.e. a re-measurement of the TM readout (+0.214 pooled), not a new channel;
(3) 1 of 3 is weak. Do **not** quote max-over-steps (0.435) — that is an argmax
over 200 candidates on the reported quantity, the same select-on-test error
already caught in the probe.

**Effect on the account.** "The sampler discards it" is too strong. Better: the
sampler is **insensitive to the conditioning while the global fold is chosen**,
and becomes mutation-sensitive only during local refinement, by which point the
fold is committed. Consistent with the β result (widening adds noise rather than
releasing information) and with flat pLDDT-at-site.

### Also landed

- **Route decomposition across all 12 assays** (24 protein × variant obs):
  median `z_direct` necessity 0.840, sufficiency 0.995; all MSA routes ≈ 0
  (`msa_bcast` sufficiency 0.160 the only non-trivial competitor). The GFP
  result is not a property of GFP. → `figures/routes12.png`
- **Mechanism figure** from `runs/distomap_gfp.npz`: WT and mutant E[d] maps,
  their difference, the symmetric-KL map, TM-aligned structure overlay, and the
  64-bin histogram behind the worst single pair. GFP + 32 core mutations moves
  individual pairs by up to **7.6 Å** in E[d] and 0.92 nats mean KL, and the two
  structures still superimpose at **TM 0.935**. → `figures/mechanism.png`
- Report renumbered contiguously (§2.1–2.6, §4.0–4.11) and rewritten per review:
  probe input data and ρ formula stated explicitly, "assay-split" defined,
  spatial-localisation measurement spelled out, RSA explained with a warning not
  to compare its 0.138 against the probe's 0.548 (second- vs first-order).
