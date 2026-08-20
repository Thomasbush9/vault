# Paper outline — reframed around the mechanism

Written 2026-08-01. Supersedes the MSA-memorisation framing. Companion to
`publication_plan.md` (what is still missing) and `methods_pairformer_interp.md`
(how everything was measured).

**The one-line change:** the MSA is no longer the subject. It is a *result* — a
ruled-out explanation, reported in one paragraph because it is the field's
default assumption and we have the intervention data to reject it.

---

## Title options

1. **Folding models represent mutations they do not express**
2. **The trunk knows: where Boltz-2 loses mutational information between
   representation and structure**
3. **Structural invariance without representational invariance in a protein
   folding model**

(1) is the thesis in five words. (3) is the most precise and the most boring.
Prefer (1), with (3) as the first sentence of the abstract.

---

## Abstract (draft)

Protein folding models predict nearly identical structures for a wild-type
sequence and its point mutants, even when the mutation is strongly
destabilising. This is usually read as the model failing to *register* the
mutation — a memorisation or alignment-retrieval effect. We show the opposite.
Using an exact JAX reimplementation of Boltz-2 that permits bit-identical
activation capture and causal intervention, we find that the Pairformer trunk
represents mutational effect well: a linear readout of per-layer pair-
representation divergence predicts measured folding ΔG at ρ = 0.548 on held-out
residue positions across 12 deep mutational scans, while the model's own
structural output reaches ρ = 0.214 and its per-residue confidence at the
mutated site ρ = 0.037 — on identical variants under an identical protocol
(internal wins in 57 of 60 assay-splits; gap +0.335, 95 % CI [+0.288, +0.380]).
Route decomposition localises where the mutation enters — the pair
representation, not the multiple sequence alignment (necessity 0.840,
sufficiency 0.995 vs ≈ 0 for all alignment routes) — and σ-resolved analysis of
the diffusion sampler localises where it is lost: the sampler's trajectory is
insensitive to the mutation-specific conditioning throughout global fold
determination and becomes sensitive only during final refinement, after the fold
is committed. We rule out two explanations: sampler under-dispersion (widening
the sampler degrades rather than releases the signal) and insufficient gain
(amplifying the mutation-specific conditioning 8× raises structural correlation
to 0.508, but a norm-matched control with the wrong direction reaches 0.396, and
the direction-specific gap is not distinguishable from zero). The information is
present, addressable, and discarded at a specific, identifiable stage.

---

## Figures

| # | figure | carries |
|---|---|---|
| **1** | `mechanism.png` | the phenomenon: WT/mutant distance maps, difference, KL map, TM-aligned overlay, one pair's histogram. E[d] moves 7.6 Å, TM 0.935. **This is the paper in one image.** |
| **2** | `routes12.png` | where it enters — pair rep, not MSA, 12 assays |
| **3** | `n12_beta.png` + `gym_summary.png` | the internal-vs-output gap, 60 assay-splits |
| **4** | `trajectory.png` | where it is lost — σ-resolved sampler insensitivity |
| **5** | `amplify.png` | not a gain problem |
| **6** | *to build* | cross-model: the internal-vs-output gap in Boltz-2, OF3, Protenix, AF2 |
| S1 | `toy_units.png` | why KL not Ångström (methods) |
| S2 | `sublayers_gfp.png`, `matrix_*.png` | per-operation attribution (n=2, mark as such) |
| S3 | `bench_gfp.png`, `depth_gfp_core32.png` | GFP behavioural anchor, MSA-depth series |

---

## Structure (agreed 2026-08-02)

Four parts, in this order:

1. **Motivation** — the phenomenon and why the standard reading is wrong
2. **Mechanism explained + consequences** — the causal core of the paper
3. **Confrontation between models** — is this Boltz-2, or is it the architecture?
4. **RSA and corrections** — supporting analyses and withdrawn conclusions

The MSA result is included **for completeness, not as the subject**: one
paragraph in part 2 ruling out the alignment-retrieval explanation, plus the
re-searched-alignment replication that removes its circularity.

---

## Section outline

### 1. Introduction
The observation everyone has: folding models are structurally invariant to
mutation. The standard reading: the model retrieves a memorised/aligned
structure and never registers the substitution. **Our claim: registration is
fine; expression is not.** State the three-step localisation up front (enters at
the pair representation → survives to the diffusion conditioning → absent from
coordinates) so the reader knows where they are going.

### 2. A harness that permits exact intervention
Short. joltz is a from-scratch JAX/equinox reimplementation; the Pairformer is a
`lax.scan` over stacked parameters, so per-layer capture is re-running the scan
with `ys` populated, ablation is `eqx.tree_at` on the stacked parameters, and
route patching is a pytree field swap. **Bit-identity is proved, not asserted**
(relative error exactly 0.0 vs joltz's own `trunk_iteration`). This is what makes
every downstream claim causal rather than correlational, and it is the reason to
use joltz rather than the PyTorch release.

### 3. The phenomenon (Fig 1)
GFP, 32 buried positions mutated to charged residues. Pairwise beliefs move by up
to 7.6 Å in expected distance and 0.92 nats mean symmetric KL; the two predicted
structures superimpose at TM 0.935. Panel F: a single pair where the WT is
confident at ~5 Å and the mutant is a broad distribution near 13 Å. **The trunk
changed its mind about that contact completely and the decoder produced the
wild-type fold anyway.**

### 4. Where the mutation enters — and where it does not (Fig 2)
Five routes, necessity (restore one to WT) and sufficiency (inject one from the
mutant), 12 assays. `z_direct` 0.840 / 0.995; all alignment routes ≈ 0.

**This is where the MSA gets its paragraph.** Frame as: the common explanation
for structural invariance is that the mutant inherits the wild type's alignment
and therefore its structure. Under exact intervention that is not what happens —
the mutation reaches the trunk output through the pair representation, and
zeroing the alignment routes changes nothing. *Caveat to state plainly:* our
variants carry a grafted wild-type MSA by design, so this result establishes
that the alignment routes are not *used*, not that a re-searched alignment would
be uninformative. (Ablation in `publication_plan.md` §1.2 — **must run before
submission.**)

### 5. The trunk represents what the output does not (Fig 3)
The probe: 250 variants × 12 assays, 256 features (4 divergence quantities × 64
layers), ridge on 16 train-selected features, position-grouped 75/25 splits, 5
partitions per assay = 60 assay-splits. ρ = 0.548 vs TM 0.214, pLDDT 0.244,
pLDDT-at-site 0.037, position-only baseline 0.069.

Emphasise: (a) the comparison is **paired** — identical rows, identical splits;
(b) the position-only baseline is what a naive random split would have leaked,
and it is near zero; (c) **pLDDT at the mutated residue is the worst predictor in
the set**, which is counterintuitive and worth its own sentence — localising
confidence to the substituted site makes it *worse* than the chain average.

(RSA is the assumption-light second view of this claim, but it now lives in
part 4 with the corrections — see §10 — because it is supporting rather than
load-bearing.)

### 6. Where it is lost (Fig 4)
Conditioning still carries it (‖Δq‖/‖q‖ = 0.285). Coordinates do not. Then the
σ-resolved result: divergence/floor flat at ~0.6 through 80 % of the schedule —
the mutant's path is *closer* to the wild type's than two wild-type runs are to
each other — separating only below σ ≈ 1 Å.

**The precise claim, which is narrower than "the sampler discards it":** the
sampler is insensitive to the mutation-specific conditioning while the global
fold is being determined, and becomes sensitive only during local refinement, by
which point the fold is committed.

Report the noise floor honestly: 4 WT keys, 6 pairs, and the endpoint correlation
significant in only 1 of 3 assays. Do **not** quote max-over-steps.

### 7. Three explanations ruled out (Fig 5)
**Under-dispersion.** Boltz-sample shows the default sampler under-explores what
the pair representation supports. Scaling the pair-derived attention biases
widens the ensemble enormously (WT spread 0.990 → 0.303 mean pairwise TM) but
degrades the stability signal monotonically (0.435 → 0.018). Widening adds noise
rather than releasing information.

**Insufficient gain.** Scaling only the mutant-minus-WT difference raises
ρ(TM, ΔG) 0.190 → 0.508 at γ = 8 — but a difference borrowed from another variant
and rescaled to the same norm reaches 0.396, and the direction-specific gap is
+0.110, 95 % CI [−0.027, +0.252]. Amplification converts perturbation *magnitude*
into structural displacement; the trunk's *directional* content stays
unexpressed. And ‖Δz‖ read straight off the trunk (0.637) beats every decoded
structure while requiring no sampling.

**Trunk-structure consistency.** Scoring the decoded structure under the trunk's
own distogram gives ρ = −0.525 raw, but 81–93 % of that is distogram *sharpness*
rather than disagreement (a sharp distogram punishes sub-Ångström error; a broad
one does not). Entropy-partialled it is −0.209 and inconsistent across assays.
Reranking 8 samples by consistency gives +0.088, 95 % CI [−0.062, +0.240], with
the sign flipping across the three assays.

This section is where the paper earns trust: three plausible fixes, all tested,
all negative, each with the control that kills it stated in full. The decoder's
insensitivity is not a matter of sampler width, not of scale, and not of
selection among samples it already generates.

### 8. Confrontation between models — **now in scope**

The single question: **is the loss a property of Boltz-2, of diffusion decoders,
or of the trunk-to-structure interface in general?** Four models, and the lineup
is chosen so the answer is forced rather than suggested.

| model | decoder | why it is in the set |
|---|---|---|
| Boltz-2 (`joltz`) | diffusion | the model everything above was measured on |
| OpenFold3 (`jopenfold3`) | diffusion | independent AF3-class implementation |
| Protenix (`protenix`) | diffusion | second independent AF3-class implementation |
| **AlphaFold2** (`mosaic.alphafold`) | **IPA regression, deterministic** | **the decisive contrast — no sampler at all** |

The three diffusion models test **replication**. AF2 tests **mechanism**, because
it is the only one whose decoder is not a sampler:

- signal lost in **all four** → the failure is the trunk→structure interface in
  general, not diffusion. The biggest version of the claim.
- signal lost in the **three diffusion models but preserved in AF2** → diffusion
  sampling is specifically at fault. Sharper, and directly actionable.

**Feasibility is settled, not speculative** (checked 2026-08-02):

- `OpenFold3.load()` is torch-free; `of3.eqx` + `of3.skeleton.pkl` are on disk.
  Its structure is `embed_inputs` → `recycle` → `pairformer_stack`, and
  `PairFormerStack` holds `stacked_params` scanned with `jax.lax.scan` — the same
  construction as joltz, so per-layer capture, `eqx.tree_at` ablation and route
  patching port by renaming attributes.
- `protenix_mini_default_v0.5.0.eqx` is on disk with a `protenij.py` scan-based
  stack.
- **AF2 is available after all.** `mosaic.alphafold` vendors DeepMind's own
  JAX/haiku implementation (`modules.py`, `folding.py`, `config.py`), and
  `weights/alphafold/params/params_model_1_ptm.npz` loads with 338 param groups.
  The `evoformer_iteration` parameters carry a **leading dimension of 48** —
  haiku's `layer_stack` scans over the 48 blocks exactly as joltz scans over 64 —
  and both `distogram_head` and `structure_module` are present. So the
  "trunk believes X, structure says Y" comparison ports directly.
  *(An earlier note in this file said AF2 needed external setup. It does not.)*

**Minimum viable comparison per model** — not the full mechanistic battery:

1. the phenomenon: TM-to-wild-type across variants
2. does the trunk represent the mutation? (probe rho on per-layer pair features)
3. does the output? (TM rho, pLDDT rho, pLDDT-at-site rho)
4. the gap, on identical variants and identical position-grouped splits

Routes and per-operation attribution stay Boltz-2-only unless time permits; say
so explicitly rather than implying a fuller comparison than was run.

### 9. Discussion
- **For interpretability:** structural invariance is not representational
  invariance. Output-level probing of a scientific foundation model can be
  actively misleading about what it encodes.
- **For practice:** if you want mutational effect from a folding model, read the
  trunk, not the structure and not pLDDT. ‖Δz‖ alone gives 0.637 with no
  sampling. Distogram entropy gives 0.483 but is subsumed by ‖Δz‖ (partial
  −0.13), so there is **one** readout to report, not a menu.
- **For architecture:** the trunk→decoder interface is the failure point, and the
  failure is specific to global fold determination. A decoder that consulted the
  distogram during high-σ steps is the natural next design.
- **Limits, stated by us not by a referee:** 12 Tsuboyama assays, 61–72 aa,
  folding stability only; recycles = 3; grafted alignments; per-operation
  attribution on 2 proteins.

### 10. RSA and corrections

Grouped deliberately: both are *supporting* material that makes the main claims
more credible without carrying them.

**RSA** — the assumption-light second view of part 2. Pairformer 0.138 >
distogram 0.023 > structure 0.000, 4/4 assays; mean variant-to-variant TM
0.96–0.99. State plainly that these are second-order quantities after three
partials and are **not comparable** to rho = 0.548; what carries the result is
the ordering and its consistency, plus the blunt fact that every variant is
predicted to within TM 0.96 of every other.

**Corrections** — **keep it**
Nine withdrawn conclusions with diagnoses, including two independent hand-rolled
superposition failures — the second made after the first was documented, which is
why every geometric primitive now self-tests on import. In a field where
invalidated runs are almost never published, this is a genuine methodological
contribution and it makes every surviving number more credible, not less.

---

## Claims we CANNOT make (guard against drift)

- ✗ "The model does not encode mutations." It does. That is the whole point.
- ✗ "The MSA is irrelevant to structure prediction." Only that the alignment
  routes do not carry *this* mutation signal, *under grafted alignments*.
- ✗ "ρ = 0.548 is a good stability predictor." It is not offered as one. The
  claim is comparative and paired. (Still needs an external reference point —
  `publication_plan.md` §1.4.)
- ✗ "The diffusion sampler discards the signal." Too strong. It is insensitive
  during fold determination.
- ✗ "Amplification fixes it." It does not; the control eats 78 % of the effect.
- ✗ Anything about AF2/AF3/OpenFold3 until those are actually run.
- ✗ Per-operation attribution as a general result — n = 2.

---

## Before submission

Ranked, from `publication_plan.md`:

1. **Re-searched-MSA ablation** (§1.2) — closes the one circularity in §4.
2. **Distogram rescoring** (§3.2) — if positive, the paper gains a fix and a
   much stronger discussion; if negative, §7 becomes "three ruled out" and the
   structural account hardens. Either way it is worth running.
3. **Production settings** (§1.3) — 15 recycles, best-of-N.
4. **External ddG reference** (§1.4) — one table row.
5. **Cross-model section** (§8) — now **in scope for this paper**, not follow-up.
   OF3 + Protenix (cheap, harness ports by renaming) and **AF2** (the decisive
   non-diffusion contrast; `mosaic.alphafold` + params confirmed present
   2026-08-02). This is the long pole and should start now.
6. Per-operation attribution to 12 assays (§1.5) — or explicitly demote to
   supplementary and label n = 2.

1–4 are days. **5 is the long pole and the one that most changes how the paper
is received** — start it in parallel rather than after.
