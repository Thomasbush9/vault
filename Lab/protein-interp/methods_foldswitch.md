# Methods — Fold-switching benchmark

Reference note. What we score, how we score it, why each metric is the right one, and where the threshold values come from. Reread this before writing up.

---

## 1. The problem we're testing

**Fold-switching proteins** adopt two distinct, stable native folds for the *same* (or near-identical) sequence — they're the cleanest probe of structure-prediction generalization because the same input must support two different correct outputs. Standard structure predictors typically lock onto one fold (usually the canonical / most-recently-deposited / most-MSA-supported one) and miss the other. Porter et al. (Chakravarty & Porter, Nat Comm 2024) argued this is **memorization-driven**: AF retrieves training PDBs rather than computing both folds from physics.

We're testing the same question on **Boltz-2** (diffusion + MSA, post-AF) and **ESMFold** (single-sequence ESM-2 + folding head, no MSA, no templates). Two complementary models — if both show the phenomenon, the mechanism isn't AF-specific.

### Cohorts

- **In-set (92 pairs).** Porter's TableS1 list of fold-switchers whose canonical PDB pair was deposited before AF's cutoff. Both folds (G1, G2) are PDB entries Boltz/AF/ESMFold likely saw in training.
- **Out-of-set (2 homologs).** Curated by us — PDBs deposited after the **Boltz-2 cutoff (2023-06-01)**:
  - `8UBH` — KaiB-TV-4 (T. elongatus vestitus, deposited 2023-09-23), homolog of canonical KaiB (`2qkeE` / `5jytA`).
  - `9IK0` — PrfaH (IncX3 plasmid, deposited 2024-06-26), homolog of canonical RfaH (`2ougC` / `6c6sD`).
  - Both homologs' families *are* in training (canonical PDBs available), but the specific sequences are post-cutoff. This tests *generalization to new homologs*, not strict isolation of memorization — but it's the closest we can construct with what's in PDB.

---

## 2. Three reference structures per protein

For each prediction we score against up to three references:

| ref | what | answers |
|---|---|---|
| **G1** | first canonical fold of the family (PDB entry, pre-cutoff, in training) | "Did the model produce fold 1 of the family?" |
| **G2** | second canonical fold of the family (also pre-cutoff, in training) | "Did the model produce fold 2 of the family?" |
| **post-cutoff self** | the homolog's own deposited PDB (out-of-set only, post-2023-06-01) | "Did the model get the new homolog's actual structure right?" |

The G1/G2 framing inherits Porter. The post-cutoff-self framing is what makes our analysis non-redundant: it separates "model is a working structure predictor on this sequence" from "model produces both folds the family is known to adopt."

---

## 3. Scoring metric — TM-score vs RMSD

### TM-score (whole chain)

For two superposed protein structures of length `L`:

$$
\mathrm{TM} = \frac{1}{L_{\text{norm}}} \sum_{i=1}^{L_{\text{aln}}} \frac{1}{1 + (d_i / d_0(L_{\text{norm}}))^2}
$$

where `d_i` is the Cα distance after optimal superposition and

$$
d_0(L) = 1.24 \, (L - 15)^{1/3} - 1.8
$$

`L_norm` is the reference length (Porter convention: `normalize_by="reference"` in `tmtools`).

**Threshold conventions**:
- `TM > 0.5` → same fold (Zhang & Skolnick 2004)
- `TM > 0.6` → confident match (Porter's threshold for whole-chain "both folds matched")
- `TM = 0.17` → random pair

### Why TM-score breaks on short fragments

`d_0(L)` shrinks fast at small `L`:

| L | d₀ (Å) | Notes |
|---|---|---|
| 200 | 5.4 | normal domain |
| 100 | 4.0 | small domain |
| 50 | 2.6 | switching-region scale (long) |
| 25 | 1.0 | typical TableS1 region length |
| 15 | undefined (formula floor) | |

At L ≈ 25, the gaussian kernel becomes so tight that even a *perfect* superposition of a flexible-loop region scores TM ≈ 0.3–0.5. Random superpositions score 0.05–0.15. The 0.6 threshold from whole-chain literature is *unreachable* in practice for switching regions.

### Porter's solution — use RMSD on the region

For short fragments, RMSD is the standard. Porter's "both folds matched on region" threshold is **RMSD < 2 Å** (after TM-align superposition).

**We compute both metrics** for the region, but the publishable apples-to-apples comparison to Porter is RMSD-based.

---

## 4. Whole-chain vs region — why the distinction matters

Whole-chain TM averages over all residues. For a 250-aa fold-switcher with a 25-aa switching region, ~90 % of residues are *scaffold* — identical in both folds. A prediction that produces the correct scaffold and a wrong (or generic) switching region can still score TM > 0.6 vs *both* G1 and G2, simply because the scaffold dominates the average.

Our data confirms this exactly:

| metric | Boltz r=3 in-set | ESMFold in-set |
|---|---|---|
| whole-chain TM > 0.6 vs both | **60.9 %** | **56.5 %** |
| region RMSD < 2 Å vs both | **37.1 %** | **32.6 %** |

Whole-chain headlines roughly double the "true" both-fold rate. Porter's All_AF 35 % is on the *region* metric — so the apples-to-apples Boltz number is 37 %, *not* 61 %. The whole-chain inflation effect must be flagged explicitly in any writeup.

---

## 5. The region-restricted scoring pipeline

Input: a **region sequence** (one short amino-acid string per fold-switcher; from Porter's TableS1 column C, or supplied manually for out-of-set). Example for `seq_00001`: `INGTDNEVVTVTDENTGEIS` (20 aa).

Reason to use a sequence motif and not a stored residue range: the same region needs to be located in three structures (pred, ref1, ref2) that differ in chain numbering, may have missing loops, and (for homologs) different sequence. A motif + local alignment survives all of those.

### Algorithm — applied independently to pred, ref1, ref2

1. **Extract observed residue sequence and Cα coordinates** from the structure's chain. Hetero residues skipped. Missing-loop residues are absent by construction (no CA → no entry).

2. **Locally align region → observed sequence** with Biopython `PairwiseAligner`:
   - `mode = "local"`
   - substitution matrix: **BLOSUM62**
   - gap open: **−10**, gap extend: **−0.5**

3. **Take the matched span on the target** — `[start_obs, end_obs)` (0-based, half-open). That's where this region sits in this specific structure.

4. **Quality filters** — drop the entry if any of {pred, ref1, ref2} fails:
   - matched span shorter than **min_len = 10** residues, or
   - alignment score per matched residue < **min_score_per_aa = 1.0** (perfect 20-aa match ≈ 4/aa; chance ≈ 0.3/aa).

   Failures tagged in `region_status` (`fail_pred`, `fail_ref1`, `fail_ref2`, combined with `+`).

5. **Slice the per-residue arrays** `coords[start:end]` and `seq[start:end]` for each of pred/ref1/ref2. Now we have three short structures.

6. **TM-align + RMSD on each (pred, ref) slice pair.** TM-align finds optimal rigid superposition; we extract:
   - `tm_g{1,2}_region` — TM-score on the slice (normalized by ref length)
   - `rmsd_g{1,2}_region` — RMSD from the same superposition

7. **Emit row** with whole-chain + region columns + region lengths + status.

### Failure modes seen in practice

3/92 entries fail (`region_status` ≠ `"ok"`). Typical cause: the *reference* PDB has a missing loop spanning the switching region, leaving too few resolved residues for a clean BLOSUM62 match. Yesterday's earlier inline analysis dropped 6/92 with a simple match/mismatch aligner; BLOSUM62 + score-per-aa filtering recovered 3 of them.

---

## 6. Sample-diversity dimension

Boltz emits **25 diffusion samples** per protein (at recycles=0). For each, we have:
- Cα coordinates → TM/RMSD against any ref
- Boltz confidence score (from `confidence_*.json`)

Three views of the same data:

| view | computation | answers |
|---|---|---|
| **top-1** | pick highest-confidence sample, score vs refs | "what does Boltz output by default?" → Porter-comparable |
| **best-of-25** (per ref) | `max(TM)` across samples, per ref independently | "can Boltz produce each fold if asked?" |
| **single-sample-both** | does any one sample score TM>0.6 vs BOTH refs simultaneously? | "does any single output match both folds?" |

For in-set: best-of-25 ≈ top-1 (59 vs 58 "both">0.6). Sample diversity does *not* rescue fold-switching — the 25 samples are tight, not bimodal.

For out-of-set: 8UBH has 0/25 samples > 0.6 vs G1; 9IK0 has 3/25 samples > 0.6 vs G2. KaiB-TV-4 collapses entirely; PrfaH samples both folds occasionally.

ESMFold produces only one prediction per protein — no sample-diversity comparison possible without running it multiple times with different random seeds.

---

## 7. ESMFold as the second test method

**Why include ESMFold:**
- It's a *single-sequence* model — uses ESM-2 (650M-param PLM) embeddings, no MSA, no templates.
- If memorization shows up here, it's encoded *in the PLM representation* — not as template retrieval. That sharpens Porter's claim significantly.
- It's smaller and faster than Boltz — easier target for downstream interpretability (linear probes on ESM-2, SAEs, activation patching).
- Training cutoff: ESMFold v1 uses PDB up to ~April 2020 — *older* than Boltz-2's 2023-06-01 cutoff. So for the in-set 92 (all pre-cutoff for both), the cutoff difference is moot. For out-of-set, only Boltz tests apply.

### Result on in-set 92

| metric | Boltz r=3 | ESMFold | Porter All_AF |
|---|---|---|---|
| whole-chain "both>0.6" TM | 60.9 % | 56.5 % | — |
| **region RMSD<2 Å vs both** | **37.1 %** | **32.6 %** | **35 %** |
| region "either" RMSD<2Å | 89.9 % | 78.7 % | — |

ESMFold sits ~5 pp below Boltz and right at Porter's All_AF level *with no MSA and no templates*. Memorization signal is encoded in the PLM embedding alone.

### Overlap of "both" successes (region-restricted)

- Boltz "both" set: 33 proteins
- ESMFold "both" set: 29 proteins
- Intersection: 20
- Boltz-only: 13 (mostly cases where MSA depth helps the alternative fold)
- ESMFold-only: 9 (cases where canonical fold-1 happens to be encoded by the PLM but Boltz wandered on fold-2)

Methods are correlated (intersection > chance), not redundant. Both are needed to argue the phenomenon is method-agnostic.

---

## 8. Provenance — file paths

- Pairs manifest (with regions): `data/foldswitch/identities_with_regions.tsv` — built by `scripts/foldswitch_make_regions_manifest.py` from Porter's `supporting tables/TableS1.xlsx`.
- Refs dir: `data/foldswitch/porterll-AF2_benchmark-40a57d7/AF2Rank/final2_and_debug2_inputs/` (per-chain extracted PDBs; full 8UBH.cif / 9IK0.cif added for out-of-set).
- Boltz r=3 in-set top-1 region scores: `data/foldswitch/tm_scores_r3_region.csv`.
- ESMFold in-set region scores: `data/foldswitch/r0_multi/tm_scores_esmfold_region.csv`.
- Boltz r=0 25-sample data: `data/foldswitch/r0_multi/tm_boltz_inset_r0_all25.csv` (whole-chain only so far).
- Out-of-set Boltz 25-sample data: `data/foldswitch/out_of_set/tm_scores_outset_r0_3refs.csv` (whole-chain only so far).

Code:
- `src/protein_interpretability/scoring/region.py` — local-alignment + slice helpers.
- `src/protein_interpretability/score_sequences.py` — pairs-mode CLI with `fold_switch_region` column support; emits both whole-chain and region columns.

---

## 9. Caveats / not-yet-done

1. **n=2 for out-of-set** — anecdotal, not statistical. Should grow the cohort if Phase 1 result holds.
2. **Memorization-via-homolog confound for out-of-set.** Both KaiB and RfaH families have canonical pairs in training, so Boltz could "memorize the family's two folds" and just replay the matching one for each new homolog. Strict memorization isolation would require a fold-switcher family with *no* PDB entries in training — currently absent from PDB.
3. **Sample-diversity ESMFold not tested.** ESMFold v1 is deterministic on a single seed; multi-seed runs would let us test if PLM-based prediction has the same bimodal-or-not structure Boltz lacks.
4. **Out-of-set + region metric still pending.** Need to look up KaiB and RfaH fold-switching region sequences from literature (KaiB: roughly residues 50–94 of canonical; RfaH: KOW C-terminal ~residues 100–162) and add manual rows to the out-of-set manifest. Then re-score 8UBH and 9IK0 region-restricted.
5. **No AF-version sweep.** We only test Boltz-2 + ESMFold. Porter has 6 AF variants. Modesty note: we extend Porter's method-generality claim by adding two non-AF architectures, not by re-running his cross-variant comparison.

---

## 10. Theory note — why fold-switching is the *right* probe

Three orthogonal hypotheses for what a structure predictor does:

| hypothesis | prediction on fold-switchers |
|---|---|
| **Template retrieval / memorization** | matches whichever fold has more / more recent PDB exemplars, fails on out-of-set |
| **Coevolution-as-physics** | should produce both folds when MSA contains signal for each (Wayment-Steele AF-Cluster premise) |
| **Sequence-mapping memorization** (no MSA needed) | PLM directly encodes sequence→fold for trained families; both folds emerge from family-level pattern matching |

A model that memorizes (Porter) will rank-order: in-set "both" >> out-of-set "both". A model that does coevolution will rank-order: deep-MSA "both" >> shallow-MSA "both" (regardless of in-set vs out-of-set). A model that does PLM-memorization will show in-set "both" rates for ESMFold (no MSA) comparable to MSA-based methods.

The 56.5 % in-set "both" rate for ESMFold (no MSA, no templates) is incompatible with pure-coevolution and consistent with PLM-memorization. The 60.9 % for Boltz (with MSA) is comparable, suggesting the MSA isn't adding much *over and above* the PLM-encoded prior on these families.

This is the Phase-3 hook: **what does ESM-2 encode about fold identity that lets it produce both folds at PLM-only resolution?** Linear probes on ESM-2 hidden states predicting per-residue fold-class membership are the natural first probe. If the probes succeed, SAEs / direction-finding on the relevant layers become tractable.
