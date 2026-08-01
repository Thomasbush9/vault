# Agenda

Current focus and the next 1–3 concrete things. When done, fold into a `log/` entry; when parked, move to `decisions.md`.

## Now — Circuit-level account of the MSA bias, on JAX/joltz Boltz-2

Restarted 2026-07-30 after the May pause. Full reasoning in
`log/2026-07-30-jax-pairformer-harness.md`. Two facts reset the plan:

1. **The fold-switch predictions on the cluster are gone** —
   `/n/holylfs06/.../protein_rsa/` no longer exists. Setup scripts survive in
   the repo; the scored CSVs survive on the Mac. Regenerating is a day of
   compute, not a rebuild.
2. **`mosaic_setup` now ships `joltz`**, a from-scratch JAX/equinox
   reimplementation of Boltz-2. Per-layer capture, path patching and full-trunk
   gradients are all available without hooks or monkeypatching. This is a
   strictly better substrate than the PyTorch extractor stack for everything in
   Phase 3 below.

**The claim under test.** Query-sequence identity reaches the pair
representation by four separable routes (`z_direct`, `msa_bcast`, `msa_query`,
`msa_prior`; see the log). `OuterProductMean` divides by the number of MSA
rows, so the query-as-MSA-row-0 route enters at ~1/S weight — but the other two
query routes are *undiluted*, so MSA dominance is not a corollary of the
architecture. It has to be measured. That measurement is the paper.

### Now (this week)

- [x] JAX harness at `/n/holylfs06/.../prot_interp_files/harness/`, validated
      **bit-identical** to joltz (`test_equivalence.py`, rel. error 0.0).
- [x] "Physics vs memory" GFP cohort: buried-core→charged mutants (n=1..32),
      count-matched surface control, nested series, WT MSA grafted onto all.
- [x] `exp_paths.py` — decomposition is **exact** (endpoints close to 0.00e+00).
      `z_direct` carries 0.72–0.93; **`msa_query` carries ~0.02**. The query's own
      row in its alignment is causally irrelevant — the naive "drowned in the
      MSA average" story is refuted.
- [x] `exp_depth.py` — single-sequence is 4.4× more mutation-sensitive than full
      depth, but the curve is **nothing like 1/S** (flat S=2→32, step at S=64).
      Composition matters, not row count.
- [x] `exp_layers.py` — divergence rises to a peak at L34 then falls over
      L37–45. **Read in Å this looked like an erasure band. It is not — see
      `exp_kl` below.**
- [x] `exp_sublayers.py` (in Å) — `transition_z` is the only op that reduces
      divergence; all four triangle ops increase it. **Ambiguous until repeated
      in KL** — `transition_z` is exactly the op that would *sharpen* a
      distribution, which in Å is indistinguishable from suppressing.
- [x] `exp_subspace.py` — **refuted my "rotated out of the readout" idea.** The
      readable fraction of Δz is flat (~0.28) across all 64 layers and its
      magnitude more than doubles. Nothing is moved out of the readout.
- [x] `exp_kl.py` — **the arbiter, and it split the phenomenon in two.**
      (a) The L37–45 band is *real* but ~1.7× smaller than Å implied
      (1.4–1.8× in KL, entropy flat there), and is fully recovered later.
      (b) The fall to the trunk output is *pure saturation*: over the last 8
      layers KL nearly doubles while |ΔE[d]| almost halves and the WT distogram
      entropy collapses 2.16 → 0.81 nats.
      Net across the stack symmetric KL grows **5.6×** and is maximal at the
      final layer. The trunk accumulates the mutation signal; it does not erase it.
- [x] `exp_sublayers.py` **re-run in KL** — `transition_z` survives as a genuine
      suppressor (−0.36 KL over 64 layers); all four triangle ops *amplify*
      (+1.10 total). The Pairformer is a competition, amplification wins ~3:1,
      and the L37–45 band is where the balance locally reverses.
- [x] `exp_ablate.py` — **causal**: deleting `transition_z` in L37–45 flips the
      band from −0.042 to **+0.063** (sign flip); a width-matched control band
      (L10–18) does nothing (final KL ×1.02). But ablation also *drops* final KL
      (×0.59 band, ×0.26 all-layers) — `transition_z` is load-bearing, not a
      dedicated eraser. Next: scale its contribution by α instead of deleting.

**Current best account.** Boltz-2 registers a destabilising mutation and carries
it through the whole trunk (‖Δz‖ ↑2.1×, KL ↑5.6×, |Δlogit| ↑2.4×), but expresses
it as a change in the **shape** of the predicted distance distribution rather
than a shift in its **mean** — i.e. as *uncertainty*, not *geometry*.
Corroborated by pLDDT falling with buried-core load (0.948 → 0.830) while
count-matched surface mutations barely move it (0.932). The emitted geometry
stays the memorised wild-type barrel.

- [x] `exp_matrix.py` / `fig_matrix.py` — **the circuit as matrices over layers**
      (residue x layer, separation x layer, op x layer, enrichment). Signal
      enters localised on the mutated residues, propagates short-range -> long-
      range, focuses until ~L43, then delocalises.
- [x] `analyze_onset.py` — no contact-graph propagation (refuted); but a clean
      spatial **magnitude** effect: divergence falls with 3-D distance from the
      mutation for buried (rho -0.56), not at all for surface (+0.03).
- [x] dose series n=1..32 — the focus/defocus profile is **dose-independent in
      shape**, inflecting at L43-45; surface never focuses.

### Now — finish the causal chain, then widen N

- [x] ProteinGym stability probe, 4 Tsuboyama assays: internal state beats the
      model's own structural output in **19/20** assay-splits (+0.480 vs +0.191,
      gap +0.289, 95% CI [+0.212, +0.366]). pLDDT at the mutated residue pools
      to −0.006 — for single substitutions the confidence head carries nothing.
- [x] RSA: Pairformer > distogram > structure in **4/4**; structure's variant
      geometry matches the experiment at pooled +0.000. All variants within
      TM 0.96 of each other.
- [x] Localisation: Pairformer carries it → conditioning still carries it
      (‖Δq‖/‖q‖ ≈ 0.28) → sampled structure does not. **The loss is in the
      diffusion sampler**, downstream of the conditioning projection.
- [x] Ensemble-spread hypothesis **refuted**: mean structure predicts ΔG 2–3×
      better than ensemble width. Shape-not-location holds at the distogram and
      fails at the structure module.
- [ ] **β at the correct insertion point** (running) — the first attempt scaled
      `z_trunk` and was absorbed by PairwiseConditioning's LayerNorm; see §24.
- [ ] **12-assay expansion** (MSAs generating) — the 1.4×–6.0× spread across
      four assays is the weakest point in the whole account.
- [ ] Denoising-trajectory divergence: mutant vs WT as a function of reverse
      step. Decides whether the sampler never grows the difference or grows then
      contracts it.
- [ ] Scramble calibration for ‖Δq‖/‖q‖ ≈ 0.28 — currently a number with no scale.

### Parked until the above is solid

Porting the toolkit to AF2 / OpenFold3 / Protenix. The method ports; the route
decomposition does not (AF2's Evoformer has MSA row/column attention, not
PWA + OuterProductMean, so `msa_query` / `msa_bcast` are Boltz-specific
categories that need redesigning per architecture). Breadth here would buy two
thin stories instead of one solid one while N is still 4 assays / 2 proteins.

### Earlier list — distogram

- [ ] Per-head resolution: which of the 4 triangle-attention heads carry the
      spreading? `pairformer_capture` already gives the hook point.
- [ ] Repeat the matrices on a second protein (the N=1 caveat is now the
      binding constraint on every claim above).
- [ ] Fix or replace `geom.tm_score` with tmtools before any structure claim.

### Then

- [ ] Generalise off N=1: regenerate the fold-switch cohort (contrastive, two
      ground truths) and rerun the same three experiments on it.
- [ ] Score predicted structures of the core mutants against a stability model
      (mosaic ships a megascale head + JAX ProteinMPNN) — makes "energetically
      unfavourable but predicted anyway" a number rather than an assertion.
- [ ] Port to AF2 (weights on disk in mosaic) — same decomposition, different
      architecture, tests whether the mechanism is Boltz-specific.

### Superseded by the above

The Phase 1/2/3 fold-switch plan below is **not abandoned** — Phase 3
("interpret") is exactly what the JAX harness is for, and the fold-switch
cohort is still the right place to generalise. What changed is the order:
build and validate the causal tooling on a cohort we control (GFP, where we can
*construct* the sequence-vs-MSA conflict), then apply it where the ground truth
is contrastive. Phase 1's recycle-confound rerun and the missing 7 out-of-set
pairs remain open if we want the replication number itself.

---

## Previous focus — Replicate Porter et al. 2024 fold-switch memorization on Boltz-2, then layer interpretability

Pivot from "single-protein GFP IG attribution" → **replicate a known memorization phenomenon on Boltz-2 first, then attribute it**. Reasoning + 3-phase plan in `log/2026-05-09-foldswitch-pivot.md`. Boltz-2 is trained later than AF2, so we expect Porter's 35 %/14 % training-vs-out-of-set memorization gap to *amplify*. The contrastive setup (each fold-switcher has G1 and G2 ground truths) is exactly what was missing in the GFP work — IG/gradient tools already built carry over unchanged.

### Phase 1 — Replicate (this week)

- [x] **MSA decision**: went with Porter's Zenodo deep MSAs (10.5281/zenodo.13221957) for tightest reproducibility against the paper's numbers. YAMLs finalised with absolute msa paths.
- [x] **Cluster wrapper**: SLURM array around `boltz predict` — one task per YAML. Predictions complete for all 92 in-set pairs (2026-05-10).
- [x] **`seq_00043` (1369 aa) survived** — predicted on H100, top cif present at `outputs/sequences/seq_00043/boltz/seq_00043_model_24.cif`.
- [x] **Pin Boltz output layout:** `outputs/sequences/seq_NNNNN/boltz/seq_NNNNN_model_*.cif`, one cif per dir.
- [x] **Extend `score_sequences.py`** with `--pairs-manifest`/`--refs-dir` (download-from-RCSB capability removed — refs are staged locally from Porter's AF2Rank dump and rsync'd if needed). Also fixed greedy seq-idx regex and added per-chain naming (`<pdb>_<chain>.pdb`) fallback in `resolve_reference`.
- [x] **Pairs-mode scoring run locally** (data is all on this Mac now). Output: `tm_scores.csv`, 92/92 pairs, no skips.
- [x] **Headline numbers + Porter cross-check + fold-switch region restriction** — all in `log/2026-05-10-phase1-scoring-plan.md` "Results" section. Whole-chain TM>0.6 vs both = 62 % (matches ACE 61 %, well above All_AF 35 %); region-restricted RMSD<2Å vs both = 33 % (scaffold inflation explains the 62→33 drop); 36 net-new "both" successes not in any Porter AF list.
- [ ] **Recycle confound — re-run at `--recycling_steps 3`** (AF default parity). Phase 1 was run at recycles=10, which is not directly comparable to Porter's numbers. This is the load-bearing baseline.
- [ ] **Locate the 7 out-of-training-set fold-switchers** (the 14 % control). Not in TableS1 — try other supporting xlsx (`Data_FigS3_revised.xlsx`, `data_FigS4_revised.xlsx`, `data_FigS5.xlsx`, `data_FigS6.xlsx`, `TableS3_revised.xlsx`) or `notes/` text in AF2_benchmark. Needed for the in-set vs out-of-set gap.
- [ ] **Lift region-restricted scoring into a `--region-mode` flag of `score_sequences.py`** (currently in-session inline script — works but should be a CLI for repeat use).
- [ ] **Spot-check oddities**: the 3 pairs AF2.3.1 hits but Boltz doesn't (`2a73b/3l5nb`, `2k0qa/2lela`, `2kxoa/3r9jc`); the 6 region-restriction drops (`seq_00009`, `seq_00026`, `seq_00040`, `seq_00054`, `seq_00064`, `seq_00069`).
- [ ] **Cleanup** — decide which GFP-arc scripts (`run_chromophore_attribution.py`, `analyze_chromophore_block.py`, `plot_chromophore_attribution.py`, `augment_*.py`, `build_seq_perturb_dataset.py`, `sample_mutations.sh`, `run_query_occlusion.py`) to delete vs keep parked.

Explore when boltz can threshold both strutures -> 

- do esm sae on predicted binders
- confermational bias with proteinmpnn
- Sergey: AF performs very local energy optim=> not doing well without msa-> complex energetic landscape-> msa makes you start at the spot closer to the truth-> synthetic proteins are predicted much better than real proteins 
- Look at the template injection
- how can you sample the energy landscape more broadly instead of using the msa: 
- model energy instead of structure 
- diffusion guided by the energy term: ranking protein stability-> better performances when stop before the full inverse diffusion process-> boltz mutation effect on the previous diffusion 
- TCR design: specificity -> give the whole complex + mask one aa in the peptide -> recover the sequence of the peptide with high confidence 
- 
### Phase 2 — Ablate (next week)

- [ ] **Single-sequence run** (`--msa_mode empty`). Already supported by `foldswitch_setup.py`; just regenerate YAMLs to a parallel out-dir.
- [ ] **SPEACH_AF replication** on Boltz-2 (alanine-mask MSA columns). This is *literally* the alanine-IG baseline we'd planned for GFP — methodological alignment is free.
- [ ] (Optional) **AF-Cluster** with shallow MSAs. Needs MSA-clustering pipeline; deprioritise unless single-seq + SPEACH_AF leave gaps.

### Phase 3 — Interpret (week 3–4)

- [ ] For each pair, run gradient/IG attribution with **two targets**: ‖predicted distogram − D(G1)‖ and ‖… − D(G2)‖. The *difference map* is the memorization-direction signal.
- [ ] Compare attribution maps for memorized vs not-memorized predictions. Specifically:
      - Does **MSA-channel** attribution dominate **query-channel** attribution in memorized cases?
      - Which Pairformer layers carry the memorization signal (vanishing-gradient artefact at layer 0 should be revisited here with cohort statistics)?
      - Which **MSA columns** are most attributed → cross-check with SPEACH_AF column-knockout outcomes (should agree if attribution is faithful).

## Conditional follow-on (only if Phase 3 yields clean results)

- [ ] **pLM-as-MSA-replacement** experiment. If Phase 3 shows MSA carries narrow conservation signal, ask whether ESM2 / ProtT5 embeddings can substitute for it without inducing memorization. Needs Boltz architectural surgery — only justified if interpretability has produced a sharp prediction first.

## Parked / reusable from the GFP arc

The tooling from the GFP work is reusable; the case study is parked.

- [ ] **GFP as a single-case demonstrator** within the broader frame. Re-cast: "here's a heavily mutated GFP — does Boltz predict the WT structure (memorize) or break (generalize)?" Reuses existing `run_chromophore_attribution.py` pipeline.
- [ ] **IG-on-WT** for GFP (`run_wt_ig.py`, never written). Was: alanine-baseline IG vs zero-baseline IG. **SPEACH_AF in Phase 2 covers the same alanine-perturbation question on a much bigger cohort**, so this script is no longer load-bearing.
- [ ] **MSA-side perturbation companion** — folded into Phase 2 (SPEACH_AF *is* a column-conservation knockout; AF-Cluster is the coevolution-pair analogue).
- [ ] Cross-reference attribution maps against PWA-attention columns from `log/2026-05-04-pwa-conservation.md` — defer until Phase 3 produces attribution maps to cross-reference.
- [ ] **Occlusion as a stand-alone scientific test** — deprioritised 2026-05-06; code preserved (`scripts/run_query_occlusion.py`), may resurface as a causal validator.
- [ ] **Old gradient-attribution input-leaf approach** — superseded.
- [ ] **Replicate PWA-conservation finding** on 2–3 more proteins — partly subsumed by Phase 1 (we'll have predictions on ~92 proteins).
- [ ] Probe 7 / Probe 8 (MI / structural-contact density per column) — defer.
- [ ] **No-MSA control** — covered by Phase 2 single-sequence run.
- [ ] **Layer-3 PWA intervention** — defer until Phase 3 attribution is in.
- [ ] **SAE direction** (parked-but-attractive). Independent track; needs many WT runs to train an SAE. Revisit only after Phase 3.

## Done — 2026-05-09

- [x] Strategic pivot to fold-switch replication. Reasoning + paper details in `log/2026-05-09-foldswitch-pivot.md`.
- [x] `scripts/foldswitch_setup.py`: parses TableS1, fetches RCSB FASTA, emits 92 Boltz YAMLs + manifest.csv.
- [x] First pass of TableS1: 92/93 sequences resolved (1 obsolete PDB skipped); length distribution median=234, max=1369.
- [x] `scripts/foldswitch_extract_zenodo_msa.py`: pulls deep ColabFold MSAs from Porter's Zenodo dump (`AFcluster_MSAs/` standalone + bulk `sub_*` archives, with auto-concat of split pieces). 92/92 coverage (88 Fold1, 4 Fold2 fallback).
- [x] `scripts/foldswitch_finalize.py`: renames to `seq_NNNNN.{yaml,a3m}` (5-digit), writes `identities.tsv`, rewrites `msa:` lines to relative or absolute (--abs_root) paths. Idempotent.
- [x] User rsynced `yamls/` + `msa/` to cluster at `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/foldswitch/`, ran `foldswitch_finalize.py --abs_root` on cluster, launched Phase 1.

## Done — 2026-05-10

- [x] **Phase 1 predictions complete** — Boltz-2 produced predicted `.cif` for all 92 in-set fold-switchers (rsynced to local Mac). Settings: `--recycling_steps 10 --diffusion_samples 25 --override`, top-confidence sample kept per protein.
- [x] **Extended `score_sequences.py`** with pairs mode (`--pairs-manifest`, `--refs-dir`) reusing `scoring/utils.py` primitives. Per-chain naming + full-structure fallback in `resolve_reference`. CLI verified under `uv run`.
- [x] **Phase 1 scoring done** locally over all 92 pairs. Whole-chain CSV + fold-switch region CSV both written under `/Users/thomasbush/tmp-data/tmp_data/foldswitch/`.
- [x] **Porter cross-check** vs `success_using_TMscore_metric_revised.xlsx` — set overlap with AF2.3.1/AF3/AF_Cluster/ACE per-method success lists. 36 net-new "both" successes not in any AF method. Recycle confound (10 vs 3) identified.

## Done — 2026-05-07

- [x] Bring up plain-gradient attribution pipeline end-to-end on Boltz2. Three debugging gotchas captured in the log.
- [x] First mutant analysed (`seq_00132`, p40, 39.9% mutation density). Loss=1.215 Å², peak 23 GB, 4.6 s forward+backward.
- [x] Wire MSA-channel attribution (`msa_module.msa_proj` hook).
- [x] Whole-structure-loss option (`--whole_structure` flag).
- [x] Block + whole runs done; comparison plot produced. Findings: argmax-layer=0 reproduces, block/whole maps highly correlated (Pearson 0.80–0.99), query/MSA dissociation in top hits (query ⇒ mutated near-chromophore residues; MSA ⇒ chromophore itself).
