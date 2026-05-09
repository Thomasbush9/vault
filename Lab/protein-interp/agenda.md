# Agenda

Current focus and the next 1–3 concrete things. When done, fold into a `log/` entry; when parked, move to `decisions.md`.

## Now — Replicate Porter et al. 2024 fold-switch memorization on Boltz-2, then layer interpretability

Pivot from "single-protein GFP IG attribution" → **replicate a known memorization phenomenon on Boltz-2 first, then attribute it**. Reasoning + 3-phase plan in `log/2026-05-09-foldswitch-pivot.md`. Boltz-2 is trained later than AF2, so we expect Porter's 35 %/14 % training-vs-out-of-set memorization gap to *amplify*. The contrastive setup (each fold-switcher has G1 and G2 ground truths) is exactly what was missing in the GFP work — IG/gradient tools already built carry over unchanged.

### Phase 1 — Replicate (this week)

- [ ] **MSA decision**: Boltz auto-MMseqs2 (current YAMLs) vs Porter's Zenodo MSAs (10.5281/zenodo.13221957). Default to auto unless we want exact reproducibility against the paper's TM-score numbers.
- [ ] **Cluster wrapper**: tiny shim around `boltz predict` for SLURM job array (one task per YAML, parallels `run_boltz_extract.slrm`). User estimates ~2 h wall time on H100 fleet.
- [ ] **Locate the 7 out-of-training-set fold-switchers** (the 14 % control). Not in TableS1 — likely in a different supplementary file. Needed for the headline `(in-set %) − (out-of-set %)` gap.
- [ ] **Decide on the 1369 aa outlier** (`4zt0C/4cmqB`): chunk, exclude, or run separately with reduced settings.
- [ ] **Score**: per prediction, compute TM-score(G1) and TM-score(G2) using `tmtools` against PDB-downloaded G1/G2 .cif structures. Report the success-rate gap.

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
- [x] User rsynced `yamls/` + `msa/` to cluster at `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/foldswitch/`. **Pending: run `foldswitch_finalize.py --abs_root …` on cluster to absolute-ize msa paths, then launch Phase 1 predictions.**

## Done — 2026-05-07

- [x] Bring up plain-gradient attribution pipeline end-to-end on Boltz2. Three debugging gotchas captured in the log.
- [x] First mutant analysed (`seq_00132`, p40, 39.9% mutation density). Loss=1.215 Å², peak 23 GB, 4.6 s forward+backward.
- [x] Wire MSA-channel attribution (`msa_module.msa_proj` hook).
- [x] Whole-structure-loss option (`--whole_structure` flag).
- [x] Block + whole runs done; comparison plot produced. Findings: argmax-layer=0 reproduces, block/whole maps highly correlated (Pearson 0.80–0.99), query/MSA dissociation in top hits (query ⇒ mutated near-chromophore residues; MSA ⇒ chromophore itself).
