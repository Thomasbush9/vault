# Agenda

Current focus and the next 1–3 concrete things. When done, fold into a `log/` entry; when parked, move to `decisions.md`.

## Now — Replicate Porter et al. 2024 fold-switch memorization on Boltz-2, then layer interpretability

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
