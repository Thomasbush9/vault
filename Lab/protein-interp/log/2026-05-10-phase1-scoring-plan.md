# 2026-05-10 — Phase 1 predictions done, scoring plan staged

## State

Boltz-2 has produced predicted `.cif` structures for **all 92 fold-switcher YAMLs** on the cluster. Outputs live under `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/foldswitch/outputs/seq_NNNNN/`. Phase 1 *compute* is complete; Phase 1 *evaluation* (TM-score vs G1 and G2, replication of Porter's 35 %/14 % gap) is the active step. Setup pipeline (manifest → Zenodo deep MSAs → renamed `seq_NNNNN.{yaml,a3m}` → absolute msa paths → predict) details are in `2026-05-09-foldswitch-pivot.md`.

To confirm before scoring:
- **MSA used.** YAMLs were finalised with Zenodo deep-MSA absolute paths, so predictions almost certainly ran against Porter's MSAs (not Boltz auto-MMseqs2). Worth confirming once from a single output's `config` / log line.
- **`seq_00043`** (4zt0C/4cmqB, 1369 aa). Did it predict on H100 or hit OOM? Decides whether it's a missing row or just a row to flag in `tm_scores.csv`.
- **Top-K.** Boltz-2 default is 5 diffusion samples — score top-1 only for the headline number. Revisit top-K only if the in-set rate is borderline.

## Existing scoring code we reuse

`src/protein_interpretability/scoring/utils.py` already provides the primitives:

- `path_tm_score(ref, pred, chain_id, normalize_by="reference")` — `tmtools.tm_align`, normalized by reference (ground-truth) length, matches Porter's convention.
- `path_rmsd(ref, pred, chain_id)` — RMSD from the same alignment.
- `load_structure`, `extract_residue_coordinates`, `extract_residue_sequence` for chain-aware extraction.

`src/protein_interpretability/score_sequences.py` is the existing CLI. **Extended in place** rather than adding a new script (keeps "the scoring CLI" as one file). Two modes now:

- `--ref <path>` — original single-reference cohort scoring.
- `--pairs-manifest <tsv> --refs-dir <dir>` — pairs mode for fold-switch (per-pair G1/G2). Reads `identities.tsv`, parses each `fold1`/`fold2` field as `<pdb><chain>`, ensures `refs-dir/<pdb>.cif` exists (`--download-missing-refs` will fetch from RCSB), and scores the prediction against both refs with their respective auth chains. Output schema: `seq_id, idx_tableS1, fold1, fold2, chain_used, primary_fold, seq_len, predicted_path, tm_g1, tm_g2, rmsd_g1, rmsd_g2`.

Side fixes during the extension: `SEQUENCE_INDEX_PATTERN` was greedy (`seq_(\w+)` matched `00066_model_24`); narrowed to `seq_(\d+)`.

The pairs-mode helpers (`parse_fold_field`, `ensure_reference_cif`, `find_prediction_for_seq`, `score_pair`, `score_pairs`, `write_pair_scores`) live in the same file. No new scripts created.

## Step-by-step scoring plan

1. **Output layout (confirmed):** `outputs/sequences/seq_NNNNN/boltz/seq_NNNNN_model_24.cif`. One cif per dir, all 92 present (incl. `seq_00043`, the 1369 aa entry).
2. **References:** `--download-missing-refs` will fetch `https://files.rcsb.org/download/<pdb>.cif` into `refs/` for each unique PDB id in `fold1`/`fold2` on first run. Idempotent.
3. **Cluster invocation** (from `/n/holylfs06/.../foldswitch/`):

    ```bash
    uv run python -m protein_interpretability.score_sequences \
        --pairs-manifest identities.tsv \
        --refs-dir       refs/ \
        --download-missing-refs \
        --predicted-dir  outputs/sequences \
        --model-subdir   boltz \
        --output-dir     . \
        --output-name    tm_scores.csv \
        --normalize-by   reference
    ```
4. **Headline numbers.** Tiny analysis cell / notebook section:
    - `% TM>0.6` against G1, G2, either, primary-only.
    - Distribution of `|tm_g1 − tm_g2|` — how often Boltz commits to one fold vs hedges.
    - Compare to Porter's **35 % in-set**. Expect amplification (Boltz-2 is trained later).
5. **Out-of-set cohort (Porter's 14 % control).** Separate thread — blocks the *gap* but not the in-set number. Hunt the 7 pairs in other supporting xlsx files (`Data_FigS3_revised.xlsx`, `data_FigS4_revised.xlsx`, `data_FigS5.xlsx`, `data_FigS6.xlsx`, `TableS3_revised.xlsx`, `success_using_TMscore_metric_revised.xlsx`) or in `notes/` text in the AF2_benchmark repo. Once located: rerun setup → MSA → predict → score on those 7.

## Open decisions

1. Top-1 only for Phase 1 headline (currently the only `.cif` per `boltz/` dir, so this is moot until top-K is regenerated).
2. Whether to delete GFP-arc one-shots (`run_chromophore_attribution.py`, `analyze_chromophore_block.py`, `plot_chromophore_attribution.py`, `augment_*.py`, `build_seq_perturb_dataset.py`, `sample_mutations.sh`, `run_query_occlusion.py`) — pending user call.

---

## Results (afternoon 2026-05-10)

Everything ran locally — predictions + Porter dump are both on this Mac.

### What we ran

- `score_sequences.py --pairs-manifest identities.tsv` over `outputs/sequences/seq_NNNNN/boltz/seq_NNNNN_model_24.cif`, refs from `porterll-AF2_benchmark-40a57d7/AF2Rank/final2_and_debug2_inputs/` (Porter's cleaned single-chain PDBs in `<pdb>_<chain>.pdb` format — covers 183/186 needed; the one truly-missing chain `1x0g_D` was patched by dropping full `1x0g.pdb` into the same dir and letting `resolve_reference` fall back).
- Result: `/Users/thomasbush/tmp-data/tmp_data/foldswitch/tm_scores.csv` — 92 rows, no skips.
- Restricted-region rescoring (script lives inline in this session, not yet a CLI): used Biopython local pairwise alignment to map TableS1 col C ("Sequence of fold-switching region") into each of `predicted`, `ref1`, `ref2` chain sequences, sliced coordinates, re-ran `tm_score`/`rmsd`. Output: `/Users/thomasbush/tmp-data/tmp_data/foldswitch/tm_scores_region.csv` — 86 / 92 pairs (6 dropped on alignment failure or too-short slice).

### Recycle confound (load-bearing)

Predictions ran with `boltz predict ... --recycling_steps 10 --diffusion_samples 25 --override`. Only the top-confidence sample (`_model_24`) was kept per protein. AF2.3.1 default is 3 recycles; Porter's recycling ablation is **zero**. The whole-cohort headline is therefore not apples-to-apples vs the paper.

**Why:** Porter's argument is that AF memorizes templates baked into the *initial* representations. Zero recycles tests whether iterative refinement is doing the memorization. 10 recycles is the opposite end and biases toward whatever the model can converge to. **Re-running at `--recycling_steps 3` (AF parity) is the right next ablation** before claiming numerical comparison; `--recycling_steps 0` replicates Porter's ablation directly.

### Whole-chain TM-score (92 pairs, top-1 sample, all recycles=10)

| metric | Boltz | reference |
|---|---|---|
| % TM>0.6 vs G1 | 87.0 % (80/92) | — |
| % TM>0.6 vs G2 | 64.1 % (59/92) | — |
| % TM>0.6 vs both | **62.0 % (57/92)** | All_AF: 35 % (32/92), ACE: 61 % (56/92) |
| % TM>0.6 vs either | 89.1 % (82/92) | — |
| median `|tm_g1 − tm_g2|` | 0.128 | — |

Porter's per-method "both" counts (`success_using_TMscore_metric_revised.xlsx → counts`): AF2.3.1=8, AF2.2.0=8, AF2_multimer=12, AF3=7, AF_Cluster=18, All_AF=32, ACE=56.

### Per-pair set overlap (Boltz both>0.6 vs Porter)

Boltz's 57 "both>0.6" set intersected with Porter's per-method success sets:

| Porter list | size | ∩ Boltz | Porter-only (Boltz misses) |
|---|---|---|---|
| AF2.3.1 'both' | 8 | 5 | `2a73b/3l5nb`, `2k0qa/2lela`, `2kxoa/3r9jc` |
| AF3 'both' | 7 | 5 | `2ougc/6c6sd`, `2qkee/5jyta` |
| AF_Cluster CF1.5.2 | 14 | 8 | 6 misses |
| ACE (best combined) | 56 | 35 | 21 misses |

**36 of Boltz's 57 "both" successes are NOT in any Porter AF success list.** Those are net-new memorization signal — pairs no AF variant could solve. Worth eyeballing 2–3 of them once we have a real-vs-artefact verdict.

But: **Boltz is not a superset of AF2.3.1.** It misses 3 pairs AF2.3.1 gets. Could be definitional (Porter's TM threshold vs ours) or genuine Boltz failures. Tie-breaker = sanity-check on those 3 specific pairs.

### Restricted to fold-switch region (86 pairs, region-only)

Using TableS1 col C sequence + Biopython local alignment to identify the residue range in each of (predicted, ref1, ref2), then TM-align/RMSD on the slice only.

| metric | Boltz |
|---|---|
| median region length | 23 aa |
| TM-region median vs G1 | 0.563 |
| TM-region median vs G2 | 0.259 |
| RMSD<2Å vs G1 | 78 % (67/86) |
| RMSD<2Å vs G2 | 47 % (40/86) |
| **RMSD<2Å vs both** | **33 % (28/86)** |
| RMSD<2Å vs either | 92 % (79/86) |

**Decision pattern** (RMSD<G1 vs <G2 by ≥0.5Å margin): picks G1 = 58 %, picks G2 = 19 %, hedges = 23 %.

### Reading

1. Scaffold *was* inflating the whole-chain "both" rate. The "Boltz produces both folds" number drops from 62 % → 33 % when restricted to the actual switching region. So the strong version of "Boltz hedges between fold-switch conformations" is overstated; the moderate version ("33 % of the time, Boltz's switching region is structurally compatible with both folds") still holds.
2. **Strong fold-1 preference** (58 % pick G1, 19 % pick G2): consistent with memorization of the canonical PDB-fold-1 template.
3. **Region-restricted RMSD<2Å vs G1 = 78 %** is still way above any single AF variant. Boltz at recycles=10 is materially better at fold-switcher prediction, but the "predicts both folds" claim needs the region-restricted view to be defensible.

### Caveats / not-yet-tested

- **Recycle confound** (above) — repeat at recycles=3 before the headline.
- **Sample diversity** — only 1 of 25 diffusion samples kept; the "Boltz is flexible" question (does sampling span both folds?) is not yet answered.
- **6 pairs dropped from region analysis** (`seq_00009`, `seq_00026`, `seq_00040`, `seq_00054`, `seq_00064`, `seq_00069`) — region didn't local-align cleanly to G1/G2 ref or slice too short. Worth eyeballing the 6.
- **The 3 Boltz-misses-but-AF2.3.1-gets pairs** — methodological pothole or real signal?

### Files

- `/Users/thomasbush/tmp-data/tmp_data/foldswitch/tm_scores.csv` — 92 rows × {tm_g1, tm_g2, rmsd_g1, rmsd_g2, primary_fold, …}.
- `/Users/thomasbush/tmp-data/tmp_data/foldswitch/tm_scores_region.csv` — 86 rows × {tm_g1_region, tm_g2_region, rmsd_g1_region, rmsd_g2_region, region_len_g1, region_len_g2}.

### Next, in priority order

1. **Recycle ablation** — re-run Phase 1 predictions with `--recycling_steps 3` (AF default parity) and `--recycling_steps 0` (Porter ablation). Score with the same pipeline. This is the load-bearing apples-to-apples baseline.
2. **Out-of-set cohort** — still pending. Without it we can't report the in-set vs out-of-set memorization gap (Porter's headline).
3. **Lift region-restricted scoring into a CLI flag** of `score_sequences.py` rather than the in-session inline script.
4. **Spot-check the 3 AF2.3.1-only pairs and 6 region-drop pairs.**
