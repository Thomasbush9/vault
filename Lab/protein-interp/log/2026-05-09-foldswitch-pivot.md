# 2026-05-09 — Pivot to fold-switch memorization replication

Strategic pivot away from "single-protein GFP interpretability" toward
**replicate Porter et al. (2024) on Boltz-2, then layer interpretability on
the memorized-vs-novel contrast**. The GFP work isn't dead — it becomes one
case study within a broader frame, and the gradient/IG tools we built carry
over unchanged.

## Why pivot

- **The previous frame was tool-rich, question-vague.** Single sequence × single ground-truth structure means our gradient attribution maps had nothing to be *measured against*. Block-vs-whole correlations of 0.80–0.99 are a soft-negative ("chromophore isn't special") and we were heading toward a methods note rather than a result paper.
- **Mech-interp papers land when the behavioural phenomenon is sharp first**, then attributed. Fold-switching gives that for free: same sequence, two known PDB folds, model picks one — "which one and why" is a clean, contrastive question that exists for ~99 proteins, not just one.
- **The replication is fast** (data fully released; see below), and even if interp turns out muddy we still have a publishable empirical result on Boltz-2.

## Reference paper

Chakravarty, Schafer, Chen, Cotten, **Porter** (2024).
"AlphaFold predictions of fold-switched conformations are driven by structure memorization."
*Nat Commun* 15:7296. https://doi.org/10.1038/s41467-024-51801-z

- **Headline numbers**: TM-score > 0.6 success rate **35 % on training-set fold-switchers** (32 / 92) vs **14 % on out-of-set fold-switchers** (1 / 7). The gap *is* the memorization signature.
- **Released artefacts**: `github.com/ncbi/AF2_benchmark` (plotting + sub-MSAs), Zenodo DOI `10.5281/zenodo.13221957` (full MSAs + log files), Supplementary Data 1 (TableS1 = the pair list), TableS3 (per-pair top-1/top-10/all success counts).
- **AF variants tested**: AF2.3.1 (mono + multimer), AF2.2.0, AF3 webserver, ColabFold 1.3 / 1.5.2 / 1.5.5.
- **Their ablations** (these are the controls that turn correlation into causation, and we get to replicate them on Boltz):
    - **Single-sequence** (no MSA) — does prediction collapse?
    - **AF-Cluster** — shallow subfamily-specific MSAs.
    - **SPEACH_AF** — alanine-mask MSA columns (this is *literally* one of the IG baselines we were considering for GFP-IG; alignment is methodologically perfect).
    - **Zero-recycling**, **AF2Rank** energy ranking, **with/without partners and cofactors**.

## 3-phase plan

1. **Phase 1 — Replicate** (~1 week). Run Boltz-2 default on all ~92 fold-switchers, score TM(G1) and TM(G2) per prediction, report success rate. **Headline question:** does Boltz-2 reproduce or *amplify* the 35 %/14 % gap? (Boltz-2 is trained later → almost everything is in-set → expect amplification. If we don't see it, that's surprising and informative.)
2. **Phase 2 — Ablate** (~1 week). Replicate **single-sequence** and **SPEACH_AF** on Boltz-2 minimum. These are the causal controls — without them the interp story has no backbone.
3. **Phase 3 — Interpret** (~1–2 weeks). Re-use the existing IG / gradient pipeline (no new ML to build, only target-loss changes):
    - For each pair, run attribution with **two targets**: ‖predicted-distogram − D(G1)‖ and ‖predicted-distogram − D(G2)‖. The *difference* between the two attribution maps is the memorization-direction signal.
    - Check: in memorized cases, does **MSA-channel** attribution dominate **query-channel** attribution? Which Pairformer layers carry the signal? Which **MSA columns** are most attributed (cross-check vs SPEACH_AF column-knockout impact — they should agree).
    - For the rare not-memorized cases, contrast attribution maps to memorized cases.

## Conditional follow-on (option 2 from the chat)

If Phase 3 shows MSA attribution concentrated on a small set of conservation-bearing columns ⇒ "can pLM embeddings (ESM2 / ProtT5) provide that signal without the memorization side-effect?" becomes a hypothesis-driven follow-up. **Not committed.** Only run this if Phase 3 produces clean results — otherwise it's Boltz architectural surgery for a guess.

## What was built today

- `scripts/foldswitch_setup.py` (~200 lines, single file).
    - Reads `TableS1.xlsx` from the AF2_benchmark repo (cloned to `/tmp/AF2_benchmark` locally).
    - For each Fold1/Fold2 pair, fetches full chain SEQRES from RCSB PDB FASTA API (`https://www.rcsb.org/fasta/entry/<PDB>`), parses out the auth-chain sequence.
    - Validates Fold1 vs Fold2 sequences: `identical`, `substring_use_longer`, `diverge_use_fold1`, etc.
    - Emits `manifest.csv` + `manifest.json` + per-pair `seq_NNNN_<fold1>_<fold2>.yaml` Boltz inputs.
    - `--msa_mode auto` (default) **omits the `msa:` field** so Boltz's built-in MMseqs2 server generates the MSA at inference. `--msa_mode empty` writes `msa: empty` for the single-sequence ablation (Phase 2).
    - Caches RCSB FASTAs under `<out_dir>/cache/` so re-runs are free.
- Output (already on disk):
    - `/Users/thom/tmp_data/tmp_data/protein_interpretability/data/foldswitch/`
        - `manifest.csv`, `manifest.json`, `cache/`, `yamls/seq_*.yaml` (92 files).

### Sequence-resolution stats (92 successful, 1 errored)

| seq_status | count | meaning |
|---|---|---|
| `identical` | 29 | F1 and F2 SEQRES match exactly — true conformational fold-switchers |
| `substring_use_longer` | 16 | one is a fragment of the other (truncated PDB) |
| `diverge_use_fold1` | 45 | F1 and F2 PDBs have non-identical sequences — used Fold1 as canonical |
| `fold1_missing_use_fold2` | 1 | chain not found in Fold1 PDB |
| `fold2_missing_use_fold1` | 1 | chain not found in Fold2 PDB |
| `error: 1FZP HTTP 404` | 1 | PDB 1FZP is obsolete; pair `2frhA / 1fzpD` skipped |

**Note on `diverge_use_fold1` (45 / 92):** the paper's "fold switchers" sometimes means *homologs that fold differently*, not strictly the same sequence in two folds. For Phase 1 we feed Fold1's sequence and score against both folds (matches what the paper does on TableS3). Revisit if results don't replicate.

### Length distribution

```
min=26  max=1369  median=234  mean=295
p25=147   p75=390   p90=510
<100        : 11
100–300     : 41
300–500     : 29
500–1000    : 10
≥1000       :  1   (4zt0C/4cmqB, 1369 aa — likely needs special handling)
```

## Open issues / cluster handoff

1. **MSA strategy.** Three options for Phase 1 default:
    - **Boltz auto-MMseqs2** (current YAMLs). Closest match to the paper's ColabFold pipeline; just run.
    - **Reuse Porter's MSAs** from Zenodo `10.5281/zenodo.13221957`. Tightest reproducibility but adds a download + path-rewrite step.
    - Mix and match later if results disagree.
    *Decision pending.*
2. **The 1369 aa entry** (`4zt0C/4cmqB`) may OOM Boltz-2 on a single H100. Either chunk to the chain-of-interest, exclude from Phase 1, or run on its own with reduced settings.
3. **`diverge` pairs** (45/92): should we also predict Fold2's sequence as a separate input? Doubles the cohort to ~137 predictions. Not for Phase 1 — defer until headline-replication numbers are in.
4. **Out-of-training set (7 fold-switchers).** The paper's 14 % control is on these. Not in TableS1 — likely in another supplementary file. Need to locate before Phase 1 reporting.

## Reproduction (local)

```bash
# Re-clone paper repo if /tmp/AF2_benchmark is gone
git clone --depth 1 https://github.com/ncbi/AF2_benchmark /tmp/AF2_benchmark

# Build manifest + YAMLs (auto-MSA mode)
python3 scripts/foldswitch_setup.py \
    --table_s1 "/tmp/AF2_benchmark/supporting tables/TableS1.xlsx" \
    --out_dir  /Users/thom/tmp_data/tmp_data/protein_interpretability/data/foldswitch

# Single-sequence ablation variant (writes a parallel yamls/ subdir or
# rerun with a different --out_dir)
python3 scripts/foldswitch_setup.py \
    --table_s1 "/tmp/AF2_benchmark/supporting tables/TableS1.xlsx" \
    --out_dir  /Users/thom/tmp_data/tmp_data/protein_interpretability/data/foldswitch_singleseq \
    --msa_mode empty
```

## Reproduction (cluster sketch — TODO once user confirms paths)

- `rsync` the `yamls/` dir + `manifest.csv` to `/n/holylfs06/.../foldswitch/`.
- Job array: 1 task per YAML; existing `run_boltz_extract.slrm` pattern works (just need a thin wrapper that calls `boltz predict <yaml> --out_dir ...` per task).
- Score: `tmtools` Python binding for TM-score against G1.cif and G2.cif (PDB downloads from RCSB).

---

## Same-day continuation (afternoon → evening 2026-05-09)

### What got built

Two more scripts to take the foldswitch dir from "list of sequences" to
"cluster-ready Boltz inputs":

1. **`scripts/foldswitch_extract_zenodo_msa.py`** — pulls the deep MSAs out of Porter's Zenodo dump and lays them down at `<foldswitch>/msa/seq_NNNN_<f1>_<f2>.a3m`. Auto-concats the split bulk archives (`sub_{0,4,5,6,8}.tar.gz??`), extracts both bulk and standalone tarballs to `_extracted/` (cached, idempotent), and rewrites each YAML's sequence to match the MSA's query line so Boltz won't reject mismatches.

2. **`scripts/foldswitch_finalize.py`** — renames everything to `seq_NNNNN.{yaml,a3m}` (5-digit, preserving TableS1 idx so `seq_00014` is left empty for the obsolete 1FZP entry), writes `identities.tsv`, and rewrites each YAML's `msa:` line to either relative (`msa/seq_NNNNN.a3m`) or absolute (`<abs_root>/msa/seq_NNNNN.a3m`) form. Idempotent — running again is safe.

### Important Zenodo-archive findings (debug-relevant)

I almost missed this on first inspection: **the Zenodo deposit contains the deep ColabFold MSAs, not just the AF-Cluster shallow ones.** The deep MSA for protein `<pdb>_<chain>` is at:

- **Standalone tarballs** (13 of them, in `AFcluster_MSAs/<pdb>_<chain>.{tar.gz,tgz}`): top-level a3m file inside the extracted dir. **Filename varies**: `<pdb>_<chain>/0.a3m` for some, but `5.a3m` / `13.a3m` / `2.a3m` / `17.a3m` / `10.a3m` for others. The current extractor handles this by globbing `<dir>/*.a3m` at the top level (excluding `<Y>_msas/` subdirs).
- **Bulk sub_X archives** (`sub_0` … `sub_9`, indexed by `info_all_runs.txt`): the deep MSA for index Y in sub_X is at `sub_X/<Y>.a3m`. The format of the index line is `<sub_dir> <Y>_msas <pdb_chain>` — the `_msas` suffix is the directory of *shallow clusters*, the *deep* MSA is the corresponding `<Y>.a3m` file at the same level.
- **Split archives** (`sub_{0,4,5,6,8}.tar.gz{aa,ab,ac,ad}`): need concat before extract. `ensure_sub_archive_concatenated()` handles this and writes a consolidated `sub_X.tar.gz` next to the pieces.

Boltz path resolution: confirmed by reading `references/boltz/src/boltz/main.py:605` — `msa_path = Path(msa_id)` is taken literally from the YAML, no implicit relative-to-yaml resolution. So either run with `cd <foldswitch_root>` and use relative `msa/seq_NNNNN.a3m`, or write absolute paths up front. We support both.

### Coverage outcome

After running both scripts:

```
foldswitch/
  yamls/seq_00001.yaml … seq_00093.yaml  (92 files; seq_00014 skipped — 1FZP obsolete)
  msa/seq_00001.a3m   … seq_00093.a3m   (~187 MB total)
  identities.tsv                          (92 rows: seq_id, idx_tableS1, fold1, fold2, chain_used, msa_status, seq_len, yaml, msa)
  manifest_with_msa.csv                   (full provenance per pair)
  _extracted/                             (~804 MB cache, exclude from rsync)
```

- **88 pairs use Fold1 MSA + Fold1 sequence**.
- **4 pairs use Fold2 MSA + Fold2 sequence** (Fold1 not present in Zenodo). For these, the YAML's `sequence:` field was rewritten from Fold1's RCSB SEQRES to the Fold2-MSA query, and the `chain_used` column in `identities.tsv` records the swap. **When scoring, score these 4 against the Fold2 PDB structure, not Fold1.**
- **3 pairs had Fold1-SEQRES ≠ MSA-query** (subtle truncations or version differences); YAMLs use the MSA query, manifest_with_msa.csv flags them as `ok_bulk_diverge` etc.

MSA depth ranges from 5 (`4g0dZ/4fu4C`) to 16 134 (`2mwfA/2nntA`), median ~3 000 — typical ColabFold MMseqs2 output.

### Cluster handoff state (where work paused)

Local foldswitch dir is **renamed + relative paths + identities.tsv written**. User confirmed cluster destination as:

```
/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/foldswitch
```

…with `yamls/`, `msa/`, `outputs/` already present on the cluster (rsync done by the user in this session). Cluster repo is at `/n/home06/tbush/protein_interpretability/` per `scripts/run_boltz_extract.slrm:32`.

**Pending: rewrite YAML msa paths to absolute on cluster.** One command to run, no GPU needed, on a login node:

```bash
cd /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/foldswitch
python /n/home06/tbush/protein_interpretability/scripts/foldswitch_finalize.py \
    --foldswitch_dir . \
    --abs_root      /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/foldswitch
```

Prerequisites on the cluster:
- The script must be present at `/n/home06/tbush/protein_interpretability/scripts/foldswitch_finalize.py` — `git pull` if needed.
- `manifest_with_msa.csv` should be in the foldswitch dir for `identities.tsv` to populate fold1/fold2/chain_used columns. If absent, rename + path-rewrite still work but identities.tsv will be sparse.

Sanity check after running — `head /n/holylfs06/.../foldswitch/yamls/seq_00001.yaml` should end with `msa: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/foldswitch/msa/seq_00001.a3m`.

### Next steps (Phase 1 launch)

1. Run `foldswitch_finalize.py --abs_root …` on cluster (above).
2. Write a tiny SLURM wrapper around `boltz predict yamls/seq_NNNNN.yaml --out_dir outputs/seq_NNNNN/` — array job, one task per yaml. Pattern: `scripts/run_boltz_extract.slrm` minus the orchestrator (we just want vanilla `boltz predict`).
3. **Outlier handling:** `seq_00043` (4zt0C/4cmqB, 1369 aa) probably OOMs an H100. Either exclude up front or give it its own job with reduced settings.
4. Score: per prediction, TM-score(predicted, G1) and TM-score(predicted, G2). Get G1/G2 .cif from RCSB. Use `tmtools` python binding (`pip install tmtools`).
5. Headline numbers wanted from Phase 1: `(% with TM>0.6 to G1)` and `(% to G2)` for in-set vs out-of-set pairs. Compare against Porter's 35 % / 14 %.

### Out-of-set fold-switchers (Porter's 14 % control)

Still **not located** in the supporting tables. The 7 out-of-set pairs are referenced in the paper text but absent from `TableS1.xlsx`. Likely candidates to check:

- Other supporting-table xlsx files: `TableS2.xlsx` (multiple sheets — already inspected, methods-specific), `Data_FigS3_revised.xlsx`, `data_FigS4_revised.xlsx`, `data_FigS5.xlsx`, `data_FigS6.xlsx`, `TableS3_revised.xlsx`, `success_using_TMscore_metric_revised.xlsx`.
- Or in the GitHub `notes/` / `scripts/` text files.
- Without these 7, Phase 1 reports only the in-set rate, not the gap. Worth digging up before writing results.

### Reproduction commands (full chain, end-to-end)

```bash
# 1. Build manifest + initial yamls (auto-MSA mode — used here only to seed
#    the manifest; we'll overwrite the yamls' msa: line in step 3)
python3 scripts/foldswitch_setup.py \
    --table_s1 "/tmp/AF2_benchmark/supporting tables/TableS1.xlsx" \
    --out_dir  /Users/thom/tmp_data/tmp_data/protein_interpretability/data/foldswitch

# 2. Extract Zenodo MSAs into msa/ + rewrite yamls with the deep MSA paths +
#    update each yaml's sequence: to the MSA query when they differ
python3 scripts/foldswitch_extract_zenodo_msa.py \
    --foldswitch_dir /Users/thom/tmp_data/tmp_data/protein_interpretability/data/foldswitch \
    --zenodo_root    /Users/thom/tmp_data/tmp_data/protein_interpretability/data/foldswitch/porterll-AF2_benchmark-40a57d7

# 3. Rename to seq_NNNNN, write identities.tsv (relative paths for local use)
python3 scripts/foldswitch_finalize.py \
    --foldswitch_dir /Users/thom/tmp_data/tmp_data/protein_interpretability/data/foldswitch

# 4. After rsync to cluster, on the cluster:
python /n/home06/tbush/protein_interpretability/scripts/foldswitch_finalize.py \
    --foldswitch_dir /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/foldswitch \
    --abs_root       /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/protein_rsa/foldswitch
```
