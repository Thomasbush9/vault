# Boltz MSA Attention — Conservation Analysis Log

Notebook: `protein_interpretability/src/protein_interpretability/notebooks/pwa_conservation.ipynb`
Data: `/Users/thomasbush/tmp-data/tmp_data/hidden_reps/boltz_results_chunk_0/`
Conditions extracted on the cluster: `original`, `seq_18553`, `random_freq`, `random_uniform`. All four share the WT MSA hits (UniRef rows are identical); only row 0 (the query) differs.
Tensor shape per condition × step × MSA layer: `pwa_weights` ∈ R^(1, 8, 238, 238) — 8 heads, 238 residues. Steps saved: 0, 5, 10. Layers: 0–3.

## 1. Why this analysis

We've already shown (msa-swap experiment, see `MSA Sanity Check.md`) that Boltz produces TM ≈ 0.33 even when the query is a uniform-random sequence, as long as the WT MSA is preserved. That's well above the random-pair baseline (~0.2) but below the same-fold threshold (0.5). Two explanations are possible:

(a) The model uses the MSA as a fold prior nearly independently of the query — a "cheat sheet" — and the residual structural signal in the prediction is whatever the MSA alone supports.

(b) The model uses the query sequence as a major input, but the MSA contributes some structural prior on top.

These differ in *what the internal computation is doing*, not just in the output. The PWA (PairWeightedAveraging) attention weights from the MSA module are the natural place to test this: each pwa softmax row says "for a given query position, which MSA-derived signal do I read?"

The leashbio article is in a different setup (Hermes / protein–ligand attention pooling), but the conceptual move is the same: does the model attend to query-informative regions, or to a query-invariant template? PWA at the MSA-pair junction is the corresponding probe for cofolding models.

## 2. Setup

Three derived signals per (condition, step, layer):

- **column attention** A_j = mean_{h, i} W[h, i, j], normalised over j. "How much position j is read from."
- **row entropy** H_i, mean over heads of −Σ_j W[h, i, j] log W[h, i, j]. Focus vs. spread.
- **per-head column attention** A_j^h, kept separately to detect head specialisation that the head-average would mask.

Conservation reference: WT MSA column distributions →
- `cons_entropy[j] = 1 − H_col[j] / log(K)` (K = 21 alphabet),
- `cons_max[j] = max_a freq[j, a]` (consensus frequency).

Mutant positions for `seq_18553`: row-0 disagreement vs WT — 96/238 ≈ 40% mutated. (This is essentially a p40 condition.)

## 3. Results (single protein, GFP)

### A_j is nearly invariant to the query (cheat-sheet at the level of *what is attended to*)

JS divergence between A_j vectors at step_10 is tiny across conditions:

| layer | orig vs seq_18553 | orig vs rand_freq | orig vs rand_uniform |
|------:|------------------:|------------------:|---------------------:|
| 0 | 0.004 | 0.008 | 0.008 |
| 1 | 0.001 | 0.003 | 0.003 |
| 2 | 0.002 | 0.007 | 0.008 |
| 3 | 0.004 | 0.010 | 0.013 |

Reference: JS between two random distributions ≈ 0.7. So A_j across all four conditions is essentially the same vector. Whatever the model is reading from the MSA, it's reading it regardless of the query — including a uniform-random query.

Two real (small) signals on top of that:
- mutant is consistently closer to WT than the random queries (model registers the query, weakly);
- divergence grows with depth — layer 3 is the most query-aware. By layer 3, JS(orig, rand_uniform) ≈ 3× JS(orig, seq_18553), suggesting late MSA layers do partially differentiate the query.

Layer 1 is the most pure-MSA layer (smallest absolute JS).

### Spearman ρ(A_j, conservation_j) is *negative*, not positive

This was the surprise. The naive cheat-sheet story predicts positive ρ — attention concentrated on highly-conserved (template-like) columns. We see the opposite at the layers that have signal:

- layer 0, 2: |ρ| < 0.15 (no clear relationship)
- layer 1: ρ ≈ −0.40 (all conditions, all steps after step_0)
- layer 3: ρ ≈ −0.30 (all conditions)

So Boltz's MSA attention concentrates on **variable** columns, not conserved ones. Variable columns are where coevolutionary information lives: paired column variability under selection is what tells you which residues are spatially proximal. So the negative-conservation correlation is consistent with the model running coevolution on the MSA — but **doing it independently of the query**, since all four conditions show the same pattern.

This sharpens the cheat-sheet interpretation: the model isn't memorising fixed-residue templates, it's running a coevolution-aware computation on the MSA, and that computation is largely query-blind. Which still explains the TM ≈ 0.33 floor on random queries — the MSA columns alone carry enough coevolutionary signal to bias toward the WT fold.

### No mutant-position-specific attention

In `seq_18553`, the 96 mutated positions receive Σ A_j ≈ 0.40, matching the uniform baseline 96/238 = 0.403 and matching what the random conditions get on the same positions. The model does not redistribute attention onto / away from positions where the query disagrees with the MSA consensus. This is a stronger version of the query-blindness result.

### Specialised heads exist; head-average obscures them

The Spearman bars in plot 1 are head-averaged; per-head heatmap (plot 4) tells a more nuanced story:
- `original` layer 2 head 4: ρ ≈ −0.6 (strongly variability-locked)
- `seq_18553` layer 2 head 0: ρ ≈ +0.4 (conservation-locked, only in this condition)
- random conditions are paler — head specialisation collapses when the query is meaningless.

The seq_18553-specific positive head suggests the model has heads that *do* condition on query content, but they're a minority and the head average blurs them out.

### Recycling stabilises after step 5

step_5 and step_10 are nearly identical for every metric. step_0 differs, particularly at layer 1 where the conservation correlation is weaker / sign-flipped at step_0. So one full recycle is enough for the MSA attention pattern to converge to its steady-state.

### Row entropy

H_i is low-amplitude across conditions but ordered consistently: original < seq_18553 < random_*. WT query reads slightly more focused than mutant, mutant slightly more focused than random. The amplitude is small (Δ ≈ 0.2 nats). Layer 3 is essentially uniform-uniform across all four (H ≈ 5.0, log(238) = 5.47).

## 4. Reading

The single-protein result reads as: **Boltz's PWA attention runs a coevolution-aware read of the MSA columns largely independently of the query.** Variable, co-varying columns are weighted up; conserved columns down. The query identity barely changes which columns get read — only later layers (especially 3) drift slightly with query content.

This is a "cheat sheet", just not the cheat sheet I expected. The cheat isn't memorisation of conserved residues — it's running coevolution on a fixed MSA regardless of whether the query is biologically related to that MSA. From the model's standpoint that's a sensible architectural prior; from an interpretability standpoint it's exactly what would explain residual structural similarity on random-query inputs.

## 5. Caveats

- **N = 1 protein.** All numbers are for GFP. Need to replicate on at least 2–3 more proteins of varying MSA depth before this is a result.
- The mutated condition (`seq_18553`) is at p40. We do not yet have a low-perturbation point (p10, p20) on this same plot. The query-aware effect at layer 3 might be larger or smaller for milder mutants.
- Spearman magnitudes are modest (|ρ| ≤ 0.5). Conservation explains at most ~25% of the column-attention rank variance.
- PWA is intra-layer but `b = proj_z(z_n)` feeds the recycled pair representation in. The recycle-step stability we see is therefore non-trivial: the pair representation evolves across recycles, but the MSA attention pattern barely budges. Worth noting.

## 6. Next probes (added to the notebook)

7. **MI / APC-corrected MI per column.** Replace conservation with a coevolution score (mean APC-corrected MI of column j with all partners at sequence separation > 5). Hypothesis: ρ flips strongly positive — i.e. attention tracks coevolution.
8. **Structural contact density.** Use the WT structure (ESMFold prediction here, but ideally Boltz's own WT prediction) to compute per-column long-range C_α-contact count. Correlate A_j with that. If the cheat is genuinely structural (the MSA tells the model where contacts are, regardless of query), we'd expect a positive ρ at the same layers (1, 3) where the conservation correlation is most negative.

## 7. Open questions / followups not yet wired up

- **Head-resolved follow-up.** Pull `seq_18553` layer-2 head 0 alone — the unusually conservation-positive head — and visualise its (N, N) attention matrix vs the same head in `original`. There's something query-specific happening there that the head average buried.
- **Replication.** Same analysis on 2–3 more proteins to check whether the layer-1 / layer-3 negative-conservation pattern is GFP-specific or general.
- **Perturbation gradient.** p10, p20, p40, p70 on the same protein, all paired with WT MSA. Plot the layer-3 JS(orig, mut) as a function of p. Where does the model start to "notice" the mutation?
- **No-MSA control.** Run the same pipeline on Boltz with no MSA (just the query). If the column-attention pattern collapses to something query-driven, that confirms the MSA is what's driving the query-invariance.
- **Coevolution vs contacts overlap.** If both the MI score and the contact density correlate positively with A_j at the same layers, decompose: is attention specifically on coevolving-and-contacting columns, or is one of the two doing all the work?
- **Intervention.** If layer 3 is where small sequence-awareness creeps in, that's the layer to inject representations at — replace the layer-3 PWA output of `random_uniform` with the corresponding output of `original` and see whether the structural prediction shifts further toward / away from WT.

## 8. Provenance

- Cluster extraction: `scripts/run_boltz_attention.py` with `boltz_attention_config.yaml`, `layer_sites: [pwa_weights]`, `recycling_steps: 10`, `recycling_steps_to_save: "0,5,-1"`, `msa_layers: all`, `average_heads: false`, `no_kernels: true`.
- Sequences and YAMLs at `/n/holylfs06/.../msa-swapping/sequences/{original,seq_18553,random_freq,random_uniform}/`.
- Local notebook reads attention `.pt`s and the WT a3m only; nothing model-runtime.
