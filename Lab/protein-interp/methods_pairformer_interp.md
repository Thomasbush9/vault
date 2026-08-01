# Methods — mechanistic interpretability of the Boltz-2 trunk under mutation

Companion to `log/2026-07-30-jax-pairformer-harness.md` (chronological log,
including the corrections). This file is the technical reference: what was
computed, exactly how, with what controls, and which figure shows it.

Code: `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/harness/`
Outputs: `../runs/*.json|npz`  ·  Figures: `../figures/*.png`
Bundle: `../figures_bundle/` (git repo) and `figures_bundle.tar.gz`

---

## 1. Model and harness

**Substrate.** Boltz-2 via **joltz**, the from-scratch JAX/equinox
reimplementation shipped in `mosaic_setup/images/mosaic.sif`
(`/opt/mosaic/.venv/.../joltz/__init__.py`). PyTorch is used exactly once, at
checkpoint load (`mosaic.losses.boltz2.load_boltz2` → `joltz.from_torch`).
Weights: `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db/boltz2_conf.ckpt`.

**Why not the PyTorch stack.** joltz stores the Pairformer and MSA stacks as a
*stacked* parameter pytree plus a `static` template, executed by `jax.lax.scan`.
Three consequences that the PyTorch hook stack cannot match: per-layer capture
is obtained by re-running the same scan with the `ys` slot populated (no
patching of the library); component ablation is a single `eqx.tree_at` on
`stacked_parameters`, leaving all other layers bit-identical; and path patching
is a field swap on an `InitialEmbedding` pytree rather than a forward hook that
must survive `torch.compile` and activation checkpointing.

**Architecture constants (boltz 2.2.x, verified against the loaded model):**
64 Pairformer blocks; 4 MSA blocks; `s` = 384, `z` = 128; distogram = 64 bins
over 2–22 Å; triangle attention 4 heads; `AttentionPairBias` 16 heads.

**Harness equivalence — this is load-bearing.** `pi_core.iteration` reimplements
`joltz.Joltz2.trunk_iteration`. `harness/test_equivalence.py` shows it is
**bit-identical**: relative error exactly 0.0 on `s`, `z`, distogram and contact
map, with capture on and off, against `model.trunk_iteration` given the same key
and state. Determinism is enforced by `deterministic=True` (dropout off,
confirmed by different keys giving identical output) and `subsample_msa=False`
so MSA depth is an exactly controlled quantity rather than a random 1024-row draw.

*Trap recorded:* an earlier `trunk_capture` ran some recycles through joltz's
jitted `trunk_iteration` and only the last through the capture path, producing a
~6e-4 relative drift that looked like float32 fusion noise but was two different
computations. There is now **one** definition of `iteration`.

**Execution.** All model work via SLURM (`harness/run.sbatch`, partition
`kempner_h100`, account `kempner_bsabatini_lab`) inside the mosaic container
(`singularity/mosaic-exec.sh`). Nothing heavier than file parsing runs on a
login node.

---

## 2. Units — the single most important methodological decision

Two rules, each of which was learned by getting it wrong first.

**2.1 Divergence, not Ångström, for anything compared across depth.**
`E[d] = Σ_b softmax(logits)_b · centre_b` is a nonlinear readout. The distogram
sharpens with depth (wild-type entropy 2.16 → 0.81 nats over the last eight
layers), so a fixed perturbation in logit space moves `E[d]` less at the output
than mid-stack. Reporting in Å therefore manufactures an apparent "suppression"
that is an artefact of the measuring instrument. **All per-layer quantities use
symmetric KL** over the 64 distogram bins, and entropy is reported alongside.
→ `figures/kl_gfp.png`

**2.2 Every quantity needs a denominator or a control.** Four conclusions in
this project were wrong on first pass for want of one: Å vs KL (units);
absolute growth vs growth relative to a scrambled-sequence control; raw vs
partial correlation; and a hand-rolled TM-score vs `tmtools`. The controls are
now built into the scripts rather than applied afterwards.

**2.3 Structure comparison is `tmtools` only.** A hand-rolled TM-score scored
two Boltz-2 predictions of the *same* sequence at 0.697 where tmtools gives
0.978, which produced two false conclusions (a spurious "sampler noise floor"
and spurious mutant-vs-WT divergence). `harness/geom.py` now wraps
`tmtools.tm_align` and carries the failure history in its docstring.
Validation: self-comparison 1.000; shuffled-residue floor 0.155; JAX wild type
vs the independent PyTorch Boltz-2 prediction of the same sequence 0.978
(RMSD 1.10 Å).

---

## 3. Cohorts

### 3.1 GFP "physics vs memory" (`harness/build_dataset.py` → `data/gfp_physics`)

avGFP, 238 aa, MSA depth 731 (from the ProtForge run at
`test_protforge/outputs/sequences/34073`).

- **core_{1,2,4,8,16,32}** — buried hydrophobic → charged. Burial = Cα
  neighbour count within 10 Å computed on the Boltz-2 wild-type prediction;
  sites taken most-buried-first among hydrophobics above the 75th burial
  percentile. Substitutions are a deterministic function of the site index, so
  the series is **nested**: the n=4 mutant is the n=32 mutant restricted to its
  first four sites. Without this, a dose-response confounds "more mutations"
  with "different substitutions".
- **surface_{1..32}** — exposed non-hydrophobic → polar, count-matched control.
- **rand_{5,10,20,40,70}%** — uniform random substitutions at the rates used in
  the adversarial-mutation literature. Necessary because core_32 is only 13.4 %
  mutated but deliberately maximally destabilising, so it is **not** comparable
  to a random-mutation rate.
- **scramble** — the wild-type composition permuted (5 % identity). The largest
  query-side perturbation available without touching the MSA; it calibrates
  every other number.

**Every variant carries the wild-type alignment verbatim**, row 0 rewritten to
the variant sequence. Two non-obvious requirements:

1. Boltz's featuriser silently substitutes a dummy MSA if row 0 ≠ the YAML
   sequence, which would make a run single-sequence without saying so.
2. Rows whose ungapped core equals the wild-type sequence are **dropped**. The
   wild type's own self-hit is present in a ColabFold alignment and Boltz
   deduplicates it against row 0 — but only for the wild type, whose query
   matches it. Left in, the wild type gets one fewer MSA row than every mutant
   (731 vs 732) and row-wise patching silently misaligns.
3. Chain id must be ≤ 5 chars (`MAX_CHAIN_NAME`); Boltz builds its chain→MSA map
   from the untruncated YAML id and raises `KeyError` deep in
   `parse_boltz_schema` otherwise, skipping the input.

### 3.2 DIO3 (`data/dio3_cohort`) — replication cohort
237 aa, unrelated fold, 3411-row alignment. Wild-type structure predicted with
`harness/predict_wt.py` (CA-only cif; geometry validated at CA–CA 3.79 Å,
Rg 17.19 Å before writing). Same construction as 3.1.

### 3.3 Ten-protein panel (`data/panel`)
Eight UniProt sequences spanning all-α, all-β, α+β, α/β (ubiquitin, lysozyme,
myoglobin, CheY, thioredoxin, barnase, staphylococcal nuclease,
acylphosphatase), plus GFP and DIO3. MSAs via ProtForge's own container and
shared MMseqs2 DB, invoked directly (`harness/msa_panel.sbatch`) rather than
through the Snakemake wrapper — one `colabfold_search` call for the batch, which
the msa_bench DECISIONS.md cost model says is the right shape (fixed ~30 min per
call + ~9 s/sequence; `--exclusive` because co-location costs 1.19×).

### 3.4 ProteinGym (`data/gym`)
All 217 substitution assays from Zenodo record 15293562. Four **Tsuboyama 2023**
folding-stability assays selected (RCRO_LAMBD 63 aa, RS15_GEOSE 63 aa,
NKX31_HUMAN 61 aa, PSAE_PICP2 68 aa; ~1200 single mutants each). Tsuboyama is
chosen deliberately: it measures ΔG of folding, the quantity this project is
about, rather than a functional proxy. MSAs built as in 3.3.

---

## 4. Experiments

### 4.1 Route decomposition — where query identity enters `z`
`harness/pi_paths.py`, `exp_paths.py` → `figures/paths_gfp.png`

Query identity reaches the pair representation by four separable routes, plus
one into the single representation:

| route | path |
|---|---|
| `z_direct` | `s_inputs → z_init_1/z_init_2` outer sum |
| `s_direct` | `s_inputs → s_init` |
| `msa_bcast` | `s_inputs → msa_module.s_proj`, added to every MSA row |
| `msa_query` | MSA row 0 (the query) → `OuterProductMean` |
| `msa_prior` | MSA rows 1..S → `OuterProductMean` |

Each is sourced from a *donor* run while everything else comes from the
*recipient* (`build_hybrid`: `eqx.tree_at` on `InitialEmbedding` fields for the
first three; `.at[:, 0]` / `.at[:, 1:]` on `feats["msa"]` for the last two).

Two directions per route:
- **necessity** — run the mutant with route r restored to wild type:
  `1 − ‖D(patched) − D(WT)‖ / ‖D(mut) − D(WT)‖`
- **sufficiency** — run the wild type with route r injected from the mutant:
  `‖D(patched) − D(WT)‖ / ‖D(mut) − D(WT)‖`

`D` is mean \|ΔE[d]\| over off-diagonal residue pairs. Å is legitimate here
because both terms are taken **at the same layer** (the trunk output), so the
sharpening cancels in the ratio.

**Blocking sanity checks**, run first: patching no routes must reproduce the
mutant and patching all must reproduce the wild type. Both close to
**0.00e+00** on every pair tested. Two further controls come out at exactly
zero by construction: `msa_prior` (all variants share an identical alignment
body) and `s_direct` (in `PairformerLayer2`, `z` is updated only by
tri-mul/tri-attention/`transition_z`, all functions of `z` alone; `s` reads `z`
as attention bias but never writes back, so the distogram is *provably*
independent of `s_init`).

### 4.2 Per-layer capture and the logit lens
`pi_core.pairformer_capture`, `msa_module_capture`; `exp_layers.py`
→ `figures/layers_gfp.png`

The scan is re-run with `ys` populated. `reduce_fn` is applied **inside** the
scan so 64 full copies of `z` (64 × N² × 128 floats, ~1.9 GB at N=238) are never
materialised. The distogram head is applied to intermediate `z` as a structural
logit lens; the absolute values are out-of-distribution and not interpreted, only
the wild-type-vs-mutant *difference* at a fixed layer, since both runs are
probed identically.

### 4.3 Per-sublayer attribution
`exp_sublayers.py` → `figures/sublayers_gfp.png`

`PairformerLayer2.__call__` is mirrored with capture after each of its five
writes into `z` (`tri_mul_out`, `tri_mul_in`, `tri_att_start`, `tri_att_end`,
`transition_z`). Per-op contribution = divergence after the op minus divergence
after the previous write; summed over ops it telescopes to the per-layer change,
which is the built-in consistency check (for `core_32`: entering KL 0.1637 +
triangle ops +1.1045 − `transition_z` 0.3594 = 0.9088 = leaving KL).

**Run in KL, not Å.** The first version used Å and could not distinguish
"`transition_z` suppresses the mutation" from "`transition_z` sharpens the
distogram" — those are indistinguishable in Ångström.

### 4.4 Causal ablation
`exp_ablate.py`

`transition_z`'s output projection (`fc3.weight`, plus bias if present — note
joltz's Transition is `fc3(silu(fc1(v)) * fc2(v))`, so **fc3** is the output and
fc1/fc2 are gated inner branches) is zeroed on a chosen slice of
`stacked_parameters`. Untouched layers are bit-identical.

Conditions: intact; ablate L37–45; ablate a width-matched control band L10–18;
ablate all 64. The control band is essential — without it, any effect could be a
generic consequence of deleting nine MLPs.

### 4.5 Readout-subspace test
`exp_subspace.py` → `figures/subspace_gfp.png`

`DistogramModule2` computes `W(z + zᵀ)` with `W` of shape (64, 128), full row
rank 64. It is therefore blind to (a) the antisymmetric part of `z` and (b) the
null space of `W` — both exactly computable, no probe required. Δz is decomposed
per layer into readable (rowspace of `W`, applied to the symmetric part) and
unreadable components. Sampling keeps both `(i,j)` and `(j,i)` so the
symmetric/antisymmetric split has its transpose partner.

### 4.6 Normalisation against a scrambled control
`exp_relative.py`

Adds the denominators §4.2 lacked: `‖Δz‖/‖z‖`, and KL as a fraction of the
scrambled-sequence control. This is what demoted "the trunk amplifies the
mutation ×5.6" to "the mutation's share is roughly conserved" — the scramble
control grows ×4.3 over the same layers.

### 4.7 Layer-resolved matrices
`exp_matrix.py`, `fig_matrix.py` → `figures/matrix_gfp_*.png`

Pair sampling is **stratified by residue** (80 partners each) so per-residue
rows are unbiased, rather than whatever a uniform pair sample happened to cover.
Emits residue × layer, sequence-separation × layer, operation × layer, and
enrichment of KL on mutated residues.

### 4.8 Spatial propagation
`analyze_onset.py`

Onset = divergence-weighted mean layer, `Σ_L L·ΔKL(r,L) / Σ_L ΔKL(r,L)`,
preferred over a threshold crossing (no arbitrary cut, less noisy for
low-divergence residues). Correlated against 3-D distance to the nearest mutated
residue, **partialling out sequence distance and total divergence** — both
required, since residues near the mutation diverge more and larger divergences
accumulate later.

### 4.9 Benchmark: structure vs confidence vs internal state
`exp_bench.py` + `score_bench.py` → `figures/bench_gfp.png`

All three measured on the same variants in one run. TM via tmtools from saved
coordinates. The **within-wild-type mean pairwise TM across diffusion samples is
computed explicitly** as the noise floor rather than assumed small — the earlier
error made exactly that assumption.

### 4.10 ProteinGym probe
`exp_gym.py`, `probe_gym.py`

Per variant, per layer: `kl_glob`, `kl_site` (pairs touching the mutated
position), `‖Δz‖` and `‖Δs‖` at that position — 4 × 64 = 256 features.

**Split is grouped by residue position, never random.** A random split puts
other substitutions at the same site in both train and test; position identity
alone carries much of the stability signal (buried sites are intolerant to
everything), so a random split flatters every model and cannot show whether the
features generalise to unseen sites. A position-only predictor is reported as
the baseline that a random split would effectively be leaking.

Two methodology bugs were caught on **synthetic data with a planted signal**
(informative feature at `kl_site` L40 plus a position effect plus 254 noise
features) before the real data was touched:

1. Multi-feature models lost to the single best feature (ridge 0.541, MLP 0.186
   vs 0.843) — 256 features overfitting ~284 rows at fixed λ. Fixed by tuning
   λ, feature count k and MLP width on an **inner position-grouped validation
   split of the training rows only**. Recovered: ridge 0.844, MLP 0.765 (and the
   MLP correctly does *not* beat ridge on linear data).
2. The "best layer" was selected on the test set — argmax over 256 candidates
   evaluated on the reported quantity. Fixed: chosen on train, scored on test.
   Visible in the control: the two pure-noise blocks fell from an inflated
   0.218/0.239 to 0.058/0.120 while the planted signal held at 0.843.

Results are averaged over **5 independent position-grouped splits**.


### 4.13 Pairwise-signal scaling (beta) — and where it must be applied

`exp_ensemble.py --beta B --beta-mode {z_trunk,bias}`

Motivated by **Boltz-sample** (Steering Conformational Sampling in Boltz-2 via
Pair Representation Scaling, bioRxiv 2026.01.23.701250), which rescales the
latent pair representation by a global scalar to widen conformational sampling.
Two uses here: (a) it is the natural causal test of whether the *sampler's*
weighting of pairwise information is what prevents mutation signal from reaching
coordinates, and (b) it controls a confound in the ensemble test — at stock
settings a null result is ambiguous between "the mutation is not in the ensemble"
and "the ensemble is too narrow for anything to show".

**The insertion point is not free, and the obvious one is a no-op.**

`--beta-mode z_trunk` scales `TrunkState.z` before `DiffusionConditioning`. This
is very nearly inert, because `PairwiseConditioning.__call__` begins with

```python
z = concat([z_trunk, token_rel_pos_feats])      # joltz __init__.py:1022
z = self.dim_pairwise_init_proj(z)              # nn.LayerNorm(...) then Linear
```

so a global scale on `z_trunk` is largely absorbed by that LayerNorm. Measured
on RCRO, 120 variants, K=6:

| beta (z_trunk) | WT ensemble spread | rho(spread, dG) | rho(TM to WT, dG) | conditioning ‖Δq‖/‖q‖ |
|---:|---:|---:|---:|---:|
| 1.0 | 0.9900 | +0.161 | +0.435 | 0.2847031 |
| 1.5 | 0.9908 | +0.132 | +0.435 | 0.2847031 |
| 2.0 | 0.9911 | +0.159 | +0.444 | 0.2847034 |

The conditioning difference is identical to **seven decimal places** across a
2× scale — the diagnostic that the intervention never reached the sampler. The
ensembles also got marginally *tighter*, not wider. **These numbers are not
evidence that widening the sampler fails; they are evidence that this insertion
point does nothing.** Recorded because the same trap applies to anyone
reimplementing the published method, and because it is worth checking where
Boltz-sample applies its scalar relative to this LayerNorm.

`--beta-mode bias` (default) scales instead the three **pair-derived attention
biases** that `DiffusionConditioning` returns — `atom_enc_bias`, `atom_dec_bias`,
`token_trans_bias`. These are added to attention logits inside the diffusion
transformer, so scaling them changes how strongly pairwise information competes
with noise in the sampler. That is downstream of the offending LayerNorm and is
the quantity a "signal-to-noise of pairwise couplings" knob has to act on.

Implementation is a wrapper rather than a reimplementation of the sampling path,
so the library's own forward runs unchanged:

```python
class ScaledConditioning(eqx.Module):
    inner: eqx.Module
    beta: float = eqx.field(static=True)
    def __call__(self, s_trunk, z_trunk, rel_pos, feats):
        q, c, to_keys, aeb, adb, ttb = self.inner(s_trunk, z_trunk, rel_pos, feats)
        return q, c, to_keys, aeb*self.beta, adb*self.beta, ttb*self.beta

model = eqx.tree_at(lambda m: m.diffusion_conditioning, model,
                    ScaledConditioning(model.diffusion_conditioning, beta),
                    is_leaf=lambda x: x is model.diffusion_conditioning)
```

**Required diagnostic before believing any beta result:** confirm that the
conditioning tensors actually differ across beta. If ‖Δq‖/‖q‖ is unchanged, the
knob is not connected and the run is uninformative regardless of its outcome.

### 4.11 RSA — pairformer vs structure module *(running)*
`exp_gym2.py`, `analyze_rsa.py`

Four dissimilarity spaces over the same single mutants: `PF[L]` (cosine distance
between variants' Δz at the mutated site, per layer), `DISTO` (Jensen-Shannon
between variants' trunk-output distograms), `STRUCT` (1 − pairwise TM between
variants' predicted coordinates), `EXP` (\|ΔG_i − ΔG_j\|).

Reported: `RSA(PF[L], EXP)`, `RSA(STRUCT, EXP)`, `RSA(PF[L], STRUCT)`, plus the
mean variant-to-variant TM (the assumption-free version — if 120 mutants all
predict to TM ≈ 0.98 of each other, the structure module has collapsed them
regardless of any correlation).

Every RSA against `EXP` is partialled for **same-position**, **BLOSUM62
distance**, and **score magnitude**; without these it would mostly measure "did
we hit the same residue", which model and experiment trivially share.

### 4.12 Ensemble spread *(running)*
`exp_ensemble.py`, `analyze_ensemble.py`

Diffusion is a sampler, not a function, and every invariance claim in this area
compares one structure per sequence. K = 6 structures are drawn from the **same
deterministic trunk state** with K different noise keys, isolating the structure
module's own variability. Per variant: `spread` = mean pairwise TM among its own
samples; `tm_to_wt` = mean TM against the wild type's samples. The wild type's
own spread is the sampler baseline and the only reference that makes a variant's
spread meaningful. Test: does `spread` track ΔG better than `tm_to_wt`?

---

## 5. Figure index

| figure | experiment | shows |
|---|---|---|
| `bench_gfp.png` | 4.9 | TM vs pLDDT vs internal KL on the same variants |
| `paths_gfp.png` | 4.1 | route decomposition, necessity and sufficiency |
| `matrix_gfp_core_32.png` | 4.7 | residue × layer, separation × layer, op × layer, enrichment |
| `matrix_gfp_surface_32.png` | 4.7 | surface control |
| `matrix_gfp_scramble.png` | 4.7 | scrambled upper bound |
| `sublayers_gfp.png` | 4.3 | per-operation attribution |
| `kl_gfp.png` | 2.1 | KL rises while Å falls; entropy collapse |
| `subspace_gfp.png` | 4.5 | readable fraction of Δz is flat |
| `layers_gfp.png` | 4.2 | per-layer divergence, two spaces |
| `dose_enrichment.png` | 4.7 | dose series, focus/defocus profile |
| `depth_gfp_core32.png` | — | sensitivity vs MSA depth |

---

## 6. Known limitations

- **Recycles = 3** throughout. Production runs at 15 recycles with 25 diffusion
  samples show stronger invariance; more recycling gives the MSA-derived prior
  more opportunity to reassert and confidence-ranked sampling selects the most
  canonical sample. Our numbers therefore *understate* the phenomenon.
- **Grafted alignments.** Every variant carries the wild-type MSA verbatim. This
  is deliberate — it makes the alignment an exactly controlled variable and is
  the condition the mechanistic question is about — but it is not what a user
  running Boltz on a mutant would get.
- **`--msa-cap`** in the ProteinGym runs changes absolute KL values, so features
  from capped and uncapped runs must not be mixed.
- **N of proteins.** Mechanistic results (4.1, 4.3, 4.4) are replicated on GFP
  and DIO3. Spatial and dose results (4.7, 4.8) are GFP only; the ten-protein
  panel exists to fix this and has not yet been run through them.
