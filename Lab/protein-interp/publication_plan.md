# What this needs to be publishable

Written 2026-08-01, after the trajectory result landed. This is the strategic
assessment, kept separate from `methods_*.md` (how things were measured) and
`log/` (what happened when). Update it as items close.

---

## 0. The framing problem, which comes before everything else

**The project's original hypothesis was MSA-driven memorisation.** The route
decomposition says that is not what is happening. Across 12 assays, median
necessity / sufficiency:

| route | necessity | sufficiency |
|---|---|---|
| `z_direct` (pair rep.) | **0.840** | **0.995** |
| `msa_bcast` | 0.002 | 0.160 |
| `msa_query` | 0.002 | 0.008 |
| `s_direct`, `msa_prior` | 0.000 | 0.000 |

The mutation reaches the trunk output through the pair representation, not
through the alignment. The MSA routes carry essentially nothing.

So the bias is **not** "the model retrieves the wild-type structure from its
alignment." It is: **the trunk registers the mutation perfectly well, and the
structure decoder does not express it.** That is a different claim, and a better
one — it is mechanistic rather than a statement about data leakage.

**The paper must say this explicitly**: we set out to test the MSA-memorisation
account, the intervention data rejected it, and here is what replaced it.
Reviewers reward a stated redirection; silently rewriting the hypothesis around
the result is the thing that looks like p-hacking. The report headline has
already moved ("Where a folding model stops representing a mutation") but the
*redirection itself* is currently invisible, which wastes it.

---

## 1. Blocking items (a reviewer will ask; we have no answer yet)

### 1.1 No intervention that recovers the signal — **highest value**
Right now the contribution is "here is where it breaks." Adding "and here is a
minimal, training-free change that recovers X % of it" roughly doubles the
paper. See §3. **Running now (`exp_amplify.py`).**

### 1.2 The grafted-MSA design is load-bearing and un-ablated
Every variant carries the wild-type MSA verbatim. This is a deliberate control —
it makes the alignment an exactly fixed variable — but it also *guarantees* the
MSA cannot carry mutation-specific information, which makes the §0 route result
partly circular. A reviewer will spot this immediately.

**Fix:** one run on a subset (say 3 assays × 40 variants) with genuinely
re-searched MSAs per variant, via ProtForge. If `z_direct` still dominates, the
result is safe and the criticism is answered in one paragraph. Cheap. **Do this
before submission regardless of what else happens.**

### 1.3 Production settings are asserted, not measured
Recycles = 3 and a single-sample structural readout, when production Boltz-2 is
15 recycles with confidence-ranked best-of-N. The report claims the numbers
"understate the phenomenon rather than exaggerate it." That is a *hypothesis*.

**Fix:** rerun the 12-assay internal-vs-output benchmark at 15 recycles with
best-of-5 confidence ranking, on at least 4 assays. If the gap narrows a lot the
claim needs softening; if it holds it becomes much stronger.

### 1.4 No external reference point for rho = 0.548
Is 0.548 good? Unknown, because nothing else was run on these 12 assays under
this protocol. If a dedicated ddG predictor gets 0.6–0.7 the internal probe is
respectable and the gap argument stands. If it gets 0.85, "the trunk represents
mutational effect well" needs qualifying.

**Fix:** run one or two off-the-shelf stability predictors on the same 12 assays
and the same position-grouped splits. This is a table row, not a project.

### 1.5 The mechanism section is n = 2
Per-sublayer attribution and the L37–45 causal ablation — the most
mechanistically interesting results — are on GFP and DIO3 only. The band
location may well be protein-specific. This is "more data", but it is the
*targeted* kind: extend the sublayer capture to all 12.

### 1.6 Scope is narrow and should be stated, not hidden
All 12 assays are Tsuboyama 2023, 61–72 aa, folding stability. The claim should
be explicitly scoped to ΔG of folding on small domains, or broadened. Do not let
a reviewer scope it for us.

---

## 2. Cross-model replication — **the models are already on disk**

`mosaic_setup/weights/` already contains:

| model | weights | JAX port in container |
|---|---|---|
| OpenFold3 | `openfold3/jax/of3.eqx` | `jopenfold3` |
| Protenix (AF3 reproduction) | `protenix_mini_default_v0.5.0.eqx` | `protenix` |
| AlphaFold2 | `alphafold/params/*.npz` | **none** — needs setup |
| Boltz-1 / Boltz-2 | `boltz*.ckpt` | `joltz` (in use) |

And critically, `jopenfold3/latent/pairformer.py` defines

```python
class PairFormerStack(eqx.Module):
    stacked_params: PairFormerBlock   # arrays stacked along dim 0
    ...
    (s, z, _), _ = jax.lax.scan(body_fn, (s, z, key), self.stacked_params)
```

— the **same** stacked-parameter + `lax.scan` construction as joltz. So per-layer
capture (re-run the scan with `ys` populated), ablation via `eqx.tree_at`, and
route patching as a pytree field swap all port by renaming attributes, not by
redesign. Going from one model to three is a small fraction of the work already
done.

### But AF2 is the more informative second model
Every AF3-class model (Boltz-2, OF3, Protenix) shares the diffusion decoder, so
they test **replication**, not **mechanism**. AF2's structure module is IPA
regression — deterministic, no sampler — and it has a distogram head, so the
"trunk believes X, structure says Y" comparison ports directly.

- signal **also lost** in AF2 → the failure is the trunk→structure interface in
  general. Much bigger claim.
- signal **preserved** in AF2 → diffusion sampling is specifically at fault.
  Sharp, and immediately actionable.

Either outcome is stronger than a third AF3 replication. Cost: AF2 params are
present but there is no JAX AF2 module in the container, so this needs real
setup (OpenFold, or DeepMind's AF2 JAX code).

**AF3 proper: skip.** Weights are gated.

**Recommended order:** OF3 + Protenix first (cheap, establishes generality) →
AF2 (expensive, separates diffusion from decoding) → never AF3.

---

## 3. The fix — what to try, and why in this order

The trajectory result narrows the search. We know the conditioning carries the
difference (‖Δq‖/‖q‖ = 0.285) and the sampler ignores it for the first 80 % of
the schedule. Two live explanations, which call for different fixes:

- **GAIN** — the mutation-specific component is real but small relative to
  everything else the decoder conditions on.
- **STRUCTURAL** — the decoder's early steps run on a prior that does not read
  that component at all.

### 3.1 Difference amplification — *running now*
`exp_amplify.py`. Scale only the mutant-minus-wild-type difference in the trunk
state and decode from it:

```
s(gamma) = s_wt + gamma * (s_mut - s_wt)
z(gamma) = z_wt + gamma * (z_mut - z_wt)
```

gamma ∈ {0, 1, 2, 4, 8}; gamma = 1 is the ordinary mutant prediction, gamma = 0
decodes the mutant's atoms from the wild type's trunk state. Diffusion key held
fixed across gamma so differences are the conditioning, not the noise draw.

**Not the same as the earlier beta experiment.** beta scaled the pair
representation in absolute terms and degraded the signal (0.435 → 0.018) because
it scaled mutation and background alike. Here the background is untouched.

**The control is the whole experiment.** Amplifying anything 8× moves the
structure, and a structure that moves more trivially has a different TM to the
wild type. So each gamma is paired with a norm-matched permuted control —
variant i gets variant j's difference vector rescaled to ‖Δ_i‖. If the true
difference raises ρ(TM, ΔG) and the norm-matched permuted one does not, the
effect is mutation-specific. If both rise together, we have only rediscovered
that bigger perturbations give bigger structural changes.

**Readout:** ρ(TM-to-WT, ΔG) and ρ(pLDDT, ΔG) per gamma, per condition.
Baseline at gamma = 1 should reproduce ≈ 0.214.

### 3.2 Distogram rescoring — training-free, do next
We have shown the mutant's distogram moves a lot (0.92 nats mean KL, individual
pairs by 7.6 Å) while the structure does not. So: sample N structures, score each
by its likelihood under the **mutant's own** distogram, rerank.

Cleanly diagnostic either way:
- reranking recovers signal → the sampler *generates* an appropriate structure
  but does not *select* it. A decode-time fix, no retraining, immediately usable.
- reranking recovers nothing → the sampler never generates it, and guidance is
  required (§3.3).

### 3.3 Distogram guidance — the actual fix if 3.2 fails
Add ∇ log p(distogram | coords) to the score function during sampling, weighted
toward high sigma where we measured the insensitivity. More work, but the σ-
resolved trajectory result *justifies the schedule weighting from our own
measurement* rather than a guess, which is a genuinely novel design argument.

---

## 4. Minimal publishable package

1. §0 redirection stated explicitly as a finding
2. Route decomposition, 12 assays (**done**)
3. Internal-vs-output gap, 60 assay-splits (**done**) + external baseline (§1.4)
4. σ-resolved trajectory localisation (**done**)
5. One intervention with a working control (§3.1, running)
6. Re-searched-MSA ablation (§1.2)
7. Replication on OF3 + Protenix (§2)
8. Corrections appendix — keep it. Two independent superposition failures caught
   by self-tests is a real methodological contribution in a field where almost
   nobody publishes their invalidated runs.

Items 1, 6, 7 are the cheap ones and close the three most obvious referee
objections.
