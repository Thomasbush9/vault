# Literature scoping — Protein interpretability + mutation invariance

Date: 2026-05-02. Quick scoping search, not a systematic review. Two topics, ~5–8 most relevant papers each, with what each contributes and where the gap is for our work.

## Topic 1 — Interpretability of deep protein models (PLMs + structure prediction)

State of the art is dominated by **sparse autoencoders on ESM2 / ESMFold**. Activation steering for PLMs landed in 2025. Almost nothing yet on layer-wise probing of cofolding diffusion models (AF3, Boltz, RoseTTAFold-AllAtom).

### Most relevant 5

1. **InterPLM — Simon & Zou (bioRxiv 2024.11.14.623630, published PNAS 2025).**
   SAEs trained on ESM2 residual stream. Up to ~2,548 interpretable latents per layer; up to 143 biological concepts (binding sites, motifs, domains) recoverable with sparse features.
   *Why it matters:* the foundational "SAE-on-PLM" paper. Establishes that PLM features decompose cleanly. Covers ESM2, not structure prediction trunks.
   <https://www.biorxiv.org/content/10.1101/2024.11.14.623630v1>
   PNAS: <https://www.pnas.org/doi/10.1073/pnas.2506316122>

2. **Towards Interpretable Protein Structure Prediction with Sparse Autoencoders — Yang et al. (ICLR 2025 GEM Workshop, arxiv 2503.08764).**
   SAEs scaled to ESM2-3B (the base of ESMFold). Matryoshka SAEs. SAE reconstruction maintains ESMFold structure prediction (RMSD 3.2 Å vs 2.9 Å for full layer 36). **Demonstrates targeted feature steering of ESMFold to increase solvent accessibility while keeping sequence fixed.**
   *Why it matters:* the closest published analog to what we want to do — intervention on a structure-predicting model via interpretable features. But it's still ESMFold (single-sequence + ESM2 trunk), not a cofolding diffusion model.
   <https://arxiv.org/abs/2503.08764>  ·  Code: <https://github.com/johnyang101/reticular-sae>  ·  Reticular write-up: <https://www.reticular.ai/research/interpretable-protein-structure-prediction>

3. **From Mechanistic Interpretability to Mechanistic Biology — Adams et al. (ICML 2025 spotlight).**
   Training/eval/interpretation protocol for SAEs on PLMs. Distinguishes generic vs family-specific features in ESM-2's residual stream.
   *Why it matters:* methodological backbone for any SAE work on PLMs. Sets standards for evaluation.
   <https://openreview.net/forum?id=zdOGBRQEbz>  ·  PMC: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11839115/>

4. **Steering Protein Language Models — Huang et al. (arxiv 2509.07983, 2025).**
   Direct port of activation steering / activation addition (ActAdd) from LLMs to PLMs. Tries to guide protein generation toward properties.
   *Why it matters:* shows that the activation-engineering toolbox transfers. Same vocabulary will apply if we want to steer Boltz pair representation.
   <https://arxiv.org/html/2509.07983v1>

5. **Interpreting and Steering Protein Language Models through Sparse Autoencoders — Garcia et al. (arxiv 2502.09135, 2025).**
   SAE latents associated with transmembrane regions, binding sites, zinc fingers; uses these to steer generation.
   *Why it matters:* second SAE-steering paper on PLMs. Confirms the InterPLM / Reticular pattern.
   <https://arxiv.org/html/2502.09135v1>

### Worth knowing but secondary

6. **Recent advances in interpretable machine learning using structure-based protein representations — survey, arxiv 2409.17726.** Good orientation for the field.
7. **Interpreting Attention in Protein Language Models — Vig et al. (2020).** Foundational: attention captures residue–residue contacts. Cited everywhere; older but still relevant for the attention-analysis style of work we've been doing on PWA.
8. **Predicting a Protein's Stability under a Million Mutations ("Mutate-Everything") — Ouyang-Zhang et al. (arxiv 2310.12979, NeurIPS 2024-ish).** Uses AF embeddings as input to a ΔΔG decoder. Implicitly probes whether AF representations carry mutation information even when its output structure is invariant. Closest in spirit to "do hidden reps encode the mutation effect."

### Gap for our work

Nobody has done **layer-wise probing of the AF3/Boltz Pairformer + MSA-module + diffusion-conditioning representations** specifically. SAE/steering work is on the PLM trunk (ESM2) or the ESMFold downstream of it. Cofolding diffusion models (AF3, Boltz-2, RoseTTAFold-AllAtom) — where the diffusion is conditioned on (s, z) from a Pairformer — are essentially untouched at the interpretability level. The MSA-module → Pairformer → diffusion conditioning is a different beast from a single-sequence transformer trunk and the existing SAE protocols don't transfer trivially.

That's the lane.

---

## Topic 2 — Mutation invariance / insensitivity of structure prediction models

Two clusters of work:
- **AF2 fold-switching** (Porter group): models memorise training-set fold and ignore MSA coevolution when they conflict.
- **General mutation insensitivity** (the "cheat sheet" critique): point/deletion mutations don't change predicted structures; MSA dominates query.

### Most relevant 5

1. **Adversarial Sequence Mutations in AlphaFold and ESMFold Reveal Nonphysical Structural Invariance — bioRxiv 2026.02.25.708002.**
   Tested 200 proteins at 5/10/20/40/70% mutation, 1/3/5/10% deletion. AF3 maintains high TM-score and high pLDDT confidence even at 40% mutation and 10% deletion. Even on experimentally-validated fold-switchers. ESMFold is more sensitive to point mutations than AF3.
   *Why it matters:* this is THE quantitative reference for what we're looking at. The user already references it. Gives the perturbation-gradient curve we should reproduce internally.
   <https://www.biorxiv.org/content/10.64898/2026.02.25.708002v1.full>

2. **AlphaFold predictions of fold-switched conformations are driven by structure memorization — Chakravarty & Porter (Nature Communications 2024).**
   Famous RfaH case: AF2 predicts the helical autoinhibited form even when MSA coevolution clearly signals the active β-sheet form. Concrete demonstration that the model overrules MSA evidence with training-set bias.
   *Why it matters:* proves the cheat sheet has a *training-set* component, not just an *MSA prior* component. Different mechanism from what we've been assuming.
   <https://www.nature.com/articles/s41467-024-51801-z>

3. **AlphaFold2's training set powers its predictions of fold-switching — Porter group bioRxiv 2024.10.11.617857.**
   Quantitative follow-up to the RfaH paper. 35% success rate on fold-switchers across >280k AF2/AF3 runs.
   *Why it matters:* gives an empirical magnitude to the memorisation effect.
   <https://www.biorxiv.org/content/10.1101/2024.10.11.617857v1.full.pdf>

4. **Several MSA perturbation methods enhance AlphaFold3 sampling of alternative protein states — bioRxiv 2026.04.02.716037.**
   Directly tests MSA perturbation as the lever that controls AF3's sensitivity to alternative conformations. Maps to our msa-swap experiments.
   *Why it matters:* exact same family of intervention as our pipeline; we should know what they tried so we don't overlap.
   <https://www.biorxiv.org/content/10.64898/2026.04.02.716037v1>

5. **Bias in the AlphaFold3 prediction of ligand-induced domain motion in enzymes — PNAS.**
   AF3 is biased toward a single predicted conformation; doesn't capture induced-fit conformational change on ligand binding.
   *Why it matters:* generalises the invariance critique beyond mutations to any conformational ensemble.
   <https://www.pnas.org/doi/10.1073/pnas.2530709123>

### Worth knowing but secondary

6. **Single-sequence vs MSA comparisons — RaptorX-Single (PNAS 2024) and ESMFold benchmarks.** RaptorX-Single beats MSA-AF on single-mutation-effect prediction; "MSA-based methods do not work well on predicting mutational effects." Confirms MSA is the source of the insensitivity, not just a contributor.
   <https://www.pnas.org/doi/10.1073/pnas.2308788121>
7. **Using AlphaFold to predict the impact of single mutations on protein stability and function — Buel & Walters (PLOS One 2022).** Older but foundational negative result.
   <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282689>
8. **Mutate-Everything — Ouyang-Zhang (NeurIPS 2024).** Crosses both topics: takes AF representations and trains a ΔΔG decoder. Implicit claim that representations carry mutation info that the AF output discards.
   <https://arxiv.org/pdf/2310.12979>

### Gap for our work

The mutation-invariance literature is about **outputs**: TM-score, RMSD, distogram, pLDDT, alternative conformations. Almost no work asks the question we're asking — "**given that the output structure is invariant to the mutation, do the internal representations encode the mutation effect anyway?**" Mutate-Everything implicitly assumes yes (since their probe works) but they predict ΔΔG, not phenotypic function.

So the angles that look novel:

- **Functional / phenotypic supervision instead of structural** — pair representation distance to WT vs |Δfluorescence| in GFP, instead of vs ΔΔG. That's not in the literature I can find.
- **Layer-wise localisation of the cheat in cofolding diffusion** — at which Pairformer layer / which recycling step does z stop carrying mutation information? No one has plotted that for Boltz / AF3.
- **Cross-model CKA between Boltz and ESM3** — tests whether mutation-sensitive ESM-style models and mutation-insensitive Boltz-style models develop comparable representations at any layer. Not in the literature.

---

## Recommendation

The pivot we discussed is well-positioned in this landscape:
- **Don't duplicate**: the SAE-on-PLM line is crowded (InterPLM, Reticular, ICML 2025 spotlights).
- **Don't reinvent**: the adversarial-mutation paper has already done the output-side perturbation gradient cleanly. We can cite it as the motivation and skip re-running it.
- **Lane that's open**: layer-wise probing of cofolding diffusion models (Boltz-2 specifically), with **functional** labels as supervision, and a causal-intervention experiment at the layer where the mutation signal is strongest.

### Suggested first reads (full-text, in order)

1. *Adversarial Sequence Mutations in AlphaFold and ESMFold* — to set the perturbation-gradient baseline.
2. *Towards Interpretable Protein Structure Prediction with SAEs* (Reticular, ICLR 2025 GEM) — closest methodological template (SAE → feature steering on a structure-prediction model). Code is available.
3. *AlphaFold predictions of fold-switched conformations are driven by structure memorization* (Chakravarty & Porter 2024) — to understand the second source of invariance (training-set memorisation) we've been ignoring.
4. *Mutate-Everything* (Ouyang-Zhang) — for the probe-on-AF-representations design pattern.

### Search caveats

This is a 1-day scoping pass, not a systematic review. Boundaries skipped:
- No Semantic Scholar / OpenAlex citation-network expansion (would surface 2026 follow-ups to the InterPLM line).
- ICML/NeurIPS/ICLR 2026 papers may already exist that aren't yet indexed in the searches I ran.
- No Boltz-2-specific interpretability papers found, but absence of evidence is not evidence of absence — would need a deeper bioRxiv crawl on "Boltz" explicitly.
- No PRISMA / quality assessment / multi-reviewer screening — this is a researcher's lane-finding doc, not a publication-grade review.

If we want to publish from this thread, we should harden it with a citation-graph crawl on the top 4–5 papers above before the experiments are framed.
