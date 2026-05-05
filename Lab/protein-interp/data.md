# Data

## Cluster paths

- MSA-swap sequences + YAMLs: `/n/holylfs06/.../msa-swapping/sequences/{original,seq_18553,random_freq,random_uniform}/`
- Boltz attention extractions (PWA + others): `/Users/thomasbush/tmp-data/tmp_data/hidden_reps/boltz_results_chunk_0/` (local mirror of cluster output)

## Conditions

- `original` — WT query + WT MSA
- `seq_18553` — p40 mutant query + WT MSA
- `random_freq` — random query (frequency-matched) + WT MSA
- `random_uniform` — uniform-random query + WT MSA

All four share the same WT UniRef MSA hits; only row 0 (the query) differs.

## Tensor shapes

- `pwa_weights` ∈ R^(1, 8, 238, 238) per condition × step × MSA layer.
- Steps saved: 0, 5, 10. Layers: 0–3.

## Extraction config

- Script: `scripts/run_boltz_attention.py` with `boltz_attention_config.yaml`.
- Key flags: `layer_sites: [pwa_weights]`, `recycling_steps: 10`, `recycling_steps_to_save: "0,5,-1"`, `msa_layers: all`, `average_heads: false`, `no_kernels: true`.
