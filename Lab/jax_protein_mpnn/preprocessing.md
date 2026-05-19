# ProteinMPNN preprocessing — input → tensors

What `tied_featurize` produces and why each piece exists. Reference for the
modeling phase. Conventions: **B** = batch size, **L** = padded length
(`L_max` = max concatenated chain length in the batch), **21** = size of the
amino-acid alphabet (`ACDEFGHIKLMNPQRSTVWYX`, where `X` = unknown/gap).

## 1. Input

The dataloader produces a `batch` = `list[dict]`. Each dict represents one
biological assembly (one PDB) and is built by `parse_PDB`:

```
{
  'name': '<pdb id or filename stem>',
  'seq':  '<concatenation of all chain sequences>',         # length = sum of chain lengths
  'seq_chain_A':    'MVKVGV...',                            # one entry per chain letter
  'coords_chain_A': {
      'N_chain_A':  [[x,y,z], ...],   # length L_A, missing atoms = NaN
      'CA_chain_A': [...],
      'C_chain_A':  [...],
      'O_chain_A':  [...],
  },
  'seq_chain_B': '...', 'coords_chain_B': {...},
  ...
  'num_of_chains': K,
}
```

Plus optional **per-sample constraint dicts**, indexed by `b['name']`:

- `chain_dict[name] = (masked_chains, visible_chains)` — which chains to
  design vs. condition on. If `None`, every chain in the dict is treated as
  masked.
- `fixed_position_dict[name][letter] = [1-indexed positions]` — within a
  designable chain, residues to keep at wild-type.
- `omit_AA_dict[name][letter] = [(positions, "ACD")]` — forbid certain AAs at
  certain positions.
- `pssm_dict[name][letter] = {'pssm_coef', 'pssm_bias', 'pssm_log_odds'}` —
  external position-specific scoring matrix.
- `bias_by_res_dict[name][letter] = [L_chain, 21]` — additive logit bias.
- `tied_positions_dict[name] = [{'A': [1,2,3], 'B': [1,2,3]}, ...]` — residues
  that must decode to the same AA (homomer symmetry).

## 2. Transformations

For each item in the batch:

1. **Sort chains** alphabetically; concatenate `masked + visible`. The
   ordering matters because positions are concatenated in this order and the
   model only "knows" chain boundaries from `chain_encoding_all` and the +100
   bumps in `residue_idx`.
2. **Per-chain extraction** (`_chain_coords`, `_fixed_pos_mask`,
   `_omit_aa_mask`, `_pssm`):
   - Replace `'-'` characters in sequence with `'X'`.
   - Stack `(N, Cα, C, O)` → `[L_chain, 4, 3]` (or just `CA` → `[L_chain, 1, 3]`
     if `ca_only=True`).
   - Build per-chain masks for design, fixed positions, forbidden AAs, PSSM.
   - Constraints (`fixed_position_dict`, `omit_AA_dict`, `pssm_dict`,
     `bias_by_res_dict`) are **only consulted for masked chains** — visible
     chains are conditioning, so per-position design constraints don't apply.
3. **Concatenate chains** along the length axis and **pad to `L_max`** with
   zeros (NaN for coords, later zeroed). Each row of the padded buffer holds
   one sample's chains laid end-to-end.
4. **`residue_idx` chain bumps**: per-chain we write
   `100*(c-1) + np.arange(L_chain)`. This guarantees a gap of at least 100
   between the last residue of chain c and the first of chain c+1. The
   downstream relative-positional encoding f(i-j) sees that gap and uses it
   to encode "different chain" implicitly (see §4).
5. **`mask` from finite atoms**: `mask = isfinite(X.sum((2,3)))` — a residue
   is observed only if every backbone atom has finite coords. After computing
   the mask, NaNs in `X` are replaced with 0.0 so the model doesn't see
   NaN-propagating arithmetic.
6. **Dihedral validity** (`_dihedral_mask`): backbone φ/ψ/ω are defined only
   when neighbouring residues are sequentially adjacent. We mark
   `jumps = (residue_idx[:,1:] - residue_idx[:,:-1] == 1)` and pad:
   - `phi_mask`: needs (i-1, i)   → pad left.
   - `psi_mask`: needs (i, i+1)   → pad right.
   - `omega_mask`: needs (i, i+1) → pad right.
   Stacked into `[B, L, 3]`. Used to gate the dihedral node features at chain
   breaks and at unobserved-residue boundaries.
7. **Tied positions** (`_tied_positions`): each entry in
   `tied_positions_dict[name]` is a dict mapping chain letter → 1-indexed
   positions. We resolve each chain-local position to a global concatenated
   index via `global_starts`, and emit a list of lists where each inner list
   is a group of positions that must decode to the same AA. The accompanying
   `tied_beta[L]` weight defaults to 1.0 and is overridden if positions are
   provided as `(positions, weights)`.
8. **Final cast**: every numeric buffer is converted from numpy to `jnp` at
   the return site. Buffers stay numpy through the loop because JAX arrays
   are immutable and slice-assignment (`X[i, :L] = ...`) is the natural idiom
   for batch packing.

## 3. Output tensors — shapes, dtypes, meaning

`tied_featurize` returns a 20-tuple. Numeric tensors are `jnp` arrays; the
five `*_list_list` outputs and `tied_pos_list_of_lists_list` are plain Python
lists (variable-length per sample → cannot be tensorized).

| #   | Name                            | Shape                         | dtype | Meaning                                                                                                                             |
| --- | ------------------------------- | ----------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `X`                             | `[B, L, 4, 3]` or `[B, L, 3]` | f32   | Backbone coords (N, Cα, C, O) per residue. NaNs zeroed; padded residues are zeros.                                                  |
| 2   | `S`                             | `[B, L]`                      | i32   | Ground-truth AA tokens. Index into `ALPHABET`. Padded with 0 (=`A`); use `mask` to ignore padding in loss.                          |
| 3   | `mask`                          | `[B, L]`                      | f32   | 1.0 if residue is fully observed, 0.0 otherwise (missing atoms **or** padding).                                                     |
| 4   | `lengths`                       | `[B]`                         | i32   | Unpadded concatenated length per sample.                                                                                            |
| 5   | `chain_M`                       | `[B, L]`                      | f32   | **Design mask.** 1.0 = residue belongs to a masked chain (we want to design it); 0.0 = visible chain (conditioning).                |
| 6   | `chain_encoding_all`            | `[B, L]`                      | i32   | Chain ID, 1-indexed in the order chains appear in the concatenated sequence.                                                        |
| 7   | `letter_list_list`              | `list[list[str]]`             | —     | Chain letters per sample, in concat order. Bookkeeping for output writing.                                                          |
| 8   | `visible_list_list`             | `list[list[str]]`             | —     | Subset of letters that are visible (conditioning).                                                                                  |
| 9   | `masked_list_list`              | `list[list[str]]`             | —     | Subset of letters that are designable.                                                                                              |
| 10  | `masked_chain_length_list_list` | `list[list[int]]`             | —     | Lengths of each masked chain per sample.                                                                                            |
| 11  | `chain_M_pos`                   | `[B, L]`                      | f32   | **Free-position mask** *within* designable chains. 1.0 = free to be redesigned, 0.0 = fixed to wild-type. Independent of `chain_M`. |
| 12  | `omit_AA_mask`                  | `[B, L, 21]`                  | i32   | 1 = forbid this AA at this position at sampling time.                                                                               |
| 13  | `residue_idx`                   | `[B, L]`                      | i32   | Per-residue absolute index with `+100` bumps at chain boundaries. Padding = -100. Drives the relative-positional encoding.          |
| 14  | `dihedral_mask`                 | `[B, L, 3]`                   | f32   | φ/ψ/ω validity (1 = neighbours are sequentially adjacent).                                                                          |
| 15  | `tied_pos_list_of_lists_list`   | `list[list[list[int]]]`       | —     | Groups of global indices that must decode to the same AA (homomer constraints).                                                     |
| 16  | `pssm_coef_all`                 | `[B, L]`                      | f32   | Mixing weight in `[0, 1]` between model logits and external PSSM at each position. Default 0 = ignore PSSM.                         |
| 17  | `pssm_bias_all`                 | `[B, L, 21]`                  | f32   | Additive logit bias from PSSM.                                                                                                      |
| 18  | `pssm_log_odds_all`             | `[B, L, 21]`                  | f32   | Hard filter at sampling: only sample AAs whose log-odds exceed a threshold. Default `1e4` = no constraint.                          |
| 19  | `bias_by_res_all`               | `[B, L, 21]`                  | f32   | Additive logit bias per (position, AA).                                                                                             |
| 20  | `tied_beta`                     | `[B, L]`                      | f32   | Relative weight of each position within its tied group. Default 1.0.                                                                |

## 4. Theory — why each tensor exists

### 4.1 The task

ProteinMPNN is an **inverse-folding** model: given a backbone fold, output a
distribution over amino-acid sequences likely to adopt that fold. Architecture
is an SE(3)-invariant message-passing GNN over a Cα-Cα **k-nearest-neighbour
graph** (k=32 or 48 by default), followed by an autoregressive decoder over a
random residue ordering.

### 4.2 Backbone coords `X` and the kNN graph

`X` carries the only information about the fold. Three things derive from it:

1. **kNN edges**: top-k Cα-Cα neighbours per residue.
2. **Edge features**: pairwise distances between {N, Cα, Cβ, C, O} atoms — 25
   distances per edge — each encoded with a radial basis (16 RBFs spaced
   between 2 Å and 22 Å). Cβ is computed from N/Cα/C via a fixed geometric
   construction. These 25 RBFs are **SE(3)-invariant**: any rigid-body
   rotation/translation of the whole protein leaves them unchanged.
3. **Node features**: backbone dihedrals φ/ψ/ω, embedded as
   `[sin, cos]` to handle the 2π wrap.

This is why coords are kept as `[L, 4, 3]` even though only Cα is used for
graph construction — the other atoms feed the edge featurizer.

### 4.3 Two masks that look similar but aren't

- `mask` = "is this residue **observed**" (atoms finite + not padding). Used
  in attention/normalization and in the loss to ignore unobserved residues.
- `chain_M` = "is this residue **designable**" (1 = predict it, 0 = condition
  on it). Used during autoregressive decoding to switch between teacher
  forcing on the ground truth (visible chains) and sampling (masked chains).
- `chain_M_pos` = "is this residue **free**" (within the designable set, 1 =
  free, 0 = pin to wild-type). Combined with `chain_M` to express "design
  this chain but keep these specific residues fixed".

These compose: effective "free to be sampled" = `mask * chain_M * chain_M_pos`.

### 4.4 `residue_idx` and the +100 chain bumps

The relative positional encoding is a function of `i - j` clipped to
`[-32, +32]` (default in MPNN), embedded into a small vector and added to
edge features. By bumping `residue_idx` by ≥100 at every chain boundary,
inter-chain residue pairs always land outside the clipping window and get the
"max distance" embedding — which the model learns to interpret as "different
chain". This way the same relative-position machinery handles both
intra-chain sequential adjacency and inter-chain "no sequential relation"
without a separate branch.

`chain_encoding_all` is the explicit chain ID; some variants of MPNN use it
directly as a learned chain-identity embedding instead of (or in addition to)
the +100 trick.

### 4.5 Dihedral mask

Backbone dihedrals are 4-atom torsions:
- φ_i needs (C_{i-1}, N_i, Cα_i, C_i)
- ψ_i needs (N_i, Cα_i, C_i, N_{i+1})
- ω_i needs (Cα_i, C_i, N_{i+1}, Cα_{i+1})

They are undefined at chain termini and across residue-index gaps (missing
density in the PDB). `dihedral_mask` is 0 at those positions; the model
multiplies its dihedral embeddings by this mask so invalid torsions don't
inject garbage into node features.

### 4.6 Constraint tensors

These don't enter the encoder — they shape the **decoder's logit
distribution**:

- `bias_by_res_all` is added to logits before softmax.
- `omit_AA_mask` subtracts a large constant from forbidden (position, AA)
  cells before softmax, driving their probability to ~0.
- `pssm_log_odds_all` is a hard mask at sampling time: AAs below threshold
  are zeroed out post-softmax.
- `pssm_coef_all` and `pssm_bias_all` linearly combine the model's predicted
  distribution with an externally supplied PSSM: roughly
  `final_logits = (1 - α) * model_logits + α * pssm_bias` where α is per-position.

For training without external constraints these all have benign defaults
(zeros / +1e4 / α=0), so the same code path works for both training and
constrained design.

### 4.7 Tied positions and `tied_beta`

For designing **homo-multimers** (multiple chains sharing a sequence) you
must enforce that tied positions decode to the same AA. The standard trick:
at each autoregressive step, **average the logits** across all tied positions
in the group, then sample once and broadcast the same token to all of them.
`tied_pos_list_of_lists_list` enumerates the groups (in global indices);
`tied_beta` lets you weight each member non-uniformly inside the average if
needed.

### 4.8 Padding strategy

We pad to `L_max` with zeros (`X = 0`, `S = 0`, `mask = 0`, `residue_idx =
-100`, `pssm_log_odds = 1e4`, etc.). Defaults are chosen so that:

- Padding contributes 0 to attention sums (via `mask`).
- The loss multiplies cross-entropy by `mask * chain_M * chain_M_pos`, so
  padded positions get zero loss weight.
- `residue_idx = -100` ensures padded positions are always >32 away from any
  real residue, so they fall outside the relative-positional encoding window
  and contribute the "out-of-range" bin (which gets masked anyway).

## 5. Training-time data flow

```
PDB file
   │ parse_PDB
   ▼
batch: list[dict] of length B
   │ tied_featurize(batch, chain_dict, ...)
   ▼
(20 outputs — see §3)
   │
   │ encoder: kNN graph from X[:,:,1,:] (Cα), edge RBFs from X[:,:,:,:],
   │          dihedral node features × dihedral_mask, relative-pos from
   │          residue_idx, chain ID from chain_encoding_all
   ▼
node embeddings h ∈ [B, L, D], edge embeddings e ∈ [B, L, K, D]
   │
   │ decoder: autoregressive over a random ordering π of [0..L),
   │          conditioned on (encoder output, S[π<t]) via teacher forcing
   │          when training. chain_M masks loss to designable positions.
   ▼
logits ∈ [B, L, 21]
   │
   │ bias + omit_AA + pssm corrections
   ▼
log-probabilities; cross-entropy vs. S, weighted by mask * chain_M * chain_M_pos
```

## 6. Practical notes for modeling

- Cβ is computed in the encoder, not in preprocessing. Standard formula:
  `Cβ = -0.58273431*a + 0.56802827*b - 0.54067466*c + Cα`
  where `a = Cα-N`, `b = Cα-C`, `c = a×b`.
- The 25 distances per edge are between ordered atom pairs
  `(N,N), (N,Cα), (N,Cβ), …, (O,O)` — see Dauparas et al. 2022 Fig. 1c.
- During training, the **decoding order** is sampled uniformly at random per
  step; during inference for partial design it's BFS from fixed → designable
  so designable residues see fixed context.
- The loss is averaged with weight `mask * chain_M * chain_M_pos`, i.e. only
  over observed, designable, free positions. Visible chains and fixed
  positions contribute zero loss but full encoder context.
- Don't `jit` `tied_featurize` — it has Python-level branching on dict
  contents and variable-shape outputs. Run it in the dataloader (CPU); only
  `jit` the encoder/decoder forward pass.
