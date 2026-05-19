# Feature extraction in ProteinMPNN

This note covers what the feature extractors in `features.py` do, why each piece exists, and the exact shapes flowing through them. The two extractor classes are `CA_ProteinFeatures` (Cα-only) and `ProteinFeatures` (full backbone N/Cα/C/O + virtual Cβ). They both turn a set of 3D coordinates into a **K-nearest-neighbor graph with per-edge feature vectors**, which is what the encoder consumes.

## 1. High-level picture

ProteinMPNN models a protein as a graph:

- **Nodes** = residues (one per position in the chain).
- **Edges** = K nearest neighbors of each residue in 3D space (default `top_k=30`).
- **Edge features** = a learned embedding built from:
  1. Inter-atom distances (RBF-encoded).
  2. Relative sequence position (with chain awareness).
  3. (Cα-only path) Local backbone orientation, encoded as a relative-frame quaternion + relative direction.

The whole pipeline is **rotation- and translation-invariant** because every feature is either a distance (scalar), a sequence offset (scalar), or a vector/rotation expressed in the local backbone frame.

## 2. Inputs and outputs

### `ProteinFeatures.__call__`

| Argument       | Shape           | Meaning                                                                                                            |
| -------------- | --------------- | ------------------------------------------------------------------------------------------------------------------ |
| `X`            | `[B, L, 4, 3]`  | Backbone coords. Atom order: N (0), Cα (1), C (2), O (3).                                                          |
| `mask`         | `[B, L]`        | 1 = valid residue, 0 = padding.                                                                                    |
| `residue_idx`  | `[B, L]`        | Sequence index per residue (with a large gap between chains so cross-chain offsets exceed `max_relative_feature`). |
| `chain_labels` | `[B, L]`        | Integer chain id.                                                                                                  |
| `noise_key`    | PRNGKey or None | If `augment_eps > 0`, adds Gaussian noise to coords (data augmentation).                                           |

Returns:

- `E`: `[B, L, K, edge_features]` — edge feature vector per (residue, neighbor).
- `E_idx`: `[B, L, K]` — index of the K nearest neighbors of each residue.

### `CA_ProteinFeatures.__call__`

Same except `Ca` is `[B, L, 3]` (single atom per residue) and the returned edge features include orientation info.

## 3. Step-by-step

### 3.1 K-nearest-neighbor graph — `_dist`

```
mask_2D = mask[:, None, :] * mask[:, :, None]        # [B, L, L]  pair mask
dX      = X[:, None, :, :] - X[:, :, None, :]        # [B, L, L, 3] pairwise vectors
D       = sqrt(sum(dX**2)) * mask_2D                 # [B, L, L]
D_adj   = D + (1 - mask_2D) * D.max(axis=-1)         # push masked pairs to far away
E_idx   = argsort(D_adj)[..., :K]                    # [B, L, K]  K nearest per residue
```

The `D_adj` trick guarantees padded positions never get picked as neighbors: they're set to the row-max so `argsort` puts them last. Both extractors use **Cα–Cα distance** to define the graph (even in the full-atom path) — neighbor selection is identity-of-atom invariant once K is fixed.

`E_idx` is then reused everywhere as the gathering index. `gather_edges` / `gather_nodes` take per-pair or per-node tensors and pull out the K-neighbor slice.

### 3.2 RBF distance encoding — `_rbf`

A single distance is a noisy, hard-to-learn feature. ProteinMPNN expands each distance into a Gaussian radial basis:

```
μ_k = linspace(2 Å, 22 Å, num_rbf)        # `num_rbf=16` centers
σ   = (22 - 2) / num_rbf                  # one bandwidth
RBF(d)_k = exp(-((d - μ_k) / σ)^2)        # [..., num_rbf]
```

This gives the network a smooth, locally-supported "distance fingerprint." Two distances close to the same center activate similar features; far-apart distances activate disjoint features. Conceptually it's the same idea as Fourier features or sinusoidal positional encoding, but localized.

### 3.3 Pairwise atom-type RBFs — `_get_rbf`

`_get_rbf(A, B, E_idx)` does: compute `[B, L, L]` pairwise distances between atom-type `A` at position `i` and atom-type `B` at position `j`, gather only the K neighbors of each `i`, then RBF-encode.

- **Cα-only path** uses `Ca_{i-1}, Ca_i, Ca_{i+1}` (a 3-window sliding over the backbone). All 9 ordered pairs (`Ca_0`-`Ca_0`, `Ca_0`-`Ca_1`, ...) give **9 × num_rbf** features per edge.
- **Full-atom path** uses `{N, Cα, C, O, Cβ}` — 25 ordered pairs give **25 × num_rbf** features. Including ordered pairs (e.g. `N→C` and `C→N`) is redundant in distance but the model gets them as separate channels and the linear projection learns the (anti-)symmetry.

### 3.4 Virtual Cβ

The full-atom path reconstructs a "virtual" Cβ from N/Cα/C using a fixed linear combo:

```
b  = Cα - N
c  = C  - Cα
a  = b × c
Cβ = -0.58273431·a + 0.56802827·b - 0.54067466·c + Cα
```

The three coefficients are tuned to place Cβ at its standard tetrahedral position relative to the backbone (the same trick used in trRosetta / AlphaFold's distogram inputs). This gives the model a sidechain-direction proxy **without needing actual sidechain coordinates** — important because at inference time you usually have only backbone.

### 3.5 Backbone orientation features (Cα-only path) — `_orientations_coarse`

This is the geometry-heavy bit and is **only** in `CA_ProteinFeatures`. Because the full-atom path already has 25 atom-pair RBFs, it doesn't need explicit orientation; for the Cα-only path orientation is the only directional info available.

Given Cα positions `X` and neighbor indices `E_idx`:

1. **Bond vectors** between consecutive Cα atoms, normalized:
   ```
   U_i = normalize(Cα_{i+1} - Cα_i)
   ```
   Pairs outside the realistic 3.6–4.0 Å Cα–Cα range are zeroed (chain breaks).

2. **Bond angles `A_i`** (between `u_{i-1}` and `u_i`) and **dihedral angles `D_i`** (signed torsion over `u_{i-2}, u_{i-1}, u_i`). These are the φ/ψ-like backbone angles in a Cα-only world.

3. **`AD` features** encode `(A, D)` as `(cos A, sin A · cos D, sin A · sin D)` — a continuous parameterization that avoids the `±π` wrap-around discontinuity you'd get from raw radians. Padded to length `L` so it aligns with the residue axis.

   (Note: `AD_features` is computed but not added to the edge features here — `V, O_features = ...` discards `V`. It's kept around because the original PyTorch code reuses it for node features in other variants.)

4. **Local frame `O_i`** — a 3×3 orthonormal frame per residue, built from the backbone normal `n_i = normalize(u_{i-1} × u_i)` and a tangent direction. Think of it as a residue-fixed coordinate system.

5. **Relative direction `dU_{ij}`** — vector from Cα_i to Cα_j **expressed in i's local frame**:
   ```
   dU_{ij} = normalize( O_i^T · (X_j - X_i) )
   ```
   This is a 3-vector that's rotation/translation-invariant.

6. **Relative orientation `R_{ij} = O_i^T · O_j`** — a 3×3 rotation matrix taking i's frame to j's frame. Encoded compactly as a **quaternion `Q ∈ R^4`** by `_quaternions(R)`. Quaternions are the standard 4-d minimal continuous parameterization of SO(3); the code computes the standard "max-trace" formula with sign-disambiguation from the off-diagonal entries.

   Final `O_features = concat([dU, Q], -1)` has shape `[B, L, K, 7]`.

### 3.6 Positional encoding — `PositionalEncodings`

Pure sequence-distance information:

```
offset_{ij} = residue_idx_i - residue_idx_j
d           = clip(offset + 32, 0, 64) * E_chains      # E_chains == 0 if cross-chain
d_onehot    = one_hot(d, num_classes=66)               # 65 in-chain bins + 1 "off-chain" slot
E_pos       = Linear(d_onehot)                          # learned projection
```

Two things to note:

- **Clipping at ±32** means residues further than 32 apart in sequence all map to the same bin — they're effectively "far in sequence" without distinguishing how far.
- **`E_chains` masks cross-chain edges to bin 0**, but because the one-hot has 66 classes (and `2·32+1 = 65`), bin 0 doubles as the "different chain" slot. So the linear layer learns a dedicated "you two are not on the same chain" embedding.

### 3.7 Final assembly

**Cα-only**:
```
E = concat([E_positional, RBF_all (9·num_rbf), O_features (7)], -1)
E = edge_embedding(E)         # Linear -> edge_features dim
E = norm_edges(E)             # LayerNorm
```

**Full-atom**:
```
E = concat([E_positional, RBF_all (25·num_rbf)], -1)
E = edge_embedding(E)
E = norm_edges(E)
```

`edge_in` in `__init__` declares the expected input dim — make sure it matches `num_positional_embeddings + num_rbf · {9,25} (+ 7 for Cα)` if you ever change a constant.

## 4. Why these specific choices?

- **KNN over fully-connected**: O(L·K) edges instead of O(L²). For typical proteins (L ≈ 200–500) K=30 covers all spatially-relevant neighbors and matches the local connectivity of actual contacts.
- **RBF over raw distance**: smoother loss landscape; the network can learn distance bands (e.g. "second-shell residue") as nearly-disjoint channels.
- **Local-frame features**: a protein's identity is invariant to global rigid motion; expressing everything in residue-local frames bakes that invariance into the architecture rather than relying on data augmentation.
- **Quaternions over rotation matrices**: 4 numbers vs 9, and a smooth manifold without the gimbal-lock issues of Euler angles.
- **Virtual Cβ**: lets the model see sidechain orientation cheaply, using only the backbone.
- **Coordinate noise (`augment_eps`)**: regularization — forces the model not to overfit to angstrom-precise crystal coordinates.

## 5. Shapes cheat-sheet

| Object | Shape |
|---|---|
| `X` (full atom) | `[B, L, 4, 3]` |
| `Ca` | `[B, L, 3]` |
| `E_idx` | `[B, L, K]` |
| `D_neighbors` | `[B, L, K]` |
| `RBF_all` (Cα) | `[B, L, K, 9·num_rbf]` |
| `RBF_all` (full) | `[B, L, K, 25·num_rbf]` |
| `O_features` (Cα only) | `[B, L, K, 7]` |
| `E_positional` | `[B, L, K, num_positional_embeddings]` |
| `E` (output) | `[B, L, K, edge_features]` |

## 6. References

- Dauparas et al., 2022. *Robust deep learning–based protein sequence design using ProteinMPNN*. Science.
- Ingraham et al., 2019. *Generative models for graph-based protein design*. NeurIPS. (Origin of the Cα-only orientation features and the local-frame trick.)
- Original PyTorch implementation: https://github.com/dauparas/ProteinMPNN
