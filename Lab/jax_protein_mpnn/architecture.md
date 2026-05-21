# Model architecture: encoder → decoder

This note covers the ProteinMPNN model proper — what happens **after** feature
extraction (`features.md`) hands off `(E, E_idx)`. Two halves: a stack of
**graph encoder layers** that turn structure-only features into rich
per-residue / per-edge hidden states, and a small stack of **autoregressive
decoder layers** that combine the encoder context with a partial sequence to
predict the next residue. Code lives in `model.py` and `layers.py`.

Conventions: **B** = batch, **L** = padded length, **K** = neighbors per
residue (default 64 in this JAX port; 30/48 in the original paper depending on
the variant), **H** = `hidden_dim` (typically 128).

## 1. Top-level wiring (`ProteinMPNN.__call__`)

The forward pass during training is:

```
(E, E_idx)        ← features(X, ...)                       # see feature_extraction.md
node_h, edge_h    ← _encode_graph(E, E_idx, mask)          # §2  encoder
seq_h             ← W_s(S)                                 # token embeddings
seq_edge_h        ← cat_neighbors_nodes(seq_h, edge_h, E_idx)
enc_context       ← _build_encoder_context(...)            # §3.4 skip pathway
fwd_mask, bwd_mask ← _build_decoder_masks(decode_noise)    # §3.5 autoreg mask
node_h            ← _run_decoder(node_h, seq_edge_h, ...)  # §3  decoder
logits            ← W_out(node_h)
return log_softmax(logits)                                  # [B, L, 21]
```

The structure-only encoder runs **once**; the decoder uses the encoder's
outputs through a "forward / backward" mask split (see §3.3) so the autoregressive
training pass remains a single parallel forward call.

## 2. Encoder

### 2.1 Inputs and initial projections

| Tensor | Shape | Source |
|---|---|---|
| `E` (edge features) | `[B, L, K, edge_features]` | feature extractor |
| `E_idx` | `[B, L, K]` | feature extractor |
| `mask` | `[B, L]` | dataloader |

Two trivial projections at the entry:

```
edge_h = W_e(E)                          # [B, L, K, H]
node_h = zeros([B, L, H])                # encoder starts with no per-residue signal
```

**Why node_h = 0?** Sequence identity is the thing we will *predict*, so it
must not leak into the encoder. All structural information arrives via the
edges. The nodes start blank and accumulate context purely from neighbor
messages over the encoder stack.

An `attend_mask` of shape `[B, L, K]` is also built here: it is 1 only when
both endpoints `(i, j)` are real residues (it's the product of `mask[i]` and
`mask` gathered along `E_idx[i, k]`). This zeros out messages from padded
neighbors.

### 2.2 EncoderStack

`EncoderStack` is just `num_encoder_layers` copies of `EncLayer` applied
sequentially (default 30 layers, but practically reduced to 3 in this repo's
configs to match the released checkpoint and keep training cheap). Each layer
**reads `(node_h, edge_h)` and writes new `(node_h, edge_h)`**, so unlike a
transformer encoder the edges are also updated layer-by-layer.

### 2.3 One `EncLayer` — message passing

Each encoder layer does three things: a **node message-passing update**, a
**FFN on nodes**, and an **edge update**. We'll walk through each line with
the explicit tensor shapes and the underlying math.

**Layer inputs** (recap):

| Name | Shape | Meaning |
|---|---|---|
| `node_h` | `[B, L, H]` | per-residue hidden, `h_i` for `i ∈ [0, L)` |
| `edge_h` | `[B, L, K, H]` | per-edge hidden, `e_ij` for `j ∈ neighbors(i)` |
| `E_idx`  | `[B, L, K]` int32 | for each `i`, the L-indices of its K neighbors |
| `attend_mask` | `[B, L, K]` | 1 if both `i` and `j = E_idx[i, k]` are real |
| `node_mask`   | `[B, L]` | 1 if residue `i` is real |

#### (a) Build the message tensor `[h_i ‖ e_ij ‖ h_j]`

Message passing needs, for every edge slot `(i, k)` (where `j = E_idx[i, k]`),
the triple `(h_i, e_ij, h_j)`. It's built in two concatenations.

**Step 1 — gather `h_j` and concat with `e_ij`** (`cat_neighbors_nodes`):

```python
gathered_h_j = gather_nodes(node_h, E_idx)             # [B, L, K, H]
node_edge    = concat([edge_h, gathered_h_j], axis=-1) # [B, L, K, 2H]   = [e_ij ‖ h_j]
```

`gather_nodes` is an indexed lookup: `node_h: [B, L, H]` + `E_idx: [B, L, K]`
→ `[B, L, K, H]` where slot `(b, i, k, :)` equals `node_h[b, E_idx[b, i, k], :]`.
For residue `i`'s `k`-th neighbor (global index `j`), it copies `h_j` into that
slot. Implementation-wise it flattens to `[B, L·K]`, does a `take_along_axis`
on the node axis, then reshapes back.

**Step 2 — broadcast `h_i` and concat on the left:**

```python
h_i_expanded = broadcast_to(node_h[:, :, None, :], (B, L, K, H))  # [B, L, K, H]
node_edge    = concat([h_i_expanded, node_edge], axis=-1)         # [B, L, K, 3H]
                                                                   # = [h_i ‖ e_ij ‖ h_j]
```

`node_h[:, :, None, :]` inserts a K axis of size 1; `broadcast_to` repeats
`h_i` along it without allocating — every neighbor slot for residue `i` sees
the same `h_i` on the left. After the concat each slot is a 3H-dim vector
holding **receiver ‖ edge ‖ sender**: `i` is the node we're about to update
(messages flow *into* it), `j` is the neighbor supplying the information.
This is the canonical MPNN message input — `M(h_v, h_w, e_vw)` in the
Gilmer et al. notation with `v = i` (receiver), `w = j` (sender).

The result is a single tensor of shape `[B, L, K, 3H]` containing `L·K`
message inputs per batch element.

#### (b) MLP → mask → aggregate → residual

```python
m_ij = W3(σ(W2(σ(W1(node_edge)))))    # [B, L, K, 3H] → H → H → H,  σ = gelu
```

`W1` is the dimensionality cruncher (`3H → H`); `W2` and `W3` are square mixes
with GELUs between. The output `m_ij` has shape `[B, L, K, H]` — one
H-dim message vector per edge slot.

```python
m_ij = attend_mask[..., None] * m_ij    # broadcast [B, L, K] → [B, L, K, 1]
                                         # zeros out messages from padded j
```

`attend_mask` is rank-3; the trailing `[..., None]` makes it broadcast along H.

```python
Δh_i = sum(m_ij, axis=-2) / scale       # [B, L, K, H] → [B, L, H],  scale = 30
```

Sum-pool over K: `Δh_i = (1/30) · Σ_k m_{i,k}`. The divisor is a **fixed
constant**, not the real-neighbor count. Two reasons:

1. **Variance control.** `Σ_k m_{i,k}` has variance roughly proportional to
   the number of nonzero terms. Without the divisor, residues with many
   neighbors would inject much larger residuals than sparse ones. `÷30` is a
   stand-in for typical K and keeps Δh near unit scale through training.
2. **Mask-friendliness.** A real mean (`Σ / count`) would need a dynamic
   `count = sum(attend_mask)` plus a zero-guard. Fixed divisor sidesteps
   both. Padded neighbors contribute zero to the numerator, so a residue with
   few real neighbors naturally gets a proportionally smaller message — which
   is the desired behavior.

```python
node_h = LayerNorm( node_h + dropout(Δh_i) )    # [B, L, H]
```

Pre-update residual + post-norm. Shape preserved.

#### (c) Position-wise FFN on nodes

```python
ff_mid = gelu( Linear(H → 4H)(node_h) )          # [B, L, 4H]
ff_out =       Linear(4H → H)(ff_mid)             # [B, L, H]
node_h = LayerNorm( node_h + dropout(ff_out) )    # [B, L, H]
node_h = node_h * node_mask[..., None]            # zero padded residues
```

Vanilla transformer 4×-expansion MLP applied independently at each position.
The final masking is belt-and-suspenders — keeps padded entries from
contributing statistics to downstream layer-norms.

#### (d) Edge update

The encoder also refines the **edge** state each layer, using the freshly
updated `node_h`:

```python
# Re-build [h_i_new ‖ e_ij ‖ h_j_new] using the updated node_h (same two-step recipe as (a))
gathered_h_j_new = gather_nodes(node_h, E_idx)                       # [B, L, K, H]   ← updated h_j
node_edge        = concat([edge_h, gathered_h_j_new], -1)            # [B, L, K, 2H]
h_i_new_expanded = broadcast_to(node_h[:, :, None, :], (B, L, K, H)) # [B, L, K, H]   ← updated h_i
node_edge        = concat([h_i_new_expanded, node_edge], -1)         # [B, L, K, 3H]
                                                                      # = [h_i_new ‖ e_ij ‖ h_j_new]

# Second MLP — same shape signature as W1/W2/W3, separate weights
m_ij   = W13(σ(W12(σ(W11(node_edge)))))           # [B, L, K, 3H] → H → H → H
edge_h = LayerNorm( edge_h + dropout(m_ij) )      # [B, L, K, H]
```

By the time this code runs, `node_h` has already been overwritten by the
message-passing + FFN updates, so both the gather and the broadcast read the
**updated** node tensor. Only `edge_h` is still the "old" (pre-update) edge
tensor — that's the whole point: we're computing how the edges should change
given the freshly updated nodes on both endpoints.

Two things to notice:

- **No aggregation here.** `m_ij` is already shaped `[B, L, K, H]` — exactly
  the shape of `edge_h` — so we add it directly. The node update needed
  `sum_k` because nodes are per-residue; edges are per-pair, so the message
  *is* the edge update.
- **`W11/W12/W13` are not the same as `W1/W2/W3`.** Different learned
  weights, same shape signature. The first MLP learns "how do edges modulate
  nodes"; the second learns "given the updated nodes, how do edges evolve."

**Layer outputs:** `(node_h, edge_h)` with the same shapes as the inputs,
`[B, L, H]` and `[B, L, K, H]`. The next layer reads them and repeats.

Same shape signature as the node MLP but **different weights** (`W11/W12/W13`),
and the residual updates `edge_h` not `node_h`. Edges getting updated each
layer is what differentiates this from a vanilla MPNN — by the end of the
stack the edges encode learned pairwise context, not just the raw geometric
features.

**Output of the encoder:** `(node_h, edge_h)` with shapes `[B, L, H]` and
`[B, L, K, H]`. These are the rotation/translation-invariant structural
embeddings the decoder will condition on.

## 3. Decoder

The decoder turns the encoder's structural embedding `(node_h_enc, edge_h_enc)`
into per-residue log-probabilities over the 21-token vocabulary. During
training it runs in one parallel pass with a permutation mask; during inference
it loops residue by residue in a chosen decoding order. The clever bit (this
is the key idea of ProteinMPNN) is that **both modes use the exact same
`DecLayer`** — the only difference is the mask and whether the loop is
materialized.

### 3.1 What the decoder computes — and why

Formally the decoder factorizes the conditional sequence likelihood as

```
p(S | X) = ∏_i p(s_{π(i)} | s_{π(<i)}, X)
```

where `π` is a permutation of residue positions (sampled fresh each step at
training time, chosen explicitly at inference). Three design choices fall out
of wanting this factorization to be cheap and order-agnostic:

1. **Autoregressive over independent.** An independent per-residue classifier
   `∏_i p(s_i | X)` discards all residue-residue coupling — charge networks,
   hydrogen-bond donor/acceptor matching, hydrophobic packing all require that
   adjacent residue identities are coordinated. The AR factorization lets each
   conditional see the structure **and** all earlier sequence decisions,
   without the model ever needing to represent the full joint.

2. **Random permutation, not N→C order.** Following Ingraham et al. 2019,
   training on a uniform distribution over permutations makes the decoder
   robust to *any* decoding order. At inference you can then start with a
   binding pocket and radiate outward, honor symmetric tied positions, or draw
   multiple sequences from independent orders for diversity. A fixed
   left-to-right order would bake in sequence-direction priors that don't
   generalize across these use cases.

3. **MPNN over cross-attention.** The decoder reuses the encoder's K-NN graph
   and the same node-update primitive — no new pairwise machinery. Sequence
   identity enters through the **edge slot** of that primitive (each encoder
   edge `e_ij` is concatenated with the neighbor identity `s_j`), not through
   a separate attention head. This keeps the decoder small (3 layers) and
   pushes the heavy structural reasoning into the encoder, which runs once.

4. **Edges frozen, only nodes evolve.** The encoder already produced a rich
   pairwise embedding; the decoder's only job is to fold sequence identity
   into the per-residue state. Updating edges again per decoder layer would
   waste compute and risk leaking *future* sequence into the structural
   channel.

The mechanism that makes the AR factorization tractable in one parallel pass
is the **forward/backward mask split** (§3.5): for each edge slot `(i, j)`,
the decoder either uses full sequence-bearing context (when `j` is earlier
than `i` in the order) or sequence-blind structural context (when `j` is
later). Summing the two over the K-neighbor axis at every position gives
exactly the AR conditional — no looping during training, no KV cache.

### 3.2 Inputs, transforms, outputs

| Stage | Tensor | Shape | Contents |
|---|---|---|---|
| **Inputs** | `node_h_enc` | `[B, L, H]` | structure-only node embedding (encoder output) |
|  | `edge_h_enc` | `[B, L, K, H]` | structure-only edge embedding (encoder output) |
|  | `S` | `[B, L]` int | sequence tokens — true labels at train time, running buffer at inference |
|  | `E_idx`, `mask`, `chain_M`, `chain_M_pos` | `[B, L, K]`, `[B, L]`, … | K-NN graph + padding + designable-position masks |
|  | `decode_noise` | `[B, L]` | per-position scalars used to draw the random permutation |
| **Transforms** | `seq_h = W_s(S)` | `[B, L, H]` | embed sequence (only place identity enters the model) |
|  | `seq_edge` | `[B, L, K, 2H]` | `[e_ij ‖ s_j]` — encoder edge tagged with neighbor identity |
|  | `enc_context` | `[B, L, K, 3H]` | `[node_i_enc ‖ e_ij ‖ 0]` — sequence-blind twin of decoder context |
|  | `decoding_order` → `(backward_mask, forward_mask)` | `[B, L, K, 1]` each | partition K-neighbors into "decoded before self" vs "after self" |
|  | per layer: `decoder_ctx` | `[B, L, K, 3H]` | `backward · [node_i_dec ‖ e_ij ‖ s_j]  +  forward · enc_context` |
|  | per layer: node MPNN update on `(node_h, decoder_ctx)` | `[B, L, H]` | only nodes evolve; edges stay frozen at encoder output |
| **Output** | `log_softmax(W_out(node_h))` | `[B, L, 21]` | `log p(s_i | s_{<i}, X)` at every position, computed in one pass |

Mental model: the decoder is the encoder's node-update stage repeated 3×, fed
an edge-slot context that has been *gated* by the autoregressive mask.
Everything else — the random permutation, the encoder skip path, the
forward/backward split — is plumbing that produces a per-edge context tensor
with exactly the right information content for AR training.

### 3.3 Sequence embedding

```
seq_h     = W_s(S)                                          # [B, L, H]  21-dim vocab → H
seq_edge  = cat_neighbors_nodes(seq_h, edge_h_enc, E_idx)   # [B, L, K, 2H]
                                                            # = [edge_ij ‖ s_j]
```

`seq_edge` carries, for each edge `(i, j)`, the encoder edge `e_ij` and the
**embedding of residue j's identity**. This is the only place sequence
identity enters the model.

### 3.4 The encoder skip pathway — `_build_encoder_context`

```
zero_seq  = zeros_like(seq_h)
enc_edge  = cat_neighbors_nodes(zero_seq, edge_h_enc, E_idx)     # [B, L, K, 2H]
                                                                  # = [edge_ij ‖ 0]
enc_context = cat_neighbors_nodes(node_h_enc, enc_edge, E_idx)   # [B, L, K, 3H]
                                                                  # = [node_i_enc ‖ edge_ij ‖ 0]
```

Compare to `seq_edge` after we concat the decoder's own node state on the
left: it would be `[node_i_dec ‖ edge_ij ‖ s_j]`. So `enc_context` is the
**sequence-blind** twin of the decoder context: same edge slot, but with
`s_j` zeroed and `node_i_enc` (purely structural) in the node slot.

This twin is what gets mixed in via the forward/backward mask split (next).

### 3.5 Autoregressive mask — `_build_decoder_masks`

This is the core ProteinMPNN trick. Given a per-residue **decode noise**
vector (one random scalar per position), we pick a permutation:

```
decoding_order = argsort((decode_mask + 1e-4) · |decode_noise|, axis=-1)
                                            # random order, with masked positions sorted last
```

The order is per-batch and changes every step (the noise is fresh each call),
so the model is trained on a **uniform distribution over all permutations** of
the protein — it has to be robust to any decoding order. This is what lets
inference start anywhere and still work.

From the order we build a permutation matrix `P` (one-hot) and combine with a
strict-lower-triangular `L`:

```
order_mask_backward[i, j] = 1  iff  rank(j) < rank(i)        # j comes before i
                                          ↑ jth comes earlier in the chosen order
```

`einsum("ij, biq, bjp -> bqp", L, P, P)` produces this in `[B, L, L]`. We
then **gather** along the K-neighbor axis to get a per-edge mask
`decoder_attend_mask: [B, L, K, 1]`. Combined with the residue mask:

```
decoder_backward_mask = mask_i · attend_mask          # neighbor j was decoded before i
decoder_forward_mask  = mask_i · (1 - attend_mask)    # neighbor j is decoded after  i
```

`backward + forward = mask_i` — these two masks partition the K neighbors of
each residue into "already decoded" and "not yet decoded."

### 3.6 Mixing encoder context with the partial sequence

```
encoder_context_forward_masked = decoder_forward_mask · enc_context
                                                 # only the "future" edges
                                                 # carry purely structural info (s_j=0)
```

Then inside each decoder layer (next subsection):

```
decoder_ctx = cat_neighbors_nodes(node_h_dec, seq_edge, E_idx)    # [B, L, K, 3H]
                                                  # = [node_i_dec ‖ edge_ij ‖ s_j]
decoder_ctx = decoder_backward_mask · decoder_ctx
            + encoder_context_forward_masked
```

So for each edge slot:

- if `j` was decoded **before** `i` → use the decoder's full
  `[node_i_dec, edge_ij, s_j]` (sequence identity of `j` is known and used).
- if `j` is decoded **after** `i` → fall back to `[node_i_enc, edge_ij, 0]`
  (no sequence leak; the only signal is the structural encoder output).

This is the **information-flow guarantee** for autoregressive training:
position `i` can only see sequence identities of strictly earlier positions in
the random decoding order. The structural part is freely available for all
neighbors (it has to be — that's the conditioning signal).

### 3.7 `DecLayer`

Each decoder layer is essentially the node half of the encoder layer — node
message-passing + FFN — but with the prepared `decoder_ctx` as the edge slot
and **no edge update**:

```
node_edge = concat([h_i_expanded, decoder_ctx], -1)    # [B, L, K, H + 3H]
m_ij      = W3(σ(W2(σ(W1(node_edge)))))                # MLP
m_ij      = attend_mask · m_ij                         # if mask provided (in DecoderStack, None)
Δh_i      = sum_k m_ij / 30
h_i       = LayerNorm(h_i + dropout(Δh_i))
h_i       = LayerNorm(h_i + dropout(FFN(h_i)))
h_i       = h_i · mask_i
```

Edge state stays frozen across decoder layers (it's the encoder's output). By
default the stack is 3 layers — enough to refine `node_h` given the now-mixed
edge context, but small enough that this is cheap to loop at inference time.

### 3.8 Output head

```
node_h ← run all decoder layers
logits ← W_out(node_h)                          # [B, L, 21]
return log_softmax(logits, axis=-1)
```

Loss is the standard label-smoothed NLL averaged over **designable** positions
(`chain_M · chain_M_pos · mask`), implemented in `losses.py`.

## 4. Training vs inference — the same DecLayer, two modes

### 4.1 Training (parallel)

One forward call computes log-probs at **every** position simultaneously. The
backward/forward mask makes each position's prediction only depend on
earlier-in-order positions for sequence info. This is the analogue of a
transformer's causal mask, but over a random per-sample permutation rather
than left-to-right.

Cost: 1× encoder + 1× decoder stack, all parallel.

### 4.2 Inference — `sample_decode` (in `decoding.py`)

The autoregressive sampling loop materializes the decoding order:

```
encode once   → (node_h_enc, edge_h_enc, E_idx)
build masks   → (decoder_backward_mask, decoder_forward_mask)
sampled_seq   = true_sequence            # positions outside design_mask stay fixed
for step in range(L):
    pos     = decoding_order[:, step]
    seq_h   = W_s(sampled_seq)           # ← uses the *currently-sampled* sequence
    seq_edge = cat_neighbors_nodes(seq_h, edge_h_enc, E_idx)
    node_h  = _run_decoder(...)          # full 3-layer pass
    logits  = W_out(node_h) / T
    step_lg = logits[:, pos, :]          # pick out the position we're sampling
    step_lg += biases (omit_AAs, bias_by_res, pssm, ...)
    token   = categorical(softmax(step_lg)) if sample_key else argmax
    sampled_seq[pos] = token
```

Each step:

1. Re-embeds the **whole sequence** (most positions are still the true
   sequence or already-sampled tokens; this position gets the freshly drawn
   one).
2. Runs the decoder stack on the whole protein.
3. Extracts the logits at `pos` and samples.

The masks were built once at the start of the loop, so positions later in the
order continue to receive only encoder-context (no sequence leak) until their
turn comes. This is the same masking machinery as training, used as a
"causal" constraint during sampling.

Cost: L × decoder stack. Encoder runs once. Per-step cost is dominated by
recomputing `seq_h` and the decoder MLPs over all L positions even though we
only use one position's logits — that's a known inefficiency vs. a true
KV-cache decoder, but acceptable for L ≈ a few hundred and a 3-layer stack.

### 4.3 Inference variants

All four entry points live in `decoding.py` and share the same encode → mask →
loop skeleton:

- **`sample_decode`** — vanilla autoregressive sampling (above). Supports
  temperature, omit-AAs, PSSM bias, per-residue bias, fixed-vs-designable
  positions via `chain_mask · chain_M_pos`.
- **`tied_sample_decode`** — for symmetric assemblies. Positions in the same
  `tied_pos` group share a single sampled identity; logits are averaged
  (β-weighted) across the group before the softmax. Decoding order is
  collapsed so each tied group is decoded **once**, at the position of its
  earliest member.
- **`conditional_probs_decode`** — score-mode, no sampling. For each
  designable position, build a one-hot `loop_decode_mask` that decodes
  *only that position* (or, if `backbone_only=True`, **all but** that
  position), run the decoder, and read out `log P(S_i | rest)`. Used for
  per-residue confidence / pseudo-perplexity.
- **`unconditional_probs_decode`** — set `decoder_backward_mask = 0` and
  `decoder_forward_mask = mask` everywhere, so every position only sees
  encoder context (no sequence at all). Gives `log P(S_i | structure)`
  unconditionally — i.e. structure-only marginals.

## 5. Why this design?

- **Encoder runs once, structure-only.** Sequence is what we want to predict
  — separating it cleanly into the decoder means the same encoder embedding
  can be reused for many sequences (sampling N candidates, scoring, etc.).
- **MPNN over attention.** With K nearest neighbors, message passing is O(L·K),
  vs O(L²) for full attention. For proteins the relevant interactions are
  local in 3D, so a KNN graph captures them with a fraction of the compute.
- **Edges updated each encoder layer, frozen in decoder.** Lets the encoder
  learn rich pairwise embeddings; the decoder's job is just to integrate them
  with sequence identity, which doesn't need further edge plasticity.
- **Random permutation autoregressive training.** Forces robustness to any
  decoding order. At inference you can pick a structurally-motivated order
  (e.g. design buried residues first, or center-out, or tied-position-aware)
  without retraining.
- **Sum / 30 instead of attention softmax.** Cheaper, mask-friendly, and
  empirically sufficient for K = 30–64. No per-edge attention weights —
  the MLP itself learns to weight messages via its output magnitude.
- **Forward/backward mask as a sequence-leak guard.** The decoder gets the
  full graph context; only the **sequence channel** is gated by the order.
  This is what makes one forward pass equivalent to the autoregressive
  factorization.

## 6. Shapes cheat-sheet

| Object | Shape |
|---|---|
| `E` (edge features in) | `[B, L, K, edge_features]` |
| `edge_h` (encoder)     | `[B, L, K, H]` |
| `node_h` (encoder)     | `[B, L, H]` |
| `attend_mask`          | `[B, L, K]` |
| `seq_h`                | `[B, L, H]` |
| `seq_edge`             | `[B, L, K, 2H]`  (edge ‖ s_j) |
| `enc_context`          | `[B, L, K, 3H]`  (node_i_enc ‖ edge ‖ 0) |
| `decoder_ctx`          | `[B, L, K, 3H]`  (node_i_dec ‖ edge ‖ s_j) |
| `decoder_backward_mask`, `decoder_forward_mask` | `[B, L, K, 1]` |
| `logits`               | `[B, L, 21]` |

## 7. References

- Dauparas et al., 2022. *Robust deep learning–based protein sequence design
  using ProteinMPNN*. Science.
- Ingraham et al., 2019. *Generative models for graph-based protein design*.
  NeurIPS. (Source of the autoregressive-with-random-order training scheme.)
- Original PyTorch implementation: https://github.com/dauparas/ProteinMPNN
