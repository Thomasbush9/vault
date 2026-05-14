# Agenda

Current focus and the next 1–3 concrete things to do. Keep this short — when something is done, move it to a `log/` entry; when it's a parked idea, move it to `decisions.md`.

## Now

- [ ] Implement the encoder (kNN graph, RBF edge features, dihedral node features). Spec in `preprocessing.md` §4.
- [ ] Implement the autoregressive decoder.

## Next

- [ ] Verify a forward pass on a small batch matches the PyTorch reference on a few samples.

## Reference

- `preprocessing.md` — what `tied_featurize` produces and why (read before touching the model).
