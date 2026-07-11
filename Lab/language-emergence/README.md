# language-emergence

Emergent communication experiments built on EGG. Two agents play a referential
pointing game over synthetic scene grids: the Sender sees a cropped target object
(shape + color), sends a short discrete message through a Gumbel-Softmax channel,
and the Receiver must point at the grid cell containing that object in the full
scene. The scientific interest is in when a communication protocol emerges at all,
and what structure (compositionality, attribute encoding) it has when it does.

## Status

`active` — game pipeline works end-to-end; currently characterizing the
initialization lottery (runs either solve the game ~0.8–0.9 or collapse to a
pooling equilibrium ~0.29) and testing interventions to raise the success rate.

## Layout

- `agenda.md` — current focus, next 1–3 things.
- `log/` — dated experiment logs.
- `lit.md` — literature review.
- `decisions.md` — design choices + dead ends.
- `data.md` — cluster paths, datasets, run IDs.
- `repo.md` — GitHub repo + setup notes.
