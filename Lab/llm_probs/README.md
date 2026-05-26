# llm_probs

Do transformers learn to *sample*? Train small autoregressive vs. masked transformers on samples from known probability distributions, then compare (1) how accurately they reproduce the target distribution, (2) sample independence across draws, and (3) how distributional properties (mean, variance, modality, entropy) are encoded in hidden states.

Motivated by *Large Language Models Are Bad Dice Players* — moves from behavioural prompting of frontier LLMs to controlled training + mechanistic analysis on tiny models.

## Status

`active` — prototyping data generation. No model yet.

## Layout

- `agenda.md` — current focus, next 1–3 things.
- `log/` — dated experiment logs.
- `lit.md` — literature review.
- `decisions.md` — design choices + dead ends.
- `data.md` — dataset format, sampler API, paths.
- `repo.md` — GitHub repo + setup notes.
