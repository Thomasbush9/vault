# llm_probs — Claude context

Small-transformer study of distributional sampling: train AR and masked transformers on samples drawn from known distributions (uniform / gaussian / bimodal-mixture to start), then compare distributional fidelity, sample independence, and how μ / σ / modality / entropy are encoded in hidden states. Code repo: `~/Documents/ML/llm_probs` (not the usual `~/code/<name>` location).

## Where things are

- Code: see `repo.md` for the GitHub repo + setup.
- Data: see `data.md` for sampler API, prompt format, dataset paths.
- Current focus: `agenda.md`.
- Latest experiment: most recent file in `log/`.
- History of decisions and dead ends: `decisions.md`.
- Lit review: `lit.md`.

## Default load

When starting a session in this project, autoload only:
- `agenda.md`
- the most recent file in `log/`

Pull in `decisions.md`, `lit.md`, or older logs only when relevant.
