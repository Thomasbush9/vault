# mosaic — Claude context

Tracks adapting Escalante-bio's `mosaic` (JAX framework for multi-objective protein design via continuous relaxation) to run on an HPC with a user + agent-assisted workflow. Upstream repo lives at `~/Documents/ML/mosaic`.

## Where things are

- Upstream code: `~/Documents/ML/mosaic` (has its own `CLAUDE.md` with the architecture overview — read that for code-level questions).
- Repo-orientation guide for this Lab: `guide.md` (architecture, conventions, HPC-relevant gotchas).
- Data: see `data.md` for cluster paths, dataset locations, run IDs.
- Current focus: `agenda.md`.
- Latest experiment: most recent file in `log/`.
- History of decisions and dead ends: `decisions.md`.
- Lit review: `lit.md`.

## Default load

When starting a session in this project, autoload only:
- `agenda.md`
- the most recent file in `log/`

Pull in `guide.md`, `decisions.md`, `lit.md`, or older logs only when relevant. For code-level work in the repo itself, prefer the upstream `CLAUDE.md` over `guide.md` — `guide.md` is more about the *adaptation* (HPC, user/agent UX) than the framework internals.
