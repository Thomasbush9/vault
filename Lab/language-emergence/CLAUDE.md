# language-emergence — Claude context

Emergent-communication project using EGG (Facebook's Emergence of lanGuage in
Games toolkit). A Sender (sees target crop) and Receiver (sees full 4x4 scene)
play a referential pointing game over a Gumbel-Softmax discrete channel
(vocab 10, max_len 2). Vision modules are pretrained separately and frozen
during the game; only the channel (sender RNN head, receiver RNN + query
projection) learns.

## Where things are

- Code: see `repo.md` for the GitHub repo + setup.
- Data: see `data.md` for cluster paths, dataset locations, run IDs.
- Current focus: `agenda.md`.
- Latest experiment: most recent file in `log/`.
- History of decisions and dead ends: `decisions.md`.
- Lit review: `lit.md`.

## Default load

When starting a session in this project, autoload only:
- `agenda.md`
- the most recent file in `log/`

Pull in `decisions.md`, `lit.md`, or older logs only when relevant.

## Project-specific gotchas

- Runs on the Harvard cluster; the env is a plain venv at
  `/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/language/envs/lang` (not uv,
  not conda) — invoke `.../envs/lang/bin/python` explicitly.
- `egg_start.py` and `basin_experiments.py` are **marimo notebooks**; they also
  run headless as scripts (`python <notebook>.py`). Opening a notebook in the
  marimo browser UI auto-runs all cells — training cells cache to disk
  (`runs/...`) so reopening loads results instead of retraining.
- Game outcomes are bimodal across seeds (see `log/`); never compare single
  runs across config changes — compare success rates over many seeds.
