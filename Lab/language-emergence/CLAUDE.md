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
- Since 2026-07-13 the code is a package (`lang_emergence/`) + scripts
  (`experiments/`); the old marimo notebooks are archived in `notebooks/`
  (frozen — the basin notebook's `import seed_sweep` no longer resolves).
  Experiment scripts cache per-run results to disk (`runs/...`), so re-running
  them loads cached results instead of retraining; delete a run file to redo it.
- Game outcomes are bimodal across seeds (see `log/`); never compare single
  runs across config changes — compare success rates over many seeds.
