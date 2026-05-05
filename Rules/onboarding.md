# Onboarding

Standing session-start instructions for working with me on any project. I'll @-reference this file at the start of a session instead of retyping the intro.

## What you have

- The **code repo** for the current project (your launch directory).
- My **Lab environment** at `~/Documents/Vault/Notes/Lab/<project>/`, added via `--add-dir`. This is the project's notes folder, synced via git separately from the code.
- The vault rules at `~/Documents/Vault/Notes/Rules/` — convention details if you need them.

## Where things go

**Markdown — notes, plans, lit, decisions, experiment logs, results — goes in the Lab folder, NOT in the repo.** I don't want the code repo overloaded with `.md` sprawl. The repo is for code; the vault is for everything else.

Specifically:

- Per-experiment / per-session findings → append to `Lab/<project>/log/YYYY-MM-DD-<slug>.md`. New dated file per experiment or significant session.
- Pivots, design decisions, dead ends → `Lab/<project>/decisions.md`.
- Current focus → `Lab/<project>/agenda.md`.
- Cluster paths, run IDs, dataset locations → `Lab/<project>/data.md`. Note: paths under `~/tmp-data/tmp_data/` are local mirrors of a Pi-hosted store, rsync'd on demand. If a referenced file doesn't exist locally, flag it — don't assume it's missing.
- Lit notes / paper summaries → `Lab/<project>/lit.md`.

## What to do on session start

1. Read `Lab/<project>/agenda.md` and the most recent file in `Lab/<project>/log/` to load current state.
2. State in one sentence what you take the current focus to be, then we proceed.

## During the session

- Code changes go in the repo as usual.
- When we hit something worth recording — a result, a pivot, a measurement, a debugging insight — write it to the appropriate vault file as we go. Don't wait to be asked. Experiments especially: open a new `log/` entry at the start, append as we work, finalize at the end.
- Don't create stray `notes.md` / `TODO.md` files in the repo. If it's a thought, it goes in the vault.

## End of session

- Today's `log/` entry should be complete (what we did, what we learned, what's next).
- Two `git commit`s when stable: one in the repo, one in the vault.

## Invocation

I'll typically start sessions with something like `@Rules/onboarding.md follow this`. Treat that as the contract for the session.
