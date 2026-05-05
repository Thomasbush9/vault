# Workflow

Day-to-day routine for working across devices with the `Lab/` + code-repo split.

## Per-session

### Start

```bash
cclab <project>           # cd to ~/code/<project> + claude --add-dir Lab/<project>
```

Claude autoloads, via the two `CLAUDE.md`s:
- repo `CLAUDE.md`
- vault `Lab/<project>/CLAUDE.md`
- vault `Lab/<project>/agenda.md`
- the most recent file in vault `Lab/<project>/log/`

That's the working context. Anything older is loaded on demand.

### During

- Claude edits **code** in the repo.
- For substantive findings, pivots, or decisions, ask Claude to update the **vault**:
  - append to today's log entry, or
  - add to `decisions.md`, or
  - update `agenda.md` if priorities shifted.
- Don't litter the repo with `notes.md` files — notes belong in the vault.

### End

- Ask Claude to write/finalize today's log entry at `Lab/<project>/log/YYYY-MM-DD-<slug>.md`.
- Two independent commits: one in the code repo, one in the vault.
- Push both before switching devices.

## Mobile sessions (ssh from phone)

When I ssh into the Mac Mini from my phone (via Tailscale + Termius/Blink), Claude should switch to **mobile mode**: 1–2 sentence responses, no wide tables, `path:line` refs instead of code blocks, confirm before long/multi-file actions, summarize long output. The convention is documented in `~/.claude/CLAUDE.md` so it travels with dotfiles. Trigger via the `$CLAUDE_MOBILE` env var (set by the phone-ssh wrapper), narrow `tput cols`, or just by saying "mobile" / "phone" in chat.

## Cross-device handoff

The vault is on GitHub, so:

1. Device A: `git push` in vault + `git push` in code repo.
2. Device B: `git pull` in both.
3. `cclab <project>` on Device B. Claude reads the freshly-pulled `agenda.md` + latest `log/` and you're back in the same state.

The vault log is the handoff — not the conversation transcript.

## Conversation sync (raw `~/.claude/` transcripts)

**Default: don't sync.** Reasons:
- Sessions are large jsonl; bloats git history.
- Concurrent edits across devices → conflict pain.
- The vault log entry already captures what matters.

**If you specifically need to resume an in-progress session on another Mac:**
- The session encoded path (e.g. `-Users-thomasbush-code-protein-interp`) is identical between MBP and Mac Mini (both `/Users/thomasbush/`). Session files are portable between them.
- Use Syncthing via the Pi on the specific encoded directory under `~/.claude/projects/`. Don't sync the whole `~/.claude/projects/` tree.
- On the other Mac: `claude --resume <session-id>`.

ThinkPad (Arch, `/home/thomasbush/`) can't share session files because the encoded path differs. Use the vault log instead.

## Bootstrapping a new project

```bash
~/Documents/Vault/Notes/Rules/bootstrap-project.sh <name>
```

This copies `Rules/project-template/` into `Lab/<name>/` and fills in placeholders. Then:

1. `mkdir ~/code/<name> && cd ~/code/<name> && git init`
2. Edit `Lab/<name>/repo.md` with the actual repo URL.
3. Drop a one-line `CLAUDE.md` in the code repo pointing at `Lab/<name>/`.

## Adding a skill to your workflow

Skills live in `Rules/skills/<name>/` (vault) and are symlinked into `~/.claude/skills/<name>` so Claude Code picks them up. See `Rules/skills/README.md` for the convention. To activate after pulling:

```bash
~/Documents/Vault/Notes/Rules/skills/bootstrap-skills.sh
```

## Sync cadence

| Layer | When to sync |
|---|---|
| Vault | Push at end of session; pull on device switch. |
| Code repos | Push when commits are stable; pull on device switch. |
| Dotfiles (`~/.claude/` minus `projects/`) | Push when settings/skills change; pull on first login. |
| Skills (`Rules/skills/`) | Travels with vault. Re-run `bootstrap-skills.sh` after pull. |
| `~/tmp-data/tmp_data/` (Pi-hosted shared data) | rsync to/from Pi on demand; never in git. |
| Claude raw transcripts (`~/.claude/projects/`) | Don't sync by default. Optional Syncthing per-project (Macs only). |

## What goes where

| Artifact | Lives in | Why |
|---|---|---|
| Code, scripts, configs | code repo | versioned with code logic |
| Experiment log, results writeup | vault `Lab/<p>/log/` | survives across reorgs of code |
| Lit review | vault `Lab/<p>/lit.md` | cross-project reuse |
| Decisions / pivots | vault `Lab/<p>/decisions.md` | so they don't get re-litigated |
| Cluster paths, run IDs | vault `Lab/<p>/data.md` | not a code concern |
| Active focus | vault `Lab/<p>/agenda.md` | what Claude reads on session start |
| Concept notes (atoms) | vault `Knowledge/` | reused across projects |
| Vault spec + templates | vault `Rules/` | meta |
| Personal skills | vault `Rules/skills/` → `~/.claude/skills/` (symlink) | authored once, used everywhere |
| Claude conversation transcript | per-device only | not worth the sync cost |
