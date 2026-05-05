# Knowledge base config

Spec for the research KB + workflow across my devices. The vault you're reading (`~/Documents/Vault/Notes/`) is the canonical Obsidian vault, synced via git/GitHub. The Mac Mini holds the always-on copy.

## Goals

- Frictionless switching between devices.
- Notes/logs/lit live in **Obsidian** (this vault). Code lives in **per-project GitHub repos** where Claude Code is launched. Two surfaces, cross-linked.
- Vault = long-term memory. Project repos = short-term workbench.
- Token-efficient context: Claude Code never autoloads the whole vault.

## Decisions (locked)

1. Each active project = an Obsidian folder under `projects/` **and** a separate GitHub repo. They cross-reference each other. Claude Code runs only in the repo.
2. Three things sync via GitHub: this vault, the project repos, and `~/.claude/` (skills, CLAUDE.md, settings).
3. A user-level `~/.claude/CLAUDE.md` enforces a common picture of this structure for every Claude Code session, on every device.
4. Local LLM on Mac Mini = personal assistant when not at work (quick answers, quick drafts, light agentic use over Tailscale). Not a batch processor.
5. References → Zotero (already on MCP). Better BibTeX export to `refs/` as markdown.
6. HPC = git sync + job launch only. No Claude Code there. No dotfile sync.

## Architecture

### Devices and roles

- **MBP** — home; primary writing/thinking.
- **Mac Mini** — work desk; always-on; hosts canonical tmux + local LLM server.
- **ThinkPad (Arch)** — work; ssh thin client into Mac Mini.
- **HPC cluster** — job runner only.
- **Raspberry Pi + 1TB disk** — sync hub for blobs that don't belong in git.

### Network

- Tailscale mesh.
- Canonical tmux on Mac Mini. From anywhere: `ssh mac-mini -t tmux a -t research`.

### Sync layers

| Layer | What | Mechanism |
|---|---|---|
| Vault | this directory | git → GitHub |
| Code | per-project repos | git → GitHub |
| Claude dotfiles | `~/.claude/` | git → GitHub (chezmoi or stow) |
| Cross-device shared data | `~/tmp-data/tmp_data/` (local mirror, canonical on Pi) | rsync on demand (no auto-sync) |
| Claude raw session transcripts | `~/.claude/projects/` | per-device; optional Syncthing per-project between Macs |

HPC syncs only the code repos.

### Cross-device shared data (`tmp-data`)

Intermediate data, model outputs, and large artifacts that don't belong in git live under `~/tmp-data/tmp_data/` on each the macbook pro, while on the macmini the tmp-data are here: `/Users/thom/tmp_data/tmp_data` . The Pi (with the 1TB disk) is the canonical store; each device pulls/pushes via **rsync on demand** (not Syncthing — explicit, not background).

- Local path on Macs: `/Users/thomasbush/tmp-data/tmp_data/`
- Canonical home: Pi.
- Project `data.md` files reference paths under `tmp-data/tmp_data/` directly. If a referenced file doesn't exist locally, pull it from the Pi.
- The user-level `~/.claude/CLAUDE.md` documents this convention so any Claude session knows to flag missing files instead of assuming they're absent.

## Project pattern: "lab member"

Each active project has two surfaces:

- **Obsidian folder**: `Notes/projects/<name>/` — notes, logs, lit, decisions, data pointers.
- **GitHub repo**: `<name>/` outside the vault — code, scripts, configs. Claude Code launched here.

The two cross-link: project folder has a `repo.md` pointing at the GitHub URL + setup; the repo's `CLAUDE.md` points back at the project folder.

## Vault refactor

### Current

```
Notes/
  BIO/  CS/  NLP/  Math/         # topic atoms
  projects_ideas/                # flat project files
  *.md                           # loose atoms at root
```

### Proposed

```
Notes/
  CLAUDE.md                      # vault-level guidance (autoloaded)
  README.md                      # human index

  atoms/                         # short evergreen concept notes (1-screen, wikilinked)
    BIO/  CS/  NLP/  Math/

  projects/                      # one folder per active "lab member"
    protein-interp/
      CLAUDE.md                  # project context: what + where + what to load
      README.md                  # human pitch + current status
      agenda.md                  # next 1–3 things
      log/                       # dated experiment logs
        2026-05-04-pwa-conservation.md
      lit.md                     # lit review
      decisions.md               # design choices + dead ends
      data.md                    # cluster paths, dataset locations, run IDs
      repo.md                    # link + setup notes for the GitHub repo

  refs/                          # Zotero (Better BibTeX) export, one md per paper

  inbox/                         # daily notes, raw drafts, things to file later

  archive/                       # parked projects, old atoms (kept out of search)
```

### Migration of existing files

- `projects_ideas/log.md` → `projects/protein-interp/log/2026-05-04-pwa-conservation.md`
- `projects_ideas/lit_review.md` → `projects/protein-interp/lit.md`
- `projects_ideas/Improving Protein Models Attention.md` → split into `agenda.md` (current focus) + `decisions.md` (parked ideas)
- `projects_ideas/Learning how Environmental Perturbation affect Protein Energy Landscapes.md` → own project folder if active, else fold into `lit.md`
- `projects_ideas/ProtForge usage.md` → either its own project folder or `data.md` of the relevant project
- `MSA Sanity Check.md` (root) → `projects/protein-interp/log/` (it's an experiment note, not an atom)
- Root atoms (`Hyperbolic Tangent tanh(x).md`, `Monotonic.md`, `Bipolar saturating.md`, `Quasi-Linear.md`, `Negative Symmetric Function.md`, `Infinite Impulse Response systems.md`) → `atoms/Math/` or `atoms/CS/` by topic
- `Perturb Seq.md`, `Model diffing.md`, `ProteinMPNN AR Improvements Ideas.md`, `Shared Latent Space for Protein Design Models.md`, `Archive DL.md` → `atoms/BIO/` or `atoms/CS/` as appropriate

## Token efficiency rules for Claude Code

- Vault `CLAUDE.md` lists *what exists where*, not content. Under ~100 lines.
- Each project's `CLAUDE.md` lists what to load *for that project*: `agenda.md` + latest `log/`. Lit/decisions loaded on demand.
- Atoms are tiny on purpose: 1-screen notes with `[[wikilinks]]`.
- Logs are append-only and dated. Old logs not autoloaded.

## Global Claude knowledge

User-level `~/.claude/CLAUDE.md` (synced via dotfiles) tells every Claude Code session:

- Vault location: `~/Documents/Vault/Notes/`.
- Project pattern: Obsidian folder + GitHub repo, cross-linked via `repo.md` ↔ project repo's `CLAUDE.md`.
- Vault layout: `projects/`, `atoms/`, `refs/`, `inbox/`, `archive/`.
- Convention: when working in a repo `<name>/`, expect a counterpart at `Notes/projects/<name>/`; check `agenda.md` and the latest `log/` entry there before starting.

This is the same on every device because dotfiles travel.

## Local LLM on Mac Mini

Use case: personal assistant when away from work. Multi-turn coherence required → not a tiny model.

- **Model**: 30B-quant minimum once RAM is checked; pick a recent instruct/coder.
- **Runtime**: `mlx-lm` (Apple-silicon native, fastest) preferred. `ollama` if model swapping matters more than throughput.
- **Server**: always-on, behind Tailscale. Reachable from MBP, ThinkPad, phone.
- **Client**: Open WebUI or a TUI; for one-off prompts, curl from the terminal.
- **Not** for batch overnight processing yet — that's a separate later question.

## References

- Zotero already installed, MCP wired. Better BibTeX → `refs/` (one md per paper).
- Citation lookups from Claude Code go through the Zotero MCP.

## Daily workflow

See `workflow.md` for the per-session routine, cross-device handoff, and conversation-sync stance (short version: don't sync raw transcripts; the vault log entry is the handoff).

## Bootstrapping

- New project: `Rules/bootstrap-project.sh <name>` scaffolds `Lab/<name>/` from the template.
- New skill: author under `Rules/skills/<name>/`, then `Rules/skills/bootstrap-skills.sh` symlinks it into `~/.claude/skills/` on the current device. Run the same script on each Mac after pulling.

## Open items

- Local LLM model + runtime: decide after checking Mac Mini RAM.
- Pi + Syncthing scope: default for now is HPC ↔ Macs intermediate data only.
- Daily-notes flow: keep an Obsidian daily note in `inbox/`, file to `projects/` on review?
- Author the user-level `~/.claude/CLAUDE.md` so the convention travels with dotfiles (text drafted in this file under "Global Claude knowledge").
- Optionally Syncthing-share specific `~/.claude/projects/<encoded>/` dirs for in-progress session continuity between Macs.
