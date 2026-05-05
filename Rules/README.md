# Rules

Spec, conventions, templates, and helper scripts for this vault.

- `Knowledge base config.md` — master spec for the cross-device knowledge-base setup. Read this first.
- `workflow.md` — daily routine, cross-device handoff, conversation-sync stance.
- `onboarding.md` — short standing instructions for Claude at session start. @-reference this on a new session instead of retyping the intro.
- `project-template/` — template directory to copy when starting a new project.
- `skills/` — personal Claude Code skills authored as part of the workflow (symlinked into `~/.claude/skills/` per device).
- `bootstrap-project.sh` — scaffold a new `Lab/<name>/` from the template.
- `skills/bootstrap-skills.sh` — symlink each `skills/<name>/` into `~/.claude/skills/`.
