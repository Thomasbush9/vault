# Skills

Personal Claude Code skills authored as part of my workflow. Source of truth lives here in the vault; each Mac symlinks `~/.claude/skills/<name>` → this folder so Claude Code picks them up.

## Why here, not directly in `~/.claude/skills/`?

- The vault is the synced surface I think in. Authoring a skill alongside the project notes that motivated it keeps context together.
- Symlinks mean every device with the vault gets the skill for free after one bootstrap step.
- Generic third-party / built-in skills stay directly in `~/.claude/skills/` (managed by dotfiles); only **my** skills live here.

## Authoring a new skill

1. Create `Rules/skills/<name>/SKILL.md` with frontmatter (`name`, `description`, optional `triggers`).
2. Add supporting scripts/resources alongside as needed.
3. Run `./bootstrap-skills.sh` to symlink it into `~/.claude/skills/`.
4. Commit the vault.

## On other devices

```bash
git pull                                            # in the vault
~/Documents/Vault/Notes/Rules/skills/bootstrap-skills.sh
```

## Index

(empty — add skills here)
