#!/usr/bin/env bash
# Symlink each Rules/skills/<name>/ directory into ~/.claude/skills/.
# Idempotent — re-run safely after pulling new skills from the vault.
set -euo pipefail

src="$HOME/Documents/Vault/Notes/Rules/skills"
dst="$HOME/.claude/skills"

if [[ ! -d "$src" ]]; then
  echo "vault skills dir not found: $src" >&2
  exit 1
fi

mkdir -p "$dst"

shopt -s nullglob
for d in "$src"/*/; do
  name="$(basename "$d")"
  link="$dst/$name"

  if [[ -L "$link" ]]; then
    target="$(readlink "$link")"
    if [[ "$target" == "${d%/}" ]]; then
      echo "ok    $name"
    else
      echo "WARN  $name symlink points elsewhere: $target (skipping)"
    fi
  elif [[ -e "$link" ]]; then
    echo "WARN  $name exists at $link and is not a symlink (skipping)"
  else
    ln -s "${d%/}" "$link"
    echo "link  $name"
  fi
done
