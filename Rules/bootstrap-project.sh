#!/usr/bin/env bash
# Bootstrap a new Lab project from the template.
# Usage: bootstrap-project.sh <project-name>
set -euo pipefail

name="${1:?usage: bootstrap-project.sh <project-name>}"
vault="$HOME/Documents/Vault/Notes"
template="$vault/Rules/project-template"
target="$vault/Lab/$name"

if [[ ! -d "$template" ]]; then
  echo "template not found: $template" >&2
  exit 1
fi

if [[ -e "$target" ]]; then
  echo "$target already exists" >&2
  exit 1
fi

cp -r "$template" "$target"

# Substitute <project-name> placeholder in all md files.
find "$target" -name '*.md' -print0 | while IFS= read -r -d '' f; do
  # macOS sed needs -i ''
  sed -i.bak "s/<project-name>/$name/g" "$f"
  rm -f "$f.bak"
done

echo "created $target"
echo "next steps:"
echo "  1. mkdir ~/code/$name && (cd ~/code/$name && git init)"
echo "  2. edit $target/repo.md with the GitHub URL"
echo "  3. drop a one-line CLAUDE.md in ~/code/$name pointing at $target"
