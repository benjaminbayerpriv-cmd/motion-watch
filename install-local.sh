#!/usr/bin/env bash
# Install this working tree as the /watch skill for every local agent host, then prepare the motion tools.
# Usage: ./install-local.sh [--dry-run] [--with-tracker]   (--with-tracker also installs the CoTracker3 point tracker, ~800 MB, CC BY-NC)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/skills/watch" && pwd)"
TARGETS=("$HOME/.claude/skills/claude-video" "$HOME/.agents/skills/claude-video")
DRY=""; TRACKER=""
for a in "$@"; do [[ "$a" == "--dry-run" ]] && DRY="--dry-run"; [[ "$a" == "--with-tracker" ]] && TRACKER=1; done
for T in "${TARGETS[@]}"; do
  mkdir -p "$T/scripts"
  rsync -a $DRY --delete --exclude '__pycache__' --exclude '*.pyc' --exclude 'build-skill.sh' --exclude '.skillignore' --exclude '.gitattributes' "$SRC/" "$T/"
  echo "installed -> $T ($(grep -m1 'version:' "$T/SKILL.md" 2>/dev/null | tr -d ' '))"
done
[[ -n "$DRY" ]] && exit 0
python3 "$HOME/.claude/skills/claude-video/scripts/setup.py" --install-motion
python3 "$HOME/.claude/skills/claude-video/scripts/setup.py" --json | grep -E '"(can_proceed|motion_ready|track_model_ready)"'
[[ -n "$TRACKER" ]] && python3 "$HOME/.claude/skills/claude-video/scripts/setup.py" --install-tracker
echo "Done. Restart Claude (or your agent) so it re-reads the skill list."
