#!/usr/bin/env bash
set -euo pipefail

remote="${GITHUB_BACKUP_REMOTE:-origin}"
branch="${GITHUB_BACKUP_BRANCH:-main}"
message="${1:-backup: VISTA-Track updates}"

git add -A

if git diff --cached --quiet; then
  echo "No changes to back up."
  exit 0
fi

git commit -m "$message"
git push "$remote" "HEAD:$branch"
