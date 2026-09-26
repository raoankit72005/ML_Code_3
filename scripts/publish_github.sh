#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v gh >/dev/null || { echo 'Install GitHub CLI and run gh auth login first.' >&2; exit 1; }
gh auth status
if [ ! -d .git ]; then git init -b main; fi
git add .
if ! git diff --cached --quiet; then git commit -m 'Add tracked hybrid entity resolution pipeline for SageMaker'; fi
gh repo create raoankit72005/ML_Code_1 --private --source=. --remote=origin --push
