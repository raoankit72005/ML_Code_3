#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys; assert (3,11)<=sys.version_info[:2]<(3,14), "Use Python 3.11-3.13"'
if [ ! -x .venv/bin/python ]; then python -m venv .venv; fi
.venv/bin/python -m pip install 'pip==25.2'
# CUDA-enabled wheel installs on CPU too; compute is attached only during GPU phases.
.venv/bin/python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements-lightning.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > environment-resolved.txt
