#!/usr/bin/env bash
# CPU-only environment for the SAEA benchmark (DDMTOLab needs numpy>=2 and torch>=2.5, which clash with .venv).
#   bash scripts/setup_saea_env.sh            (run once, on a login or CPU node, from the repo root)
set -euo pipefail
REPO_ROOT="${DIFFUSIONNAS_ROOT:-$(pwd)}"
cd "${REPO_ROOT}"
PY="${PYTHON:-python3}"
"${PY}" -m venv .venv-saea 2>/dev/null || { rm -rf .venv-saea; virtualenv -p "${PY}" .venv-saea; }
source .venv-saea/bin/activate
pip install --upgrade pip -q
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements-saea.txt
pip freeze > .venv-saea/freeze.txt
python -c "import ddmtolab, moocore, sklearn, torch; print('ddmtolab ok; torch', torch.__version__, 'threads', torch.get_num_threads())"
