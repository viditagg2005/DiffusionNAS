#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_PYTHON="${CLUSTER_PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

cd "${REPO_ROOT}"
git submodule update --init --recursive

"${CLUSTER_PYTHON}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install torch --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements-gpu.txt
python -m pip install -e .

python -m unittest discover -s tests -v
diffusionnas preflight

echo "Cluster environment is ready at ${REPO_ROOT}/.venv"

