#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment.yml"
ENV_PREFIX="${1:-${REPO_ROOT}/.conda/envs/qwen-context-bloat}"
KERNEL_NAME="${2:-qwen-context-bloat}"
KERNEL_DISPLAY_NAME="${3:-Qwen Context Bloat (Python 3.11)}"
PKGS_DIR="${REPO_ROOT}/.conda/pkgs"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENV_PREFIX}")"
mkdir -p "${PKGS_DIR}"

export CONDA_PKGS_DIRS="${PKGS_DIR}"

if [ -d "${ENV_PREFIX}" ]; then
  echo "Updating existing conda environment at: ${ENV_PREFIX}"
  conda --no-plugins env update \
    --prefix "${ENV_PREFIX}" \
    --file "${ENV_FILE}" \
    --prune \
    --override-channels \
    --channel conda-forge
else
  echo "Creating conda environment at: ${ENV_PREFIX}"
  conda --no-plugins env create \
    --prefix "${ENV_PREFIX}" \
    --file "${ENV_FILE}" \
    --override-channels \
    --channel conda-forge
fi

ENV_PREFIX="$(conda --no-plugins run --prefix "${ENV_PREFIX}" python -c 'import sys; print(sys.prefix)' | tail -n 1)"
ACTIVATE_DIR="${ENV_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${ENV_PREFIX}/etc/conda/deactivate.d"

mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

sed "s|__PROJECT_ROOT__|${REPO_ROOT}|g" \
  "${REPO_ROOT}/conda_hooks/qwen_context_bloat_activate.sh.template" \
  > "${ACTIVATE_DIR}/qwen_context_bloat.sh"

cp "${REPO_ROOT}/conda_hooks/qwen_context_bloat_deactivate.sh" \
  "${DEACTIVATE_DIR}/qwen_context_bloat.sh"

chmod +x "${ACTIVATE_DIR}/qwen_context_bloat.sh" "${DEACTIVATE_DIR}/qwen_context_bloat.sh"

echo "Registering a Jupyter kernel for ${ENV_PREFIX}"
conda --no-plugins run --prefix "${ENV_PREFIX}" python -m ipykernel install --user \
  --name "${KERNEL_NAME}" \
  --display-name "${KERNEL_DISPLAY_NAME}"

echo
echo "Environment setup complete."
echo "Next steps:"
echo "  1. conda activate ${ENV_PREFIX}"
echo "  2. python ${REPO_ROOT}/scripts/generate_qwen_context_bloat_notebook.py"
echo "  3. jupyter lab ${REPO_ROOT}/notebooks/qwen_context_bloat_experiments.ipynb"
