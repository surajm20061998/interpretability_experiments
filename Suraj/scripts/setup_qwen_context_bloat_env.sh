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
export CONDA_NO_PLUGINS="true"

run_conda_env() {
  if conda env --help >/dev/null 2>&1; then
    conda env "$@"
    return
  fi

  if command -v conda-env >/dev/null 2>&1; then
    conda-env "$@"
    return
  fi

  echo "Could not find a usable 'conda env' or 'conda-env' command." >&2
  exit 1
}

if [ -d "${ENV_PREFIX}" ]; then
  echo "Updating existing conda environment at: ${ENV_PREFIX}"
  run_conda_env update \
    --prefix "${ENV_PREFIX}" \
    --file "${ENV_FILE}" \
    --prune
else
  echo "Creating conda environment at: ${ENV_PREFIX}"
  run_conda_env create \
    --prefix "${ENV_PREFIX}" \
    --file "${ENV_FILE}"
fi

PYTHON_BIN="${ENV_PREFIX}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Expected Python executable was not found at ${PYTHON_BIN}" >&2
  exit 1
fi

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
"${PYTHON_BIN}" -m ipykernel install --user \
  --name "${KERNEL_NAME}" \
  --display-name "${KERNEL_DISPLAY_NAME}"

echo
echo "Environment setup complete."
echo "Next steps:"
echo "  1. conda activate ${ENV_PREFIX}"
echo "  2. python ${REPO_ROOT}/scripts/generate_qwen_context_bloat_notebook.py"
echo "  3. jupyter lab ${REPO_ROOT}/notebooks/qwen_context_bloat_experiments.ipynb"
