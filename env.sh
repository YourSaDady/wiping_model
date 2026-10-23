# Source this from the project root: `source env.sh`
export PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJ_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJ_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
