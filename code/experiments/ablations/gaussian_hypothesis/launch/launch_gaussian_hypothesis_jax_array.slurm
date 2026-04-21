#!/bin/bash
# Jean Zay SLURM launcher for this experiment package.
# Adjust account, modules, constraints, and environment activation for other HPC systems.

#SBATCH --job-name=ADR_JAX_GAUSS_HYP
#SBATCH --output=slurm/log/%x_%A_%a.out
#SBATCH --error=slurm/log/%x_%A_%a.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --array=0-11

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${EXPERIMENT_DIR}/../../.." && pwd)"

cd "${ROOT_DIR}"

module purge
module load pytorch-gpu/py3/2.1.1

JAX_ADR_VENV="${JAX_ADR_VENV:-}"
if [ -n "${JAX_ADR_VENV}" ] && [ -f "${JAX_ADR_VENV}/bin/activate" ]; then
  VENV_ACTIVATE="${JAX_ADR_VENV}/bin/activate"
else
  for candidate in \
    "/lustre/fsn1/projects/rech/fdb/usv13rn/venvs/jax_adr/bin/activate" \
    "/lustre/fswork/projects/rech/fdb/usv13rn/venvs/jax_adr/bin/activate" \
    "${WORK:-}/venvs/jax_adr/bin/activate" \
    "${SCRATCH:-}/venvs/jax_adr/bin/activate"
  do
    if [ -f "${candidate}" ]; then
      VENV_ACTIVATE="${candidate}"
      break
    fi
  done
fi

if [ -z "${VENV_ACTIVATE:-}" ]; then
  echo "JAX ADR virtualenv not found. Falling back to module python." >&2
else
  echo "Using JAX venv: ${VENV_ACTIVATE}"
  source "${VENV_ACTIVATE}"
fi

python - <<'PY'
from importlib.util import find_spec
mods = ["jax", "optax", "scipy"]
missing = [m for m in mods if find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")
print("Python environment OK for JAX benchmark.")
PY

mkdir -p slurm/log

echo "Job on $SLURMD_NODENAME"
date
nvidia-smi
python -c "import jax; print('jax_devices=', jax.devices())"
python -c "import jax; print('jax_backend=', jax.default_backend())"
python -c "import scipy; print('scipy_version=', scipy.__version__)"

export PYTHONUNBUFFERED=1

VARIANTS=(
  "free_lbfgs_off ${EXPERIMENT_DIR}/configs/benchmarks/benchmark_gaussian_hyp_free_lbfgs_off.yaml ${EXPERIMENT_DIR}/configs/jax/config_ADR_jax_t1_gaussian_hyp_free_lbfgs_off.yaml"
  "free_lbfgs_on ${EXPERIMENT_DIR}/configs/benchmarks/benchmark_gaussian_hyp_free_lbfgs_on.yaml ${EXPERIMENT_DIR}/configs/jax/config_ADR_jax_t1_gaussian_hyp_free_lbfgs_on.yaml"
  "ansatz_lbfgs_off ${EXPERIMENT_DIR}/configs/benchmarks/benchmark_gaussian_hyp_ansatz_lbfgs_off.yaml ${EXPERIMENT_DIR}/configs/jax/config_ADR_jax_t1_gaussian_hyp_ansatz_lbfgs_off.yaml"
  "ansatz_lbfgs_on ${EXPERIMENT_DIR}/configs/benchmarks/benchmark_gaussian_hyp_ansatz_lbfgs_on.yaml ${EXPERIMENT_DIR}/configs/jax/config_ADR_jax_t1_gaussian_hyp_ansatz_lbfgs_on.yaml"
)
SEEDS=(0 1 2)

VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_INDEX]}

read -r LABEL BENCHMARK_CFG MODEL_CFG <<< "${VARIANTS[$VARIANT_INDEX]}"

echo "Variant=${LABEL} Seed=${SEED}"
echo "Benchmark=${BENCHMARK_CFG}"
echo "Model=${MODEL_CFG}"

srun python -u code/benchmarks/jax/train_fulltrainer_benchmark.py \
  --benchmark-config "${BENCHMARK_CFG}" \
  --model-config "${MODEL_CFG}" \
  --seed "${SEED}"

srun python -u code/benchmarks/jax/eval_benchmark.py \
  --benchmark-config "${BENCHMARK_CFG}" \
  --model-config "${MODEL_CFG}" \
  --seed "${SEED}"

srun python -u code/benchmarks/jax/inference_benchmark.py \
  --benchmark-config "${BENCHMARK_CFG}" \
  --model-config "${MODEL_CFG}" \
  --seed "${SEED}"
