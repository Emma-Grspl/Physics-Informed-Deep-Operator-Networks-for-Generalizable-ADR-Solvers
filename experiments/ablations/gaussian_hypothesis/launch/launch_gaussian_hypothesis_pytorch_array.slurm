#!/bin/bash
#SBATCH --job-name=ADR_PT_GAUSS_HYP
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

mkdir -p slurm/log

echo "Job on $SLURMD_NODENAME"
date
nvidia-smi
python -c "import torch; print('torch_cuda_available=', torch.cuda.is_available())"
python -c "import torch; print('torch_device_count=', torch.cuda.device_count())"

export PYTHONUNBUFFERED=1

VARIANTS=(
  "free_lbfgs_off ${EXPERIMENT_DIR}/configs/benchmarks/benchmark_gaussian_hyp_free_lbfgs_off.yaml ${EXPERIMENT_DIR}/configs/pytorch/config_ADR_t1_gaussian_hyp_free_lbfgs_off.yaml"
  "free_lbfgs_on ${EXPERIMENT_DIR}/configs/benchmarks/benchmark_gaussian_hyp_free_lbfgs_on.yaml ${EXPERIMENT_DIR}/configs/pytorch/config_ADR_t1_gaussian_hyp_free_lbfgs_on.yaml"
  "ansatz_lbfgs_off ${EXPERIMENT_DIR}/configs/benchmarks/benchmark_gaussian_hyp_ansatz_lbfgs_off.yaml ${EXPERIMENT_DIR}/configs/pytorch/config_ADR_t1_gaussian_hyp_ansatz_lbfgs_off.yaml"
  "ansatz_lbfgs_on ${EXPERIMENT_DIR}/configs/benchmarks/benchmark_gaussian_hyp_ansatz_lbfgs_on.yaml ${EXPERIMENT_DIR}/configs/pytorch/config_ADR_t1_gaussian_hyp_ansatz_lbfgs_on.yaml"
)
SEEDS=(0 1 2)

VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_INDEX]}

read -r LABEL BENCHMARK_CFG MODEL_CFG <<< "${VARIANTS[$VARIANT_INDEX]}"

echo "Variant=${LABEL} Seed=${SEED}"
echo "Benchmark=${BENCHMARK_CFG}"
echo "Model=${MODEL_CFG}"

srun python -u benchmarks/pytorch/train_fulltrainer_benchmark.py \
  --benchmark-config "${BENCHMARK_CFG}" \
  --model-config "${MODEL_CFG}" \
  --seed "${SEED}"

srun python -u benchmarks/pytorch/eval_benchmark.py \
  --benchmark-config "${BENCHMARK_CFG}" \
  --model-config "${MODEL_CFG}" \
  --seed "${SEED}"

srun python -u benchmarks/pytorch/inference_benchmark.py \
  --benchmark-config "${BENCHMARK_CFG}" \
  --model-config "${MODEL_CFG}" \
  --seed "${SEED}"
