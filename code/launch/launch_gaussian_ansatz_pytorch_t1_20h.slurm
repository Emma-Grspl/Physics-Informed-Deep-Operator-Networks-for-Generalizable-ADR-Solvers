#!/bin/bash
#SBATCH --job-name=ADR_PT_GAUSS_ANSATZ
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

mkdir -p slurm/log

echo "Job on $SLURMD_NODENAME"
date
nvidia-smi
python -c "import torch; print('torch_cuda_available=', torch.cuda.is_available())"
python -c "import torch; print('torch_device_count=', torch.cuda.device_count())"

export PYTHONUNBUFFERED=1

srun python -u code/benchmarks/pytorch/train_fulltrainer_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_gaussian_ansatz_t1.yaml \
  --model-config code/configs/config_ADR_t1_gaussian_ansatz.yaml

srun python -u code/benchmarks/pytorch/eval_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_gaussian_ansatz_t1.yaml \
  --model-config code/configs/config_ADR_t1_gaussian_ansatz.yaml

srun python -u code/benchmarks/pytorch/inference_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_gaussian_ansatz_t1.yaml \
  --model-config code/configs/config_ADR_t1_gaussian_ansatz.yaml
