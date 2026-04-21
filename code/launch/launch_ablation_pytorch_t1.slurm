#!/bin/bash
#SBATCH --job-name=ADR_ABL_PT_T1
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=15:00:00
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

echo "=== Fixed horizon baseline: T=1.0 ==="
srun python -u code/benchmarks/pytorch/train_short_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_t1_20k.yaml
srun python -u code/benchmarks/pytorch/eval_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_t1_20k.yaml
srun python -u code/benchmarks/pytorch/inference_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_t1_20k.yaml

echo "=== Curriculum v1: T=1.0 ==="
srun python -u code/benchmarks/pytorch/train_curriculum_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_curriculum_t1.yaml
srun python -u code/benchmarks/pytorch/eval_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_curriculum_t1.yaml
srun python -u code/benchmarks/pytorch/inference_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_curriculum_t1.yaml

echo "=== Time marching v2: T=1.0 ==="
srun python -u code/benchmarks/pytorch/train_timemarching_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_timemarch_t1.yaml
srun python -u code/benchmarks/pytorch/eval_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_timemarch_t1.yaml
srun python -u code/benchmarks/pytorch/inference_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_timemarch_t1.yaml
