#!/bin/bash
#SBATCH --job-name=ADR_FULL_JAX_T1
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
source /lustre/fswork/projects/rech/fdb/usv13rn/venvs/jax_adr/bin/activate

mkdir -p slurm/log

echo "Job on $SLURMD_NODENAME"
date
nvidia-smi
python -c "import jax; print('jax_devices=', jax.devices())"
python -c "import jax; print('jax_backend=', jax.default_backend())"

export PYTHONUNBUFFERED=1

srun python -u code/benchmarks/jax/train_fulltrainer_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_fulltrainer_t1.yaml \
  --model-config code/configs_jax/config_ADR_jax_t1_compare.yaml

srun python -u code/benchmarks/jax/eval_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_fulltrainer_t1.yaml \
  --model-config code/configs_jax/config_ADR_jax_t1_compare.yaml

srun python -u code/benchmarks/jax/inference_benchmark.py \
  --benchmark-config code/benchmarks/configs/benchmark_fulltrainer_t1.yaml \
  --model-config code/configs_jax/config_ADR_jax_t1_compare.yaml
