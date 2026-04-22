#!/bin/bash
#SBATCH --job-name=ADR_CURR_JAX_T05
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
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
python -c "import jaxlib; print('jaxlib_version=', jaxlib.__version__)"

export PYTHONUNBUFFERED=1

srun python -u benchmarks/jax/train_curriculum_benchmark.py \
  --benchmark-config benchmarks/configs/benchmark_curriculum_t05.yaml

srun python -u benchmarks/jax/eval_benchmark.py \
  --benchmark-config benchmarks/configs/benchmark_curriculum_t05.yaml

srun python -u benchmarks/jax/inference_benchmark.py \
  --benchmark-config benchmarks/configs/benchmark_curriculum_t05.yaml
