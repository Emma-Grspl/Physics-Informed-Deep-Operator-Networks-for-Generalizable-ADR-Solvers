#!/bin/bash
#SBATCH --job-name=ADR_reprise     # Nom du job
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100                 # <--- TON COMPTE PROJET ICI
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3

# Nettoyage
module purge
module load pytorch-gpu/py3/2.1.1

# Création du dossier de logs (au cas où)
mkdir -p slurm/log

echo "🚀 Job lancé sur $SLURMD_NODENAME"
nvidia-smi

# Lancement de ton script d'entraînement
srun python train_reprise.py
