#!/bin/bash
#SBATCH --job-name=ADR_NewC     # Nom du job
#SBATCH --output=logs/%x_%j.out     # Log standard (créer le dossier logs/ avant !)
#SBATCH --error=logs/%x_%j.err      # Log d'erreur
#SBATCH --time=20:00:00             # Temps max (20h pour être large)
#SBATCH --nodes=1                   # 1 noeud
#SBATCH --ntasks=1                  # 1 tache MPI
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --cpus-per-task=10          # 10 CPU pour le chargement des données
#SBATCH --hint=nomultithread        # Pas d'hyperthreading
#SBATCH --partition=gpu_p13         # Partition GPU (Vérifie si tu as accès à p13 ou p2)

# Chargement des modules (Adapte selon ton environnement)
module purge
module load pytorch-gpu/py3/2.0.0   # Ou ton module habituel

# Activation de ton environnement Conda (si tu en utilises un spécifique)
# source activate ton_env_conda

# Echo des infos pour le debug
echo "Job lancé sur $(hostname)"
echo "Début : $(date)"

# Lancement du script python
# Le -u permet d'avoir les prints en temps réel dans le fichier log
python -u train.py

echo "Fin : $(date)"
