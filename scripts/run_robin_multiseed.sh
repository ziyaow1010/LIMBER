#!/bin/bash
#SBATCH --job-name=robin_multiseed
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/robin_multiseed_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/robin_multiseed_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate unimol
cd "$REPO"
mkdir -p logs

echo "[robin_multiseed] Node : $(hostname)"
echo "[robin_multiseed] Start: $(date)"

python -u scripts/robin_train_multiseed.py

echo "[robin_multiseed] Done : $(date)"
