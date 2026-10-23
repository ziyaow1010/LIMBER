#!/bin/bash
#SBATCH --job-name=robin_unimol_embed
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/robin_unimol_embed_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/robin_unimol_embed_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate unimol
cd "$REPO"
mkdir -p logs

echo "[robin_unimol_embed] Node : $(hostname)"
echo "[robin_unimol_embed] Start: $(date)"

python -u scripts/robin_extract_mol_embeddings.py

echo "[robin_unimol_embed] Done : $(date)"
