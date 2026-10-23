#!/bin/bash
#SBATCH --job-name=extract_cls_full
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/extract_cls_full_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/extract_cls_full_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

echo "[extract_cls_full] Node  : $(hostname)"
echo "[extract_cls_full] Start : $(date)"

python -u scripts/extract_family_cls_full.py

echo "[extract_cls_full] Done : $(date)"
