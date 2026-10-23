#!/bin/bash
#SBATCH --job-name=layerimp_10m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/layerimp_10m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/layerimp_10m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT=$REPO/outputs/hybrid_10m/epoch_checkpoints/epoch6
OUT=$REPO/outputs/layer_importance/hybrid_10m_ep6

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

echo "[layerimp_10m] Start: $(date)"
echo "[layerimp_10m] Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -u scripts/layer_importance.py \
    --ckpt "$CKPT" \
    --out  "$OUT"  \
    --n_val 4096   \
    --batch 32

echo "[layerimp_10m] Done: $(date)"
