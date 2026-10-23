#!/bin/bash
# Zero-shot Jacobian SS evaluation — 10M ep6, baseline (full model).
#SBATCH --job-name=jac_10m_base
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/jac_10m_base_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/jac_10m_base_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT=$REPO/outputs/hybrid_10m/epoch_checkpoints/epoch6
OUT=$REPO/outputs/jacobian_eval/hybrid_10m_ep6

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

echo "[jac_10m_base] Start : $(date)"
export TOKENIZERS_PARALLELISM=false

python -u scripts/jacobian_ss_eval.py \
    --ckpt    "$CKPT" \
    --out     "$OUT"  \
    --mode    baseline \
    --max_rfam 1000

echo "[jac_10m_base] Done : $(date)"
