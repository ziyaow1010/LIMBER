#!/bin/bash
# Jacobian layer ablation — RFAM dataset only (adds to existing 5-dataset ablation).
# 18 layers × 1000 RFAM seqs × avg_len ~200 × 4 fwd passes ≈ 8-10h on L40S.
#
#SBATCH --job-name=jac_rfam_abl
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/jac_rfam_abl_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/jac_rfam_abl_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT=/beacon-projects/rnallm/checkpoints/fm_hybrid_100m/epoch_checkpoints/epoch6
OUT=$REPO/outputs/jacobian_eval/hybrid_100m_ep6

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

echo "[jac_rfam_abl] Node  : $(hostname)"
echo "[jac_rfam_abl] GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[jac_rfam_abl] Start : $(date)"

export TOKENIZERS_PARALLELISM=false

python -u scripts/jacobian_ss_eval.py \
    --ckpt         "$CKPT" \
    --out          "$OUT"  \
    --mode         ablation \
    --abl_datasets RFAM \
    --abl_max      1000

echo "[jac_rfam_abl] Done : $(date)"
