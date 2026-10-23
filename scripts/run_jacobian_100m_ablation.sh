#!/bin/bash
# Zero-shot Jacobian SS evaluation — 100M ep6, layer ablation.
# Skips each of the 18 layers in turn; evaluates on 5 small datasets.
# Each ablation config ~ 30 min; total ~10h on a single L40S.
#
#SBATCH --job-name=jac_100m_abl
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/jac_100m_abl_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/jac_100m_abl_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT=/beacon-projects/rnallm/checkpoints/fm_hybrid_100m/epoch_checkpoints/epoch6
OUT=$REPO/outputs/jacobian_eval/hybrid_100m_ep6

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

echo "[jac_100m_abl] Node  : $(hostname)"
echo "[jac_100m_abl] GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[jac_100m_abl] Start : $(date)"
echo "[jac_100m_abl] Ckpt  : $CKPT"
echo "[jac_100m_abl] Out   : $OUT"

export TOKENIZERS_PARALLELISM=false

# Small datasets only (PDB/RNP/SRP/tmRNA/tRNAdb); all sequences (no cap)
python -u scripts/jacobian_ss_eval.py \
    --ckpt         "$CKPT" \
    --out          "$OUT"  \
    --mode         ablation \
    --abl_datasets PDB RNP SRP tmRNA tRNAdb

echo "[jac_100m_abl] Done : $(date)"
