#!/bin/bash
# Baseline-only zero-shot Jacobian SS eval for Hybrid-650M at a given epoch,
# RFAM capped at 1000 (matches the existing epoch6 baseline) for a clean
# epoch-progression comparison (3 / 6 / 9 / 12).
#SBATCH --job-name=jac_650m_epsweep
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/jac_650m_epsweep_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/jac_650m_epsweep_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
EPOCH=$1
CKPT=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch${EPOCH}
OUT=$REPO/outputs/jacobian_eval/hybrid_650m_ep${EPOCH}_epsweep

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

echo "[jac_650m_epsweep] epoch : $EPOCH"
echo "[jac_650m_epsweep] Node  : $(hostname)"
echo "[jac_650m_epsweep] Ckpt  : $CKPT"
echo "[jac_650m_epsweep] Out   : $OUT"

export TOKENIZERS_PARALLELISM=false

python -u scripts/jacobian_ss_eval.py \
    --ckpt  "$CKPT" \
    --out   "$OUT"  \
    --mode  baseline

echo "[jac_650m_epsweep] Done : $(date)"
