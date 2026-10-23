#!/bin/bash
# Zero-shot Jacobian SS evaluation — 300M ep6, full layer ablation (all 7 datasets).
# 15 layers (TTMTTMTTMTTMTTM); evaluates all 7 bpRNA datasets.
# RFAM/CRW capped at 1000 seqs; others uncapped.
# Estimated: ~45 min/layer × 15 layers ≈ 11h on a single L40S.
#
#SBATCH --job-name=jac_300m_abl
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/jac_300m_abl_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/jac_300m_abl_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT=/beacon-projects/rnallm/checkpoints/fm_hybrid_300m/epoch_checkpoints/epoch6
OUT=$REPO/outputs/jacobian_eval/hybrid_300m_ep6

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

echo "[jac_300m_abl] Node  : $(hostname)"
echo "[jac_300m_abl] GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[jac_300m_abl] Start : $(date)"
echo "[jac_300m_abl] Ckpt  : $CKPT"
echo "[jac_300m_abl] Out   : $OUT"

export TOKENIZERS_PARALLELISM=false

# Baseline: skip if already done
if [ -f "$OUT/baseline_summary.json" ]; then
    echo "[jac_300m_abl] Baseline already done, skipping."
else
    python -u scripts/jacobian_ss_eval.py \
        --ckpt         "$CKPT" \
        --out          "$OUT"  \
        --mode         baseline
fi

# Full ablation: all 7 datasets, CRW/RFAM capped at 1000
python -u scripts/jacobian_ss_eval.py \
    --ckpt         "$CKPT" \
    --out          "$OUT"  \
    --mode         ablation \
    --abl_datasets PDB RNP SRP tmRNA tRNAdb RFAM CRW \
    --abl_max      1000

echo "[jac_300m_abl] Done : $(date)"
