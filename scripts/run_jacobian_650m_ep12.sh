#!/bin/bash
# Zero-shot Jacobian SS evaluation — 650M ep12 (final), baseline + full layer
# ablation (all 7 datasets, all capped at 1000). 33 layers.
#
#SBATCH --job-name=jac_650m_ep12
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/jac_650m_ep12_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/jac_650m_ep12_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch12
OUT=$REPO/outputs/jacobian_eval/hybrid_650m_ep12

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

echo "[jac_650m_ep12] Node  : $(hostname)"
echo "[jac_650m_ep12] Start : $(date)"
echo "[jac_650m_ep12] Ckpt  : $CKPT"
echo "[jac_650m_ep12] Out   : $OUT"

export TOKENIZERS_PARALLELISM=false

if [ -f "$OUT/baseline_summary.json" ]; then
    echo "[jac_650m_ep12] Baseline already done, skipping."
else
    python -u scripts/jacobian_ss_eval.py \
        --ckpt  "$CKPT" \
        --out   "$OUT"  \
        --mode  baseline \
        --max_rfam 0
fi

python -u scripts/jacobian_ss_eval.py \
    --ckpt         "$CKPT" \
    --out          "$OUT"  \
    --mode         ablation \
    --abl_datasets PDB RNP SRP tmRNA tRNAdb RFAM CRW \
    --abl_max      1000

echo "[jac_650m_ep12] Done : $(date)"
