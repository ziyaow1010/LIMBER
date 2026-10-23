#!/bin/bash
#SBATCH --job-name=jac_crw_abl
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/jac_crw_abl_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/jac_crw_abl_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO" && mkdir -p logs

echo "[jac_crw_abl] Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[jac_crw_abl] Start: $(date)"
export TOKENIZERS_PARALLELISM=false

python -u scripts/jacobian_ss_eval.py \
    --ckpt  /beacon-projects/rnallm/checkpoints/fm_hybrid_100m/epoch_checkpoints/epoch6 \
    --out   $REPO/outputs/jacobian_eval/hybrid_100m_ep6 \
    --mode  ablation \
    --abl_datasets CRW \
    --abl_max 1000

echo "[jac_crw_abl] Done: $(date)"
