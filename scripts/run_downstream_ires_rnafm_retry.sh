#!/bin/bash
# Retry: rnafm_100m IRES fine-tune (crashed earlier due to a max-length cap bug, now fixed).
#
#SBATCH --job-name=ires_ft_rnafm_retry
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ires_ft_rnafm_retry_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ires_ft_rnafm_retry_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

python scripts/downstream_finetune.py \
    --model_type rnafm_100m --task ires \
    --out_dir "outputs/downstream_ft/ires/rnafm_100m" \
    > "logs/ires_ft_rnafm_100m.log" 2>&1

echo "[ires_ft] rnafm_100m retry done: $(date)"
