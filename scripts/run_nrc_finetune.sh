#!/bin/bash
#SBATCH --job-name=nrc_ft_650m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/nrc_ft_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/nrc_ft_%j.err

set -euo pipefail
cd /beacon-homes/ziyaow/RNA_FM

echo "[nrc_ft] Node: $(hostname)"
echo "[nrc_ft] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "[nrc_ft] Start: $(date)"

CKPT=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch5

source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm

python scripts/nrc_finetune.py \
    --ckpt_dir "$CKPT" \
    --train_fasta data/nrc/dataset_Rfam_6320_13classes.fasta \
    --val_fasta   data/nrc/dataset_Rfam_validated_2600_13classes.fasta \
    --tokenizer   tokenizers/single \
    --out_dir     outputs/nrc_finetune \
    --batch_size  64 \
    --phase1_epochs 10 \
    --phase2_epochs 20 \
    --lr_head 5e-4 \
    --lr_full 1e-5

echo "[nrc_ft] Done: $(date)"
