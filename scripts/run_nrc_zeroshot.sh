#!/bin/bash
#SBATCH --job-name=nrc_zero
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/nrc_zero_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/nrc_zero_%j.err

set -euo pipefail
cd /beacon-homes/ziyaow/RNA_FM

echo "[zero-shot] Node: $(hostname)"
echo "[zero-shot] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "[zero-shot] Start: $(date)"

source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm

python scripts/nrc_zeroshot.py \
    --ckpt_dir /beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch5 \
    --train_fasta data/nrc/dataset_Rfam_6320_13classes.fasta \
    --val_fasta   data/nrc/dataset_Rfam_validated_2600_13classes.fasta \
    --tokenizer   tokenizers/single \
    --out_dir     outputs/nrc_zeroshot \
    --batch_size  128

echo "[zero-shot] Done: $(date)"
