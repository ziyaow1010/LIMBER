#!/bin/bash
# Pure-Mamba-650M pretraining: 33 all-M layers, hidden=1280 (688M params).
# Baseline for Hybrid-650M comparison (same depth/width, all Mamba).
# Effective batch = 24 × 4 GPU × 14 accum = 1344 (same as Hybrid-650M target).
#
#SBATCH --job-name=rnafm_pm650m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_pm650m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_pm650m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_pure_mamba_650m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[pm650m] Already complete, exiting."
    exit 0
fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" "$REPO/scripts/pretrain_h200_pure_mamba_650m.sh"
echo "[pm650m] Queued follow-up (afterany:$SLURM_JOB_ID)"

mkdir -p "$CKPT_DIR"
if [ ! -L outputs/fm_pure_mamba_650m ]; then
    rm -rf outputs/fm_pure_mamba_650m
    ln -s "$CKPT_DIR" outputs/fm_pure_mamba_650m
fi

echo "[pm650m] Node : $(hostname)"
echo "[pm650m] GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[pm650m] Start: $(date)"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=29663 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_pure_mamba_650m \
    --epoch_dir  "$DATA_DIR/epoch_seqs_full" \
    --layer_pattern "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM" \
    --hidden_size 1280 \
    --num_attention_heads 20 \
    --intermediate_size 5120 \
    --mamba_expand 4 \
    --mamba_d_state 16 \
    --micro_batch 24 \
    --grad_accum 14 \
    --epochs 6 \
    --save_steps 2000 --val_steps 2000 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

echo "[pm650m] Done: $(date)"
