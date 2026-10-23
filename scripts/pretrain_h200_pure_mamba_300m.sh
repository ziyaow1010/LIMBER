#!/bin/bash
# Pure-Mamba-300M pretraining: 15 all-M layers, hidden=1280.
# Baseline for Hybrid-300M comparison (same depth/width, all Mamba).
# Effective batch = 48 × 4 GPU × 7 accum = 1344 (identical to Hybrid-300M).
#
#SBATCH --job-name=rnafm_pm300m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_pm300m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_pm300m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_pure_mamba_300m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[pm300m] Already complete, exiting."
    exit 0
fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" "$REPO/scripts/pretrain_h200_pure_mamba_300m.sh"
echo "[pm300m] Queued follow-up (afterany:$SLURM_JOB_ID)"

mkdir -p "$CKPT_DIR"
if [ ! -L outputs/fm_pure_mamba_300m ]; then
    rm -rf outputs/fm_pure_mamba_300m
    ln -s "$CKPT_DIR" outputs/fm_pure_mamba_300m
fi

echo "[pm300m] Node : $(hostname)"
echo "[pm300m] GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[pm300m] Start: $(date)"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=29661 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_pure_mamba_300m \
    --epoch_dir  "$DATA_DIR/epoch_seqs_full" \
    --layer_pattern "MMMMMMMMMMMMMMM" \
    --hidden_size 1280 \
    --num_attention_heads 20 \
    --intermediate_size 5120 \
    --mamba_expand 4 \
    --mamba_d_state 16 \
    --micro_batch 48 \
    --grad_accum 7 \
    --epochs 6 \
    --save_steps 2000 --val_steps 2000 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

echo "[pm300m] Done: $(date)"
