#!/bin/bash
# Hybrid-50M pretraining — scaling law data point below 100M.
# Architecture: hidden=512, heads=8, FFN=2048, mamba_expand=4, TTM×5 (15 layers) → ~49.5M params.
# Effective batch = 42*4*8 = 1344 (same as all other runs).
#
# Usage: sbatch scripts/pretrain_l40s_50m.sh
#SBATCH --job-name=rnafm_50m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_50m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_50m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_50m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
mkdir -p logs "$CKPT_DIR"

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[pretrain-50m] Already complete."
    cat "$CKPT_DIR/TRAINING_COMPLETE"
    exit 0
fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,EPOCHS="${EPOCHS:-6}" \
    "$REPO/scripts/pretrain_l40s_50m.sh"
echo "[pretrain-50m] Queued follow-up job (afterany:$SLURM_JOB_ID)"

if [ ! -L outputs/fm_hybrid_50m ] && [ ! -d outputs/fm_hybrid_50m ]; then
    ln -s "$CKPT_DIR" outputs/fm_hybrid_50m
fi

EPOCH_DIR=$DATA_DIR/epoch_seqs_full
if [ ! -f "$EPOCH_DIR/meta.json" ]; then
    echo "[pretrain-50m] ERROR: $EPOCH_DIR/meta.json not found" >&2
    exit 1
fi

echo "[pretrain-50m] Node: $(hostname)"
echo "[pretrain-50m] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[pretrain-50m] Start: $(date)"

MICRO_BATCH=${MICRO_BATCH:-42}
GRAD_ACCUM=${GRAD_ACCUM:-8}
echo "[pretrain-50m] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=4  eff_batch=$((MICRO_BATCH*4*GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=29655 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_hybrid_50m \
    --epoch_dir "$EPOCH_DIR" \
    --layer_pattern "TTMTTMTTMTTMTTM" \
    --hidden_size 512 \
    --num_attention_heads 8 \
    --intermediate_size 2048 \
    --mamba_expand 4 \
    --micro_batch "$MICRO_BATCH" \
    --grad_accum "$GRAD_ACCUM" \
    --epochs "${EPOCHS:-6}" \
    --save_steps 2000 --val_steps 2000 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

echo "[pretrain-50m] Done: $(date)"
