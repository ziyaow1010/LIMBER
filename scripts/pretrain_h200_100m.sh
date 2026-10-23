#!/bin/bash
# Hybrid-100M pretraining — same data/optim as 650M/300M, reduced hidden for ~92M params.
# Architecture: hidden=640, heads=10, FFN=2560, mamba_expand=4, TTM×6 (18 layers) → ~92M params.
# Effective batch = 84*4*4 = 1344 (same as 650M run).
#
# Usage: sbatch scripts/pretrain_h200_100m.sh
#SBATCH --job-name=rnafm_100m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_100m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_100m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_100m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
mkdir -p logs "$CKPT_DIR"

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[pretrain-100m] Already complete."
    cat "$CKPT_DIR/TRAINING_COMPLETE"
    exit 0
fi

# Self-chain follow-up job
sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,EPOCHS="${EPOCHS:-6}" \
    "$REPO/scripts/pretrain_h200_100m.sh"
echo "[pretrain-100m] Queued follow-up job (afterany:$SLURM_JOB_ID)"

# Symlink checkpoint dir
if [ ! -L outputs/fm_hybrid_100m ] && [ ! -d outputs/fm_hybrid_100m/.git ]; then
    rm -rf outputs/fm_hybrid_100m
    ln -s "$CKPT_DIR" outputs/fm_hybrid_100m
    echo "[pretrain-100m] Symlinked outputs/fm_hybrid_100m -> $CKPT_DIR"
fi

EPOCH_DIR=$DATA_DIR/epoch_seqs_full
if [ ! -f "$EPOCH_DIR/meta.json" ]; then
    echo "[pretrain-100m] ERROR: $EPOCH_DIR/meta.json not found" >&2
    exit 1
fi
echo "[pretrain-100m] Using per-epoch data: $EPOCH_DIR"

echo "[pretrain-100m] Node: $(hostname)"
echo "[pretrain-100m] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[pretrain-100m] Start: $(date)"

# 100M model is much smaller; micro_batch=84 grad_accum=4 → effective batch = 84*4*4 = 1344
MICRO_BATCH=${MICRO_BATCH:-42}
GRAD_ACCUM=${GRAD_ACCUM:-8}

echo "[pretrain-100m] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=4"
echo "[pretrain-100m] Effective batch = $((MICRO_BATCH * 4 * GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=29653 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_hybrid_100m \
    --epoch_dir "$EPOCH_DIR" \
    --layer_pattern "TTMTTMTTMTTMTTMTTM" \
    --hidden_size 640 \
    --num_attention_heads 10 \
    --intermediate_size 2560 \
    --mamba_expand 4 \
    --micro_batch "$MICRO_BATCH" \
    --grad_accum "$GRAD_ACCUM" \
    --epochs "${EPOCHS:-6}" \
    --save_steps "${SAVE_STEPS:-2000}" --val_steps "${VAL_STEPS:-2000}" \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

echo "[pretrain-100m] Done: $(date)"
