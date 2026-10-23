#!/bin/bash
# Hybrid-300M pretraining — same setup as 650M, half the depth (TTM×5 = 15 layers).
# Architecture: hidden=1280, heads=20, FFN=5120, mamba_expand=4, 15 layers → ~297M params.
# Data/optim/batch identical to 650M run for fair comparison.
#
# Usage: sbatch scripts/pretrain_h200_300m.sh
#        sbatch --gres=gpu:nvidia_l40s:4 --export=ALL,BENCH_FILE=l40s_bench.json scripts/pretrain_h200_300m.sh
#SBATCH --job-name=rnafm_300m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_300m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_300m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_300m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
mkdir -p logs "$CKPT_DIR"

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[pretrain-300m] Already complete."
    cat "$CKPT_DIR/TRAINING_COMPLETE"
    exit 0
fi

# Self-chain follow-up job
sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,EPOCHS="${EPOCHS:-6}" \
    "$REPO/scripts/pretrain_h200_300m.sh"
echo "[pretrain-300m] Queued follow-up job (afterany:$SLURM_JOB_ID)"

# Symlink checkpoint dir
if [ ! -L outputs/fm_hybrid_300m ] && [ ! -d outputs/fm_hybrid_300m/.git ]; then
    rm -rf outputs/fm_hybrid_300m
    ln -s "$CKPT_DIR" outputs/fm_hybrid_300m
    echo "[pretrain-300m] Symlinked outputs/fm_hybrid_300m -> $CKPT_DIR"
fi

EPOCH_DIR=$DATA_DIR/epoch_seqs_full
if [ ! -f "$EPOCH_DIR/meta.json" ]; then
    echo "[pretrain-300m] ERROR: $EPOCH_DIR/meta.json not found" >&2
    exit 1
fi
echo "[pretrain-300m] Using per-epoch data: $EPOCH_DIR"

echo "[pretrain-300m] Node: $(hostname)"
echo "[pretrain-300m] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[pretrain-300m] Start: $(date)"

# 300M is smaller than 650M, so 650M bench values (micro_batch=48, grad_accum=7)
# are a safe conservative default → effective batch = 48*4*7 = 1344 (same as 650M).
MICRO_BATCH=${MICRO_BATCH:-48}
GRAD_ACCUM=${GRAD_ACCUM:-7}

echo "[pretrain-300m] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=4"
echo "[pretrain-300m] Effective batch = $((MICRO_BATCH * 4 * GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --master_port=29652 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_hybrid_300m \
    --epoch_dir "$EPOCH_DIR" \
    --layer_pattern "TTMTTMTTMTTMTTM" \
    --hidden_size 1280 \
    --num_attention_heads 20 \
    --intermediate_size 5120 \
    --mamba_expand 4 \
    --micro_batch "$MICRO_BATCH" \
    --grad_accum "$GRAD_ACCUM" \
    --epochs "${EPOCHS:-6}" \
    --save_steps "${SAVE_STEPS:-2000}" --val_steps "${VAL_STEPS:-2000}" \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

echo "[pretrain-300m] Done: $(date)"
