#!/bin/bash
# Hybrid-100M-Thin: tall & thin variant for depth vs width study.
# hidden=512, heads=8, FFN=2048, mamba_expand=4, TTM×10 (30 layers) → ~94M params.
# vs baseline: hidden=640, heads=10, FFN=2560, TTM×6 (18 layers) → ~92M params.
# Effective batch = 42×4×8 = 1344 (identical to baseline).
#
# Usage: sbatch scripts/pretrain_hybrid_100m_thin.sh
#SBATCH --job-name=h100m_thin
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_h100m_thin_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_h100m_thin_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/hybrid_100m_thin

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
mkdir -p logs "$CKPT_DIR"

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[h100m-thin] Already complete."
    cat "$CKPT_DIR/TRAINING_COMPLETE"
    exit 0
fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,EPOCHS="${EPOCHS:-6}" \
    "$REPO/scripts/pretrain_hybrid_100m_thin.sh"
echo "[h100m-thin] Queued follow-up job (afterany:$SLURM_JOB_ID)"

if [ ! -L outputs/hybrid_100m_thin ] && [ ! -d outputs/hybrid_100m_thin ]; then
    ln -s "$CKPT_DIR" outputs/hybrid_100m_thin
fi

EPOCH_DIR=$DATA_DIR/epoch_seqs_full
if [ ! -f "$EPOCH_DIR/meta.json" ]; then
    echo "[h100m-thin] ERROR: $EPOCH_DIR/meta.json not found" >&2
    exit 1
fi

echo "[h100m-thin] Node: $(hostname)"
echo "[h100m-thin] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[h100m-thin] Start: $(date)"

MICRO_BATCH=${MICRO_BATCH:-42}
GRAD_ACCUM=${GRAD_ACCUM:-8}
echo "[h100m-thin] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=4  eff_batch=$((MICRO_BATCH*4*GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=29658 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/hybrid_100m_thin \
    --epoch_dir "$EPOCH_DIR" \
    --layer_pattern "TTMTTMTTMTTMTTMTTMTTMTTMTTMTTM" \
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

echo "[h100m-thin] Done: $(date)"
