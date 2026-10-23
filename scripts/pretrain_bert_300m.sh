#!/bin/bash
# BERT-300M baseline — all-Transformer, same dims/layers as Hybrid-300M.
# hidden=1280, heads=20, FFN=5120, 15×T layers → ~297M params.
# Effective batch = 24×4×14 = 1344 (identical to hybrid).
#
# Usage: sbatch scripts/pretrain_bert_300m.sh
#SBATCH --job-name=bert_300m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_bert_300m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_bert_300m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/bert_300m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
mkdir -p logs "$CKPT_DIR"

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[bert-300m] Already complete."
    cat "$CKPT_DIR/TRAINING_COMPLETE"
    exit 0
fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,EPOCHS="${EPOCHS:-6}" \
    "$REPO/scripts/pretrain_bert_300m.sh"
echo "[bert-300m] Queued follow-up job (afterany:$SLURM_JOB_ID)"

if [ ! -L outputs/bert_300m ] && [ ! -d outputs/bert_300m ]; then
    ln -s "$CKPT_DIR" outputs/bert_300m
fi

EPOCH_DIR=$DATA_DIR/epoch_seqs_full
if [ ! -f "$EPOCH_DIR/meta.json" ]; then
    echo "[bert-300m] ERROR: $EPOCH_DIR/meta.json not found" >&2
    exit 1
fi

echo "[bert-300m] Node: $(hostname)"
echo "[bert-300m] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[bert-300m] Start: $(date)"

MICRO_BATCH=${MICRO_BATCH:-24}
GRAD_ACCUM=${GRAD_ACCUM:-14}
echo "[bert-300m] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=4  eff_batch=$((MICRO_BATCH*4*GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=29657 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/bert_300m \
    --epoch_dir "$EPOCH_DIR" \
    --layer_pattern "TTTTTTTTTTTTTTT" \
    --hidden_size 1280 \
    --num_attention_heads 20 \
    --intermediate_size 5120 \
    --micro_batch "$MICRO_BATCH" \
    --grad_accum "$GRAD_ACCUM" \
    --epochs "${EPOCHS:-6}" \
    --save_steps 2000 --val_steps 2000 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

echo "[bert-300m] Done: $(date)"
