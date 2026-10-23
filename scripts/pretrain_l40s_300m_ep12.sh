#!/bin/bash
# Hybrid-300M continued pretraining: epoch 7-12 (resume from epoch 6 checkpoint).
# L40S variant. Effective batch = 8*4*42 = 1344 (small micro_batch for L40S 44GiB VRAM).
#
#SBATCH --job-name=rnafm_300m_ep12
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_300m_ep12_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_300m_ep12_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_300m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
mkdir -p logs

if [ -f "$CKPT_DIR/TRAINING_COMPLETE_EP12" ]; then
    echo "[300m-ep12] Already complete (ep12)."
    cat "$CKPT_DIR/TRAINING_COMPLETE_EP12"
    exit 0
fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    "$REPO/scripts/pretrain_l40s_300m_ep12.sh"
echo "[300m-ep12] Queued follow-up job (afterany:$SLURM_JOB_ID)"

EPOCH_DIR=$DATA_DIR/epoch_seqs_full
echo "[300m-ep12] Node: $(hostname)"
echo "[300m-ep12] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[300m-ep12] Start: $(date)"

MICRO_BATCH=8
GRAD_ACCUM=42
echo "[300m-ep12] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=4  eff_batch=$((MICRO_BATCH*4*GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3


torchrun --nproc_per_node=4 --master_port=29653 \
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
    --epochs 12 \
    --save_steps 2000 --val_steps 2000 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    cp "$CKPT_DIR/TRAINING_COMPLETE" "$CKPT_DIR/TRAINING_COMPLETE_EP12"
fi

echo "[300m-ep12] Done: $(date)"
