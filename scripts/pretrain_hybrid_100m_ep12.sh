#!/bin/bash
# Hybrid-100M continue pretraining: epochs 7-12 (resume from ep6 checkpoint).
# Sets --epochs 12; HF Trainer resumes from checkpoint-180804 and runs to step 361608.
# Uses TRAINING_COMPLETE_EP12 as completion marker to avoid conflicting with ep6 marker.
#
#SBATCH --job-name=rnafm_100m_ep12
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_100m_ep12_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_100m_ep12_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_100m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

if [ -f "$CKPT_DIR/TRAINING_COMPLETE_EP12" ]; then
    echo "[100m-ep12] Already complete at ep12, exiting."
    exit 0
fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" "$REPO/scripts/pretrain_hybrid_100m_ep12.sh"
echo "[100m-ep12] Queued follow-up (afterany:$SLURM_JOB_ID)"

# Symlink so train script writes to CKPT_DIR
if [ ! -L outputs/fm_hybrid_100m ] && [ ! -d outputs/fm_hybrid_100m/.git ]; then
    rm -rf outputs/fm_hybrid_100m
    ln -s "$CKPT_DIR" outputs/fm_hybrid_100m
fi

echo "[100m-ep12] Node: $(hostname)"
echo "[100m-ep12] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[100m-ep12] Start: $(date)"
echo "[100m-ep12] Resuming from step 180804 (ep6) → target ep12 (step 361608)"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=29654 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_hybrid_100m \
    --epoch_dir "$DATA_DIR/epoch_seqs_full" \
    --layer_pattern "TTMTTMTTMTTMTTMTTM" \
    --hidden_size 640 \
    --num_attention_heads 10 \
    --intermediate_size 2560 \
    --mamba_expand 4 \
    --micro_batch 42 \
    --grad_accum 8 \
    --epochs 12 \
    --save_steps 2000 --val_steps 2000 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

# Rename completion marker so the original ep6 check isn't confused
if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    cp "$CKPT_DIR/TRAINING_COMPLETE" "$CKPT_DIR/TRAINING_COMPLETE_EP12"
fi

echo "[100m-ep12] Done: $(date)"
