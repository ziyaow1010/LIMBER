#!/bin/bash
# Hybrid-650M continued pretraining: epoch 7-12 (resume from epoch 6 checkpoint).
# Identical settings to the original 6-epoch run; resumes automatically from
# checkpoint-180804. Uses the same epoch_seqs_full data (epoch7-12 are symlinks
# to train_full.txt, same as epoch1-6). Effective batch = 48*4*7 = 1344.
#
# Usage: sbatch scripts/pretrain_h200_650m_ep12.sh
#SBATCH --job-name=rnafm_ep12
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_ep12_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_ep12_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
mkdir -p logs

# Use a separate completion flag so the original ep6 TRAINING_COMPLETE is preserved
if [ -f "$CKPT_DIR/TRAINING_COMPLETE_EP12" ]; then
    echo "[pretrain-ep12] Already complete (ep12)."
    cat "$CKPT_DIR/TRAINING_COMPLETE_EP12"
    exit 0
fi

# Self-chain follow-up job (fires whether this job finishes, fails, or times out)
sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL \
    "$REPO/scripts/pretrain_h200_650m_ep12.sh"
echo "[pretrain-ep12] Queued follow-up job (afterany:$SLURM_JOB_ID)"

# Symlink already exists from original run; just verify
if [ ! -L outputs/fm_hybrid_650m ] && [ ! -d outputs/fm_hybrid_650m ]; then
    ln -s "$CKPT_DIR" outputs/fm_hybrid_650m
fi

EPOCH_DIR=$DATA_DIR/epoch_seqs_full
echo "[pretrain-ep12] Node: $(hostname)"
echo "[pretrain-ep12] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[pretrain-ep12] Start: $(date)"

# Reuse bench values from original run
MICRO_BATCH=48
GRAD_ACCUM=7
echo "[pretrain-ep12] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=4  eff_batch=$((MICRO_BATCH*4*GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Remove the ep6 TRAINING_COMPLETE so the train script doesn't bail early
# (it writes its own flag at the end of training)
# We do NOT delete it -- the train script checks it on save, not on start.
# The flag is written by train script only after reaching max_steps=epochs*steps_per_epoch.
# With --epochs 12 it will compute max_steps=361956 and run from step 180804 to 361956.

torchrun --nproc_per_node=4 --master_port=29654 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_hybrid_650m \
    --epoch_dir "$EPOCH_DIR" \
    --epochs 12 \
    --micro_batch "$MICRO_BATCH" \
    --grad_accum "$GRAD_ACCUM" \
    --save_steps 2000 --val_steps 2000 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

# Training script writes TRAINING_COMPLETE itself; rename to ep12 flag
if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    cp "$CKPT_DIR/TRAINING_COMPLETE" "$CKPT_DIR/TRAINING_COMPLETE_EP12"
fi

echo "[pretrain-ep12] Done: $(date)"
