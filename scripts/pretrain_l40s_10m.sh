#!/bin/bash
# Hybrid-10M pretraining, full RNAcentral (epoch_seqs_full).
# Architecture: TTM×4 (12 layers), hidden=256, heads=4, FFN=1024, expand=4 → ~10.5M params.
# Effective batch = 336 × 1 GPU × 4 accum = 1344 (same as other models).
#
#SBATCH --job-name=rnafm_10m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_10m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_10m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=$REPO/outputs/hybrid_10m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[10m] Already complete, exiting."
    exit 0
fi

# Queue a follow-up in case we hit the time limit before finishing
sbatch --dependency=afterany:"$SLURM_JOB_ID" "$REPO/scripts/pretrain_l40s_10m.sh"
echo "[10m] Queued follow-up (afterany:$SLURM_JOB_ID)"

EPOCH_DIR=$DATA_DIR/epoch_seqs_full
MICRO_BATCH=84
GRAD_ACCUM=4

echo "[10m] Node: $(hostname)"
echo "[10m] GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[10m] Start: $(date)"
echo "[10m] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=1  eff_batch=$((MICRO_BATCH*1*GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --standalone --nproc_per_node=4 --master_port=29670 \
    scripts/train_hybrid650_mlm.py \
    --output_dir "$CKPT_DIR" \
    --epoch_dir  "$EPOCH_DIR" \
    --layer_pattern "TTMTTMTTMTTM" \
    --hidden_size 256 \
    --num_attention_heads 4 \
    --intermediate_size 1024 \
    --mamba_expand 4 \
    --mamba_d_state 16 \
    --micro_batch "$MICRO_BATCH" \
    --grad_accum  "$GRAD_ACCUM" \
    --epochs 6 \
    --save_steps 3000 \
    --val_steps  3000 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 4

echo "[10m] Done: $(date)"
