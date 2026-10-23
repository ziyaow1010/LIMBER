#!/bin/bash
# Pure-Mamba-100M pretraining: 18 all-M layers, hidden=640.
# Architecture: "MMMMMMMMMMMMMMMMMM" (18×M), ~95.8M params.
# Baseline for hybrid comparison: same depth/width as Hybrid-100M (91.7M).
# Effective batch = 42 × 4 GPU × 8 accum = 1344 (identical to Hybrid-100M).
#
#SBATCH --job-name=rnafm_pm100m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_pm100m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_pm100m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_pure_mamba_100m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[pm100m] Already complete, exiting."
    exit 0
fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" "$REPO/scripts/pretrain_pure_mamba_100m.sh"
echo "[pm100m] Queued follow-up (afterany:$SLURM_JOB_ID)"

# Symlink so train script writes to CKPT_DIR
mkdir -p "$CKPT_DIR"
if [ ! -L outputs/fm_pure_mamba_100m ]; then
    rm -rf outputs/fm_pure_mamba_100m
    ln -s "$CKPT_DIR" outputs/fm_pure_mamba_100m
fi

echo "[pm100m] Node : $(hostname)"
echo "[pm100m] GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[pm100m] Start: $(date)"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=29671 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_pure_mamba_100m \
    --epoch_dir  "$DATA_DIR/epoch_seqs_full" \
    --layer_pattern "MMMMMMMMMMMMMMMMMM" \
    --hidden_size 640 \
    --num_attention_heads 10 \
    --intermediate_size 2560 \
    --mamba_expand 4 \
    --mamba_d_state 16 \
    --micro_batch 42 \
    --grad_accum 8 \
    --epochs 6 \
    --save_steps 2000 --val_steps 2000 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

echo "[pm100m] Done: $(date)"
