#!/bin/bash
# Hybrid-100M pretraining on 60% of data (stratified by RNA category).
# Architecture identical to baseline hybrid_100m (18L TTM×6, hidden=640).
# Effective batch = 42*4*8 = 1344.
#
#SBATCH --job-name=h100m_60pct
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_100m_60pct_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_100m_60pct_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/hybrid_100m_60pct

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[h100m-60pct] Already complete."
    exit 0
fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    "$REPO/scripts/pretrain_hybrid_100m_60pct.sh"
echo "[h100m-60pct] Queued follow-up (afterany:$SLURM_JOB_ID)"

EPOCH_DIR=$DATA_DIR/epoch_seqs_60pct
MICRO_BATCH=42
GRAD_ACCUM=8
echo "[h100m-60pct] Node: $(hostname)"
echo "[h100m-60pct] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[h100m-60pct] Start: $(date)"
echo "[h100m-60pct] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=4  eff_batch=$((MICRO_BATCH*4*GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --master_port=29662 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/hybrid_100m_60pct \
    --epoch_dir "$EPOCH_DIR" \
    --layer_pattern "TTMTTMTTMTTMTTMTTM" \
    --hidden_size 640 \
    --num_attention_heads 10 \
    --intermediate_size 2560 \
    --mamba_expand 4 \
    --micro_batch "$MICRO_BATCH" \
    --grad_accum "$GRAD_ACCUM" \
    --epochs 6 \
    --save_steps 1000 --val_steps 1000 \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

echo "[h100m-60pct] Done: $(date)"
