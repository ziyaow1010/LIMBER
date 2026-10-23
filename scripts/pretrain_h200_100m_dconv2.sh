#!/bin/bash
# Hybrid-100M pretraining, Mamba local-conv ablation: mamba_d_conv=2 instead of
# the default 4 (smallest width the fused causal_conv1d CUDA kernel supports --
# width=1 errors out: "causal_conv1d only supports width between 2 and 4").
# Same data/optim/architecture as pretrain_h200_100m.sh otherwise, so it's a
# clean comparison against the existing d_conv=4 100M checkpoint.
#
# Usage: sbatch scripts/pretrain_h200_100m_dconv2.sh
#SBATCH --job-name=rnafm_100m_dconv2
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_100m_dconv2_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_100m_dconv2_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_100m_dconv2

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
mkdir -p logs "$CKPT_DIR"

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[pretrain-100m-dconv2] Already complete."
    cat "$CKPT_DIR/TRAINING_COMPLETE"
    exit 0
fi

# Self-chain follow-up job
sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,EPOCHS="${EPOCHS:-6}" \
    "$REPO/scripts/pretrain_h200_100m_dconv2.sh"
echo "[pretrain-100m-dconv2] Queued follow-up job (afterany:$SLURM_JOB_ID)"

# Symlink checkpoint dir
if [ ! -L outputs/fm_hybrid_100m_dconv2 ] && [ ! -d outputs/fm_hybrid_100m_dconv2/.git ]; then
    rm -rf outputs/fm_hybrid_100m_dconv2
    ln -s "$CKPT_DIR" outputs/fm_hybrid_100m_dconv2
    echo "[pretrain-100m-dconv2] Symlinked outputs/fm_hybrid_100m_dconv2 -> $CKPT_DIR"
fi

EPOCH_DIR=$DATA_DIR/epoch_seqs_full
if [ ! -f "$EPOCH_DIR/meta.json" ]; then
    echo "[pretrain-100m-dconv2] ERROR: $EPOCH_DIR/meta.json not found" >&2
    exit 1
fi
echo "[pretrain-100m-dconv2] Using per-epoch data: $EPOCH_DIR"

echo "[pretrain-100m-dconv2] Node: $(hostname)"
echo "[pretrain-100m-dconv2] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[pretrain-100m-dconv2] Start: $(date)"

MICRO_BATCH=${MICRO_BATCH:-42}
GRAD_ACCUM=${GRAD_ACCUM:-8}

echo "[pretrain-100m-dconv2] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=4"
echo "[pretrain-100m-dconv2] Effective batch = $((MICRO_BATCH * 4 * GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=29654 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_hybrid_100m_dconv2 \
    --epoch_dir "$EPOCH_DIR" \
    --layer_pattern "TTMTTMTTMTTMTTMTTM" \
    --hidden_size 640 \
    --num_attention_heads 10 \
    --intermediate_size 2560 \
    --mamba_expand 4 \
    --mamba_d_conv 2 \
    --micro_batch "$MICRO_BATCH" \
    --grad_accum "$GRAD_ACCUM" \
    --epochs "${EPOCHS:-6}" \
    --save_steps "${SAVE_STEPS:-2000}" --val_steps "${VAL_STEPS:-2000}" \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

echo "[pretrain-100m-dconv2] Done: $(date)"
