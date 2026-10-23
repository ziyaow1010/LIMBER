#!/bin/bash
# Hybrid-650M pretraining, 4-GPU single node. RiNALMo-aligned: single-nuc
# tokenizer, ctx 1022, MLM 15%, eff-batch 1344 (regardless of GPU type --
# micro_batch x grad_accum is tuned per-GPU to hit the same effective batch).
# Clustering step (HANDOFF §3b) recommended first: run cluster_rnacentral.sh.
#
# Usage: sbatch scripts/pretrain_h200.sh                                       # H200 x4 (default)
#        sbatch --gres=gpu:nvidia_l40s:4 --export=ALL,BENCH_FILE=l40s_bench.json scripts/pretrain_h200.sh   # L40S x4
#        EPOCHS=10 sbatch --export=ALL,EPOCHS=10 scripts/pretrain_h200.sh      # extend a finished run
# Resumes automatically if outputs/fm_hybrid_650m/checkpoint-* exists (picks
# up at the right per-epoch cluster-sample file, not from epoch1 again --
# see start_epoch in train_hybrid650_mlm.py). epoch_dir must already contain
# enough epoch{N}.txt files for the requested --epochs (sample_cluster_epochs.sh).
#SBATCH --job-name=rnafm_pretrain
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_h200_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/pretrain_h200_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA_DIR=/beacon-projects/rnallm/data/rnacentral
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
mkdir -p logs "$CKPT_DIR"

if [ -f "$CKPT_DIR/TRAINING_COMPLETE" ]; then
    echo "[pretrain] $CKPT_DIR/TRAINING_COMPLETE already exists -- nothing to do."
    cat "$CKPT_DIR/TRAINING_COMPLETE"
    exit 0
fi

# Self-chain a follow-up job now, before training starts: multi-day full-data
# runs span multiple SLURM time-limit windows, and `afterany` fires whether
# this job finishes cleanly, fails, or gets killed at the time limit -- the
# only way to guarantee a follow-up gets queued no matter how this job ends.
# The follow-up checks TRAINING_COMPLETE itself and is a no-op if we're done.
sbatch --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,EPOCHS="${EPOCHS:-6}",EPOCH_DIR_NAME="${EPOCH_DIR_NAME:-epoch_seqs}",BENCH_FILE="${BENCH_FILE:-h200_bench.json}" \
    "$REPO/scripts/pretrain_h200.sh"
echo "[pretrain] Queued follow-up job (afterany:$SLURM_JOB_ID) in case this one hits the time limit"

# Symlink checkpoint dir so train script writes there (avoids home dir space issue)
if [ ! -L outputs/fm_hybrid_650m ] && [ ! -d outputs/fm_hybrid_650m/.git ]; then
    rm -rf outputs/fm_hybrid_650m
    ln -s "$CKPT_DIR" outputs/fm_hybrid_650m
    echo "[pretrain] Symlinked outputs/fm_hybrid_650m -> $CKPT_DIR"
fi

# Per-epoch cluster-resampled training data (run sample_cluster_epochs.sh first)
EPOCH_DIR=$DATA_DIR/${EPOCH_DIR_NAME:-epoch_seqs}
if [ ! -f "$EPOCH_DIR/meta.json" ]; then
    echo "[pretrain] ERROR: $EPOCH_DIR/meta.json not found — run sample_cluster_epochs.sh first" >&2
    exit 1
fi
echo "[pretrain] Using per-epoch cluster-resampled data: $EPOCH_DIR"

echo "[pretrain] Node: $(hostname)"
echo "[pretrain] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi -L | wc -l)"
echo "[pretrain] Start: $(date)"

BENCH_PATH="$CKPT_DIR/${BENCH_FILE:-h200_bench.json}"
# Step 1: Throughput benchmark to pick optimal micro_batch
if [ ! -f "$BENCH_PATH" ]; then
    echo "[pretrain] Running throughput benchmark..."
    python scripts/bench_throughput.py
fi

# Read recommended micro_batch from bench results
MICRO_BATCH=$(python3 -c "
import json
r = json.load(open('$BENCH_PATH'))
print(r['best_micro_batch'])
" 2>/dev/null || echo 32)
GRAD_ACCUM=$(python3 -c "
import json
r = json.load(open('$BENCH_PATH'))
print(r['best_grad_accum'])
" 2>/dev/null || echo 5)

echo "[pretrain] micro_batch=$MICRO_BATCH  grad_accum=$GRAD_ACCUM  n_gpu=4"
echo "[pretrain] Effective batch = $((MICRO_BATCH * 4 * GRAD_ACCUM))"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --master_port=29651 \
    scripts/train_hybrid650_mlm.py \
    --output_dir outputs/fm_hybrid_650m \
    --epoch_dir "$EPOCH_DIR" \
    --micro_batch "$MICRO_BATCH" \
    --grad_accum "$GRAD_ACCUM" \
    --epochs "${EPOCHS:-6}" \
    --save_steps "${SAVE_STEPS:-2000}" --val_steps "${VAL_STEPS:-2000}" \
    --lr 5e-5 --min_lr 1e-5 --warmup_steps 2000 \
    --max_len 1022 --num_workers 2

echo "[pretrain] Done: $(date)"
