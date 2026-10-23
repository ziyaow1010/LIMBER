#!/bin/bash
# nRC 13-class ncRNA classification, two-phase fine-tuning.
# Usage: sbatch --job-name=nrc_<run> --export=ALL,RUN=<name>[,CKPT=...|MODEL_TYPE=...][,NLAYERS=N] nrc_ft.sbatch
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/nrc/%x_%j.out
set -uo pipefail
: "${RUN:?}"
REPO=/beacon-homes/ziyaow/RNA_FM
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ARGS="--out_dir /beacon-projects/rnallm/outputs/nrc_finetune/$RUN"
[ -n "${CKPT:-}" ]       && ARGS="$ARGS --ckpt_dir $CKPT"
[ -n "${MODEL_TYPE:-}" ] && ARGS="$ARGS --model_type $MODEL_TYPE"
[ -n "${NLAYERS:-}" ]    && ARGS="$ARGS --n_layers $NLAYERS"
echo "[nrc] run=$RUN node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date)"
echo "[nrc] args: $ARGS"
python scripts/nrc_finetune.py $ARGS ${EXTRA:-}
echo "[nrc] exit=$? done=$(date)"
