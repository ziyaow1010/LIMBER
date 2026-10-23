#!/bin/bash
# Zero-shot Jacobian layer ablation for hybrid-650M ep12, restricted to a layer range.
# Usage: sbatch --export=ALL,START=17,END=19 scripts/run_jacobian_650m_ep12_range.sh
# Parallel-safe: layers are independent, per-dataset CSVs are reused, summary json is locked.
#
#SBATCH --job-name=jac650_rng
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/jac_650m_ep12_rng_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/jac_650m_ep12_rng_%j.err

set -euo pipefail
: "${START:?set START}" "${END:?set END}"
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch12
OUT=$REPO/outputs/jacobian_eval/hybrid_650m_ep12

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
echo "[jac650_rng] Node: $(hostname)  layers $START-$END  Start: $(date)"
export TOKENIZERS_PARALLELISM=false

python -u scripts/jacobian_ss_eval.py \
    --ckpt "$CKPT" --out "$OUT" --mode ablation \
    --abl_datasets PDB RNP SRP tmRNA tRNAdb RFAM CRW --abl_max 1000 \
    --layer_start "$START" --layer_end "$END"

echo "[jac650_rng] Done: $(date)"
