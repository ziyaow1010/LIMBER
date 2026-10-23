#!/bin/bash
# Prepare full (unclustered, exact-deduped) RNAcentral for pretraining from
# scratch -- no mmseqs clustering, no per-epoch resampling, same data reused
# every epoch (matches the original full-data design).
# Input:  rnacentral_active.dedup.fasta (exact-dedup only, from cluster_rnacentral.sh)
# Output: /beacon-projects/rnallm/data/rnacentral/epoch_seqs_full/{epoch1..6,val}.txt + meta.json
#SBATCH --job-name=prep_full
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/prep_full_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/prep_full_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA=/beacon-projects/rnallm/data/rnacentral
FASTA=$DATA/rnacentral_active.dedup.fasta
OUT_DIR=$DATA/epoch_seqs_full

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

echo "[prep] Start: $(date)"
python "$REPO/scripts/prep_full_data_epochs.py" "$FASTA" "$OUT_DIR" \
    --epochs 6 --val_mod 2048
echo "[prep] Done: $(date)"
ls -la "$OUT_DIR"
