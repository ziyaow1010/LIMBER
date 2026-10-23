#!/bin/bash
# Pre-sample one random member sequence per mmseqs cluster, independently for
# each pretraining epoch (RiNALMo-style dynamic cluster sampling, instead of
# reusing the fixed mmseqs representative every epoch). Deterministic in the
# epoch index, so re-running with a higher --epochs reproduces epoch1..N
# identically and just adds the new epochs on top (used to extend a finished
# 6-epoch run to 10 epochs without invalidating the original epoch1-6 data).
# Input:  mmseqs_clust_all_seqs.fasta (from cluster_rnacentral.sh, job 81311)
# Output: /beacon-projects/rnallm/data/rnacentral/epoch_seqs/{epoch1..N,val}.txt + meta.json
#SBATCH --job-name=sample_epochs
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/sample_epochs_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/sample_epochs_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA=/beacon-projects/rnallm/data/rnacentral
ALL_SEQS=$DATA/mmseqs_clust_all_seqs.fasta
OUT_DIR=$DATA/epoch_seqs

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

echo "[sample] Start: $(date)"
python "$REPO/scripts/sample_cluster_epochs.py" "$ALL_SEQS" "$OUT_DIR" \
    --epochs "${SAMPLE_EPOCHS:-6}" --val_mod 2048 --seed 42
echo "[sample] Done: $(date)"
ls -la "$OUT_DIR"
