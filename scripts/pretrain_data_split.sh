#!/bin/bash
# CPU-only job: split rnacentral_active.fasta into 20/40/60/80% stratified subsets.
#
#SBATCH --job-name=data_split
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/data_split_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/data_split_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

cd "$REPO"
echo "[split] Start: $(date)"
python3 scripts/split_data_by_category.py
echo "[split] Done:  $(date)"
