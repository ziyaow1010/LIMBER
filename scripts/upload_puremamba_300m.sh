#!/bin/bash
#SBATCH --job-name=hf_upload_puremamba300m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/hf_upload_puremamba300m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/hf_upload_puremamba300m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

python scripts/upload_to_hf.py \
    --token "$HF_TOKEN" \
    --username Ziyao1010 \
    --only kmer1-PureMamba-300M
