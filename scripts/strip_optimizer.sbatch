#!/bin/bash
#SBATCH --job-name=strip_opt
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/strip_opt_%j.out
set -uo pipefail
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate /beacon-projects/rnallm/software/miniconda3/envs/zfold
export PYTHONUNBUFFERED=1
python -I /beacon-homes/ziyaow/RNA_FM/scripts/strip_optimizer.py
echo "[strip] exit=$? done=$(date)"
df -h /beacon-projects/rnallm | tail -1
