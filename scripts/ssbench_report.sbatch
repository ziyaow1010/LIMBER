#!/bin/bash
#SBATCH --job-name=ssb_report
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_bench/report_%j.out
set -uo pipefail
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm
export PYTHONUNBUFFERED=1
python -I /beacon-homes/ziyaow/RNA_FM/scripts/ssbench_report.py
echo "[report] exit=$? done=$(date)"
ls -la /beacon-homes/ziyaow/RNA_FM/_ssbench_out/
