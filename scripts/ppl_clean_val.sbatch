#!/bin/bash
#SBATCH --job-name=pplclean
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/valcheck/%x_%j.out
set -uo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
D=/beacon-projects/rnallm/data/rnacentral
C=/beacon-projects/rnallm/checkpoints
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
for P in 20 40 60 80; do
  CK=$C/hybrid_100m_${P}pct/epoch_checkpoints/epoch6
  [ -d "$CK" ] || CK=$C/hybrid_100m_${P}pct
  echo "=== ${P}pct  ckpt=$CK"
  ls -d "$CK" >/dev/null 2>&1 || { echo "MISSING checkpoint for ${P}pct"; continue; }
  python scripts/ppl_clean_val.py --ckpt "$CK" \
      --train_file "$D/epoch_seqs_${P}pct/epoch1.txt" --tag "${P}pct"
done
echo "ALLDONE $(date)"
