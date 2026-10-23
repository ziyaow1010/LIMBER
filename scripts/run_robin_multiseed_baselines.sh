#!/bin/bash
# Per-molecule-split (all 27 RNA targets kept, compound-level 70/15/15 split, 10 seeds)
# comparison across frozen RNA encoders. Reuses molecule/RNA embeddings already
# extracted in data/robin/emb by run_robin_loto_baselines.sh.
#SBATCH --job-name=robin_ms_bl
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/robin_ms_bl_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/robin_ms_bl_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
export TOKENIZERS_PARALLELISM=false
R=data/robin; mkdir -p $R/multiseed_multi

if [ -z "$(ls -A $R/emb 2>/dev/null)" ]; then
    python -u scripts/robin_extract_rna_embeddings_multi.py
fi
cp -n $R/rna_target_embeddings.npy $R/emb/fm_hybrid_650m_cls_orig.npy

for emb in $R/emb/*.npy; do
  n=$(basename $emb .npy)
  out=$R/multiseed_multi/${n}.json
  [ -f "$out" ] && continue
  echo "[multiseed] $n"
  python -u scripts/robin_train_multiseed.py --rna_emb $emb --out $out 2>&1 | tail -3
done
python scripts/robin_aggregate_multiseed.py
