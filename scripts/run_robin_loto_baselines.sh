#!/bin/bash
# LOTO (hold out 1 of 27 RNA targets, train on 26) with different frozen RNA encoders,
# same Uni-Mol molecule embeddings and same MLP head. 3 seeds each.
#SBATCH --job-name=robin_loto_bl
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/robin_loto_bl_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/robin_loto_bl_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
export TOKENIZERS_PARALLELISM=false
R=data/robin; mkdir -p $R/loto_multi $R/emb

python -u scripts/robin_extract_rna_embeddings_multi.py ${MODELS:+--models $MODELS}

# original LIMBER-650M CLS embedding (regression check vs loto_results.json: AUROC 0.747)
cp -n $R/rna_target_embeddings.npy $R/emb/fm_hybrid_650m_cls_orig.npy
for emb in $R/emb/*.npy; do
  n=$(basename $emb .npy)
  for seed in 0 1 2; do
    out=$R/loto_multi/${n}_seed${seed}.json
    [ -f "$out" ] && continue
    echo "[loto] $n seed=$seed"
    python -u scripts/robin_train_loto.py --rna_emb $emb --out $out --seed $seed 2>&1 | grep -E "FINAL"
  done
done
python scripts/robin_aggregate_loto.py
