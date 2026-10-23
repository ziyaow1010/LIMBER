#!/bin/bash
# Zero-shot nRC sweep for 650M epoch 1-11 (skips epochs with existing results).
# Runs sequentially on 1 GPU; ~5 min/epoch → ~50 min total.
#SBATCH --job-name=zs_650m_sweep
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/zs_650m_sweep_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/zs_650m_sweep_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "[zs_sweep] Start: $(date)"

for ep in $(seq 1 11); do
    OUT="outputs/nrc_zeroshot/650m_ep${ep}"
    if [ -f "$OUT/results.json" ]; then
        echo "[zs_sweep] ep${ep}: already done, skipping"
        continue
    fi
    CKPT="$CKPT_DIR/epoch_checkpoints/epoch${ep}"
    if [ ! -d "$CKPT" ]; then
        echo "[zs_sweep] ep${ep}: checkpoint not found, skipping"
        continue
    fi
    echo "[zs_sweep] ep${ep}: running..."
    python scripts/nrc_zeroshot.py \
        --ckpt_dir "$CKPT" \
        --out_dir  "$OUT" \
        --batch_size 128
    echo "[zs_sweep] ep${ep}: done"
done

echo ""
echo "[zs_sweep] === SUMMARY 650M epoch sweep ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

base = Path("outputs/nrc_zeroshot")
print(f"\n{'epoch':<8}{'knn_k1':>10}{'knn_k5':>10}{'knn_k10':>10}")
print("-" * 40)
for ep in range(1, 12):
    p = base / f"650m_ep{ep}" / "results.json"
    if p.exists():
        d = json.load(open(p))
        print(f"  ep{ep:<6}{d['knn_k1']:>10.4f}{d['knn_k5']:>10.4f}{d['knn_k10']:>10.4f}")
    else:
        print(f"  ep{ep:<6}{'—':>10}{'—':>10}{'—':>10}")

# Also print size scaling
print(f"\n{'model':<10}{'knn_k1':>10}{'knn_k5':>10}{'knn_k10':>10}")
print("-" * 42)
for tag, path in [("100M ep6","100m_ep6"), ("300M ep6","300m_ep6"), ("650M ep6","650m_ep6")]:
    p = base / path / "results.json"
    if p.exists():
        d = json.load(open(p))
        print(f"  {tag:<10}{d['knn_k1']:>10.4f}{d['knn_k5']:>10.4f}{d['knn_k10']:>10.4f}")
PYEOF

echo ""
echo "[zs_sweep] Done: $(date)"
