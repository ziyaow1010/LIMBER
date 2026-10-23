#!/bin/bash
# Zero-shot nRC classification for all scaling-law models (100M, 300M, 650M).
# Runs sequentially on a single GPU; each model takes ~5 min.
#SBATCH --job-name=nrc_zs_all
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/nrc_zeroshot_scaling_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/nrc_zeroshot_scaling_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "[nrc_zs] Start: $(date)"

declare -A CKPTS=(
    ["100m"]="/beacon-projects/rnallm/checkpoints/fm_hybrid_100m/epoch_checkpoints/epoch6"
    ["300m"]="/beacon-projects/rnallm/checkpoints/fm_hybrid_300m/epoch_checkpoints/epoch6"
    ["650m"]="/beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch6"
)

for model in 100m 300m 650m; do
    ckpt="${CKPTS[$model]}"
    out="outputs/nrc_zeroshot/${model}_ep6"
    echo ""
    echo "[nrc_zs] === $model ==="
    python scripts/nrc_zeroshot.py \
        --ckpt_dir "$ckpt" \
        --out_dir  "$out" \
        --batch_size 128
done

echo ""
echo "[nrc_zs] === SUMMARY ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

models = ["100m", "300m", "650m"]
results = {}
for m in models:
    p = Path(f"outputs/nrc_zeroshot/{m}_ep6/results.json")
    if p.exists():
        results[m] = json.load(open(p))

if results:
    # header
    first = next(iter(results.values()))
    metrics = [k for k in first if "knn" in k or "kmeans" in k]
    print(f"\n{'metric':<20}", end="")
    for m in models:
        if m in results: print(f"{m:>10}", end="")
    print()
    print("-" * (20 + 10*len(results)))
    for k in metrics:
        print(f"  {k:<18}", end="")
        for m in models:
            if m in results:
                v = results[m].get(k, float("nan"))
                print(f"{v:>10.4f}" if isinstance(v, float) else f"{v:>10}", end="")
        print()
PYEOF

echo ""
echo "[nrc_zs] Done: $(date)"
