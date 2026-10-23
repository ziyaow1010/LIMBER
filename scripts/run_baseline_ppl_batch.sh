#!/bin/bash
# Full-model (no ablation) perplexity for: 100M/300M/650M final (epoch12)
# checkpoints, and the 100M data-scaling series (20/40/60/80pct, each at its
# own TRAINING_COMPLETE epoch). Cheap (baseline-only, no layer loop) -> one
# GPU, run serially.
#
#SBATCH --job-name=baseline_ppl_batch
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/baseline_ppl_batch_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/baseline_ppl_batch_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs outputs/baseline_ppl_final

run_one () {
    local tag=$1; local ckpt=$2
    echo "=== $tag ($ckpt) ==="
    python -u scripts/baseline_ppl_only.py \
        --ckpt "$ckpt" \
        --out  "outputs/baseline_ppl_final/$tag" \
        --n_val 4096 --batch 32
}

run_one "hybrid_100m_epoch12" "/beacon-projects/rnallm/checkpoints/fm_hybrid_100m/epoch_checkpoints/epoch12"
run_one "hybrid_300m_epoch12" "/beacon-projects/rnallm/checkpoints/fm_hybrid_300m/epoch_checkpoints/epoch12"
run_one "hybrid_650m_epoch12" "/beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch12"
run_one "hybrid_100m_20pct"   "outputs/hybrid_100m_20pct/epoch_checkpoints/epoch8"
run_one "hybrid_100m_40pct"   "outputs/hybrid_100m_40pct/epoch_checkpoints/epoch8"
run_one "hybrid_100m_60pct"   "outputs/hybrid_100m_60pct/epoch_checkpoints/epoch7"
run_one "hybrid_100m_80pct"   "outputs/hybrid_100m_80pct/epoch_checkpoints/epoch6"

echo "[baseline_ppl_batch] All done: $(date)"
python3 - <<'PYEOF'
import json
from pathlib import Path
root = Path("outputs/baseline_ppl_final")
for d in sorted(root.iterdir()):
    jf = d / "baseline_ppl.json"
    if jf.exists():
        r = json.load(open(jf))
        print(f"  {d.name:<22s} PPL={r['baseline_ppl']:.4f}  n_layers={r['n_layers']}")
PYEOF
