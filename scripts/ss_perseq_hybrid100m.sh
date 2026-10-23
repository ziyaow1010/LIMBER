#!/bin/bash
# SS fine-tune Hybrid-100M ep12, LFO + within-family, per-seq F1 for violin plots.
#
#SBATCH --job-name=ss_perseq_h100m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_perseq_h100m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ss_perseq_h100m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
EPOCH_CKPT=$REPO/outputs/fm_hybrid_100m/epoch_checkpoints/epoch12

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

CT_ROOT="data/contact_eval/raw/ct/fam-fold"
if [ ! -d "$CT_ROOT/5s/train" ]; then python scripts/make_ct_splits.py; fi

N_GPU=4
FAMILIES="5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA"

run_families() {
    local OUT=$1; local EXTRA=${2:-}
    mkdir -p "$OUT" logs
    local i=0
    for fam in $FAMILIES; do
        local gpu=$(( i % N_GPU ))
        ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py \
            --family "$fam" --model_dir "$EPOCH_CKPT" --out_dir "$OUT" \
            --epochs 30 --base_lr 5e-4 --unfreeze_min_layer 3 \
            $EXTRA \
            > "logs/ss_perseq_h100m_${fam}_$(basename $OUT).log" 2>&1
          echo "[h100m] $fam $(basename $OUT) done" ) &
        i=$(( i + 1 ))
        if (( i % N_GPU == 0 )); then wait; fi
    done
    wait
}

echo "[h100m] === LFO ==="
run_families "outputs/fm_hybrid_100m/ss_eval_perseq/epoch12_lfo"

echo "[h100m] === Within-family ==="
run_families "outputs/fm_hybrid_100m/ss_eval_perseq/epoch12_within" "--within_family"

echo "[h100m] All done: $(date)"
python3 - <<'PYEOF'
import json, statistics
from pathlib import Path
for tag, d in [("LFO", "epoch12_lfo"), ("Within", "epoch12_within")]:
    out = Path(f"outputs/fm_hybrid_100m/ss_eval_perseq/{d}")
    rows = []
    for fam in ["5s","16s","23s","grp1","srp","telomerase","RNaseP","tmRNA","tRNA"]:
        jf = out / f"{fam}.json"
        if jf.exists():
            rows.append(json.load(open(jf))["mean_F1"])
    if rows:
        print(f"  Hybrid-100M {tag} macro F1: {statistics.mean(rows):.3f}")
PYEOF
