#!/bin/bash
# SS fine-tune StructRFM (86M, heqin-zhu/structRFM), LFO + within-family, per-seq F1.
#
#SBATCH --job-name=ss_perseq_structrfm
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_perseq_structrfm_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ss_perseq_structrfm_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

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
        ( CUDA_VISIBLE_DEVICES=$gpu python scripts/ss_perseq_external.py \
            --model_type structrfm_86m \
            --family "$fam" --out_dir "$OUT" \
            --epochs 30 --base_lr 5e-4 --unfreeze_min_layer 3 \
            $EXTRA \
            > "logs/ss_perseq_structrfm_${fam}_$(basename $OUT).log" 2>&1
          echo "[structrfm] $fam $(basename $OUT) done" ) &
        i=$(( i + 1 ))
        if (( i % N_GPU == 0 )); then wait; fi
    done
    wait
}

echo "[structrfm] === LFO ==="
run_families "outputs/structrfm_86m/ss_eval_perseq/lfo"

echo "[structrfm] === Within-family ==="
run_families "outputs/structrfm_86m/ss_eval_perseq/within" "--within_family"

echo "[structrfm] All done: $(date)"
python3 - <<'PYEOF'
import json, statistics
from pathlib import Path
for tag, d in [("LFO", "lfo"), ("Within", "within")]:
    out = Path(f"outputs/structrfm_86m/ss_eval_perseq/{d}")
    rows = []
    for fam in ["5s","16s","23s","grp1","srp","telomerase","RNaseP","tmRNA","tRNA"]:
        jf = out / f"{fam}.json"
        if jf.exists():
            rows.append(json.load(open(jf))["mean_F1"])
    if rows:
        print(f"  StructRFM {tag} macro F1: {statistics.mean(rows):.3f}")
PYEOF
