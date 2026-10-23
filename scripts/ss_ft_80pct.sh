#!/bin/bash
# SS fine-tune LFO for hybrid_100m_80pct epoch6 checkpoint.
# Same arch as hybrid_100m (18L TTM×6, hidden=640). unfreeze_min_layer=3.
#
#SBATCH --job-name=ssft_80pct
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_80pct_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_80pct_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT_DIR=/beacon-homes/ziyaow/RNA_FM/outputs/hybrid_100m_80pct

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

EPOCH=${EPOCH:-6}
EPOCH_CKPT="$CKPT_DIR/epoch_checkpoints/epoch${EPOCH}"

if [ ! -d "$EPOCH_CKPT" ]; then
    echo "[ssft_80pct] ERROR: checkpoint not found: $EPOCH_CKPT" >&2
    exit 1
fi
echo "[ssft_80pct] Checkpoint: $EPOCH_CKPT"

CT_ROOT="data/contact_eval/raw/ct/fam-fold"
if [ ! -d "$CT_ROOT/5s/train" ]; then
    python scripts/make_ct_splits.py
fi

OUT="outputs/hybrid_100m_80pct/ss_eval/epoch${EPOCH}"
mkdir -p "$OUT" logs

echo "[ssft_80pct] Start: $(date)"

N_GPU=4
FAMILIES="5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA"
i=0
for fam in $FAMILIES; do
    gpu=$(( i % N_GPU ))
    LOG="logs/ss_ft_80pct_${fam}.log"
    ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py \
        --family "$fam" \
        --model_dir "$EPOCH_CKPT" \
        --out_dir "$OUT" \
        --epochs 30 \
        --base_lr 5e-4 \
        --unfreeze_min_layer 3 \
        > "$LOG" 2>&1
      echo "[ssft_80pct] $fam done (gpu $gpu)" ) &
    i=$(( i + 1 ))
    if (( i % N_GPU == 0 )); then
        wait
        echo "[ssft_80pct] Wave $((i/N_GPU)) done"
    fi
done
wait

echo "[ssft_80pct] All families done: $(date)"

python3 - <<'PYEOF'
import json, statistics
from pathlib import Path

EPOCH = 6
out = Path(f"outputs/hybrid_100m_80pct/ss_eval/epoch{EPOCH}")
rinalmo = {"5s":0.88,"16s":0.74,"23s":0.85,"grp1":0.66,"srp":0.70,
           "telomerase":0.12,"RNaseP":0.80,"tmRNA":0.80,"tRNA":0.93}

rows = []
print(f"\n{'family':<12}{'80pct':>10}{'RiNALMo':>10}")
print("-" * 34)
for jf in sorted(out.glob("*.json")):
    d = json.load(open(jf))
    fam = jf.stem
    f1 = d.get("mean_F1", 0.0)
    rows.append(f1)
    print(f"  {fam:<12}{f1:>10.3f}{rinalmo.get(fam,float('nan')):>10.3f}")

if rows:
    macro = statistics.mean(rows)
    print("-" * 34)
    print(f"  {'MACRO':<12}{macro:>10.3f}{statistics.mean(rinalmo.values()):>10.3f}")
    json.dump({"model":"hybrid_100m_80pct","epoch":EPOCH,"macro_f1":macro},
              open(out/"summary.json","w"), indent=2)
    print(f"[ssft_80pct] Summary: {out}/summary.json")
PYEOF
