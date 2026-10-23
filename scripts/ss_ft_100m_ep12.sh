#!/bin/bash
# SS fine-tune LFO for Hybrid-100M epoch12 checkpoint.
# Same settings as ep6 (unfreeze_min_layer 3, lr 5e-4, 30 epochs).
#
#SBATCH --job-name=ss_ft_100m12
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_100m_ep12_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_100m_ep12_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_100m
EPOCH_CKPT="$CKPT_DIR/epoch_checkpoints/epoch12"

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

if [ ! -d "$EPOCH_CKPT" ]; then
    echo "[ss_ft_100m_ep12] ERROR: checkpoint not found: $EPOCH_CKPT" >&2
    exit 1
fi
echo "[ss_ft_100m_ep12] Checkpoint: $EPOCH_CKPT"

CT_ROOT="data/contact_eval/raw/ct/fam-fold"
if [ ! -d "$CT_ROOT/5s/train" ]; then
    python scripts/make_ct_splits.py
fi

OUT="outputs/fm_hybrid_100m/ss_eval/epoch12"
mkdir -p "$OUT" logs

echo "[ss_ft_100m_ep12] Start: $(date)"

N_GPU=4
FAMILIES="5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA"
i=0
for fam in $FAMILIES; do
    gpu=$(( i % N_GPU ))
    LOG="logs/ss_ft_100m_ep12_${fam}.log"
    ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py \
        --family "$fam" \
        --model_dir "$EPOCH_CKPT" \
        --out_dir "$OUT" \
        --epochs 30 \
        --base_lr 5e-4 \
        --unfreeze_min_layer 3 \
        > "$LOG" 2>&1
      echo "[ss_ft_100m_ep12] $fam done (gpu $gpu)" ) &
    i=$(( i + 1 ))
    if (( i % N_GPU == 0 )); then
        wait
        echo "[ss_ft_100m_ep12] Wave $((i/N_GPU)) done"
    fi
done
wait

echo "[ss_ft_100m_ep12] All families done: $(date)"
echo "[ss_ft_100m_ep12] Aggregating..."

python3 - <<'PYEOF'
import json, statistics
from pathlib import Path

out = Path("outputs/fm_hybrid_100m/ss_eval/epoch12")
ep6         = {"5s":0.891,"16s":0.726,"23s":0.850,"grp1":0.422,
               "srp":0.690,"telomerase":0.111,"RNaseP":0.800,"tmRNA":0.832,"tRNA":0.906}
rinalmo     = {"5s":0.88,"16s":0.74,"23s":0.85,"grp1":0.66,
               "srp":0.70,"telomerase":0.12,"RNaseP":0.80,"tmRNA":0.80,"tRNA":0.93}

rows = []
print(f"\n{'family':<12}{'ep12':>8}{'ep6':>8}{'RiNALMo':>10}")
print("-" * 42)
for jf in sorted(out.glob("*.json")):
    d = json.load(open(jf))
    fam = jf.stem
    f1  = d.get("mean_F1", 0.0)
    rows.append(f1)
    print(f"  {fam:<12}{f1:>8.3f}{ep6.get(fam,float('nan')):>8.3f}"
          f"{rinalmo.get(fam,float('nan')):>10.3f}")

if rows:
    macro = statistics.mean(rows)
    print("-" * 42)
    print(f"  {'MACRO':<12}{macro:>8.3f}{statistics.mean(ep6.values()):>8.3f}"
          f"{statistics.mean(rinalmo.values()):>10.3f}")
    json.dump({"model":"hybrid_100m","epoch":12,"macro_f1":macro,
               "ep6_macro":statistics.mean(ep6.values()),
               "rinalmo_macro":statistics.mean(rinalmo.values())},
              open(out/"summary.json","w"), indent=2)
    print(f"\n[ss_ft_100m_ep12] Summary saved.")
PYEOF
