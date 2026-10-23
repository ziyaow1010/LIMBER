#!/bin/bash
# SS fine-tune LFO for Hybrid-100M epoch6 checkpoint.
# 9 families, 4 L40S GPUs, RiNALMo-aligned pipeline.
# --unfreeze_min_layer 3: freeze bottom 3/18 layers (proportional to 9/33 in 650M).
#
# Usage: sbatch scripts/ss_ft_100m.sh
#SBATCH --job-name=ss_ft_100m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_100m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_100m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_100m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

EPOCH=${EPOCH:-6}
EPOCH_CKPT="$CKPT_DIR/epoch_checkpoints/epoch${EPOCH}"

if [ ! -d "$EPOCH_CKPT" ]; then
    echo "[ss_ft_100m] ERROR: checkpoint not found: $EPOCH_CKPT" >&2
    exit 1
fi
echo "[ss_ft_100m] Checkpoint: $EPOCH_CKPT"

CT_ROOT="data/contact_eval/raw/ct/fam-fold"
if [ ! -d "$CT_ROOT/5s/train" ]; then
    python scripts/make_ct_splits.py
fi

OUT="outputs/fm_hybrid_100m/ss_eval/epoch${EPOCH}"
mkdir -p "$OUT" logs

echo "[ss_ft_100m] Start: $(date)"

N_GPU=4
FAMILIES="5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA"
i=0
for fam in $FAMILIES; do
    gpu=$(( i % N_GPU ))
    LOG="logs/ss_ft_100m_${fam}.log"
    ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py \
        --family "$fam" \
        --model_dir "$EPOCH_CKPT" \
        --out_dir "$OUT" \
        --epochs 30 \
        --base_lr 5e-4 \
        --unfreeze_min_layer 3 \
        > "$LOG" 2>&1
      echo "[ss_ft_100m] $fam done (gpu $gpu)" ) &
    i=$(( i + 1 ))
    if (( i % N_GPU == 0 )); then
        wait
        echo "[ss_ft_100m] Wave $((i/N_GPU)) done"
    fi
done
wait

echo "[ss_ft_100m] All families done: $(date)"
echo "[ss_ft_100m] Aggregating..."

python3 - <<'PYEOF'
import json, statistics
from pathlib import Path

out = Path("outputs/fm_hybrid_100m/ss_eval/epoch6")
rinalmo     = {"5s":0.88,"16s":0.74,"23s":0.85,"grp1":0.66,
               "srp":0.70,"telomerase":0.12,"RNaseP":0.80,"tmRNA":0.80,"tRNA":0.93}
m300        = {"5s":0.874,"16s":0.680,"23s":0.738,"grp1":0.324,
               "srp":0.620,"telomerase":0.109,"RNaseP":0.766,"tmRNA":0.820,"tRNA":0.862}
m650_ep6    = {"5s":0.884,"16s":0.759,"23s":0.879,"grp1":0.416,
               "srp":0.690,"telomerase":0.101,"RNaseP":0.797,"tmRNA":0.844,"tRNA":0.933}

rows = []
print(f"\n{'family':<12}{'100M':>8}{'300M':>8}{'650M_ep6':>10}{'RiNALMo':>10}")
print("-" * 52)
for jf in sorted(out.glob("*.json")):
    d = json.load(open(jf))
    fam = jf.stem
    f1  = d.get("mean_F1", 0.0)
    rows.append(f1)
    print(f"  {fam:<12}{f1:>8.3f}{m300.get(fam,float('nan')):>8.3f}"
          f"{m650_ep6.get(fam,float('nan')):>10.3f}{rinalmo.get(fam,float('nan')):>10.3f}")

if rows:
    macro = statistics.mean(rows)
    print("-" * 52)
    print(f"  {'MACRO':<12}{macro:>8.3f}{statistics.mean(m300.values()):>8.3f}"
          f"{statistics.mean(m650_ep6.values()):>10.3f}{statistics.mean(rinalmo.values()):>10.3f}")
    summary = {"model":"hybrid_100m","epoch":6,"macro_f1":macro,
               "rinalmo_macro":statistics.mean(rinalmo.values())}
    json.dump(summary, open(out/"summary.json","w"), indent=2)
    print(f"\n[ss_ft_100m] Summary saved: {out}/summary.json")
PYEOF
