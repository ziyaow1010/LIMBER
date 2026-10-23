#!/bin/bash
# SS fine-tune LFO for Pure-Mamba-100M epoch6 checkpoint.
# 18 all-M layers; same unfreeze ratio as Hybrid-100M (3/18).
#
#SBATCH --job-name=ss_ft_pm100m
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_pm100m_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_pm100m_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
EPOCH_CKPT=/beacon-projects/rnallm/checkpoints/fm_pure_mamba_100m/epoch_checkpoints/epoch6

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

CT_ROOT="data/contact_eval/raw/ct/fam-fold"
if [ ! -d "$CT_ROOT/5s/train" ]; then python scripts/make_ct_splits.py; fi

OUT="outputs/fm_pure_mamba_100m/ss_eval/epoch6"
mkdir -p "$OUT" logs
echo "[ss_ft_pm100m] Start: $(date)"

N_GPU=4
FAMILIES="5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA"
i=0
for fam in $FAMILIES; do
    gpu=$(( i % N_GPU ))
    ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py \
        --family "$fam" --model_dir "$EPOCH_CKPT" --out_dir "$OUT" \
        --epochs 30 --base_lr 5e-4 --unfreeze_min_layer 3 \
        > "logs/ss_ft_pm100m_${fam}.log" 2>&1
      echo "[ss_ft_pm100m] $fam done" ) &
    i=$(( i + 1 ))
    if (( i % N_GPU == 0 )); then wait; fi
done
wait

echo "[ss_ft_pm100m] All done: $(date)"
python3 - <<'PYEOF'
import json, statistics
from pathlib import Path

out = Path("outputs/fm_pure_mamba_100m/ss_eval/epoch6")
h100m   = {"5s":0.891,"16s":0.726,"23s":0.850,"grp1":0.422,"srp":0.690,
            "telomerase":0.111,"RNaseP":0.800,"tmRNA":0.832,"tRNA":0.906}
rinalmo = {"5s":0.88,"16s":0.74,"23s":0.85,"grp1":0.66,"srp":0.70,
            "telomerase":0.12,"RNaseP":0.80,"tmRNA":0.80,"tRNA":0.93}

rows = []
print(f"\n{'family':<12}{'PureMamba':>10}{'Hybrid100M':>12}{'RiNALMo':>10}")
print("-" * 46)
for jf in sorted(out.glob("*.json")):
    d = json.load(open(jf)); fam = jf.stem; f1 = d.get("mean_F1", 0.0)
    rows.append(f1)
    print(f"  {fam:<12}{f1:>10.3f}{h100m.get(fam,float('nan')):>12.3f}{rinalmo.get(fam,float('nan')):>10.3f}")
if rows:
    macro = statistics.mean(rows)
    print("-" * 46)
    print(f"  {'MACRO':<12}{macro:>10.3f}{statistics.mean(h100m.values()):>12.3f}{statistics.mean(rinalmo.values()):>10.3f}")
    json.dump({"model":"pure_mamba_100m","epoch":6,"macro_f1":macro}, open(out/"summary.json","w"), indent=2)
PYEOF
