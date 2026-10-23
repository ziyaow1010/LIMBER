#!/bin/bash
#SBATCH --job-name=ss_ft_650m_ep5
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_650m_ep5_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ss_ft_650m_ep5_%j.err

set -uo pipefail
cd /beacon-homes/ziyaow/RNA_FM
export TOKENIZERS_PARALLELISM=false

echo "[ss_ft_650m] Node: $(hostname)"
echo "[ss_ft_650m] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "[ss_ft_650m] Start: $(date)"

source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm

CKPT=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch5
OUT=outputs/contact_pred/rinalmo_ss_ft_650m_ep5
FAMILIES="16s 23s grp1 telomerase srp tmRNA RNaseP 5s tRNA"

mkdir -p logs $OUT

i=0
for fam in $FAMILIES; do
    gpu=$(( i % 4 ))
    echo "[ss_ft_650m] Launching $fam on GPU $gpu"
    ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py \
        --family "$fam" \
        --model_dir "$CKPT" \
        --out_dir "$OUT" \
        --epochs 26 \
        > logs/ss_ft_650m_ep5_${fam}.log 2>&1 ) &
    i=$(( i + 1 ))
    # launch in batches of 4
    (( i % 4 == 0 )) && wait
done
wait

echo "[ss_ft_650m] All families done, aggregating..."

# aggregate and compare with 300M results
python - << 'PYEOF'
import json, statistics
from pathlib import Path

CP = Path("outputs/contact_pred")
FAMS = "5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA".split()

def load(sub, f):
    p = CP / sub / f"{f}.json"
    return json.load(open(p)) if p.exists() else None

old = {f: load("rinalmo_ss_ft", f) for f in FAMS}
new = {f: load("rinalmo_ss_ft_650m_ep5", f) for f in FAMS}

print(f"\n{'family':12}{'300M_F1':>10}{'650M_ep5':>11}{'Δ':>8}")
print("-" * 44)
old_vals, new_vals = [], []
for f in FAMS:
    o = old[f]["mean_F1"] if old[f] else None
    n = new[f]["mean_F1"] if new[f] else None
    delta = f"{n - o:+.4f}" if (o and n) else "  —"
    print(f"  {f:<12}{str(round(o,4)) if o else '—':>10}{str(round(n,4)) if n else '—':>11}{delta:>8}")
    if o: old_vals.append(o)
    if n: new_vals.append(n)

print("-" * 44)
print(f"  {'macro avg':<12}{statistics.mean(old_vals):>10.4f}{statistics.mean(new_vals):>11.4f}"
      f"{statistics.mean(new_vals)-statistics.mean(old_vals):>+8.4f}")

result = {
    "backbone": "hybrid-650M-epoch5",
    "macro_F1": statistics.mean(new_vals),
    "vs_300M_delta": statistics.mean(new_vals) - statistics.mean(old_vals),
    "per_family": {f: new[f] for f in FAMS if new[f]},
}
with open(CP / "compare_ss_ft_650m_ep5.json", "w") as fp:
    json.dump(result, fp, indent=2)
print(f"\nSaved to {CP}/compare_ss_ft_650m_ep5.json")
PYEOF

echo "[ss_ft_650m] Done: $(date)"
