#!/bin/bash
# SS fine-tune LIMBER-650M ep12, LFO split, saves per-sequence F1 for violin plots.
#
#SBATCH --job-name=ss_650m_lfo
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_650m_lfo_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ss_650m_lfo_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
EPOCH_CKPT=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch12

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

CT_ROOT="data/contact_eval/raw/ct/fam-fold"
if [ ! -d "$CT_ROOT/5s/train" ]; then python scripts/make_ct_splits.py; fi

OUT="outputs/fm_hybrid_650m/ss_eval_perseq/epoch12_lfo"
mkdir -p "$OUT" logs
echo "[ss_650m_lfo] Start: $(date)"

N_GPU=4
FAMILIES="5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA"
i=0
for fam in $FAMILIES; do
    gpu=$(( i % N_GPU ))
    ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py \
        --family "$fam" --model_dir "$EPOCH_CKPT" --out_dir "$OUT" \
        --epochs 30 --base_lr 5e-4 --unfreeze_min_layer 9 \
        > "logs/ss_650m_lfo_${fam}.log" 2>&1
      echo "[ss_650m_lfo] $fam done" ) &
    i=$(( i + 1 ))
    if (( i % N_GPU == 0 )); then wait; fi
done
wait

echo "[ss_650m_lfo] All done: $(date)"
python3 - <<'PYEOF'
import json, statistics
from pathlib import Path

out = Path("outputs/fm_hybrid_650m/ss_eval_perseq/epoch12_lfo")
fams = ["5s","16s","23s","grp1","srp","telomerase","RNaseP","tmRNA","tRNA"]
rows = []
print(f"\n{'family':<12}{'mean_F1':>9}{'median':>9}{'q25':>7}{'q75':>7}{'n_test':>8}")
print("-"*54)
for fam in fams:
    jf = out / f"{fam}.json"
    if not jf.exists(): print(f"  {fam:<12}  MISSING"); continue
    d = json.load(open(jf))
    rows.append(d["mean_F1"])
    print(f"  {fam:<12}{d['mean_F1']:>9.3f}{d['median_F1']:>9.3f}"
          f"{d['q25_F1']:>7.3f}{d['q75_F1']:>7.3f}{d['num_test']:>8}")
if rows:
    print("-"*54)
    print(f"  {'MACRO':<12}{statistics.mean(rows):>9.3f}")
PYEOF
