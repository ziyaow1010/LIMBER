#!/bin/bash
# IRES classification fine-tune, all 7 backbones (3 LIMBER + 4 baselines).
#
#SBATCH --job-name=ires_ft_all
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ires_ft_all_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ires_ft_all_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs outputs/downstream_ft/ires

N_GPU=4
MODELS="fm_hybrid_100m fm_hybrid_300m fm_hybrid_650m rinalmo_giga aido_rna_650m rnafm_100m structrfm_86m"

i=0
for m in $MODELS; do
    gpu=$(( i % N_GPU ))
    ( CUDA_VISIBLE_DEVICES=$gpu python scripts/downstream_finetune.py \
        --model_type "$m" --task ires \
        --out_dir "outputs/downstream_ft/ires/$m" \
        > "logs/ires_ft_${m}.log" 2>&1
      echo "[ires_ft] $m done" ) &
    i=$(( i + 1 ))
    if (( i % N_GPU == 0 )); then wait; fi
done
wait

echo "[ires_ft] All done: $(date)"
python3 - <<'PYEOF'
import json
from pathlib import Path
def g(r, k):
    v = r.get(k, {}).get("auroc")
    return f"{v:.4f}" if isinstance(v, (int, float)) else "?"
root = Path("outputs/downstream_ft/ires")
for m in sorted(root.iterdir()):
    jf = m / "results.json"
    if jf.exists():
        d = json.load(open(jf))
        r = d["results"]
        print(f"  {d['model_name']:<16s} test_bal_auroc={g(r,'test_bal')} "
              f"test_imbal_auroc={g(r,'test_imbal')} test_main_auroc={g(r,'test_main')}")
PYEOF
