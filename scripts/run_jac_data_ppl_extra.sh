#!/bin/bash
# jac_data_perplexity.py for the extra configs: 100M/300M/650M @ epoch12,
# and the 100M training-data-fraction series (20/40/60/80%) @ epoch6.
#
#SBATCH --job-name=jac_data_ppl_extra
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/jac_data_ppl_extra_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/jac_data_ppl_extra_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs outputs/jac_data_ppl

N_GPU=4
MODELS="fm_hybrid_100m_ep12 fm_hybrid_300m_ep12 fm_hybrid_650m_ep12 hybrid_100m_20pct hybrid_100m_40pct hybrid_100m_60pct hybrid_100m_80pct"

i=0
for m in $MODELS; do
    gpu=$(( i % N_GPU ))
    ( CUDA_VISIBLE_DEVICES=$gpu python scripts/jac_data_perplexity.py \
        --model_type "$m" \
        --out_dir "outputs/jac_data_ppl/$m" \
        --max_rfam 0 \
        > "logs/jac_data_ppl_${m}.log" 2>&1
      echo "[jac_data_ppl] $m done" ) &
    i=$(( i + 1 ))
    if (( i % N_GPU == 0 )); then wait; fi
done
wait

echo "[jac_data_ppl_extra] All done: $(date)"
python3 - <<'PYEOF'
import json
from pathlib import Path
root = Path("outputs/jac_data_ppl")
for m in ["fm_hybrid_100m_ep12","fm_hybrid_300m_ep12","fm_hybrid_650m_ep12",
          "hybrid_100m_20pct","hybrid_100m_40pct","hybrid_100m_60pct","hybrid_100m_80pct"]:
    jf = root / m / "jac_data_ppl.json"
    if jf.exists():
        d = json.load(open(jf))
        print(f"  {m:<22s} macro_ppl={d['macro_mean_ppl']:.4f}")
    else:
        print(f"  {m}: MISSING")
PYEOF
