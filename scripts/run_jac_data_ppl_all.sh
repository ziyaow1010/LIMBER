#!/bin/bash
# Full-model (no ablation) perplexity on the categorical-Jacobian SS eval
# sequences (bpRNA PDB/RNP/SRP/tmRNA/tRNAdb/RFAM/CRW), across all 9 backbones:
# our LIMBER-10M/50M/100M/300M/650M (epoch6, matching the jac/layer-importance
# convention) + the 4 baselines.
#
#SBATCH --job-name=jac_data_ppl_all
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/jac_data_ppl_all_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/jac_data_ppl_all_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs outputs/jac_data_ppl

N_GPU=4
MODELS="fm_hybrid_10m fm_hybrid_50m fm_hybrid_100m fm_hybrid_300m fm_hybrid_650m rinalmo_giga aido_rna_650m rnafm_100m structrfm_86m"

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

echo "[jac_data_ppl_all] All done: $(date)"
python3 - <<'PYEOF'
import json
from pathlib import Path
root = Path("outputs/jac_data_ppl")
for m in ["fm_hybrid_10m","fm_hybrid_50m","fm_hybrid_100m","fm_hybrid_300m","fm_hybrid_650m",
          "rinalmo_giga","aido_rna_650m","rnafm_100m","structrfm_86m"]:
    jf = root / m / "jac_data_ppl.json"
    if jf.exists():
        d = json.load(open(jf))
        print(f"  {d['model_name']:<16s} macro_ppl={d['macro_mean_ppl']:.4f}")
    else:
        print(f"  {m}: MISSING")
PYEOF
