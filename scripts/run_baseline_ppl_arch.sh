#!/bin/bash
# Full-model baseline perplexity for the pure-architecture ablations (Fig 4:
# scaling across model size AND architecture): pure-Transformer and
# pure-Mamba at 100M/300M(/650M where available), matched against LIMBER.
#
#SBATCH --job-name=baseline_ppl_arch
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/baseline_ppl_arch_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/baseline_ppl_arch_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
mkdir -p logs outputs/baseline_ppl_final

run_one () {
    local gpu=$1; local tag=$2; local ckpt=$3
    ( CUDA_VISIBLE_DEVICES=$gpu python -u scripts/baseline_ppl_only.py \
        --ckpt "$ckpt" \
        --out  "outputs/baseline_ppl_final/$tag" \
        --n_val 4096 --batch 32 \
        > "logs/baseline_ppl_${tag}.log" 2>&1
      echo "[baseline_ppl_arch] $tag done" ) &
}

run_one 0 "bert_100m_epoch6"             "/beacon-projects/rnallm/checkpoints/bert_100m/epoch_checkpoints/epoch6"
run_one 1 "bert_300m_epoch6"             "/beacon-projects/rnallm/checkpoints/bert_300m/epoch_checkpoints/epoch6"
run_one 2 "pure_mamba_100m_epoch6"       "/beacon-projects/rnallm/checkpoints/fm_pure_mamba_100m/epoch_checkpoints/epoch6"
run_one 3 "pure_mamba_300m_epoch6"       "/beacon-projects/rnallm/checkpoints/fm_pure_mamba_300m/epoch_checkpoints/epoch6"
wait
run_one 0 "pure_transformer_650m_epoch4" "/beacon-projects/rnallm/checkpoints/fm_pure_transformer_650m/epoch_checkpoints/epoch4"
wait

echo "[baseline_ppl_arch] All done: $(date)"
python3 - <<'PYEOF'
import json
from pathlib import Path
root = Path("outputs/baseline_ppl_final")
for tag in ["bert_100m_epoch6","bert_300m_epoch6","pure_mamba_100m_epoch6",
            "pure_mamba_300m_epoch6","pure_transformer_650m_epoch4"]:
    jf = root / tag / "baseline_ppl.json"
    if jf.exists():
        r = json.load(open(jf))
        print(f"  {tag:<28s} PPL={r['baseline_ppl']:.4f}  n_layers={r['n_layers']}")
    else:
        print(f"  {tag}: MISSING")
PYEOF
