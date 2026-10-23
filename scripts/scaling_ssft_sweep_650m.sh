#!/bin/bash
# SS FT sweep for 650M epochs 1-11. Skips epochs with existing summary.json.
# Runs 9 families in 4-GPU waves per epoch, epochs sequentially.
# Submit after ep11 SS FT finishes: sbatch --dependency=afterany:<ep11_job> this_script.sh
#SBATCH --job-name=ssft_650m_sweep
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ssft_650m_sweep_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/ssft_650m_sweep_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
CKPT_DIR=/beacon-projects/rnallm/checkpoints/fm_hybrid_650m

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"

CT_ROOT="data/contact_eval/raw/ct/fam-fold"
if [ ! -d "$CT_ROOT/5s/train" ]; then
    python scripts/make_ct_splits.py
fi

FAMILIES="5s 16s 23s grp1 srp telomerase RNaseP tmRNA tRNA"
N_GPU=4

echo "[ssft_sweep] Start: $(date)"

for ep in $(seq 1 11); do
    OUT="outputs/fm_hybrid_650m/ss_eval/epoch${ep}"
    if [ -f "$OUT/summary.json" ]; then
        echo "[ssft_sweep] ep${ep}: already done, skipping"
        continue
    fi
    CKPT="$CKPT_DIR/epoch_checkpoints/epoch${ep}"
    if [ ! -d "$CKPT" ]; then
        echo "[ssft_sweep] ep${ep}: checkpoint not found, skipping"
        continue
    fi
    mkdir -p "$OUT"
    echo "[ssft_sweep] ep${ep}: running SS FT... ($(date))"

    i=0
    for fam in $FAMILIES; do
        gpu=$(( i % N_GPU ))
        LOG="logs/ssft_650m_ep${ep}_${fam}.log"
        ( CUDA_VISIBLE_DEVICES=$gpu python scripts/rinalmo_ss_finetune.py \
            --family "$fam" \
            --model_dir "$CKPT" \
            --out_dir "$OUT" \
            --epochs 30 \
            --base_lr 5e-4 \
            --unfreeze_min_layer 9 \
            > "$LOG" 2>&1
          echo "[ssft_sweep] ep${ep} $fam done (gpu $gpu)" ) &
        i=$(( i + 1 ))
        if (( i % N_GPU == 0 )); then
            wait
            echo "[ssft_sweep] ep${ep} wave $((i/N_GPU)) done"
        fi
    done
    wait

    # Aggregate this epoch
    python3 - <<PYEOF
import json, statistics
from pathlib import Path

ep = $ep
out = Path(f"outputs/fm_hybrid_650m/ss_eval/epoch{ep}")
rinalmo = {"5s":0.88,"16s":0.74,"23s":0.85,"grp1":0.66,
           "srp":0.70,"telomerase":0.12,"RNaseP":0.80,"tmRNA":0.80,"tRNA":0.93}

rows = []
for jf in sorted(out.glob("*.json")):
    d = json.load(open(jf))
    fam = jf.stem
    f1 = d.get("mean_F1", 0.0)
    rows.append((fam, f1))

if rows:
    macro = statistics.mean(f for _,f in rows)
    summary = {"model":"hybrid_650m","epoch":ep,"macro_f1":macro,
               "per_family":{fam:f1 for fam,f1 in rows},
               "rinalmo_macro":statistics.mean(rinalmo.values())}
    json.dump(summary, open(out/"summary.json","w"), indent=2)
    print(f"[ssft_sweep] ep{ep} macro_F1={macro:.4f}")
PYEOF

    echo "[ssft_sweep] ep${ep}: done ($(date))"
done

echo ""
echo "[ssft_sweep] === FULL RESULTS ==="
python3 - <<'PYEOF'
import json, statistics
from pathlib import Path

rinalmo = {"5s":0.88,"16s":0.74,"23s":0.85,"grp1":0.66,
           "srp":0.70,"telomerase":0.12,"RNaseP":0.80,"tmRNA":0.80,"tRNA":0.93}
fams = ["5s","16s","23s","grp1","srp","telomerase","RNaseP","tmRNA","tRNA"]

# --- Epoch sweep (650M) ---
print("\n=== 650M epoch sweep (SS FT macro F1) ===")
print(f"{'epoch':<8}", end="")
for f in fams: print(f"{f:>12}", end="")
print(f"{'MACRO':>8}")
print("-" * (8 + 12*len(fams) + 8))

for ep in range(1, 12):
    p = Path(f"outputs/fm_hybrid_650m/ss_eval/epoch{ep}/summary.json")
    if not p.exists():
        print(f"  ep{ep:<6}{'—':>12}" * len(fams) + f"{'—':>8}")
        continue
    d = json.load(open(p))
    pf = d.get("per_family", {})
    print(f"  ep{ep:<6}", end="")
    for f in fams:
        v = pf.get(f, float("nan"))
        print(f"{v:>12.3f}", end="")
    print(f"{d['macro_f1']:>8.3f}")

ref_macro = statistics.mean(rinalmo.values())
print(f"  {'RiNALMo':<6}", end="")
for f in fams: print(f"{rinalmo[f]:>12.3f}", end="")
print(f"{ref_macro:>8.3f}")

# --- Model size scaling (ep6) ---
print("\n=== Model size scaling (epoch 6, SS FT macro F1) ===")
size_results = {}
for m, path in [("100M","fm_hybrid_100m"), ("300M","fm_hybrid_300m"), ("650M","fm_hybrid_650m")]:
    p = Path(f"outputs/{path}/ss_eval/epoch6/summary.json")
    if p.exists():
        size_results[m] = json.load(open(p))

print(f"{'model':<10}", end="")
for f in fams: print(f"{f:>12}", end="")
print(f"{'MACRO':>8}")
print("-" * (10 + 12*len(fams) + 8))
for m, d in size_results.items():
    pf = d.get("per_family", {})
    print(f"  {m:<8}", end="")
    for f in fams:
        v = pf.get(f, float("nan"))
        print(f"{v:>12.3f}", end="")
    print(f"{d['macro_f1']:>8.3f}")
print(f"  {'RiNALMo':<8}", end="")
for f in fams: print(f"{rinalmo[f]:>12.3f}", end="")
print(f"{ref_macro:>8.3f}")
PYEOF

echo ""
echo "[ssft_sweep] All done: $(date)"
