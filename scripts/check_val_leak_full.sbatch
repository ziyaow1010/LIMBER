#!/bin/bash
#SBATCH --job-name=valleakfull
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/valcheck/%x_%j.out
set -uo pipefail
D=/beacon-projects/rnallm/data/rnacentral
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm
python - <<'PY'
D = "/beacon-projects/rnallm/data/rnacentral"
val = {l.strip() for l in open(f"{D}/epoch_seqs_full/val.txt") if l.strip()}
print(f"val unique: {len(val)}", flush=True)
for sub in ("full", "20pct", "40pct", "60pct", "80pct"):
    hit, n = set(), 0
    with open(f"{D}/epoch_seqs_{sub}/epoch1.txt") as fh:
        for line in fh:
            s = line.strip()
            n += 1
            if s in val:
                hit.add(s)
    print(f"{sub}: train_lines={n}  val_overlap={len(hit)}  "
          f"pct_of_train={100*len(hit)/n:.4f}%  pct_of_val={100*len(hit)/len(val):.2f}%", flush=True)
PY
echo "ALLDONE $(date)"
