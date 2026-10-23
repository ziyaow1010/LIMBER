#!/bin/bash
# Fold predicted geometries with PyRosetta (5 decoys, lowest energy kept) and score against natives.
# Usage: sbatch --job-name=z3f_<lm> --export=ALL,LM=<lm> fold_score.sbatch
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/rna3d/%x_%j.out
set -uo pipefail
: "${LM:?}"; LM=${RUN:-$LM}
R=/beacon-projects/rnallm/rna3d
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate /beacon-projects/rnallm/software/miniconda3/envs/zfold
export PYTHONUNBUFFERED=1 TMPDIR=$R/tmp
cd /beacon-homes/ziyaow/RNA_FM/rna3d/Zfold/src/Zfold
fold_one() {   # $1=set $2=name $3=seq
  d=$R/preds/$LM/$1; [ -s $d/$2.npz ] || return 0; [ -s $d/$2.pdb ] && return 0
  mkdir -p $d/tmp_$2; printf '>%s\n%s\n' $2 $3 > $d/tmp_$2/seq.fasta
  python fold.py -npz $d/$2.npz -fa $d/tmp_$2/seq.fasta -out $d/$2.pdb -nm 5 -tmp $d/tmp_$2 --CPU 5 > $d/$2.fold.log 2>&1
  echo "$1/$2 fold exit=$?"
}
export -f fold_one; export R LM
tail -n +2 $R/tests/all.tsv | cut -f1-3 | xargs -P 3 -L 1 bash -c 'fold_one "$0" "$1" "$2"'
python /beacon-homes/ziyaow/RNA_FM/rna3d/score.py --lm $LM
echo "[z3f] done=$(date)"
