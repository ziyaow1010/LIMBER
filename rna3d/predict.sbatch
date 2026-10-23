#!/bin/bash
# Predict geometry NPZs for every test target with one trained Zfold run (final epoch).
# Usage: sbatch --job-name=z3p_<lm> --export=ALL,LM=<lm> predict.sbatch
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/rna3d/%x_%j.out
set -uo pipefail
: "${LM:?}"
R=/beacon-projects/rnallm/rna3d
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate /beacon-projects/rnallm/software/miniconda3/envs/zfold
export PYTHONUNBUFFERED=1 TMPDIR=$R/tmp
NAME=${RUN:-$LM}; RUN=$R/runs/$NAME; FIN=$RUN/final; mkdir -p $FIN
cp $RUN/config.json $FIN/; last=$(ls $RUN/checkpoints/model_ep*.pth | sort | tail -1); ln -sfn $last $FIN/final.pth
echo "[z3p] LM=$LM ckpt=$(basename $last) start=$(date)"
if [ "$LM" = SS ]; then FEAT="--stru_feat_type SS --LM_name none"; else FEAT="--stru_feat_type LM --LM_name $LM"; fi
cd /beacon-homes/ziyaow/RNA_FM/rna3d/Zfold/src/Zfold
# empty SS fields become '-' first: tab is IFS whitespace, so `read` would otherwise collapse them
tail -n +2 $R/tests/all.tsv | awk -F'\t' 'BEGIN{OFS="\t"}{if($5=="")$5="-";print}' | while IFS=$'\t' read -r set name seq msa ss native; do
  [ "$ss" = - ] && ss=""
  out=$R/preds/$NAME/$set; mkdir -p $out
  [ -s $out/$name.npz ] && continue
  if [ -n "$ss" ]; then SS="-ss $ss -ss_fmt ct"
  elif [ "$LM" = SS ]; then echo "[skip] $set/$name: no SS for the SPOT-RNA baseline"; continue
  else printf '%s\n' "$(printf '.%.0s' $(seq 1 ${#seq}))" > $out/$name.dummy.dbn; SS="-ss $out/$name.dummy.dbn -ss_fmt dot_bracket"; fi
  t0=$(date +%s)
  python predict.py -i $msa -o $out/$name.npz $SS --para_dir / --Zfold_para_name ${FIN#/} --LM_para_name ${R#/}/code/ckpt/structRFM_checkpoint -gpu 0 --cpu 8 $FEAT > $out/$name.predict.log 2>&1
  echo "$set/$name L=${#seq} exit=$? $(( $(date +%s)-t0 ))s"
done
echo "[z3p] done=$(date)"
