#!/bin/bash
#SBATCH --job-name=z3d_smoke
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/rna3d/smoke_%j.out
R=/beacon-projects/rnallm/rna3d
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate /beacon-projects/rnallm/software/miniconda3/envs/zfold
export PYTHONUNBUFFERED=1 TMPDIR=$R/tmp
NPZ=$(cat $R/data/NPZ_DIR); ls $NPZ | sed 's/.npz$//' | sort | head -130 > $R/tmp/smoke130.lst
cd /beacon-homes/ziyaow/RNA_FM/rna3d/Zfold/src/Zfold/training
for LM in ${LMS:-SS none structRFM hybrid_650m rinalmo_giga rnafm_100m aido_rna_650m}; do
  if [ "$LM" = SS ]; then FEAT="--stru_feat_type SS --LM_name none"; else FEAT="--stru_feat_type LM --LM_name $LM --LM_path $R/code/ckpt/structRFM_checkpoint"; fi
  t0=$(date +%s)
  python train.py $NPZ $R/tmp/smoke_$LM${SFX:-} -lst $R/tmp/smoke130.lst --max_epochs 1 --gpu 0 --init_lr 0.0005 --crop_size 200 $FEAT ${EXTRA:-} > $R/tmp/smoke_$LM${SFX:-}.log 2>&1
  echo "### $LM exit=$? $(( $(date +%s)-t0 ))s peakmem=$(grep -o 'peak[^ ]*' $R/tmp/smoke_$LM${SFX:-}.log | tail -1)"
  grep -E "train:test|train loss|val loss|OOM|Error|Traceback" $R/tmp/smoke_$LM${SFX:-}.log | tail -6
  tail -3 $R/tmp/smoke_$LM${SFX:-}.log | tr '\r' '\n' | tail -2 | cut -c1-200
done
