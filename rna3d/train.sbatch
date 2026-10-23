#!/bin/bash
# Train Zfold (trRosettaRNA network) with one RNA LM's H@H.T as the 2D structure feature.
# Usage: sbatch --job-name=z3d_<run> --export=ALL,LM=<structRFM|hybrid_650m|...|none|SS>[,RUN=<name>,EXTRA="--e2e --lm_lr 1e-5"] train.sbatch
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/rna3d/%x_%j.out
set -euo pipefail
: "${LM:?}"
R=/beacon-projects/rnallm/rna3d
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate /beacon-projects/rnallm/software/miniconda3/envs/zfold
export PYTHONUNBUFFERED=1 TMPDIR=$R/tmp TOKENIZERS_PARALLELISM=false
NPZ=$(cat $R/data/NPZ_DIR)                 # set once the training tarball layout is known
RUN=${RUN:-$LM}
OUT=$R/runs/$RUN
mkdir -p $OUT
cd /beacon-homes/ziyaow/RNA_FM/rna3d/Zfold/src/Zfold/training
if [ "$LM" = SS ]; then FEAT="--stru_feat_type SS --LM_name none --LM_path $R/code/ckpt/structRFM_checkpoint"     # original trRosettaRNA input (SPOT-RNA SS)
else FEAT="--stru_feat_type LM --LM_name $LM --LM_path $R/code/ckpt/structRFM_checkpoint"; fi
echo "[z3d] LM=$LM node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date)"
RES=$(ls $OUT/checkpoints/model_ep*.pth 2>/dev/null | sort | tail -1 || true)
if [ -n "$RES" ]; then FEAT="$FEAT --resume $RES"; echo "[z3d] resume $RES"; fi
python train.py $NPZ $OUT --warning --gpu 0 --init_lr 0.0005 --crop_size 200 $FEAT ${EXTRA:-} 2>&1 | grep -v "it/s\]$" | tee -a $OUT/stdout.log | grep -E "Epoch:|loss|OOM|Error|error|Traceback|train:test" 
echo "[z3d] done=$(date)"
