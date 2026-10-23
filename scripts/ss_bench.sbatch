#!/bin/bash
# Secondary-structure fine-tuning on the standard benchmarks, RiNALMo protocol.
#   BENCH=bprna : TR0 -> VL0 -> TS0            (SPOT-RNA / Zhou lab, intra-family)
#   BENCH=rivas : TrainSetA -> TestSetA/TestSetB (Rivas; TestSetB = cross-family)
# Usage: sbatch --job-name=ssb_<bench>_<model> --export=ALL,BENCH=...,MODEL=... ss_bench.sbatch
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_bench/%x_%j.out
set -uo pipefail
: "${BENCH:?}" "${MODEL:?}"
REPO=/beacon-homes/ziyaow/RNA_FM
D=/beacon-projects/rnallm/data/ss_bench
OUT=/beacon-projects/rnallm/outputs/ss_bench/$BENCH
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate rnafm
cd "$REPO"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"

case $BENCH in
  bprna) SPLITS="--train_dir $D/spotrna_bprna/bpRNA_dataset/TR0 --valid_dir $D/spotrna_bprna/bpRNA_dataset/VL0 \
                 --test_dir $D/spotrna_bprna/bpRNA_dataset/TS0 --test_name TS0 --split_name bpRNA_TR0-VL0-TS0" ;;
  rivas) SPLITS="--train_dir $D/rivas/data/TrainSetA --test_dir $D/rivas/data/TestSetA \
                 --test_dir2 $D/rivas/data/TestSetB --test_name TestSetA --test_name2 TestSetB \
                 --val_frac 0.10 --split_name Rivas_TrainSetA" ;;
  *) echo "unknown BENCH=$BENCH" >&2; exit 1 ;;
esac

# bottom 9 layers stay frozen for 33-layer encoders (RiNALMo's published ft schedule), bottom 3 otherwise
case $MODEL in
  hybrid_100m)  SCRIPT=rinalmo_ss_finetune.py; M="--model_dir /beacon-projects/rnallm/checkpoints/fm_hybrid_100m/epoch_checkpoints/epoch12"; MIN=3 ;;
  hybrid_300m)  SCRIPT=rinalmo_ss_finetune.py; M="--model_dir /beacon-projects/rnallm/checkpoints/fm_hybrid_300m/epoch_checkpoints/epoch12"; MIN=3 ;;
  hybrid_650m)  SCRIPT=rinalmo_ss_finetune.py; M="--model_dir /beacon-projects/rnallm/checkpoints/fm_hybrid_650m/epoch_checkpoints/epoch12"; MIN=9 ;;
  rinalmo_giga) SCRIPT=ss_perseq_external.py;  M="--model_type rinalmo_giga";  MIN=9 ;;
  aido_rna_650m)SCRIPT=ss_perseq_external.py;  M="--model_type aido_rna_650m"; MIN=9 ;;
  rnafm_100m)   SCRIPT=ss_perseq_external.py;  M="--model_type rnafm_100m";    MIN=3 ;;
  structrfm_86m)SCRIPT=ss_perseq_external.py;  M="--model_type structrfm_86m"; MIN=3 ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

echo "[ssb] bench=$BENCH model=$MODEL node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date)"
python scripts/$SCRIPT $M $SPLITS --tag $MODEL --out_dir "$OUT" \
    --epochs ${EPOCHS:-30} --base_lr 5e-4 --unfreeze_min_layer $MIN --max_len 512 \
    --tune_every ${TUNE_EVERY:-5} --seed 42 ${EXTRA:-}
echo "[ssb] exit=$? done=$(date)"
