#!/bin/bash
#SBATCH --job-name=z3d_evalprep
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/rna3d/evalprep_%j.out
set -uo pipefail
R=/beacon-projects/rnallm/rna3d; C=/beacon-homes/ziyaow/RNA_FM/rna3d
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
conda activate /beacon-projects/rnallm/software/miniconda3/envs/zfold
export PYTHONUNBUFFERED=1 TMPDIR=$R/tmp PIP_CACHE_DIR=/beacon-projects/rnallm/cache/pip
mkdir -p $R/data/casp16_native && tar -xzf $C/_stage/casp16.targets_RNA.tgz -C $R/data/casp16_native && echo "[casp16] natives: $(ls $R/data/casp16_native/Targets_RNA | wc -l)"
pip install -q barnaba && python -c "import barnaba;print('[env] barnaba ok')"
python $C/prepare_tests.py
# end-to-end test on one CASP15 target with the smoke-test hybrid_650m checkpoint
T=$R/tmp/e2e; mkdir -p $T/final; cp $R/tmp/smoke_hybrid_650m/config.json $T/final/; ln -sfn $(ls $R/tmp/smoke_hybrid_650m/checkpoints/*.pth | tail -1) $T/final/final.pth
line=$(grep -P "^CASP15\tR1107\t" $R/tests/all.tsv); IFS=$'\t' read -r set name seq msa ss native <<< "$line"
cd $C/Zfold/src/Zfold
python predict.py -i $msa -o $T/$name.npz -ss $ss -ss_fmt ct --para_dir / --Zfold_para_name ${T#/}/final --LM_para_name x -gpu 0 --cpu 8 --stru_feat_type LM --LM_name hybrid_650m > $T/predict.log 2>&1; echo "[e2e] predict exit=$?"; tail -3 $T/predict.log
printf '>%s\n%s\n' $name $seq > $T/seq.fasta
t0=$(date +%s); python fold.py -npz $T/$name.npz -fa $T/seq.fasta -out $T/$name.pdb -nm 5 -tmp $T/tmp --CPU 5 > $T/fold.log 2>&1; echo "[e2e] fold exit=$? $(( $(date +%s)-t0 ))s"; tail -3 $T/fold.log
python - <<PY
import sys; sys.path.insert(0, "$C"); import score
print("[e2e] TM, RMSD =", score.tmscore("$T/$name.pdb", "$native"))
print("[e2e] bp =", score.bp_metrics("$T/$name.pdb", "$native", "$seq"))
print("[e2e] native self-check =", score.bp_metrics("$native", "$native", "$seq"), score.tmscore("$native", "$native"))
PY
echo "[evalprep] done $(date)"
