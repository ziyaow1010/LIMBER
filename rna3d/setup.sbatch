#!/bin/bash
# One-off setup for the RNA 3D (trRosettaRNA / Zfold) evaluation: data, code, conda env.
#SBATCH --job-name=rna3d_setup
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/rna3d/setup_%j.out
set -uo pipefail
R=/beacon-projects/rnallm/rna3d
mkdir -p $R/data $R/code $R/tmp
export TMPDIR=$R/tmp PIP_CACHE_DIR=/beacon-projects/rnallm/cache/pip CONDA_PKGS_DIRS=/beacon-projects/rnallm/cache/conda_pkgs
B=https://yanglab.qd.sdu.edu.cn/trRosettaRNA/benchmark
cd $R/data
for f in training_npzs_light.tar.bz2 20_RNA_Puzzles.tar.bz2 CASP15_RNAs.tar.bz2 RNA_Puzzles_latest.tar.bz2 30_independent_RNAs.tar.bz2 trRNA2_bench.tar.bz2 \
         before2010-12.lst before2013-07.lst before2016-07.lst before2017-01.lst before2019-04.lst train10699.lst params_201904.tar.bz2; do
  for i in 1 2 3 4 5; do curl -sS -L -C - --retry 5 -o $f $B/$f && break; sleep 30; done
  echo "[dl] $f $(stat -c %s $f)"
done
for f in *.tar.bz2; do d=${f%.tar.bz2}; [ -d $d ] || { mkdir -p $d && tar -xjf $f -C $d && echo "[untar] $f ok"; }; done
cd $R/code
[ -d structRFM ] || git clone -q https://github.com/heqin-zhu/structRFM.git
mkdir -p ckpt && cd ckpt && [ -f structRFM_checkpoint.tar.gz ] || curl -sS -L -o structRFM_checkpoint.tar.gz https://github.com/heqin-zhu/structRFM/releases/latest/download/structRFM_checkpoint.tar.gz; tar -xzf structRFM_checkpoint.tar.gz && echo "[ckpt] structRFM ok"
source /beacon-projects/rnallm/software/miniconda3/etc/profile.d/conda.sh
E=/beacon-projects/rnallm/software/miniconda3/envs/zfold
[ -d $E ] || conda create -y -q -p $E --clone /beacon-projects/rnallm/software/miniconda3/envs/rnafm
conda activate $E
conda install -y -q -c https://conda.rosettacommons.org -c conda-forge pyrosetta && echo "[env] pyrosetta ok"
pip install -q einops biopython BPfold structRFM multimolecule && echo "[env] pip ok"
python -c "import torch,pyrosetta,einops,Bio,mamba_ssm;print('[env] import ok',torch.__version__,torch.version.cuda)"
echo "[tree]"; find $R/data -maxdepth 3 | head -80; du -sh $R/data/* 2>/dev/null
echo "[setup] done $(date)"
