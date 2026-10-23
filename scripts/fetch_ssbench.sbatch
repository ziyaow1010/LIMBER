#!/bin/bash
#SBATCH --job-name=ssb_fetch
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/ss_bench/fetch_%j.out
set -uo pipefail
S=/beacon-homes/ziyaow/RNA_FM/_stage_ssbench
rm -rf "$S"; mkdir -p "$S"
cp -r /beacon-projects/rnallm/outputs/ss_bench/. "$S"/
echo "jsons: $(find $S -name '*.json' | wc -l)"; du -sh "$S"; ls "$S"/*
