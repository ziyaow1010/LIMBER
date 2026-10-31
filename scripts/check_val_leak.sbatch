#!/bin/bash
# Does the held-out validation set leak into the data-scaling training subsets?
#SBATCH --job-name=valleak
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/valcheck/%x_%j.out
set -uo pipefail
D=/beacon-projects/rnallm/data/rnacentral
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
export LC_ALL=C

echo "host=$(hostname)  start=$(date)"
VAL=$D/epoch_seqs_full/val.txt
echo "val.txt lines: $(wc -l < "$VAL")"
sort -u -S 8G -T "$TMP" "$VAL" > "$TMP/val.srt"
NVAL=$(wc -l < "$TMP/val.srt"); echo "val unique: $NVAL"

for SUB in 20pct 40pct 60pct 80pct full; do
  DIR=$D/epoch_seqs_$SUB
  EP=$(ls "$DIR"/epoch1.txt 2>/dev/null)
  [ -z "$EP" ] && { echo "$SUB: no epoch file in $DIR"; ls "$DIR" | head; continue; }
  N=$(wc -l < "$EP")
  sort -u -S 8G -T "$TMP" "$EP" > "$TMP/tr.srt"
  OV=$(comm -12 "$TMP/val.srt" "$TMP/tr.srt" | wc -l)
  echo "$SUB: file=$(basename "$EP") train_lines=$N  val_overlap=$OV  pct_of_train=$(awk -v a=$OV -v b=$N 'BEGIN{printf "%.4f%%", 100*a/b}')  pct_of_val=$(awk -v a=$OV -v b=$NVAL 'BEGIN{printf "%.2f%%", 100*a/b}')"
done
echo "end=$(date)"
