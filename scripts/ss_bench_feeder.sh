#!/bin/bash
# Submit all SS-benchmark runs, retrying while the account's submit limit is hit.
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH USER=${USER:-ziyaow}
C=/beacon-homes/ziyaow/RNA_FM; OUT=/beacon-projects/rnallm/outputs/ss_bench
MODELS="hybrid_650m hybrid_300m hybrid_100m rinalmo_giga aido_rna_650m rnafm_100m structrfm_86m"
todo=""
for b in bprna rivas; do for m in $MODELS; do todo="$todo $b:$m"; done; done
while [ -n "$todo" ]; do
  left=""
  for pair in $todo; do
    b=${pair%%:*}; m=${pair##*:}
    [ -f "$OUT/$b/$m.json" ] && { echo "$(date) $b/$m already done"; continue; }
    squeue -h -u $USER -n ssb_${b}_${m} -o %i 2>/dev/null | grep -q . && { left="$left $pair"; continue; }
    if j=$(sbatch --parsable --job-name=ssb_${b}_${m} --export=ALL,BENCH=$b,MODEL=$m $C/scripts/ss_bench.sbatch 2>&1); then
      echo "$(date) submitted $b/$m -> $j"
    else
      left="$left $pair"
    fi
  done
  todo=$(echo $left)
  [ -n "$todo" ] && { echo "$(date) queued-limit, waiting. left:$todo"; sleep 900; }
done
echo "$(date) all SS-bench runs submitted"
