#!/bin/bash
# Pick up the two runs whose smoke-test JSONs were mistakenly treated as finished.
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH USER=${USER:-ziyaow}
C=/beacon-homes/ziyaow/RNA_FM; OUT=/beacon-projects/rnallm/outputs/ss_bench
todo="bprna:hybrid_100m rivas:rnafm_100m rivas:structrfm_86m"
while [ -n "$todo" ]; do
  left=""
  for pair in $todo; do
    b=${pair%%:*}; m=${pair##*:}
    [ -f "$OUT/$b/$m.json" ] && continue
    squeue -h -u $USER -n ssb_${b}_${m} -o %i 2>/dev/null | grep -q . && { left="$left $pair"; continue; }
    if j=$(sbatch --parsable --job-name=ssb_${b}_${m} --export=ALL,BENCH=$b,MODEL=$m $C/scripts/ss_bench.sbatch 2>&1); then
      echo "$(date) submitted $b/$m -> $j"
    else
      left="$left $pair"
    fi
  done
  todo=$(echo $left)
  [ -n "$todo" ] && sleep 900
done
echo "$(date) remaining SS-bench runs submitted"
