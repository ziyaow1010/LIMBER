#!/bin/bash
# For each "run:lm" pair: once training has written the epoch-30 checkpoint and its job has left the
# queue, submit predict + dependent fold/score. If the job died early, resubmit training (it resumes).
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH USER=${USER:-ziyaow}
C=/beacon-homes/ziyaow/RNA_FM/rna3d; R=/beacon-projects/rnallm/rna3d
todo="$*"
while [ -n "$todo" ]; do
  left=""
  for pair in $todo; do
    run=${pair%%:*}; lm=${pair##*:}; extra=""; case $run in *_e2e) extra="--e2e --lm_lr 1e-5";; esac
    queued=$(squeue -h -u $USER -n z3d_$run -o %i 2>/dev/null) || { left="$left $pair"; continue; }
    if [ -n "$queued" ]; then left="$left $pair"; continue; fi
    if ls $R/runs/$run/checkpoints/model_ep029_*.pth >/dev/null 2>&1; then
      P=$(sbatch --parsable --job-name=z3p_$run --export=ALL,LM=$lm,RUN=$run $C/predict.sbatch) &&
      F=$(sbatch --parsable --job-name=z3f_$run --dependency=afterok:$P --export=ALL,LM=$lm,RUN=$run $C/fold_score.sbatch) &&
      echo "$(date) $run predict=$P fold=$F" || { echo "$(date) $run eval submit failed"; left="$left $pair"; }
    elif ls $R/runs/$run >/dev/null 2>&1; then
      J=$(sbatch --parsable --job-name=z3d_$run --export=ALL,LM=$lm,RUN=$run,EXTRA="$extra" $C/train.sbatch) && echo "$(date) $run training not finished, resubmitted $J (resumes)"
      left="$left $pair"
    else echo "$(date) $run: project disk unreadable, retry"; left="$left $pair"; fi
  done
  todo=$(echo $left); echo "$(date) waiting on: $todo"; [ -n "$todo" ] && sleep 900
done
echo "$(date) all chained"
