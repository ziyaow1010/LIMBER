#!/bin/bash
# Submit the NRC fine-tuning runs that fit on an L40S, retrying while the submit limit is hit.
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH USER=${USER:-ziyaow}
C=/beacon-homes/ziyaow/RNA_FM; CK=/beacon-projects/rnallm/checkpoints
OUT=/beacon-projects/rnallm/outputs/nrc_finetune
# run|ckpt-or-model|extra   (effective batch 64 = 8 x 8 everywhere)
SPECS="
limber_100m|CKPT=$CK/fm_hybrid_100m/epoch_checkpoints/epoch12|
limber_300m|CKPT=$CK/fm_hybrid_300m/epoch_checkpoints/epoch12|
limber_650m|CKPT=$CK/fm_hybrid_650m/epoch_checkpoints/epoch12|
limber_650m_L2|CKPT=$CK/fm_hybrid_650m/epoch_checkpoints/epoch12|NLAYERS=2
limber_650m_L14|CKPT=$CK/fm_hybrid_650m/epoch_checkpoints/epoch12|NLAYERS=14
rnafm_100m|MODEL_TYPE=rnafm_100m|
structrfm_86m|MODEL_TYPE=structrfm_86m|
"
todo=$(echo "$SPECS" | grep -v '^$')
while [ -n "$todo" ]; do
  left=""
  while IFS= read -r spec; do
    [ -z "$spec" ] && continue
    run=$(echo "$spec" | cut -d'|' -f1)
    mdl=$(echo "$spec" | cut -d'|' -f2)
    nl=$(echo  "$spec" | cut -d'|' -f3)
    [ -f "$OUT/$run/results.json" ] && { echo "$(date) $run already done"; continue; }
    squeue -h -u $USER -n nrc_$run -o %i 2>/dev/null | grep -q . && { left="$left$spec"$'\n'; continue; }
    if j=$(sbatch --parsable --job-name=nrc_$run \
             --export=ALL,RUN=$run,$mdl${nl:+,$nl},EXTRA="--batch_size 8 --accum_steps 8" \
             $C/scripts/nrc_ft.sbatch 2>&1); then
      echo "$(date) submitted $run -> $j"
    else
      left="$left$spec"$'\n'
    fi
  done <<< "$todo"
  todo=$(echo "$left" | grep -v '^$')
  [ -n "$todo" ] && { echo "$(date) limit hit, waiting"; sleep 900; }
done
echo "$(date) all NRC runs submitted"
