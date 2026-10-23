#!/bin/bash
# Wait for the metric smoke test, then resubmit all 9 NRC runs with full metrics.
export PATH=/usr/bin:/bin:/usr/local/bin:$PATH USER=${USER:-ziyaow}
C=/beacon-homes/ziyaow/RNA_FM; CK=/beacon-projects/rnallm/checkpoints
OUT=/beacon-projects/rnallm/outputs/nrc_finetune
SMOKE=$OUT/smoke_metrics/results.json

while squeue -h -u $USER -n nrc_mtest -o %i 2>/dev/null | grep -q .; do sleep 60; done
if ! grep -q '"mcc"' "$SMOKE" 2>/dev/null; then
  echo "$(date) smoke test did not produce metrics; aborting"; exit 1
fi
echo "$(date) smoke ok, metrics present"

SPECS="
limber_650m|CKPT=$CK/fm_hybrid_650m/epoch_checkpoints/epoch12||8|8
limber_300m|CKPT=$CK/fm_hybrid_300m/epoch_checkpoints/epoch12||8|8
limber_100m|CKPT=$CK/fm_hybrid_100m/epoch_checkpoints/epoch12||8|8
limber_650m_L14|CKPT=$CK/fm_hybrid_650m/epoch_checkpoints/epoch12|NLAYERS=14|8|8
limber_650m_L2|CKPT=$CK/fm_hybrid_650m/epoch_checkpoints/epoch12|NLAYERS=2|8|8
rnafm_100m|MODEL_TYPE=rnafm_100m||8|8
structrfm_86m|MODEL_TYPE=structrfm_86m||8|8
rinalmo_giga|MODEL_TYPE=rinalmo_giga||4|16
aido_rna_650m|MODEL_TYPE=aido_rna_650m||4|16
"
todo=$(echo "$SPECS" | grep -v '^$')
while [ -n "$todo" ]; do
  left=""
  while IFS= read -r spec; do
    [ -z "$spec" ] && continue
    run=$(echo "$spec"|cut -d'|' -f1); mdl=$(echo "$spec"|cut -d'|' -f2)
    nl=$(echo "$spec"|cut -d'|' -f3); bs=$(echo "$spec"|cut -d'|' -f4); ac=$(echo "$spec"|cut -d'|' -f5)
    grep -q '"mcc"' "$OUT/$run/results.json" 2>/dev/null && { echo "$(date) $run done"; continue; }
    squeue -h -u $USER -n nrc_$run -o %i 2>/dev/null | grep -q . && { left="$left$spec"$'\n'; continue; }
    if j=$(sbatch --parsable --job-name=nrc_$run \
             --export=ALL,RUN=$run,$mdl${nl:+,$nl},EXTRA="--batch_size $bs --accum_steps $ac" \
             $C/scripts/nrc_ft.sbatch 2>&1); then
      echo "$(date) submitted $run -> $j"
    else
      left="$left$spec"$'\n'
    fi
  done <<< "$todo"
  todo=$(echo "$left" | grep -v '^$')
  [ -n "$todo" ] && sleep 600
done
echo "$(date) all 9 submitted"
