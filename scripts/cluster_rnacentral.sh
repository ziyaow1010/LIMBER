#!/bin/bash
# Cluster RNAcentral to ~17M representative sequences, matching RiNALMo's exact
# preprocessing: (1) exact-dedup raw RNAcentral to unique sequences (paper: 40.7M -> 36M),
# (2) mmseqs easy-linclust --min-seq-id 0.7 -c 0.8 on the deduped set (paper: 36M -> 17M).
# Input:  /beacon-projects/rnallm/data/rnacentral/rnacentral_active.fasta.gz
# Output: /beacon-projects/rnallm/data/rnacentral/clustered_17m.fasta
#SBATCH --job-name=cluster_rna
#SBATCH --partition=beacon
#SBATCH --account=angliece
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-homes/ziyaow/RNA_FM/logs/cluster_rnacentral_%j.out
#SBATCH --error=/beacon-homes/ziyaow/RNA_FM/logs/cluster_rnacentral_%j.err

set -euo pipefail
REPO=/beacon-homes/ziyaow/RNA_FM
CONDA_BASE=/beacon-projects/rnallm/software/miniconda3
DATA=/beacon-projects/rnallm/data/rnacentral
INPUT=$DATA/rnacentral_active.fasta.gz
OUT_FASTA=$DATA/clustered_17m.fasta
TMP=$DATA/mmseqs_tmp
CLUST_PREFIX=$DATA/mmseqs_clust
DEDUP_FASTA=$DATA/rnacentral_active.dedup.fasta

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate rnafm

# mmseqs should have been installed in build_env step
if ! command -v mmseqs &>/dev/null; then
    # Fallback: download static binary
    echo "[cluster] mmseqs not found, downloading static binary..."
    MMSEQS_BIN=$DATA/mmseqs_bin
    mkdir -p $MMSEQS_BIN
    curl -L -o $MMSEQS_BIN/mmseqs.tar.gz \
        "https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-avx2.tar.gz"
    tar -xf $MMSEQS_BIN/mmseqs.tar.gz -C $MMSEQS_BIN --strip-components=1
    export PATH="$MMSEQS_BIN/bin:$PATH"
fi

echo "[cluster] mmseqs version: $(mmseqs --version 2>/dev/null || echo unknown)"

# Decompress to temp fasta for mmseqs
PLAIN_FASTA=$DATA/rnacentral_active.fasta
if [ ! -f "$PLAIN_FASTA" ]; then
    echo "[cluster] Decompressing fasta..."
    pigz -d -k -p 8 "$INPUT" 2>/dev/null || gzip -d -k "$INPUT"
fi

echo "[cluster] Input sequences: $(grep -c '^>' $PLAIN_FASTA)"

# Step 1: exact-dedup (RiNALMo dedupes RNAcentral to ~36M unique seqs before clustering)
if [ ! -f "$DEDUP_FASTA" ]; then
    echo "[cluster] Deduplicating to unique sequences..."
    python "$REPO/scripts/dedup_fasta.py" "$PLAIN_FASTA" "$DEDUP_FASTA"
fi
echo "[cluster] Deduped sequences: $(grep -c '^>' $DEDUP_FASTA)"

mkdir -p "$TMP"
echo "[cluster] Running mmseqs easy-linclust (--min-seq-id 0.7, RiNALMo paper setting)..."
mmseqs easy-linclust \
    "$DEDUP_FASTA" \
    "$CLUST_PREFIX" \
    "$TMP" \
    --min-seq-id 0.7 \
    --cov-mode 1 \
    -c 0.8 \
    --threads 4 \
    -v 1

REP_FASTA="${CLUST_PREFIX}_rep_seq.fasta"
N_REPS=$(grep -c '^>' "$REP_FASTA")
echo "[cluster] Representatives: $N_REPS"
cp "$REP_FASTA" "$OUT_FASTA"

echo "[cluster] Done: $OUT_FASTA  ($N_REPS sequences)"
echo "[cluster] RiNALMo paper: 40.7M raw -> 36M unique -> 17M clustered reps"
