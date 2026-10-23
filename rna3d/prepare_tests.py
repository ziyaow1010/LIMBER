#!/usr/bin/env python3
"""Build the three test sets used in the structRFM paper (CASP15-natural, 20 RNA-Puzzles, CASP16)
as tests/<set>/list.tsv: name, seq, msa, ss (CT path or empty), native PDB."""
import glob, os, re
import pandas as pd

D = "/beacon-projects/rnallm/rna3d/data"
OUT = "/beacon-projects/rnallm/rna3d/tests"
CASP16 = ["R1203", "R1205", "R1209", "R1211", "R1212", "R1242", "R1260", "R1261",
          "R1263", "R1283v1", "R1285", "R1286", "R1296"]          # the paper's list


def first_seq(msa):
    seq, started = [], False
    for line in open(msa):
        if line.startswith(">"):
            if started: break
            started = True; continue
        seq.append(line.strip())
    return re.sub(r"[^ACGUT]", "", "".join(seq).upper()).replace("T", "U")


rows = []
for name, base in [("CASP15", f"{D}/CASP15_RNAs/CASP15_RNAs"), ("Puzzles20", f"{D}/20_RNA_Puzzles/20_RNA_Puzzles")]:
    for pdb in sorted(glob.glob(f"{base}/pdb/*.pdb")):
        t = os.path.basename(pdb)[:-4]
        msa = [m for m in glob.glob(f"{base}/msa/{t}.*")][0]
        ct = f"{base}/ss/{t}.ct"
        rows.append(dict(set=name, name=t, seq=first_seq(msa), msa=msa, ss=ct if os.path.exists(ct) else "", native=pdb))
nat = f"{D}/casp16_native/Targets_RNA"
for t in CASP16:
    msa = glob.glob(f"{D}/trRNA2_bench/CASP16/msa/{t}.*")
    natives = sorted(glob.glob(f"{nat}/{t}.pdb") + glob.glob(f"{nat}/{t}_updated.pdb") + glob.glob(f"{nat}/{t}v1.pdb"))
    if not msa or not natives:
        print(f"[skip] CASP16 {t}: msa={msa} native={natives}"); continue
    rows.append(dict(set="CASP16", name=t, seq=first_seq(msa[0]), msa=msa[0], ss="", native=natives[0]))
df = pd.DataFrame(rows)
for s, g in df.groupby("set"):
    os.makedirs(f"{OUT}/{s}", exist_ok=True)
    g.to_csv(f"{OUT}/{s}/list.tsv", sep="\t", index=False)
    print(s, len(g), "targets, lengths", g.seq.str.len().min(), "-", g.seq.str.len().max(), "| with SS:", (g.ss != "").sum())
df.to_csv(f"{OUT}/all.tsv", sep="\t", index=False)
