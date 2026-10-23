#!/usr/bin/env python3
"""Score folded models against natives: TM-score and RMSD (TM-score program, sequence-dependent
superposition, as in the structRFM evaluation), plus base-pair INF / F1 / DI from barnaba annotation.
Native residues are mapped onto the target sequence by global alignment before comparing pairs."""
import argparse, re, subprocess
from pathlib import Path
import numpy as np, pandas as pd

R = Path("/beacon-projects/rnallm/rna3d")
TMSCORE = "/beacon-homes/ziyaow/RNA_FM/rna3d/Zfold/src/Zfold/RNAeval/TMscore_cpp"


def tmscore(model, native):
    out = subprocess.run([TMSCORE, "-seq", str(model), str(native)], capture_output=True, text=True).stdout
    tm = re.search(r"TM-score\s+=\s+([\d.]+)", out); rmsd = re.search(r"RMSD of\s+the common residues=\s+([\d.]+)", out)
    return (float(tm.group(1)) if tm else np.nan, float(rmsd.group(1)) if rmsd else np.nan)


def pairs(pdb):
    """Base pairs as (resA, resB, annotation) keyed by residue order in the file."""
    import barnaba as bb
    stack, pair, res = bb.annotate(str(pdb))
    resnames = [r.split("_")[0][-1] if "_" in r else r[0] for r in res]
    out = set()
    for (idx, ann) in zip(pair[0][0], pair[0][1]):
        if ann in ("WCc", "GUc"):                       # canonical Watson-Crick and G-U wobble
            out.add(tuple(sorted(idx)))
    return out, res


def seq_of(res):
    return "".join(r.split("_")[0][0] if not r.startswith("_") else "N" for r in res)


def map_to_target(res_seq, target):
    from Bio import Align
    al = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1, open_gap_score=-3, extend_gap_score=-1)
    a = al.align(res_seq, target)[0]
    m = {}
    for (s0, s1), (t0, t1) in zip(*a.aligned):
        for k in range(s1 - s0):
            m[s0 + k] = t0 + k
    return m


def bp_metrics(pred_pdb, native_pdb, target):
    pp, pres = pairs(pred_pdb); npairs, nres = pairs(native_pdb)
    pm, nm = map_to_target(seq_of(pres), target), map_to_target(seq_of(nres), target)
    P = {tuple(sorted((pm[i], pm[j]))) for i, j in pp if i in pm and j in pm}
    N = {tuple(sorted((nm[i], nm[j]))) for i, j in npairs if i in nm and j in nm}
    tp = len(P & N); ppv = tp / len(P) if P else 0; sty = tp / len(N) if N else 0
    f1 = 2 * ppv * sty / (ppv + sty) if ppv + sty else 0
    return dict(INF=np.sqrt(ppv * sty), F1_2D=f1, n_pairs_native=len(N), n_pairs_pred=len(P))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--lm", required=True); a = ap.parse_args()
    t = pd.read_csv(R / "tests/all.tsv", sep="\t")
    rows = []
    for r in t.itertuples():
        pdb = R / "preds" / a.lm / r.set / f"{r.name}.pdb"
        d = dict(lm=a.lm, set=r.set, name=r.name, L=len(r.seq), folded=pdb.exists())
        if pdb.exists():
            d["TM"], d["RMSD"] = tmscore(pdb, r.native)
            try:
                d.update(bp_metrics(pdb, r.native, r.seq))
                d["DI"] = d["RMSD"] / d["INF"] if d["INF"] > 0 else np.nan
            except Exception as e:
                d["bp_error"] = str(e)[:120]
        rows.append(d)
    df = pd.DataFrame(rows); out = R / "preds" / a.lm / "scores.csv"; df.to_csv(out, index=False)
    print(df.groupby("set")[["TM", "RMSD", "INF", "F1_2D"]].mean().round(3)); print("saved", out)
