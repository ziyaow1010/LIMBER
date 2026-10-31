#!/usr/bin/env python3
"""Render the README banner and benchmark figure (light + dark variants).

    python scripts/make_readme_figures.py          # writes assets/*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS = Path(__file__).resolve().parent.parent / "assets"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", blue="#2a78d6", aqua="#1baf7a", gray="#898781",
                  card="#f2f1ec", on_fill="#ffffff"),
    "dark":  dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", blue="#3987e5", aqua="#199e70", gray="#898781",
                  card="#242422", on_fill="#ffffff"),
}

SANS = ["DejaVu Sans"]


def box(ax, x, y, w, h, fc, ec="none", r=0.06, lw=0, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))


def arrow(ax, x0, y0, x1, y1, c, lw=1.4, z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=lw, color=c, shrinkA=0, shrinkB=0, zorder=z))


# ── banner ───────────────────────────────────────────────────────────────────

LAYERS = ["T", "T", "M", "T", "T", "M"]          # one TTM period, drawn twice
PROBES = [
    ("Categorical Jacobian", "contact maps from output-logit sensitivity"),
    ("Embedding perturbation", "layer-wise structural information"),
    ("Spectral family probe", "homologous vs non-homologous pairs"),
    ("Likelihood-ratio fitness", "33 deep mutational scanning assays"),
]
TASKS = [
    ("Secondary structure", "TS0  ·  TestSetB  ·  ArchiveII"),
    ("ncRNA classification", "nRC, 13 classes"),
    ("IRES identification", "DeepIRES bicistronic assay"),
    ("Mean ribosome load", "5'UTR MPRA library"),
    ("RNA-small molecule", "SMRTnet multimodal fusion"),
]


def banner(theme: str):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(14, 5.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.4); ax.axis("off")

    # wordmark
    ax.text(0.45, 4.78, "LIMBER", fontsize=38, fontweight="bold", color=t["ink"],
            family=SANS, va="center")
    ax.text(3.35, 4.90, "Layer-Interleaved Mamba-Bidirectional Encoder for RNA",
            fontsize=13.5, color=t["ink"], family=SANS, va="center")
    ax.text(3.35, 4.56, "A hybrid RNA foundation model pretrained on 40.5 million RNAcentral sequences",
            fontsize=10.5, color=t["ink2"], family=SANS, va="center")
    ax.plot([0.45, 13.55], [4.26, 4.26], lw=1, color=t["grid"], zorder=1)

    # ── panel 1: architecture
    x0 = 0.45
    ax.text(x0, 3.92, "HYBRID ARCHITECTURE", fontsize=9, color=t["muted"],
            family=SANS, fontweight="bold")
    bw, bh, gap = 2.45, 0.235, 0.055
    box(ax, x0, 0.16, bw, 0.28, t["card"])
    ax.text(x0 + bw / 2, 0.30, "5'  A U G [MASK] C  ...  G A U C  3'", fontsize=8,
            color=t["ink2"], family=SANS, ha="center", va="center")

    y = 0.56
    for i, kind in enumerate(LAYERS):                      # TTM x 2, bottom block
        fc = t["blue"] if kind == "T" else t["aqua"]
        box(ax, x0, y, bw, bh, fc)
        ax.text(x0 + bw / 2, y + bh / 2, "Transformer" if kind == "T" else "Mamba",
                fontsize=7.5, color=t["on_fill"], family=SANS,
                ha="center", va="center", zorder=3)
        y += bh + gap
    ax.text(x0 + bw / 2, y + 0.10, "⋮", fontsize=13, color=t["muted"],
            family=SANS, ha="center", va="center")
    y += 0.26
    for kind in LAYERS[:3]:                                 # final TTM period
        fc = t["blue"] if kind == "T" else t["aqua"]
        box(ax, x0, y, bw, bh, fc)
        ax.text(x0 + bw / 2, y + bh / 2, "Transformer" if kind == "T" else "Mamba",
                fontsize=7.5, color=t["on_fill"], family=SANS,
                ha="center", va="center", zorder=3)
        y += bh + gap

    box(ax, x0, y + 0.04, bw, 0.28, t["card"])
    ax.text(x0 + bw / 2, y + 0.18, "masked token prediction", fontsize=8.5,
            color=t["ink2"], family=SANS, ha="center", va="center")
    ax.text(x0 + bw + 0.16, 2.00, "TTM\nrepeat", fontsize=8.5, color=t["muted"],
            family=SANS, ha="left", va="center", linespacing=1.4)

    # ── panel 2: zero-shot framework
    x1 = 4.20
    ax.text(x1, 3.92, "TASK-AGNOSTIC EVALUATION", fontsize=9, color=t["muted"],
            family=SANS, fontweight="bold")
    cw, ch = 4.35, 0.74
    for i, (head, sub) in enumerate(PROBES):
        yy = 3.52 - i * (ch + 0.11) - ch
        box(ax, x1, yy, cw, ch, t["card"])
        ax.add_patch(FancyBboxPatch((x1, yy), 0.055, ch,
                                    boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=t["blue"], ec="none", zorder=3))
        ax.text(x1 + 0.22, yy + ch * 0.62, head, fontsize=10.5, color=t["ink"],
                family=SANS, va="center")
        ax.text(x1 + 0.22, yy + ch * 0.26, sub, fontsize=8.5, color=t["ink2"],
                family=SANS, va="center")

    # ── panel 3: downstream
    x2 = 8.95
    ax.text(x2, 3.92, "DOWNSTREAM APPLICATIONS", fontsize=9, color=t["muted"],
            family=SANS, fontweight="bold")
    cw2, ch2 = 4.60, 0.585
    for i, (head, sub) in enumerate(TASKS):
        yy = 3.52 - i * (ch2 + 0.095) - ch2
        box(ax, x2, yy, cw2, ch2, t["card"])
        ax.add_patch(FancyBboxPatch((x2, yy), 0.055, ch2,
                                    boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=t["aqua"], ec="none", zorder=3))
        ax.text(x2 + 0.22, yy + ch2 * 0.64, head, fontsize=10.5, color=t["ink"],
                family=SANS, va="center")
        ax.text(x2 + 0.22, yy + ch2 * 0.25, sub, fontsize=8.5, color=t["ink2"],
                family=SANS, va="center")

    arrow(ax, 3.48, 2.00, 4.05, 2.00, t["muted"])
    arrow(ax, 8.65, 2.00, 8.84, 2.00, t["muted"])

    fig.savefig(ASSETS / f"limber_banner_{theme}.png", facecolor=t["surface"],
                bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


# ── benchmark panels ─────────────────────────────────────────────────────────

PANELS = [
    ("RNA secondary structure", "bpRNA TR0 -> TS0, median flexible F1", "bar", [
        ("LIMBER-macro", 0.895, True), ("LIMBER-meso", 0.889, True),
        ("LIMBER-nano", 0.870, True), ("AIDO.RNA-650M", 0.851, False),
        ("RiNALMo-giga", 0.851, False), ("structRFM", 0.792, False),
        ("RNA-FM", 0.702, False)], (0.0, 1.0)),
    ("Cross-family generalization", "Rivas TrainSetA -> TestSetB, median flexible F1", "bar", [
        ("LIMBER-macro", 0.816, True), ("AIDO.RNA-650M", 0.810, False),
        ("LIMBER-meso", 0.791, True), ("LIMBER-nano", 0.772, True),
        ("RiNALMo-giga", 0.720, False), ("structRFM", 0.613, False),
        ("RNA-FM", 0.492, False)], (0.0, 1.0)),
    ("ncRNA classification", "nRC 13-class test set, accuracy", "dot", [
        ("LIMBER-macro", 0.992, True), ("LIMBER-meso", 0.992, True),
        ("LIMBER-nano", 0.987, True), ("AIDO.RNA-650M", 0.987, False),
        ("LIMBER-macro (L14)", 0.984, True), ("RiNALMo-giga", 0.979, False),
        ("RNA-FM", 0.965, False), ("structRFM", 0.964, False)], (0.960, 1.000)),
]


def benchmarks(theme: str):
    t = THEMES[theme]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    for ax, (title, sub, kind, rows, xlim) in zip(axes, PANELS):
        ax.set_facecolor(t["surface"])
        names = [r[0] for r in rows]
        ypos = list(range(len(rows)))[::-1]
        span = xlim[1] - xlim[0]

        for yy, (_, v, is_ours) in zip(ypos, rows):
            c = t["blue"] if is_ours else t["gray"]
            if kind == "bar":
                ax.barh(yy, v, height=0.56, color=c, zorder=3)
            else:
                ax.plot([xlim[0], v], [yy, yy], lw=1, color=t["grid"], zorder=2)
                ax.plot([v], [yy], marker="o", ms=9, color=c, zorder=4,
                        mec=t["surface"], mew=2)
            ax.text(v + span * 0.022, yy, f"{v:.3f}", fontsize=9.5, color=t["ink"],
                    family=SANS, va="center", ha="left", zorder=5)

        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=9.5, color=t["ink2"], family=SANS)
        ax.set_xlim(xlim[0], xlim[1] + span * 0.14)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        if kind == "dot":
            ticks = [0.96, 0.97, 0.98, 0.99, 1.00]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{v:.2f}" for v in ticks], fontsize=8.5,
                               color=t["muted"], family=SANS)
            ax.tick_params(axis="x", length=0, pad=4)
        else:
            ax.set_xticks([])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(t["grid"])
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_title(title, fontsize=12.5, color=t["ink"], family=SANS,
                     loc="left", pad=26)
        ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=9,
                color=t["muted"], family=SANS, va="bottom")

    handles = [plt.Line2D([], [], marker="s", ms=8, ls="none", color=t["blue"],
                          label="LIMBER (this work)"),
               plt.Line2D([], [], marker="s", ms=8, ls="none", color=t["gray"],
                          label="Published RNA language models")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.035), labelcolor=t["ink2"],
               prop={"family": "DejaVu Sans", "size": 10})
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ASSETS / f"limber_benchmarks_{theme}.png", facecolor=t["surface"],
                bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


# ── scaling panels ────────────────────────────────────────────────

SCALING = [
    ("Model capacity", "LIMBER at 6 epochs, full corpus",
     "zero-shot structure recovery (normalized AUPR)",
     ["10M", "50M", "100M", "300M", "650M"],
     [0.301, 0.429, 0.590, 0.642, 0.646], "up"),
    ("Pretraining data", "LIMBER-nano at 6 epochs",
     "validation perplexity (lower is better)",
     ["20%", "40%", "60%", "80%", "100%"],
     [1.563, 1.498, 1.467, 1.447, 1.434], "down"),
    ("Training compute", "LIMBER-macro, full corpus",
     "validation perplexity (lower is better)",
     ["3 ep", "6 ep", "9 ep", "12 ep"],
     [1.399, 1.375, 1.365, 1.358], "down"),
]


def scaling(theme: str):
    t = THEMES[theme]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    for ax, (title, sub, ylab, xs, ys, direction) in zip(axes, SCALING):
        ax.set_facecolor(t["surface"])
        x = list(range(len(xs)))
        ax.plot(x, ys, lw=2, color=t["blue"], zorder=3, solid_capstyle="round")
        ax.plot(x, ys, marker="o", ms=9, ls="none", color=t["blue"],
                mec=t["surface"], mew=2, zorder=4)

        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.28 or 0.1
        ax.set_ylim(lo - pad, hi + pad * 1.15)
        for i, (xx, yy) in enumerate(zip(x, ys)):            # label the two ends only
            if i not in (0, len(ys) - 1):
                continue
            up = (direction == "up") == (i == len(ys) - 1)
            ax.annotate(f"{yy:.3f}", (xx, yy), textcoords="offset points",
                        xytext=(0, 11 if up else -19), ha="center",
                        fontsize=10, color=t["ink"], family=SANS, zorder=5)

        ax.set_xticks(x); ax.set_xticklabels(xs, fontsize=9.5, color=t["ink2"], family=SANS)
        ax.set_xlim(-0.35, len(xs) - 0.65)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0, pad=5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        ax.text(0, 1.255, title, transform=ax.transAxes, fontsize=12.5,
                color=t["ink"], family=SANS, va="bottom")
        ax.text(0, 1.145, sub, transform=ax.transAxes, fontsize=9,
                color=t["muted"], family=SANS, va="bottom")
        ax.text(0, 1.045, ylab, transform=ax.transAxes, fontsize=8.5,
                color=t["ink2"], family=SANS, va="bottom", style="italic")

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(ASSETS / f"limber_scaling_{theme}.png", facecolor=t["surface"],
                bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    for th in ("light", "dark"):
        banner(th)
        benchmarks(th)
        scaling(th)
    print("wrote:", *sorted(p.name for p in ASSETS.glob("*.png")))
