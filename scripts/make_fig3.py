#!/usr/bin/env python3
"""Redraw Figure 3 (layer-wise logit lens) from the raw layer logits.

Per-layer softmax over the nine digit tokens, averaged over N = 10 trials.
Colours follow the shared scheme: crimson = geometric ground truth,
pale cyan = the reference band.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/mnt/project"
OUT = "/mnt/user-data/outputs/fig3.png"

GT_RED = "#dc143c"       # canonical "geometric ground truth" red
BAND = "#e5ffff"         # canonical reference-band cyan
BAND_EDGE = "#4dd0e1"
BLUE = "#1f77b4"         # the biased response, digit '5'
GREEN = "#2e7d32"        # digit '3', only annotated in v3
GREY = "0.62"

BAND_LO, BAND_HI = 35, 45

PANELS = [
    ("v1", "v1: visual judgment"),
    ("v2", "v2: concept recall"),
    ("v3", "v3: irrelevant prior"),
    ("v4", "v4: visual judgment + numerical cue"),
]


def load(v):
    d = json.load(open(f"{SRC}/{v}_layer_logits.json"))
    m = json.load(open(f"{SRC}/{v}_layer_logits_meta.json"))
    lg = np.array(d["lm_logits_avg"])                       # (n_layers, 9)
    e = np.exp(lg - lg.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True), m


fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.6), dpi=150)

for ax, (v, subtitle) in zip(axes.ravel(), PANELS):
    p, m = load(v)
    digits = m["digits"]
    layers = np.arange(p.shape[0])
    i1, i5, i3 = digits.index("1"), digits.index("5"), digits.index("3")

    ax.axvspan(BAND_LO, BAND_HI, color=BAND, zorder=0)
    for x in (BAND_LO, BAND_HI):
        ax.axvline(x, color=BAND_EDGE, linestyle="--", linewidth=1.0, zorder=1)

    for j, dg in enumerate(digits):
        if j in (i1, i5) or (v == "v3" and j == i3):
            continue
        ax.plot(layers, p[:, j], color=GREY, linewidth=0.9, zorder=2,
                label="other digits (2,3,4,6-9)" if j == digits.index("2") else None)
    if v == "v3":
        ax.plot(layers, p[:, i3], color=GREEN, linewidth=2.0, zorder=3,
                label="digit '3' (early-band peak)")
    ax.plot(layers, p[:, i5], color=BLUE, linewidth=2.4, zorder=4,
            label="digit '5'")
    ax.plot(layers, p[:, i1], color=GT_RED, linewidth=2.4, zorder=5,
            label="digit '1' (ground truth)")

    k = int(p[-1].argmax())
    ax.text(0.985, 0.955,
            f"final-layer argmax: '{digits[k]}' (P = {p[-1, k]:.2f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.6", alpha=0.95), zorder=6)

    if v == "v3":
        peak = int(p[:, i3].argmax())
        ax.annotate(f"early-band peak\ndigit '3' (P = {p[peak, i3]:.2f})",
                    xy=(peak, p[peak, i3]), xytext=(peak + 9, p[peak, i3] + 0.24),
                    fontsize=9, color=GREEN, ha="left",
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1),
                    zorder=6)

    ax.set_title(f"{subtitle}\nCONTINUATION = \"{m['continuation']}\"",
                 fontsize=11, loc="left", fontweight="bold")
    ax.set_xlim(0, p.shape[0] - 1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 10, 20, 30, 40, 50, p.shape[0] - 1])
    ax.set_xlabel("LM layer (early → late)", fontsize=10)
    ax.set_ylabel("softmax probability (per layer, over 9 digits)", fontsize=10)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92,
              bbox_to_anchor=(0.0, 0.94))

fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
