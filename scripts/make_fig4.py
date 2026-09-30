#!/usr/bin/env python3
"""Redraw Figure 4 (prompt cancellation) from cancellation_stats.json.

All bars share the neutral grey of the P0 control; the figure-level title is
omitted so the caption carries the description.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS = "/mnt/project/cancellation_stats.json"
OUT = "/mnt/user-data/outputs/fig4.png"

LABELS = {
    "P0": "P0 Control",
    "P1": 'P1 "Watanabe Illusion"',
    "P2": 'P2 "visual illusion"',
    "P3": "P3 Suppress prior",
    "P4": "P4 Counter-hint (top)",
    "P5": "P5 Force visual CoT",
    "P6": "P6 GT disclosure",
    "P7": "P7 Random distractor",
}
BAR = "#6e6e6e"          # the P0 control grey, used for every bar
BAND = "#e5ffff"         # same pale cyan as the mid-layer band in Figure 3
GT_RED = "#dc143c"       # canonical "geometric ground truth" red, shared by all figures
GT = 1                   # veridical response

d = json.load(open(STATS))
pp = d["per_prompt"]
keys = [k for k in LABELS if k in pp]

base = pp["P0"]["mean"]
base_sd = pp["P0"]["sd"]

fig, ax = plt.subplots(figsize=(15.5, 7.0), dpi=150)

# P0 reference band and line
ax.axhspan(base - base_sd, base + base_sd, color=BAND, zorder=0,
           label="P0 ± 1 SD")
ax.axhline(base, color="0.35", linestyle="--", linewidth=1.4, zorder=1,
           label=f"P0 baseline mean = {base:.2f}")
ax.axhline(GT, color=GT_RED, linewidth=2.2, zorder=6,
           label="Ground truth = 1 (topmost dot)")

rng = np.random.default_rng(0)
for i, k in enumerate(keys):
    st = pp[k]
    ax.bar(i, st["mean"], width=0.62, color=BAR, edgecolor="black",
           linewidth=0.7, zorder=2)
    ax.errorbar(i, st["mean"], yerr=st["sd"], fmt="none", ecolor="black",
                elinewidth=1.5, capsize=5, zorder=4)
    ax.text(i, st["mean"] + st["sd"] + 0.22, f"{st['mean']:.2f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold", zorder=5)

    # individual trials, jittered
    vals = []
    for v, c in st["distribution"].items():
        vals += [int(v)] * c
    x = i + rng.uniform(-0.22, 0.22, len(vals))
    ax.scatter(x, vals, s=30, color="0.10", alpha=0.85,
               edgecolors="white", linewidths=0.6, zorder=3)

ax.set_xticks(range(len(keys)))
ax.set_xticklabels([LABELS[k] for k in keys], rotation=18, ha="right",
                   fontsize=12)
ax.set_ylabel("Q1 response (dot # from top)", fontsize=13)
ax.set_ylim(0, 8.6)
ax.set_yticks(range(0, 9))
ax.tick_params(axis="y", labelsize=12)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(False)
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
for k in keys:
    print(f"  {k}: mean={pp[k]['mean']:.2f} sd={pp[k]['sd']:.2f} n={pp[k]['n']}")
