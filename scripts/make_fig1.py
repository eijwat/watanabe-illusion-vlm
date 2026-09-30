#!/usr/bin/env python3
"""Regenerate Figure 1 for the Watanabe Illusion paper.

Layout: left column = stimulus (a) and generative probe (b), both rendered at
identical width; right column = Q1 / Q2 / II dot-whisker panels (c-e).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

STATS = "/mnt/project/stats_extended.json"
IMG_A = "/mnt/project/104.jpg"
IMG_B = "/mnt/project/104_jpg_ChatGPT_Image.png"
OUT = "/mnt/user-data/outputs/fig1.png"

GT_RED = "#dc143c"   # canonical "geometric ground truth" red, shared by all figures

SOURCES = [
    ("human",        "Human",             "#2e7d4f"),
    ("qwen_a100",    "Qwen2.5-VL-32B",    "#1f77b4"),
    ("qwen_spark",   "Qwen2.5-VL-7B",     "#ff7f0e"),
    ("sonnet46",     "Claude Sonnet 4.6", "#d62728"),
    ("gpt55",        "GPT-5.5 Instant",   "#9467bd"),
    ("gemini35flash","Gemini 3.5 Flash",  "#17becf"),
]
PANELS = [("Q1", "(c) Q1 (from top)", 1.0),
          ("Q2", "(d) Q2 (from bottom)", 11.0),
          ("II", "(e) Illusion Index II", 0.0)]

stats = json.load(open(STATS))

FW, FH = 15.5, 7.6
fig = plt.figure(figsize=(FW, FH), dpi=150)

# ---- left column: both images at identical width -------------------------
MARGIN = 0.02                      # same white margin on both sides
LEFT, WIDTH = MARGIN, 0.385        # identical width for (a) and (b)
TITLE_PAD = 0.045                  # space reserved for each panel title

def load_cropped(path, margin_frac=0.02):
    """Crop to the drawn content, then re-pad by a uniform margin, so that both
    panels show their figure at the same width regardless of the white space
    each source file happens to carry."""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    dark = np.asarray(im.convert("L")) < 128
    ys, xs = np.nonzero(dark)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    m = int(round((x1 - x0) * margin_frac))
    x0 = max(0, x0 - m); x1 = min(a.shape[1] - 1, x1 + m)
    y0 = max(0, y0 - m); y1 = min(a.shape[0] - 1, y1 + m)
    return a[y0:y1 + 1, x0:x1 + 1]


imgs = [(load_cropped(IMG_A), "(a) Watanabe Illusion stimulus (104.jpg)"),
        (load_cropped(IMG_B), "(b) Generative probe")]

# height each axes needs so that the image is undistorted at WIDTH
heights = [WIDTH * (im.shape[0] / im.shape[1]) * (FW / FH) for im, _ in imgs]

RH, RGAP = 0.215, 0.075            # height and gap of the right-hand panels

def frame_rows(im):
    """First and last rows of drawn content, as fractions of the panel height."""
    dark = np.asarray(Image.fromarray(im).convert("L")) < 128
    rows = np.nonzero(dark.any(axis=1))[0]
    return rows[0] / im.shape[0], rows[-1] / im.shape[0]


block = sum(heights) + 2 * TITLE_PAD
gap = 0.10
top = 0.485 + (block + gap) / 2      # vertical placement of panel (a)

# (a) sits at the top of the left column; the top line of its frame fixes the
# top of panel (c), which in turn fixes the bottom of panel (e)
y_a = top - TITLE_PAD - heights[0]
r0_a, _ = frame_rows(imgs[0][0])
frame_top = y_a + heights[0] * (1 - r0_a)

rtop = frame_top
e_bottom = rtop - 2 * (RH + RGAP) - RH

# (b) is then raised so that the bottom line of its frame meets panel (e)
_, r1_b = frame_rows(imgs[1][0])
y_b = e_bottom - heights[1] * (1 - r1_b)

for (im, title), h, yy in zip(imgs, heights, [y_a, y_b]):
    ax = fig.add_axes([LEFT, yy, WIDTH, h])
    ax.imshow(im, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=11, pad=6, loc="center")

# ---- right column: dot-whisker panels ------------------------------------
RL, RW = 0.555, 1.0 - MARGIN - 0.555

for i, (key, title, veridical) in enumerate(PANELS):
    ax = fig.add_axes([RL, rtop - i * (RH + RGAP) - RH, RW, RH])
    ypos = list(range(len(SOURCES)))[::-1]
    for yv, (src, label, colour) in zip(ypos, SOURCES):
        st = stats[src]["stats"][key]
        ax.errorbar(st["mean"], yv, xerr=st["sd"], fmt="o", color=colour,
                    ecolor=colour, elinewidth=1.4, capsize=3, markersize=7)
    ax.axvline(veridical, color=GT_RED, linewidth=1.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([s[1] for s in SOURCES], fontsize=9.5)
    ax.set_ylim(-0.7, len(SOURCES) - 0.3)
    ax.set_xlim(-0.6, 11.6)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis="x", labelsize=9.5)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, pad=6)

ax.set_xlabel("response (dots) — mean ± SD; red line = veridical", fontsize=10)

fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
for (im, t), h in zip(imgs, heights):
    print(f"  {t}: {im.shape[1]}x{im.shape[0]} -> axes h={h:.3f}, w={WIDTH}")
