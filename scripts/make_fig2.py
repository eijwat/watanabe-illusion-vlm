#!/usr/bin/env python3
"""Compose Figure 2 (vision encoder probe) from the two source panels.

Takes A.angle from fig_angle_probe.png and C.dot_num from fig_spatial_probe.png,
drops every figure-level and row-level title, rescales the second panel so its
text matches the first, and aligns the two axes boxes vertically.

results/qwen25vl_vit_probe_a100/probe_results.csv
"""
import numpy as np
from PIL import Image

ANGLE = "/mnt/project/fig_angle_probe.png"
SPATIAL = "/mnt/project/fig_spatial_probe.png"
OUT = "/mnt/user-data/outputs/fig2.png"

# crop boxes (left, top, right, bottom) in source pixels, chosen to include the
# panel title, y-label, tick labels and x-label, but no figure-level title
CROP_A = (10, 95, 781, 733)          # A.angle  (axes span x 81-773)
CROP_C = (952, 443, 1424, 838)       # C.dot_num (axes span x 1024-1419)

GAP = 70                             # horizontal gap between panels
MARGIN = 24

def axes_band(img):
    """Row range occupied by the plot box (the tallest dark row group)."""
    d = np.asarray(img.convert("L")) < 160
    rows = d.sum(axis=1)
    nz = np.nonzero(rows > 0)[0]
    groups, s, p = [], nz[0], nz[0]
    for r in nz[1:]:
        if r - p > 3:
            groups.append((s, p))
            s = r
        p = r
    groups.append((s, p))
    return max(groups, key=lambda g: g[1] - g[0])


a = Image.open(ANGLE).convert("RGB").crop(CROP_A)
c = Image.open(SPATIAL).convert("RGB").crop(CROP_C)

# scale the second panel so that its plot box has exactly the same height as
# the first one: the two x-axes then sit at the same level, and the text sizes
# come out within a few percent of each other
ta, ba = axes_band(a)
tc0, bc0 = axes_band(c)
SCALE_C = (ba - ta) / (bc0 - tc0)
c = c.resize((round(c.width * SCALE_C), round(c.height * SCALE_C)),
             Image.LANCZOS)
tc, bc = axes_band(c)
print(f"scale for C: {SCALE_C:.4f}")
print(f"axes band A: {ta}-{ba} (h={ba - ta})")
print(f"axes band C: {tc}-{bc} (h={bc - tc})")

# align the tops of the two plot boxes
off_a = max(0, tc - ta)
off_c = max(0, ta - tc)

W = MARGIN * 2 + a.width + GAP + c.width
H = MARGIN * 2 + max(a.height + off_a, c.height + off_c)
canvas = Image.new("RGB", (W, H), "white")
canvas.paste(a, (MARGIN, MARGIN + off_a))
canvas.paste(c, (MARGIN + a.width + GAP, MARGIN + off_c))
canvas.save(OUT)
print("wrote", OUT, canvas.size)
