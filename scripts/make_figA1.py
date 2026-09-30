#!/usr/bin/env python3
"""Compose the three classical control stimuli into a single Figure A1.

The three source files have different aspect ratios, so each panel is scaled to
a common height and centred in an equal-width cell, with a lettered caption
underneath.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = "/mnt/project"
OUT = "/mnt/user-data/outputs/figA1.png"

PANELS = [
    ("control_muller_lyer.png", "(a) Müller-Lyer"),
    ("control_poggendorff.png", "(b) Poggendorff"),
    ("control_kanizsa.png", "(c) Kanizsa"),
]

PANEL_H = 460          # height of each cell
CELL_W = 620           # width of each cell
GAP = 60               # gap between cells
MARGIN = 30
LABEL_GAP = 16         # space between image and its caption
FONT_SIZE = 30


def get_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


font = get_font(FONT_SIZE)

def crop_to_content(im, margin_frac=0.04):
    """Trim each source file's own white border so the three drawings are
    scaled by their content, not by however much padding they happen to have."""
    d = np.asarray(im.convert("L")) < 240
    ys, xs = np.nonzero(d)
    m = int(round(max(xs.max() - xs.min(), ys.max() - ys.min()) * margin_frac))
    return im.crop((max(0, xs.min() - m), max(0, ys.min() - m),
                    min(im.width, xs.max() + 1 + m),
                    min(im.height, ys.max() + 1 + m)))


imgs = []
for fname, label in PANELS:
    im = crop_to_content(Image.open(f"{SRC}/{fname}").convert("RGB"))
    # fit inside the cell, preserving aspect: wide panels are width-limited,
    # square ones height-limited
    k = min(CELL_W / im.width, PANEL_H / im.height)
    imgs.append((im.resize((round(im.width * k), round(im.height * k)),
                           Image.LANCZOS), label))

cell_w = CELL_W
label_h = FONT_SIZE + 10
W = MARGIN * 2 + cell_w * 3 + GAP * 2
H = MARGIN * 2 + PANEL_H + LABEL_GAP + label_h

canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)

for i, (im, label) in enumerate(imgs):
    cx = MARGIN + i * (cell_w + GAP) + cell_w // 2
    cy = MARGIN + (PANEL_H - im.height) // 2
    canvas.paste(im, (cx - im.width // 2, cy))
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.text((cx - (bbox[2] - bbox[0]) // 2, MARGIN + PANEL_H + LABEL_GAP),
              label, fill="black", font=font)

canvas.save(OUT)
print("wrote", OUT, canvas.size)
for im, label in imgs:
    print(f"  {label}: {im.width}x{im.height}")
