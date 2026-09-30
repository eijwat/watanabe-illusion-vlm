"""Measure the Watanabe Illusion stimulus geometry from the image file itself.

Usage: python3 measure_stimulus_v2.py <path to 104.jpg>

v2: removes the rectangle frame before fitting the circle. In v1 the frame
lines were included in the circle fit, which biased the centre and radius and
therefore corrupted the chord angle.
"""
import sys, math
import numpy as np
from PIL import Image
from scipy import ndimage

path = sys.argv[1]
im = Image.open(path)
a = np.asarray(im.convert("L")).astype(float)
dark = a < 128
H, W = dark.shape
print("size: %s   file: %s" % (im.size, path))

ys, xs = np.nonzero(dark)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
print("content bbox: x %d-%d  y %d-%d  (w=%d h=%d)" % (x0, x1, y0, y1, x1 - x0, y1 - y0))

# ---- 1. locate the dot column (rightmost dense band) -------------------
colsum = dark.sum(axis=0)
xd = int(np.argmax(colsum[x1 - 12:x1 + 1])) + x1 - 12
band = dark[:, xd - 6:xd + 7].copy()
band[:, 4:9] = False                      # kill the vertical frame line
lab, n = ndimage.label(band)
sz = ndimage.sum(band, lab, range(1, n + 1))
ct = ndimage.center_of_mass(band, lab, range(1, n + 1))
cy = sorted(c[0] for c, s in zip(ct, sz) if s > 15)
dots = [float(np.mean(cy[i:i + 2])) for i in range(0, len(cy), 2)]
spacing = np.diff(dots)
print("dot column x = %d" % xd)
print("n_dots: %d" % len(dots))
print("dot y: %s" % [round(d, 2) for d in dots])
print("dot spacing: mean %.3f  sd %.3f" % (spacing.mean(), spacing.std(ddof=1)))

# ---- 2. remove the rectangle frame -------------------------------------
work = dark.copy()
rowsum = work.sum(axis=1)
colsum2 = work.sum(axis=0)
frame_rows = np.nonzero(rowsum > 0.5 * (x1 - x0))[0]
frame_cols = np.nonzero(colsum2 > 0.5 * (y1 - y0))[0]
work[frame_rows, :] = False
work[:, frame_cols] = False
print("frame rows removed: %s   frame cols removed: %s"
      % (list(frame_rows), list(frame_cols)))

# ---- 3. isolate the circle component (lower-left) -----------------------
lab2, n2 = ndimage.label(work, structure=np.ones((3, 3)))
sizes = ndimage.sum(work, lab2, range(1, n2 + 1))
coms = ndimage.center_of_mass(work, lab2, range(1, n2 + 1))
cand = [(i + 1, s, c) for i, (s, c) in enumerate(zip(sizes, coms))
        if s > 50 and c[1] < 0.4 * W]
cand.sort(key=lambda t: -t[1])
if not cand:
    sys.exit("circle component not found")
cid = cand[0][0]
yy, xx = np.nonzero(lab2 == cid)
pts = np.c_[xx.astype(float), yy.astype(float)]
print("circle component: %d px  centroid (%.1f, %.1f)"
      % (len(pts), cand[0][2][1], cand[0][2][0]))

# ---- 4. algebraic circle fit, then chord --------------------------------
x, y = pts[:, 0], pts[:, 1]
sol = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones(len(x))], x ** 2 + y ** 2,
                      rcond=None)[0]
ccx, ccy = sol[0], sol[1]
r = math.sqrt(sol[2] + ccx ** 2 + ccy ** 2)
d = np.hypot(x - ccx, y - ccy)
inner = pts[d < r - 3.5]
print("circle: cx=%.2f cy=%.2f r=%.2f" % (ccx, ccy, r))

# total-least-squares line fit on the chord
mx, my = inner[:, 0].mean(), inner[:, 1].mean()
u = inner - [mx, my]
_, _, vt = np.linalg.svd(u, full_matrices=False)
dxv, dyv = vt[0]
slope = dyv / dxv
ang = math.degrees(math.atan(-slope))
print("chord: n=%d  x %.1f-%.1f  angle above horizontal = %.3f deg"
      % (len(inner), inner[:, 0].min(), inner[:, 0].max(), ang))

# ---- 5. extrapolate to the dot column ----------------------------------
yhit = my + slope * (xd - mx)
k = int(np.argmin([abs(yhit - dd) for dd in dots]))
print("y at dot column: %.2f" % yhit)
print("nearest dot from top: %d   (dot %d y=%.2f, offset %.2f px = %.3f dot units)"
      % (k + 1, k + 1, dots[k], yhit - dots[k], (yhit - dots[k]) / spacing.mean()))
print("continuous dot index: %.3f" % (1 + (yhit - dots[0]) / spacing.mean()))
