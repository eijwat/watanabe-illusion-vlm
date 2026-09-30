"""
plot_vlm_vs_human.py
====================
Standalone reproduction of paper_v12 Figure 2 (top row) for ANY single VLM run,
compared directly against the individual-level human distribution.

Three panels: Q1 (from top), Q2 (from bottom), Illusion Index II.
  II        = ((Q1 - 1) + (11 - Q2)) / 2          (per observer/trial)
  II_Q1     =  Q1 - 1     (veridical Q1 = 1)
  II_Q2     = 11 - Q2     (veridical Q2 = 11)
Veridical (no illusion): Q1 = 1, Q2 = 11, II = 0  (red vertical line).

Human histogram uses individual-level responses (NO Gaussian approximation),
matching paper_v12. Both series are plotted as DENSITY so the N=133 human and
the N=60 VLM are comparable despite different sample sizes. Group means are
shown as triangle markers under each panel.

This script is intentionally self-contained: it does not import or depend on
any other project file.

Usage
-----
  python plot_vlm_vs_human.py \
      --vlm   results/qwen25vl_n60_a100_q1q2_7b/n60_progress.json \
      --human human_responses.csv \
      --label "Qwen2.5-VL-7B" \
      --out   fig_7b_vs_human.png

  # stats only (no human file needed), prints Q1/Q2/II summary for the VLM:
  python plot_vlm_vs_human.py --vlm .../n60_progress.json --stats-only

Human CSV format
----------------
One row per human observer, with a Q1 (from top, 1-11) column and a Q2
(from bottom, 1-11) column. Column names are auto-detected from common
spellings (q1/Q1/q1_from_top/q1_top ... and q2/...); override with
--human-q1-col / --human-q2-col if needed. Rows outside [1,11] are dropped.
"""

import argparse, csv, json, re, sys
import numpy as np

N_DOTS = 11
VER_Q1, VER_Q2 = 1, 11
ORD = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,
       "seventh":7,"eighth":8,"ninth":9,"tenth":10,"eleventh":11}


# ---------- VLM response parsing (paper-faithful normalization) ----------
def parse_response(text):
    """Return (k, frame) where k in 1..11 and frame in {'top','bottom'}.
    Prefers the quoted final answer; falls back to last 'Nth dot from the X'.
    Handles digits, spelled ordinals, 'last', 'second-to-last', and frame-flips."""
    q = re.findall(r'"the\s+([\w-]+)\s+dot\s+from\s+the\s+(top|bottom)"', text, re.I)
    src = q[-1] if q else None
    if src is None:
        m = re.findall(r'([\w-]+)\s+dot\s+from\s+the\s+(top|bottom)', text, re.I)
        src = m[-1] if m else None
    if src is None:
        return None
    w, frame = src[0].lower(), src[1].lower()
    if w == "last":
        k = N_DOTS
    elif w in ("second-to-last", "secondtolast"):
        k = N_DOTS - 1
    else:
        mn = re.match(r'(\d+)', w)
        k = int(mn.group(1)) if mn else ORD.get(w)
    if k is None or not (1 <= k <= N_DOTS):
        return None
    return k, frame


def to_top(k, frame):    # express as 'from top' index
    return k if frame == "top" else (N_DOTS + 1 - k)


def to_bottom(k, frame):
    return k if frame == "bottom" else (N_DOTS + 1 - k)


def load_vlm(path):
    """Return arrays q1_top, q2_bot, and per-trial II where both are present."""
    data = json.load(open(path))
    q1, q2, ii = [], [], []
    n_trials = 0
    for t in data.get("trials", []):
        if "Q1" not in t or "Q2" not in t:
            continue
        n_trials += 1
        a = parse_response(t["Q1"]["response"])
        b = parse_response(t["Q2"]["response"])
        a_top = to_top(*a) if a else None
        b_bot = to_bottom(*b) if b else None
        if a_top is not None:
            q1.append(a_top)
        if b_bot is not None:
            q2.append(b_bot)
        if a_top is not None and b_bot is not None:
            ii.append(((a_top - 1) + (N_DOTS - b_bot)) / 2.0)
    return (np.array(q1, float), np.array(q2, float), np.array(ii, float),
            data.get("model_id", "VLM"), n_trials)


# ---------- human loader ----------
def _find_col(fieldnames, override, candidates):
    if override:
        if override in fieldnames:
            return override
        sys.exit(f"[ERROR] --human column '{override}' not in CSV header {fieldnames}")
    low = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in low:
            return low[c]
    sys.exit(f"[ERROR] could not auto-detect a column among {candidates} in {fieldnames}. "
             f"Use --human-q1-col / --human-q2-col.")


def load_human(path, q1_col=None, q2_col=None):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        fields = rdr.fieldnames
        c1 = _find_col(fields, q1_col, ["q1", "q1_from_top", "q1_top", "g1", "from_top"])
        c2 = _find_col(fields, q2_col, ["q2", "q2_from_bottom", "q2_bottom", "g2", "from_bottom"])
        q1, q2, ii = [], [], []
        for row in rdr:
            try:
                v1, v2 = float(row[c1]), float(row[c2])
            except (TypeError, ValueError):
                continue
            if not (1 <= v1 <= N_DOTS and 1 <= v2 <= N_DOTS):
                continue
            q1.append(v1); q2.append(v2)
            ii.append(((v1 - 1) + (N_DOTS - v2)) / 2.0)
    return np.array(q1), np.array(q2), np.array(ii), (c1, c2)


def summ(name, a):
    return f"{name}: n={len(a)}  mean={a.mean():.2f}  SD={a.std(ddof=0):.2f}"


# ---------- plotting ----------
def make_figure(vlm, human, label, out, human_label="Human (N=%d)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vq1, vq2, vii, model_id, _ = vlm
    HUMAN_GREEN = "#639922"
    VLM_BLUE = "#185FA5"
    VER_RED = "#A32D2D"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        ("Q1 — from top", vq1, (human[0] if human else None), 1, VER_Q1,
         np.arange(0.5, 12.5, 1.0)),
        ("Q2 — from bottom", vq2, (human[1] if human else None), 1, VER_Q2,
         np.arange(0.5, 12.5, 1.0)),
        ("Illusion Index  II = ((Q1-1)+(11-Q2))/2", vii, (human[2] if human else None),
         0, 0, np.arange(-0.25, 10.5, 0.5)),
    ]
    hlabel = (human_label % len(human[0])) if human else None

    for ax, (title, vdata, hdata, _w, ver, bins) in zip(axes, panels):
        if hdata is not None:
            ax.hist(hdata, bins=bins, density=True, color=HUMAN_GREEN, alpha=0.45,
                    label=hlabel, edgecolor="none")
        ax.hist(vdata, bins=bins, density=True, histtype="step", color=VLM_BLUE,
                linewidth=2.0, label=f"{label} (N={len(vdata)})")
        ax.axvline(ver, color=VER_RED, linestyle="--", linewidth=1.4)
        ax.text(ver, ax.get_ylim()[1] * 0.97, "veridical", color=VER_RED,
                fontsize=8, ha="center", va="top", rotation=90)

        ymin = ax.get_ylim()[0]
        if hdata is not None:
            ax.plot(hdata.mean(), -0.02 * ax.get_ylim()[1], marker="^",
                    color=HUMAN_GREEN, markersize=11, clip_on=False)
        ax.plot(vdata.mean(), -0.02 * ax.get_ylim()[1], marker="^",
                color=VLM_BLUE, markersize=11, clip_on=False)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("dot index" if ver else "II")
        ax.set_ylabel("density")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(f"Watanabe Illusion (104.jpg): {label} vs human", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True, help="VLM n60_progress.json")
    ap.add_argument("--human", help="human responses CSV (omit for --stats-only)")
    ap.add_argument("--human-q1-col", default=None)
    ap.add_argument("--human-q2-col", default=None)
    ap.add_argument("--label", default=None, help="VLM display label")
    ap.add_argument("--out", default="fig_vlm_vs_human.png")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    vlm = load_vlm(args.vlm)
    vq1, vq2, vii, model_id, n_trials = vlm
    label = args.label or model_id.split("/")[-1]

    print(f"== {label}  (file trials={n_trials}) ==")
    print(" ", summ("Q1 from-top   ", vq1), f"(veridical {VER_Q1})")
    print(" ", summ("Q2 from-bottom", vq2), f"(veridical {VER_Q2})")
    print(" ", summ("II            ", vii), "(veridical 0; human II=4.15, SD=2.26)")
    iq1 = vq1 - 1; iq2 = N_DOTS - vq2
    print(f"  II_Q1 mean={iq1.mean():.2f} (human 3.63)   II_Q2 mean={iq2.mean():.2f} (human 4.67)")

    human = None
    if args.human:
        h = load_human(args.human, args.human_q1_col, args.human_q2_col)
        hq1, hq2, hii, cols = h
        human = (hq1, hq2, hii)
        print(f"\n== Human (cols {cols}) ==")
        print(" ", summ("Q1 from-top   ", hq1))
        print(" ", summ("Q2 from-bottom", hq2))
        print(" ", summ("II            ", hii))

    if args.stats_only:
        return
    make_figure(vlm, human, label, args.out)


if __name__ == "__main__":
    main()
