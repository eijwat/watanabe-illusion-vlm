#!/usr/bin/env python3
"""
analyze_commercial_vlm.py
==========================
Watanabe Illusion behavioral data: a six-subplot figure that places
the manual commercial-VLM data (Sonnet 4.6 / GPT-5.5 Instant / Gemini 3.5
Flash, N=30 each, Web UI) alongside the Qwen2.5-VL-32B replications
(A100 / DGX Spark, N=60 each) and the human reference (N=132), and an
extended Table 2 (CSV + Markdown) with all six data sources.

Layout:
                Q1          Q2          II
   Top row:    [Human + Qwen A100 + Qwen Spark]      <- paper_v8 Fig 2
   Bottom row: [Human + Sonnet46 + GPT55 + Gemini35Flash]  <- new

Inputs (defaults; override via CLI):
  - n60_progress_a100.json        (Qwen on A100, N=60)
  - n60_progress_spark.json       (Qwen on DGX Spark, N=60)
  - n60_progress_sonnet46.json    (manual, N=30)
  - n60_progress_gpt55.json       (manual, N=30)
  - n60_progress_gemini35flash.json (manual, N=30)
  - human_2nd_test.csv            (N=132 wide-format Q1/Q2/Q3)

Outputs:
  - fig_six_panel_commercial_vs_qwen_vs_human.png  + .pdf
  - table2_extended.csv
  - table2_extended.md
  - stats_extended.json

Usage:
  python analyze_commercial_vlm.py \\
      --qwen-a100 n60_progress_a100.json \\
      --qwen-spark n60_progress_spark.json \\
      --sonnet46 results/manual/n60_progress_sonnet46.json \\
      --gpt55    results/manual/n60_progress_gpt55.json \\
      --gemini35flash results/manual/n60_progress_gemini35flash.json \\
      --human human_2nd_test.csv \\
      --out-dir figures/

Author: Watanabe lab, NIBB
Date: 2026-05-26
"""

import argparse
import json
import csv
import re
import sys
from pathlib import Path
from collections import OrderedDict
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ========== Constants ==========

N_DOTS = 11
GT_Q1 = 1     # 104.jpg: line hits the 1st dot from the top
GT_Q2 = 11    # equivalently the 11th from the bottom
GT_SUM = N_DOTS + 1  # 12

# Illusion Index II definitions (paper_v8 §2.4):
#   IIQ1 = Q1 - 1
#   IIQ2 = 11 - Q2
#   II   = (IIQ1 + IIQ2) / 2   (when both Q1 and Q2 are present)


# ========== Number extraction (re-used from analyze_n60_q1q2.py) ==========

SPELLED_MAP = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12,
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
    'eleventh': 11, 'twelfth': 12,
}
SPELLED_REGEX = '|'.join(SPELLED_MAP.keys())


def extract_number_and_direction(text, expected_direction=None):
    """Extract (number, direction) from a model response.

    direction is 'top' / 'bottom' / None
    number is int or None

    If expected_direction is given ("top" or "bottom"), and the response
    contains multiple "Xth dot from the top|bottom" patterns, prefer
    the one that matches expected_direction. This handles responses
    like:
        "**the 3rd dot from the top** (or about the 8th from the bottom)"
    where, if Q2 asked for "from the bottom", we want 8, not 3.

    Priority:
      1. All "Xth dot from the top|bottom" matches; if expected_direction
         is set, prefer matches with that direction; otherwise use last.
      2. \\boxed{...top|bottom...} with embedded number
      3. \\boxed{N} + nearby direction word
      4. fallback: last "Xth dot" with no direction
    """
    if not text:
        return None, None
    t = text

    # 1a. Collect ALL "the Xth dot from the top|bottom" matches (digit form)
    digit_matches = list(re.finditer(
        rf"\b(\d+)(?:st|nd|rd|th)?\s+dot\s+from\s+(?:the\s+)?(top|bottom)\b",
        t, re.IGNORECASE,
    ))
    # 1b. ... and spelled form
    spelled_matches = list(re.finditer(
        rf"\b({SPELLED_REGEX})\s+dot\s+from\s+(?:the\s+)?(top|bottom)\b",
        t, re.IGNORECASE,
    ))

    candidates = []  # list of (value, direction, position_in_text)
    for m in digit_matches:
        candidates.append((int(m.group(1)), m.group(2).lower(), m.start()))
    for m in spelled_matches:
        candidates.append((SPELLED_MAP[m.group(1).lower()],
                           m.group(2).lower(), m.start()))

    if candidates:
        # Sort by position so we can reason about order
        candidates.sort(key=lambda c: c[2])
        if expected_direction is not None:
            # Prefer matches with expected direction
            matched = [c for c in candidates if c[1] == expected_direction]
            if matched:
                # Among expected-direction matches, take the last one
                # (final answer usually appears last in chain-of-thought).
                v, d, _ = matched[-1]
                return v, d
        # No expected direction or no matching: take the last candidate
        v, d, _ = candidates[-1]
        return v, d

    # 2. \boxed{...top|bottom...}
    m = re.search(r"\\boxed\{([^}]*?(?:top|bottom)[^}]*?)\}", t, re.IGNORECASE)
    if m:
        inner = m.group(1)
        direction = "top" if "top" in inner.lower() else "bottom"
        n_match = re.search(rf"\b(\d+)(?:st|nd|rd|th)?\b", inner)
        if n_match:
            return int(n_match.group(1)), direction
        sp_match = re.search(rf"\b({SPELLED_REGEX})\b", inner, re.IGNORECASE)
        if sp_match:
            return SPELLED_MAP[sp_match.group(1).lower()], direction

    # 3. \boxed{N} + nearby direction
    m = re.search(r"\\boxed\{(\d+)\}", t)
    if m:
        n = int(m.group(1))
        pos = m.start()
        window = t[max(0, pos - 200):pos + 200].lower()
        if "from the top" in window or "from top" in window:
            return n, "top"
        if "from the bottom" in window or "from bottom" in window:
            return n, "bottom"
        return n, None

    # 4. fallback: last "Xth dot"
    matches = list(re.finditer(r"\b(\d+)(?:st|nd|rd|th)\s+dot\b", t.lower()))
    if matches:
        return int(matches[-1].group(1)), None

    return None, None


# ========== Loading: Qwen JSON ==========

def load_qwen_progress(path):
    """Load a Qwen n60_progress_*.json and return parsed (Q1_list, Q2_list)
    with direction normalization (Q1=top, Q2=bottom convention).

    Trials where extraction fails are skipped (with a warning print).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    q1_list = []
    q2_list = []
    skipped = []

    for trial in data["trials"]:
        idx = trial.get("trial_idx", "?")

        for q_label in ("Q1", "Q2"):
            if q_label not in trial:
                continue
            q_block = trial[q_label]
            # canonical direction for this question
            expected = "top" if q_label == "Q1" else "bottom"

            # Manual-log JSONs include parsed_value directly; but it was
            # generated with the simpler parser in parse_manual_log.py,
            # so we RE-parse from the raw response with the more robust
            # extractor here, which can handle "the 3rd dot from the top
            # (or about the 8th from the bottom)" style responses.
            resp = q_block.get("response", "") or ""
            value, direction = extract_number_and_direction(
                resp, expected_direction=expected,
            )
            if value is None:
                # Last fallback: trust the upstream parsed_value if present
                if "parsed_value" in q_block and q_block["parsed_value"] is not None:
                    value = int(q_block["parsed_value"])
                    direction = q_block.get("direction_detected")
                else:
                    skipped.append((path.name, idx, q_label, "extract_failed"))
                    continue

            # Direction normalization to canonical convention
            #   Q1 should be "from the top"
            #   Q2 should be "from the bottom"
            if direction is not None and direction != expected:
                # Flip: value becomes N_DOTS - value + 1
                value = N_DOTS - value + 1

            if not (1 <= value <= N_DOTS):
                skipped.append((path.name, idx, q_label, f"out_of_range:{value}"))
                continue

            if q_label == "Q1":
                q1_list.append(value)
            else:
                q2_list.append(value)

    if skipped:
        print(f"  [load] {path.name}: {len(skipped)} skipped trials")
        for s in skipped[:5]:
            print(f"    {s}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
    print(f"  [load] {path.name}: Q1={len(q1_list)} Q2={len(q2_list)}")

    return q1_list, q2_list


def load_human_csv(path):
    """Load human_2nd_test.csv (wide format Q1,Q2,Q3,...), apply listwise
    deletion for missing Q1 or Q2.
    """
    q1_list = []
    q2_list = []
    with open(path, "r", encoding="utf-8-sig") as f:
        # utf-8-sig handles the BOM
        reader = csv.DictReader(f)
        for row in reader:
            q1_raw = row.get("Q1", "").strip()
            q2_raw = row.get("Q2", "").strip()
            if not q1_raw or not q2_raw:
                continue
            try:
                q1 = float(q1_raw)
                q2 = float(q2_raw)
            except ValueError:
                continue
            if not (1 <= q1 <= N_DOTS and 1 <= q2 <= N_DOTS):
                continue
            q1_list.append(q1)
            q2_list.append(q2)

    print(f"  [load] {path.name}: Q1={len(q1_list)} Q2={len(q2_list)} (listwise complete)")
    return q1_list, q2_list


# ========== Per-source statistics ==========

def compute_stats(q1, q2):
    """Compute distribution stats for a single source.
    Returns a dict with Q1/Q2/IIQ1/IIQ2/II/Sum mean, SD, median, range, n.
    """
    q1 = np.array(q1, dtype=float)
    q2 = np.array(q2, dtype=float)

    n_q1 = len(q1)
    n_q2 = len(q2)

    iiq1 = q1 - GT_Q1
    iiq2 = GT_Q2 - q2

    # II requires paired Q1+Q2 from the same trial; for human, q1 and q2
    # are aligned (same participant); for Qwen, trials are aligned by index
    # which we've maintained by appending in order. We take min(n_q1, n_q2)
    # for II computation.
    n_paired = min(n_q1, n_q2)
    ii = (iiq1[:n_paired] + iiq2[:n_paired]) / 2.0
    sum_q = q1[:n_paired] + q2[:n_paired]

    def summary(arr):
        if len(arr) == 0:
            return {
                "mean": None, "sd": None, "median": None,
                "min": None, "max": None, "n": 0,
            }
        return {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(len(arr)),
        }

    return {
        "Q1": summary(q1),
        "Q2": summary(q2),
        "IIQ1": summary(iiq1),
        "IIQ2": summary(iiq2),
        "II": summary(ii),
        "Sum": summary(sum_q),
        "n_q1": n_q1,
        "n_q2": n_q2,
        "n_paired": n_paired,
    }


# ========== Plotting ==========

def plot_density(ax, values, color, label, bins, linestyle="-",
                 linewidth=2.0, alpha=1.0, filled=False):
    """Plot a step-histogram normalized to density on `ax`.

    If filled=True, fills the histogram (used for the human reference).
    """
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return None
    if filled:
        ax.hist(values, bins=bins, density=True, color=color, alpha=alpha,
                label=label, edgecolor="none")
    else:
        # Step histogram via plt.hist with histtype='step'
        ax.hist(values, bins=bins, density=True, color=color,
                histtype="step", linewidth=linewidth, label=label,
                linestyle=linestyle)


def plot_mean_tick(ax, mean_value, color, y_position=-0.04, marker="v",
                   markersize=9):
    """Plot a small downward marker below the x-axis at the mean value,
    in the same color as that source's line. Multiple sources stack
    vertically (one tier per source) below the axis.
    """
    if mean_value is None:
        return
    ax.plot(mean_value, y_position, marker=marker, color=color,
            markersize=markersize, clip_on=False,
            markeredgecolor="black", markeredgewidth=0.5)


def make_six_panel_figure(sources, out_path_png, out_path_pdf):
    """Build the 2x3 figure.

    Layout: 2 rows x 3 cols of histogram axes, plus a thin "tick strip"
    below each histogram showing the mean of each source as a colored
    triangle. The tick strip uses a separate axes to avoid clashing with
    the histogram's y-axis.

    sources: ordered dict
      {key: {"q1": [...], "q2": [...], "color": "#...",
             "label": "...", "tier": "top"|"bottom"|"both"}}
    """
    # Two-level GridSpec: outer 2 rows (with comfortable spacing between
    # top and bottom panels), inner 2 rows per panel (histogram + tick
    # strip, kept close together).
    fig = plt.figure(figsize=(13.5, 9.0))
    outer = fig.add_gridspec(
        nrows=2, ncols=1,
        hspace=0.40,
        left=0.06, right=0.83, top=0.93, bottom=0.06,
    )

    hist_axes = [[None, None, None], [None, None, None]]
    tick_axes = [[None, None, None], [None, None, None]]
    for row in range(2):
        inner = outer[row].subgridspec(
            nrows=2, ncols=3,
            height_ratios=[6, 0.9],
            hspace=0.05, wspace=0.18,
        )
        for col in range(3):
            hist_axes[row][col] = fig.add_subplot(inner[0, col])
            tick_axes[row][col] = fig.add_subplot(inner[1, col],
                                                  sharex=hist_axes[row][col])

    # Bins
    bins_q = np.arange(0.5, N_DOTS + 1.5 + 0.001, 1.0)  # 0.5, 1.5, ..., 11.5
    bins_ii = np.linspace(0, 10, 21)                     # 0, 0.5, ..., 10

    panel_specs = [
        ("Q1", "Q1 (dot from top)", bins_q, GT_Q1, "veridical Q1=1", (0.5, 11.5)),
        ("Q2", "Q2 (dot from bottom)", bins_q, GT_Q2, "veridical Q2=11", (0.5, 11.5)),
        ("II", "Illusion Index II", bins_ii, 0, "no illusion", (-0.3, 10.3)),
    ]

    row_titles = [
        "Open-weights (Qwen2.5-VL-32B) vs Human",
        "Commercial VLMs (Web UI, default mode) vs Human",
    ]

    for row_idx, tier in enumerate(["top", "bottom"]):
        row_sources = [
            (k, v) for k, v in sources.items()
            if v["tier"] == tier or v["tier"] == "both"
        ]
        # Plot "both" (Human) first as background filled hist
        row_sources.sort(key=lambda kv: 0 if kv[1]["tier"] == "both" else 1)

        for col_idx, (key, label_x, bins, gt_x, gt_label, xlim) in enumerate(panel_specs):
            ax = hist_axes[row_idx][col_idx]
            tax = tick_axes[row_idx][col_idx]

            max_density = 0.0

            # Plot each source histogram
            for src_key, src in row_sources:
                q1 = src["q1"]
                q2 = src["q2"]

                if key == "Q1":
                    values = q1
                elif key == "Q2":
                    values = q2
                else:  # II
                    n_paired = min(len(q1), len(q2))
                    if n_paired == 0:
                        continue
                    q1a = np.array(q1[:n_paired], dtype=float)
                    q2a = np.array(q2[:n_paired], dtype=float)
                    values = ((q1a - GT_Q1) + (GT_Q2 - q2a)) / 2.0

                filled = (src["tier"] == "both")
                alpha = 0.30 if filled else 1.0
                lw = 1.2 if filled else 2.0
                ls = src.get("linestyle", "-")

                plot_density(
                    ax, values,
                    color=src["color"],
                    label=src["label"],
                    bins=bins,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=alpha,
                    filled=filled,
                )

                # Track maximum density for axis scaling
                if len(values) > 0:
                    hist_counts, _ = np.histogram(values, bins=bins, density=True)
                    if hist_counts.max() > max_density:
                        max_density = hist_counts.max()

            # Set x-limits
            ax.set_xlim(xlim)
            # Slightly above the max density
            if max_density > 0:
                ax.set_ylim(0, max_density * 1.18)

            # Veridical / no-illusion vertical reference
            ax.axvline(gt_x, color="#d62728", linewidth=1.5, alpha=0.7,
                       zorder=-1)
            # Text near top of axis, horizontal, small font
            ylim = ax.get_ylim()
            ax.text(
                gt_x + (xlim[1] - xlim[0]) * 0.01,
                ylim[1] * 0.95,
                gt_label,
                color="#d62728", fontsize=8, va="top", ha="left",
            )

            # Labels and styling for histogram
            ax.tick_params(labelsize=9, labelbottom=False)  # bottom labels on tick strip
            if col_idx == 0:
                ax.set_ylabel("density", fontsize=10)
            ax.grid(True, alpha=0.25)

            # ---- Tick strip below ----
            tax.set_xlim(xlim)
            tax.set_ylim(-0.5, len(row_sources) - 0.5)
            tax.set_yticks([])
            tax.set_xlabel(label_x, fontsize=10)
            tax.tick_params(labelsize=9)

            # One vertical row per source, each at a different y level
            for tier_count, (src_key, src) in enumerate(row_sources):
                stats_src = src["_stats"]
                if key in ("Q1", "Q2"):
                    m = stats_src[key]["mean"]
                else:
                    m = stats_src["II"]["mean"]
                if m is None:
                    continue
                y_level = tier_count
                # Vertical line marker
                tax.plot(
                    [m, m], [y_level - 0.35, y_level + 0.35],
                    color=src["color"], linewidth=2.5, solid_capstyle="round",
                )
                # Small text label of the mean value
                tax.text(
                    m, y_level + 0.45,
                    f"{m:.2f}",
                    color=src["color"], fontsize=7, va="bottom", ha="center",
                    fontweight="bold",
                )

            # Faint horizontal separator between source tracks
            for tier_count in range(len(row_sources)):
                tax.axhline(tier_count - 0.5, color="gray", linewidth=0.3, alpha=0.4)

            # Reference vertical line on tick strip too
            tax.axvline(gt_x, color="#d62728", linewidth=1.0, alpha=0.5)

            # Hide spines on tick strip for cleanness
            for spine_name in ("top", "right", "left"):
                tax.spines[spine_name].set_visible(False)

        # Row title above the row's middle histogram
        hist_axes[row_idx][1].set_title(
            row_titles[row_idx], fontsize=11, pad=10, fontweight="bold",
        )

        # Legend to the right of the rightmost histogram in this row
        hist_axes[row_idx][-1].legend(
            loc="center left", bbox_to_anchor=(1.04, 0.5),
            fontsize=9, frameon=True, framealpha=0.95,
        )

    fig.suptitle(
        "Watanabe Illusion (104.jpg): Q1, Q2, and Illusion Index II "
        "across humans, open-weights VLM, and commercial VLMs",
        fontsize=12.5, fontweight="bold", y=0.985,
    )

    fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    print(f"  [plot] wrote {out_path_png}")
    print(f"  [plot] wrote {out_path_pdf}")
    plt.close(fig)


# ========== Table writers ==========

def write_table_csv(sources, out_path):
    """Write Table 2 extended as CSV. Rows = sources, columns = measures."""
    fieldnames = [
        "Source", "N (Q1)", "N (Q2)", "N (paired II)",
        "Q1 mean", "Q1 SD", "Q1 median", "Q1 min", "Q1 max",
        "Q2 mean", "Q2 SD", "Q2 median", "Q2 min", "Q2 max",
        "IIQ1 mean", "IIQ1 SD",
        "IIQ2 mean", "IIQ2 SD",
        "II mean", "II SD", "II median", "II min", "II max",
        "Sum mean", "Sum SD",
    ]

    def fmt(x, digits=2):
        if x is None:
            return ""
        return f"{x:.{digits}f}"

    rows = []
    for key, src in sources.items():
        s = src["_stats"]
        rows.append({
            "Source": src["label"],
            "N (Q1)": s["n_q1"],
            "N (Q2)": s["n_q2"],
            "N (paired II)": s["n_paired"],
            "Q1 mean": fmt(s["Q1"]["mean"]),
            "Q1 SD": fmt(s["Q1"]["sd"]),
            "Q1 median": fmt(s["Q1"]["median"], 1),
            "Q1 min": fmt(s["Q1"]["min"], 1),
            "Q1 max": fmt(s["Q1"]["max"], 1),
            "Q2 mean": fmt(s["Q2"]["mean"]),
            "Q2 SD": fmt(s["Q2"]["sd"]),
            "Q2 median": fmt(s["Q2"]["median"], 1),
            "Q2 min": fmt(s["Q2"]["min"], 1),
            "Q2 max": fmt(s["Q2"]["max"], 1),
            "IIQ1 mean": fmt(s["IIQ1"]["mean"]),
            "IIQ1 SD": fmt(s["IIQ1"]["sd"]),
            "IIQ2 mean": fmt(s["IIQ2"]["mean"]),
            "IIQ2 SD": fmt(s["IIQ2"]["sd"]),
            "II mean": fmt(s["II"]["mean"]),
            "II SD": fmt(s["II"]["sd"]),
            "II median": fmt(s["II"]["median"]),
            "II min": fmt(s["II"]["min"]),
            "II max": fmt(s["II"]["max"]),
            "Sum mean": fmt(s["Sum"]["mean"]),
            "Sum SD": fmt(s["Sum"]["sd"]),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [table] wrote {out_path}")


def write_table_md(sources, out_path):
    """Write Table 2 extended as Markdown.

    Two compact tables: one for Q1/Q2/II central tendency, one for SDs.
    This is paste-ready into paper_v8.md.
    """
    def fmt(x, digits=2):
        if x is None:
            return "—"
        return f"{x:.{digits}f}"

    lines = []
    lines.append("# Table 2 (extended): Watanabe Illusion behavioral statistics")
    lines.append("")
    lines.append("Each row is one data source. Q1=dot from top, Q2=dot from bottom; ")
    lines.append("IIQ1=Q1−1, IIQ2=11−Q2, II=(IIQ1+IIQ2)/2 (paired only). ")
    lines.append(f"Veridical: Q1={GT_Q1}, Q2={GT_Q2}, II=0. ")
    lines.append("All values are dots (Q1/Q2/Sum) or dot-magnitude (II).")
    lines.append("")

    # ---- Central tendency table ----
    lines.append("## Means and medians")
    lines.append("")
    lines.append("| Source | N | Q1 mean | Q1 med | Q2 mean | Q2 med | "
                 "IIQ1 mean | IIQ2 mean | **II mean** | II med | Sum mean |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, src in sources.items():
        s = src["_stats"]
        n_str = f"{s['n_q1']}/{s['n_q2']}/{s['n_paired']}"
        lines.append(
            f"| {src['label']} | {n_str} | "
            f"{fmt(s['Q1']['mean'])} | {fmt(s['Q1']['median'], 1)} | "
            f"{fmt(s['Q2']['mean'])} | {fmt(s['Q2']['median'], 1)} | "
            f"{fmt(s['IIQ1']['mean'])} | {fmt(s['IIQ2']['mean'])} | "
            f"**{fmt(s['II']['mean'])}** | {fmt(s['II']['median'])} | "
            f"{fmt(s['Sum']['mean'])} |"
        )
    lines.append("")
    lines.append("N column shows: Q1 trials / Q2 trials / paired (used for II, Sum).")
    lines.append("")

    # ---- Spread table ----
    lines.append("## Variance and range")
    lines.append("")
    lines.append("| Source | Q1 SD | Q2 SD | IIQ1 SD | IIQ2 SD | **II SD** | "
                 "II range | Sum SD |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, src in sources.items():
        s = src["_stats"]
        ii_range = ""
        if s["II"]["min"] is not None and s["II"]["max"] is not None:
            ii_range = f"[{fmt(s['II']['min'], 1)}, {fmt(s['II']['max'], 1)}]"
        lines.append(
            f"| {src['label']} | "
            f"{fmt(s['Q1']['sd'])} | {fmt(s['Q2']['sd'])} | "
            f"{fmt(s['IIQ1']['sd'])} | {fmt(s['IIQ2']['sd'])} | "
            f"**{fmt(s['II']['sd'])}** | {ii_range} | "
            f"{fmt(s['Sum']['sd'])} |"
        )
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [table] wrote {out_path}")


def write_stats_json(sources, out_path):
    """Write full machine-readable stats."""
    obj = {}
    for key, src in sources.items():
        s = src["_stats"]
        obj[key] = {
            "label": src["label"],
            "tier": src["tier"],
            "stats": s,
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  [stats] wrote {out_path}")


# ========== Main ==========

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qwen-a100", type=str,
                    default="n60_progress_a100.json")
    ap.add_argument("--qwen-7b", type=str, dest="qwen_7b",
                    default="n60_progress_7b.json")
    ap.add_argument("--sonnet46", type=str,
                    default="results/manual/n60_progress_sonnet46.json")
    ap.add_argument("--gpt55", type=str,
                    default="results/manual/n60_progress_gpt55.json")
    ap.add_argument("--gemini35flash", type=str,
                    default="results/manual/n60_progress_gemini35flash.json")
    ap.add_argument("--human", type=str,
                    default="human_2nd_test.csv")
    ap.add_argument("--out-dir", type=str, default="figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    inputs = OrderedDict()

    # Human (both rows reference)
    human_q1, human_q2 = load_human_csv(Path(args.human))
    inputs["human"] = {
        "q1": human_q1, "q2": human_q2,
        "color": "#2ca02c",
        "label": f"Human (N={len(human_q1)})",
        "tier": "both",
    }

    # Qwen A100
    a100_q1, a100_q2 = load_qwen_progress(Path(args.qwen_a100))
    inputs["qwen_a100"] = {
        "q1": a100_q1, "q2": a100_q2,
        "color": "#1f77b4",
        "label": f"Qwen2.5-VL-32B (N={len(a100_q1)})",
        "tier": "top",
    }

    # Qwen 7B (model-size axis, replaces former DGX Spark hardware axis)
    q7_q1, q7_q2 = load_qwen_progress(Path(args.qwen_7b))
    inputs["qwen_7b"] = {
        "q1": q7_q1, "q2": q7_q2,
        "color": "#ff7f0e",
        "label": f"Qwen2.5-VL-7B (N={len(q7_q1)})",
        "tier": "top",
        "linestyle": "--",
    }

    # Commercial: Sonnet 4.6
    sonnet_q1, sonnet_q2 = load_qwen_progress(Path(args.sonnet46))
    inputs["sonnet46"] = {
        "q1": sonnet_q1, "q2": sonnet_q2,
        "color": "#d62728",
        "label": f"Claude Sonnet 4.6 (N={len(sonnet_q1)})",
        "tier": "bottom",
    }

    # Commercial: GPT-5.5 Instant
    gpt_q1, gpt_q2 = load_qwen_progress(Path(args.gpt55))
    inputs["gpt55"] = {
        "q1": gpt_q1, "q2": gpt_q2,
        "color": "#9467bd",
        "label": f"GPT-5.5 Instant (N={len(gpt_q1)})",
        "tier": "bottom",
    }

    # Commercial: Gemini 3.5 Flash
    gemini_q1, gemini_q2 = load_qwen_progress(Path(args.gemini35flash))
    inputs["gemini35flash"] = {
        "q1": gemini_q1, "q2": gemini_q2,
        "color": "#17becf",  # cyan - clearly distinct from Sonnet red & GPT purple
        "label": f"Gemini 3.5 Flash (N={len(gemini_q1)})",
        "tier": "bottom",
    }

    # Compute stats
    print("\nComputing statistics...")
    for key, src in inputs.items():
        src["_stats"] = compute_stats(src["q1"], src["q2"])
        s = src["_stats"]
        print(f"  {src['label']}: II mean={s['II']['mean']:.2f}, "
              f"SD={s['II']['sd']:.2f}, n_paired={s['n_paired']}"
              if s['II']['mean'] is not None else f"  {src['label']}: no II")

    print("\nWriting outputs...")
    make_six_panel_figure(
        inputs,
        out_dir / "fig_six_panel_commercial_vs_qwen_vs_human.png",
        out_dir / "fig_six_panel_commercial_vs_qwen_vs_human.pdf",
    )
    write_table_csv(inputs, out_dir / "table2_extended.csv")
    write_table_md(inputs, out_dir / "table2_extended.md")
    write_stats_json(inputs, out_dir / "stats_extended.json")

    print(f"\nAll outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
