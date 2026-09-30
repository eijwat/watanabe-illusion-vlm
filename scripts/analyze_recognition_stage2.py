"""
analyze_recognition_stage2.py
==============================
Aggregate Phase A (Qwen2.5-VL-32B on A100) and Phase B (commercial APIs)
recognition probe results into the tables and figures that go into the
paper's "Watanabe Illusion is unknown to current VLMs" claim.

Inputs:
    /workspace/results/qwen25vl_recognition_stage2_a100/summary.csv
    ~/watanabe-illusion/results/commercial_recognition_stage2/summary.csv
    (paths overridable via CLI)

Outputs (default in --out-dir = figures/):
    recognition_stage2_combined.csv       merged data (model x stim x Q x trial)
    recognition_stage2_naming_rates.csv   per-model x per-stimulus naming hit rate
    recognition_stage2_naming_rates.md    same as Markdown table for paper
    recognition_stage2_yesno.csv          per-model x per-stimulus QV1 yes/no/unsure breakdown
    recognition_stage2_misidentifications.csv  rows where Watanabe got a wrong name
    fig_recognition_stage2.png            grouped bar: naming hit rate per (model, stim)
    fig_recognition_stage2.pdf            same in PDF

Console summary:
    - per-model naming hit rate on Watanabe vs each control
    - per-model QV1 yes/no/unsure rate on Watanabe
    - candidate paper-ready sentence summarizing findings

Usage:
    python analyze_recognition_stage2.py \\
        --qwen   /workspace/results/qwen25vl_recognition_stage2_a100/summary.csv \\
        --commercial ~/watanabe-illusion/results/commercial_recognition_stage2/summary.csv \\
        --out-dir figures/

    # Phase A only (when commercial data is not yet collected)
    python analyze_recognition_stage2.py --qwen .../summary.csv

Notes:
    - "Hit" = the response named the *correct* illusion at least once
      across naming-style questions (QV2 + QV3). QV1 (yes/no) is reported
      separately as it asks about familiarity, not naming.
    - For Watanabe (the target), "hit" means the response contained
      "watanabe" (case-insensitive). For controls it means the response
      contained one of the registered aliases.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import OrderedDict, Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ========== Constants ==========
NAMING_QUESTIONS = ["QV2", "QV3"]  # ones that elicit a name
YESNO_QUESTION = "QV1"

STIM_ORDER = ["watanabe", "muller_lyer", "poggendorff", "kanizsa"]
STIM_DISPLAY = {
    "watanabe":     "Watanabe (target)",
    "muller_lyer":  "Müller-Lyer (control)",
    "poggendorff":  "Poggendorff (control)",
    "kanizsa":      "Kanizsa (control)",
}

# Display order: Qwen first (open-weights, paper main subject),
# then commercial in canonical order
MODEL_ORDER_PREFERRED = [
    "qwen", "anthropic", "openai", "google",
]
MODEL_DISPLAY = {
    "qwen":      "Qwen2.5-VL-32B",
    "anthropic": "Claude Opus 4.7",
    "openai":    "GPT-5.5",
    "google":    "Gemini 3.5 Flash",
}


# ========== CSV loading ==========
def _to_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "t")


def load_csv(path: Path, default_model_key: str | None = None) -> list[dict]:
    """Load one summary.csv, return list of dict rows.

    If the file has no `model_key` column (Phase A output), inject the
    given `default_model_key` (e.g. "qwen").
    """
    if not path.exists():
        print(f"  [WARN] {path} does not exist, skipping")
        return []

    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if "model_key" not in r or not r.get("model_key"):
                r["model_key"] = default_model_key or "unknown"
                r["model_id"] = r.get("model_id", "")
                r["model_label"] = r.get(
                    "model_label",
                    MODEL_DISPLAY.get(default_model_key, default_model_key
                                      or "unknown")
                )
            # Normalize booleans
            for col in ("is_control", "named_watanabe", "named_muller_lyer",
                        "named_poggendorff", "named_kanizsa",
                        "named_target", "named_wrong"):
                if col in r:
                    r[col] = _to_bool(r[col])
            # Numeric columns
            for col in ("trial_idx", "seed", "response_len_chars"):
                if col in r and r[col] not in ("", None):
                    try:
                        r[col] = int(r[col])
                    except ValueError:
                        pass
            for col in ("elapsed_sec",):
                if col in r and r[col] not in ("", None):
                    try:
                        r[col] = float(r[col])
                    except ValueError:
                        pass
            rows.append(r)
    print(f"  loaded {len(rows):>4d} rows from {path}")
    return rows


# ========== Aggregation ==========
def naming_hit_rate(rows: list[dict], model_key: str,
                    stimulus_id: str) -> tuple[int, int]:
    """Count naming hits for (model, stimulus) across QV2 + QV3.

    A trial counts as a hit if the response named the *correct* illusion
    (named_target == True). The denominator is the number of QV2/QV3
    trials present for this (model, stimulus) pair.
    """
    hits = 0
    total = 0
    for r in rows:
        if r["model_key"] != model_key or r["stimulus_id"] != stimulus_id:
            continue
        if r["question_id"] not in NAMING_QUESTIONS:
            continue
        total += 1
        if r["named_target"]:
            hits += 1
    return hits, total


def yesno_breakdown(rows: list[dict], model_key: str,
                    stimulus_id: str) -> dict:
    """Distribution of QV1 yes/no/unsure for (model, stimulus)."""
    counts = Counter()
    for r in rows:
        if r["model_key"] != model_key or r["stimulus_id"] != stimulus_id:
            continue
        if r["question_id"] != YESNO_QUESTION:
            continue
        label = r.get("yes_no_label", "")
        counts[label or "unknown"] += 1
    return dict(counts)


def collect_misidentifications(rows: list[dict],
                               target_stimulus: str = "watanabe") -> list[dict]:
    """Rows where the response, when shown the target stimulus, named
    a different illusion. These are the 'false guesses' — useful for
    showing what VLMs *think* Watanabe is.
    """
    out = []
    for r in rows:
        if r["stimulus_id"] != target_stimulus:
            continue
        if r["question_id"] not in NAMING_QUESTIONS:
            continue
        if r["named_target"]:
            continue
        # Did it name any *wrong* illusion?
        wrong_names = []
        for other_stim in STIM_ORDER:
            if other_stim == target_stimulus:
                continue
            col = f"named_{other_stim}"
            if r.get(col):
                wrong_names.append(other_stim)
        if wrong_names:
            out.append({
                "model_key": r["model_key"],
                "model_label": r.get("model_label", ""),
                "question_id": r["question_id"],
                "trial_idx": r["trial_idx"],
                "named_what_instead": "|".join(wrong_names),
                "response_excerpt": (r.get("response_full", "") or "")[:300]
                                    .replace("\n", " ↵ "),
            })
    return out


# ========== Output: tables ==========
def write_naming_rates_csv(out_path: Path, rows: list[dict],
                           models: list[str], stimuli: list[str]):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["model_key", "model_label"] + \
                 [f"{s}_hits" for s in stimuli] + \
                 [f"{s}_total" for s in stimuli] + \
                 [f"{s}_rate" for s in stimuli]
        writer.writerow(header)
        for mkey in models:
            mlabel = MODEL_DISPLAY.get(mkey, mkey)
            hits_list, totals_list, rates_list = [], [], []
            for stim in stimuli:
                h, t = naming_hit_rate(rows, mkey, stim)
                hits_list.append(h)
                totals_list.append(t)
                rates_list.append(f"{h/t:.3f}" if t > 0 else "—")
            writer.writerow([mkey, mlabel] + hits_list + totals_list +
                            rates_list)


def write_naming_rates_md(out_path: Path, rows: list[dict],
                          models: list[str], stimuli: list[str]):
    lines = []
    lines.append("# Recognition probe — naming hit rate (Stage 2)")
    lines.append("")
    lines.append("Each cell shows hits / total (rate), pooled across QV2 "
                 "(forced naming) and QV3 (describe + recognition). A *hit* "
                 "means the response named the correct illusion.")
    lines.append("")

    # Header
    hdr = "| Model | " + " | ".join(STIM_DISPLAY[s] for s in stimuli) + " |"
    sep = "|---|" + "|".join("---" for _ in stimuli) + "|"
    lines.append(hdr)
    lines.append(sep)

    for mkey in models:
        mlabel = MODEL_DISPLAY.get(mkey, mkey)
        row_cells = [mlabel]
        for stim in stimuli:
            h, t = naming_hit_rate(rows, mkey, stim)
            if t == 0:
                row_cells.append("—")
            else:
                row_cells.append(f"{h}/{t} ({h/t*100:.0f}%)")
        lines.append("| " + " | ".join(row_cells) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_yesno_csv(out_path: Path, rows: list[dict],
                    models: list[str], stimuli: list[str]):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_key", "model_label", "stimulus_id",
                         "n_yes", "n_no", "n_unsure", "n_unknown", "n_total"])
        for mkey in models:
            mlabel = MODEL_DISPLAY.get(mkey, mkey)
            for stim in stimuli:
                br = yesno_breakdown(rows, mkey, stim)
                n_yes = br.get("yes", 0)
                n_no = br.get("no", 0)
                n_unsure = br.get("unsure", 0)
                n_unknown = br.get("unknown", 0)
                total = n_yes + n_no + n_unsure + n_unknown
                writer.writerow([mkey, mlabel, stim,
                                 n_yes, n_no, n_unsure, n_unknown, total])


def write_misidentifications_csv(out_path: Path, misids: list[dict]):
    if not misids:
        out_path.write_text("model_key,model_label,question_id,trial_idx,"
                            "named_what_instead,response_excerpt\n",
                            encoding="utf-8")
        return
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(misids[0].keys()))
        writer.writeheader()
        writer.writerows(misids)


# ========== Output: figure ==========
def make_figure(out_png: Path, out_pdf: Path, rows: list[dict],
                models: list[str], stimuli: list[str]):
    """Grouped bar: x = stimulus, hue = model, y = naming hit rate."""
    n_models = len(models)
    n_stims = len(stimuli)
    bar_width = 0.8 / max(n_models, 1)
    x_pos = np.arange(n_stims)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c",
               "#ff7f0e", "#8c564b"]

    for i, mkey in enumerate(models):
        rates, labels = [], []
        for j, stim in enumerate(stimuli):
            h, t = naming_hit_rate(rows, mkey, stim)
            rates.append((h / t) if t > 0 else 0.0)
            labels.append(f"{h}/{t}" if t > 0 else "—")
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x_pos + offset, rates, bar_width,
                      label=MODEL_DISPLAY.get(mkey, mkey),
                      color=palette[i % len(palette)],
                      edgecolor="black", linewidth=0.5)
        for b, lbl in zip(bars, labels):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    lbl, ha="center", va="bottom", fontsize=8,
                    rotation=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([STIM_DISPLAY[s] for s in stimuli],
                       rotation=0, fontsize=10)
    ax.set_ylabel("Naming hit rate (QV2+QV3 pooled)", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Stage 2 — recognition probe across 4 VLMs and 4 stimuli",
                 fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_pdf)
    plt.close()


# ========== Output: combined CSV ==========
def write_combined_csv(out_path: Path, rows: list[dict]):
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    # union of keys, deterministic order: standard cols first, rest alpha
    standard_order = [
        "model_key", "model_id", "model_label",
        "stimulus_id", "true_name", "is_control",
        "question_id", "question_type",
        "trial_idx", "seed",
        "yes_no_label",
        "named_watanabe", "named_muller_lyer", "named_poggendorff",
        "named_kanizsa", "named_target", "named_wrong",
        "response_len_chars", "elapsed_sec",
        "error", "response_full",
    ]
    keys = list(standard_order)
    extra = sorted({k for r in rows for k in r.keys()} - set(keys))
    keys = keys + extra
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ========== Console summary (paper-ready text) ==========
def print_console_summary(rows: list[dict], models: list[str],
                          stimuli: list[str], misids: list[dict]):
    print()
    print("=" * 70)
    print("Recognition probe — console summary")
    print("=" * 70)

    # Naming hit rates
    print("\nNaming hit rate (QV2 + QV3 pooled):")
    hdr = f"  {'Model':<22s}" + "".join(
        f"  {STIM_DISPLAY[s][:18]:<20s}" for s in stimuli
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for mkey in models:
        mlabel = MODEL_DISPLAY.get(mkey, mkey)
        cells = [f"  {mlabel:<22s}"]
        for stim in stimuli:
            h, t = naming_hit_rate(rows, mkey, stim)
            if t == 0:
                cells.append(f"  {'—':<20s}")
            else:
                cells.append(f"  {h:>2d}/{t:>2d} ({h/t*100:>3.0f}%)" + " " * 11)
        print("".join(cells))

    # QV1 yes/no/unsure on Watanabe
    print("\nQV1 'Have you seen this?' on Watanabe (the target):")
    for mkey in models:
        br = yesno_breakdown(rows, mkey, "watanabe")
        total = sum(br.values())
        if total == 0:
            continue
        mlabel = MODEL_DISPLAY.get(mkey, mkey)
        y, n, u, x = (br.get("yes", 0), br.get("no", 0),
                      br.get("unsure", 0), br.get("unknown", 0))
        print(f"  {mlabel:<22s}  yes={y:>2d}  no={n:>2d}  "
              f"unsure={u:>2d}  unknown={x:>2d}  (N={total})")

    # Misidentifications on Watanabe
    if misids:
        print(f"\nWatanabe misidentifications (named something else): "
              f"{len(misids)} trial(s)")
        # Aggregate what they got named instead
        wrong_counter = Counter()
        for r in misids:
            for w in r["named_what_instead"].split("|"):
                wrong_counter[w] += 1
        for w, c in wrong_counter.most_common():
            print(f"  named {w:<14s} : {c} time(s)")
    else:
        print("\nWatanabe misidentifications: none "
              "(no response named another registered illusion)")

    # Paper-ready summary sentence
    print("\nCandidate paper sentence:")
    print("-" * 70)

    # Watanabe naming totals across all models
    wat_hits = 0
    wat_total = 0
    for mkey in models:
        h, t = naming_hit_rate(rows, mkey, "watanabe")
        wat_hits += h
        wat_total += t

    # Control hit rates
    ctrl_summaries = []
    for stim in stimuli:
        if stim == "watanabe":
            continue
        ch, ct = 0, 0
        for mkey in models:
            h, t = naming_hit_rate(rows, mkey, stim)
            ch += h
            ct += t
        if ct > 0:
            ctrl_summaries.append(f"{STIM_DISPLAY[stim].replace(' (control)', '')} "
                                  f"{ch}/{ct} ({ch/ct*100:.0f}%)")

    print(f"  We probed {len(models)} VLMs ({', '.join(MODEL_DISPLAY.get(m, m) for m in models)})")
    print(f"  with N=10 trials per (stimulus, question) pair on 4 stimuli")
    print(f"  (104.jpg + 3 classical controls).  On naming-style probes "
          f"(QV2+QV3),")
    print(f"  the Watanabe stimulus was named correctly in "
          f"{wat_hits}/{wat_total} trials")
    if wat_total > 0:
        print(f"  ({wat_hits/wat_total*100:.1f}%) across all VLMs combined,")
    print(f"  while classical controls were named at:")
    for cs in ctrl_summaries:
        print(f"    {cs}")
    print("-" * 70)


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qwen", type=str, default=None,
                        help="path to Phase A summary.csv "
                             "(/workspace/results/qwen25vl_recognition_stage2_a100/summary.csv)")
    parser.add_argument("--commercial", type=str, default=None,
                        help="path to Phase B summary.csv "
                             "(commercial_recognition_stage2/summary.csv)")
    parser.add_argument("--out-dir", type=str, default="figures",
                        help="output directory (default: figures/)")
    parser.add_argument("--models", type=str, default=None,
                        help="comma-separated subset of model keys")
    parser.add_argument("--stimuli", type=str, default=None,
                        help="comma-separated subset of stimulus IDs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Analyze Stage 2 recognition probe")
    print("=" * 70)

    # Load
    rows = []
    if args.qwen:
        rows += load_csv(Path(args.qwen).expanduser(), default_model_key="qwen")
    if args.commercial:
        rows += load_csv(Path(args.commercial).expanduser())

    if not rows:
        print("[ERROR] no rows loaded; specify --qwen and/or --commercial.")
        sys.exit(1)

    # Filter
    if args.models:
        wanted = set(s.strip() for s in args.models.split(","))
        rows = [r for r in rows if r["model_key"] in wanted]
        models_present = [m for m in MODEL_ORDER_PREFERRED if m in wanted]
    else:
        present = {r["model_key"] for r in rows}
        models_present = [m for m in MODEL_ORDER_PREFERRED if m in present]
        # Append any unknowns at the end
        unknown = sorted(present - set(MODEL_ORDER_PREFERRED))
        models_present += unknown

    if args.stimuli:
        wanted = set(s.strip() for s in args.stimuli.split(","))
        rows = [r for r in rows if r["stimulus_id"] in wanted]
        stimuli_present = [s for s in STIM_ORDER if s in wanted]
    else:
        present = {r["stimulus_id"] for r in rows}
        stimuli_present = [s for s in STIM_ORDER if s in present]

    print(f"\nTotal rows after filtering: {len(rows)}")
    print(f"Models present:   {models_present}")
    print(f"Stimuli present:  {stimuli_present}")

    # Outputs
    combined_csv = out_dir / "recognition_stage2_combined.csv"
    naming_csv = out_dir / "recognition_stage2_naming_rates.csv"
    naming_md = out_dir / "recognition_stage2_naming_rates.md"
    yesno_csv = out_dir / "recognition_stage2_yesno.csv"
    misid_csv = out_dir / "recognition_stage2_misidentifications.csv"
    fig_png = out_dir / "fig_recognition_stage2.png"
    fig_pdf = out_dir / "fig_recognition_stage2.pdf"

    write_combined_csv(combined_csv, rows)
    print(f"\nWrote {combined_csv}")

    write_naming_rates_csv(naming_csv, rows, models_present, stimuli_present)
    write_naming_rates_md(naming_md, rows, models_present, stimuli_present)
    print(f"Wrote {naming_csv}")
    print(f"Wrote {naming_md}")

    write_yesno_csv(yesno_csv, rows, models_present, stimuli_present)
    print(f"Wrote {yesno_csv}")

    misids = collect_misidentifications(rows, target_stimulus="watanabe")
    write_misidentifications_csv(misid_csv, misids)
    print(f"Wrote {misid_csv}  ({len(misids)} entries)")

    make_figure(fig_png, fig_pdf, rows, models_present, stimuli_present)
    print(f"Wrote {fig_png}")
    print(f"Wrote {fig_pdf}")

    print_console_summary(rows, models_present, stimuli_present, misids)


if __name__ == "__main__":
    main()
