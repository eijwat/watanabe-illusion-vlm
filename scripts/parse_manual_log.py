#!/usr/bin/env python3
"""
parse_manual_log.py

Parse a manual Markdown log of VLM responses (collected by hand from
Web UIs) and produce Qwen2.5-VL-32B-compatible n60_progress_<model>.json
files.

Usage:
    python parse_manual_log.py manual_log.md
    python parse_manual_log.py manual_log.md --output-dir results/manual

Input format (manual_log.md):

    # ... arbitrary preamble, ignored until first ## heading ...

    ## opus47 / trial 00 / Q1
    the 7th dot from the top.

    ## opus47 / trial 00 / Q2
    Looking at the line, it would hit the 4th dot from the bottom.

    ## gpt55 / trial 00 / Q1
    the 3rd dot from the top

    ...

Output: one JSON file per model, with structure matching
n60_progress_a100.json so that analyze_n60_q1q2.py works on it directly.

Author: Watanabe lab, NIBB
Date: 2026-05-25
"""

import sys
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict


# ===== Model registry =====
#
# Strategy shift (2026-05-26): from "thinking-mode flagships" to
# "default/instant Web UI models" -- i.e. what users actually see when
# they open claude.ai / chatgpt.com / gemini.google.com without changing
# any settings. This better matches the paper's central question:
# "what do today's everyday-use VLMs see?"

MODEL_REGISTRY = {
    # --- Current registry (2026-05-26 onwards) ---
    "sonnet46": {
        "model_id": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6 (Web UI, default)",
        "ui": "claude.ai",
    },
    "gpt55": {
        "model_id": "gpt-5.5",
        "label": "GPT-5.5 Instant (Web UI, default Instant mode)",
        "ui": "chatgpt.com",
    },
    "gemini35flash": {
        "model_id": "gemini-3-5-flash",
        "label": "Gemini 3.5 Flash (Web UI, current Gemini app default)",
        "ui": "gemini.google.com",
    },
    # --- Legacy keys, kept for backward compatibility with earlier logs ---
    "opus47": {
        "model_id": "claude-opus-4-7",
        "label": "Claude Opus 4.7 (Web UI, Thinking) [legacy]",
        "ui": "claude.ai",
    },
    "gemini3pro": {
        "model_id": "gemini-3-pro",
        "label": "Gemini 3 Pro (Web UI) [legacy]",
        "ui": "gemini.google.com",
    },
}

PROMPTS = {
    "Q1": (
        "Look at the image below straight on from the front. "
        "Please answer based on your visual impression alone "
        "\u2014 do not use any tools such as Python, image processing, "
        "a ruler, or a protractor to measure.\n\n"
        "Estimate which dot on the right edge the line in the circle "
        "would hit if extended. Answer like 'the Xth dot from the top.'"
    ),
    "Q2": (
        "Look at the image below straight on from the front. "
        "Please answer based on your visual impression alone "
        "\u2014 do not use any tools such as Python, image processing, "
        "a ruler, or a protractor to measure.\n\n"
        "Estimate which dot on the right edge the line in the circle "
        "would hit if extended. Answer like 'the Xth dot from the bottom.'"
    ),
}

N_TRIALS_TARGET = 60
QUESTIONS = ["Q1", "Q2"]

NUMBER_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11,
}


# ===== Heading regex =====

# Matches:  ## opus47 / trial 23 / Q1
#           ##  GPT55  /  Trial 7  /  q2
# Case-insensitive, flexible whitespace around the slashes.
HEADING_RE = re.compile(
    r"^\s*##\s*(?P<model>[A-Za-z0-9_]+)\s*/\s*trial\s*(?P<trial>\d+)\s*/\s*(?P<question>[Qq][12])\s*$",
    re.MULTILINE
)


# ===== Parsing the Markdown =====

def parse_log(md_text: str) -> dict:
    """Split the log on heading lines. Returns
    {(model_key, trial_idx, question): response_text, ...}.

    Lines before the first heading are ignored (preamble/instructions).

    Level-1 markdown headings (lines starting with a single "# ", not "## ")
    inside a response body are stripped out, since these are used in the
    template as visual section dividers between models (e.g. "# ----- gpt55 -----")
    and are not part of any model's answer.
    """
    matches = list(HEADING_RE.finditer(md_text))
    if not matches:
        return {}

    result = {}
    for i, m in enumerate(matches):
        model = m.group("model").lower()
        trial = int(m.group("trial"))
        question = m.group("question").upper()
        # response is from end of heading to start of next heading
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        response = md_text[start:end]
        # Strip out any level-1 heading lines (visual section dividers)
        response = re.sub(r"^\s*#(?!#).*$", "", response, flags=re.MULTILINE)
        response = response.strip()
        key = (model, trial, question)
        if key in result:
            # Duplicate heading -- keep the last one but warn later
            result[key] = response
        else:
            result[key] = response
    return result


# ===== Response interpretation =====

def parse_dot_value(text: str, question: str) -> dict:
    """Extract dot number, direction, compliance, parse method from a response.

    Returns dict with:
      - boxed_value: int or None
      - direction_detected: "top" / "bottom" / None
      - direction_compliant: bool or None
      - parse_method: string
      - skipped: bool
      - skip_reason: str or None
      - not_yet_collected: bool  (heading exists but response is empty,
                                  meaning this trial is just not done yet)
    """
    out = {
        "boxed_value": None,
        "direction_detected": None,
        "direction_compliant": None,
        "parse_method": None,
        "skipped": False,
        "skip_reason": None,
        "not_yet_collected": False,
    }
    if not text or not text.strip():
        # Empty heading body -- just a placeholder, not yet collected.
        # We treat this differently from SKIPPED (which means the trial
        # was attempted but unusable). The parser will not warn about these.
        out["not_yet_collected"] = True
        return out

    # SKIPPED marker (user-entered)
    skip_match = re.match(r"^\s*SKIPPED\s*:?\s*(.*)", text, re.IGNORECASE)
    if skip_match:
        out["skipped"] = True
        out["skip_reason"] = skip_match.group(1).strip() or "user marked as skipped"
        return out

    text_lower = text.lower()

    # Direction detection
    if "from the top" in text_lower or "from top" in text_lower:
        out["direction_detected"] = "top"
    elif "from the bottom" in text_lower or "from bottom" in text_lower:
        out["direction_detected"] = "bottom"

    # Compliance check
    expected = "top" if question == "Q1" else "bottom"
    if out["direction_detected"] is not None:
        out["direction_compliant"] = (out["direction_detected"] == expected)

    # \boxed{N}
    m = re.search(r"\\boxed\{(\d+)\}", text)
    if m:
        out["boxed_value"] = int(m.group(1))
        out["parse_method"] = "boxed"
        return out

    # \boxed{spelled}
    m = re.search(r"\\boxed\{[^}]*?(\w+)[^}]*?\}", text)
    if m and m.group(1).lower() in NUMBER_WORDS:
        out["boxed_value"] = NUMBER_WORDS[m.group(1).lower()]
        out["parse_method"] = "boxed"
        return out

    # "5th dot" / "the 5th dot"
    m = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+dot\b", text_lower)
    if m:
        out["boxed_value"] = int(m.group(1))
        out["parse_method"] = "ordinal_digit"
        return out

    # "fifth dot"
    for word, num in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+dot\b", text_lower):
            out["boxed_value"] = num
            out["parse_method"] = "spelled_ordinal"
            return out

    # Fallback: bare digit somewhere in the text (last resort)
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 11:
            out["boxed_value"] = val
            out["parse_method"] = "bare_digit_fallback"
            return out

    out["parse_method"] = "FAILED"
    return out


# ===== Pretty-print progress =====

def print_progress(by_model: dict, parsed: dict):
    """Print progress summary."""
    print()
    print("=" * 70)
    print(" Progress")
    print("=" * 70)
    for model_key in MODEL_REGISTRY:
        if model_key not in by_model:
            print(f"  {model_key}: no entries")
            continue

        trials = by_model[model_key]
        q1_count = sum(
            1 for t in trials
            if "Q1" in trials[t] and parsed[(model_key, t, "Q1")]["boxed_value"] is not None
        )
        q2_count = sum(
            1 for t in trials
            if "Q2" in trials[t] and parsed[(model_key, t, "Q2")]["boxed_value"] is not None
        )
        complete = sum(
            1 for t in trials
            if "Q1" in trials[t] and "Q2" in trials[t]
            and parsed[(model_key, t, "Q1")]["boxed_value"] is not None
            and parsed[(model_key, t, "Q2")]["boxed_value"] is not None
        )
        print(
            f"  {model_key:12s}: "
            f"complete pairs {complete:3d}/{N_TRIALS_TARGET}  "
            f"(Q1: {q1_count}/{N_TRIALS_TARGET}, Q2: {q2_count}/{N_TRIALS_TARGET})"
        )
    print()


def print_warnings(parsed: dict):
    """Print direction-compliance issues, parse failures, skipped trials."""
    print("=" * 70)
    print(" Warnings")
    print("=" * 70)

    direction_issues = []
    parse_failures = []
    skipped = []
    bare_digit_fallback = []

    for (model, trial, q), p in sorted(parsed.items()):
        if p.get("not_yet_collected"):
            # Empty placeholder heading -- don't warn, just count as
            # not collected in the progress view.
            continue
        if p["skipped"]:
            skipped.append((model, trial, q, p["skip_reason"]))
        elif p["boxed_value"] is None:
            parse_failures.append((model, trial, q))
        else:
            if p["direction_compliant"] is False:
                direction_issues.append(
                    (model, trial, q, p["direction_detected"])
                )
            if p["parse_method"] == "bare_digit_fallback":
                bare_digit_fallback.append((model, trial, q, p["boxed_value"]))

    if not (direction_issues or parse_failures or skipped or bare_digit_fallback):
        print("  No warnings. All parsed cleanly.")
        print()
        return

    if direction_issues:
        print(f"  Direction-compliance issues ({len(direction_issues)}):")
        for model, trial, q, dir_got in direction_issues:
            expected = "top" if q == "Q1" else "bottom"
            print(f"    {model} / trial {trial:02d} / {q}: "
                  f"says '{dir_got}' (expected '{expected}'). "
                  f"Will be direction-normalized in analysis.")
        print()

    if parse_failures:
        print(f"  Parse failures ({len(parse_failures)}):")
        for model, trial, q in parse_failures:
            print(f"    {model} / trial {trial:02d} / {q}: "
                  f"no recognizable number found")
        print()

    if skipped:
        print(f"  Skipped trials ({len(skipped)}):")
        for model, trial, q, reason in skipped:
            print(f"    {model} / trial {trial:02d} / {q}: {reason}")
        print()

    if bare_digit_fallback:
        print(f"  Bare-digit fallback used ({len(bare_digit_fallback)}) "
              f"(no dot/ordinal context; may be incorrect):")
        for model, trial, q, val in bare_digit_fallback:
            print(f"    {model} / trial {trial:02d} / {q}: parsed as {val}")
        print()


# ===== JSON writer (Qwen-compatible schema) =====

def build_json_record(model_key: str, trials_by_idx: dict,
                      parsed: dict, raw_responses: dict) -> dict:
    """Build a dict matching n60_progress_a100.json structure."""
    reg = MODEL_REGISTRY[model_key]
    timestamp = datetime.now().isoformat()

    record = {
        "started_at": timestamp,
        "model_id": reg["model_id"],
        "model_label": reg["label"],
        "stimulus": "./104.jpg",
        "preprocessing": f"Web UI ({reg['ui']}), default behavior",
        "prompt_set": "human_G1G2_aligned_v1_manual",
        "structure": "2-turn independent (manual, fresh chat per question)",
        "machine": f"Manual via {reg['ui']}",
        "n_trials_target": N_TRIALS_TARGET,
        "data_collection_method": "manual_paste_from_web_ui",
        "ui_settings": {
            "memory": "off",
            "temporary_chat": True,
            "thinking_mode": "default (whatever the UI does by default)"
        },
        "prompts": PROMPTS,
        "trials": [],
    }

    # Iterate over trials in order
    for trial_idx in sorted(trials_by_idx.keys()):
        trial_record = {"trial_idx": trial_idx}
        questions_present = trials_by_idx[trial_idx]
        any_data = False
        for q in QUESTIONS:
            if q not in questions_present:
                continue
            key = (model_key, trial_idx, q)
            p = parsed[key]
            # Skip placeholder headings with no response yet.
            if p.get("not_yet_collected"):
                continue
            any_data = True
            raw = raw_responses[key]
            trial_record[q] = {
                "prompt": PROMPTS[q],
                "response": raw,
                "parsed_value": p["boxed_value"],
                "direction_detected": p["direction_detected"],
                "direction_compliant": p["direction_compliant"],
                "parse_method": p["parse_method"],
                "skipped": p["skipped"],
                "skip_reason": p["skip_reason"],
            }
        if any_data:
            record["trials"].append(trial_record)

    record["completed_at"] = timestamp
    return record


# ===== Main =====

def main():
    ap = argparse.ArgumentParser(description="Parse manual_log.md.")
    ap.add_argument("log_file", type=str, help="Path to manual_log.md")
    ap.add_argument("--output-dir", type=str, default="results/manual",
                    help="Directory for JSON output (default: results/manual)")
    ap.add_argument("--no-write", action="store_true",
                    help="Only show progress/warnings, don't write JSONs")
    args = ap.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}")
        sys.exit(1)

    md_text = log_path.read_text(encoding="utf-8")
    raw_responses = parse_log(md_text)

    if not raw_responses:
        print("No headings found. Make sure your log has entries like:")
        print("    ## opus47 / trial 00 / Q1")
        print("    the 5th dot from the top")
        sys.exit(0)

    # Parse each response
    parsed = {}
    by_model = defaultdict(dict)  # model -> {trial_idx -> set of questions}
    for key, raw in raw_responses.items():
        model, trial, q = key
        if model not in MODEL_REGISTRY:
            print(f"  WARNING: unknown model key '{model}' "
                  f"in heading 'trial {trial} / {q}'. Skipping.")
            continue
        p = parse_dot_value(raw, q)
        parsed[key] = p
        if trial not in by_model[model]:
            by_model[model][trial] = set()
        by_model[model][trial].add(q)

    # Print progress & warnings
    print_progress(by_model, parsed)
    print_warnings(parsed)

    if args.no_write:
        return

    # Write per-model JSONs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Writing JSON files")
    print("=" * 70)
    for model_key, trials_by_idx in by_model.items():
        if model_key not in MODEL_REGISTRY:
            continue
        record = build_json_record(model_key, trials_by_idx, parsed, raw_responses)
        out_path = out_dir / f"n60_progress_{model_key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {out_path}")
    print()
    print("Done. Re-run anytime as you add more data to the log.")


if __name__ == "__main__":
    main()
