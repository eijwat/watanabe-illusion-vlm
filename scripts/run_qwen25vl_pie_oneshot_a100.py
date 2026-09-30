"""
run_qwen25vl_pie_oneshot_a100.py
=================================
PIE one-shot experiment for fair comparison with human N=130.

Each condition uses 130 trials, where each trial uses a DIFFERENT prompt
and a DIFFERENT seed (no within-prompt repetitions). This mirrors the human
experiment where each of 130 participants answered once.

Conditions (7 total, each N=130, total 910 trials):
  A. Baseline:        no preamble, bare Q1
  B. CharacterSystem: 130 character prompts as system message
  C. CharacterUser:   130 character prompts in user message
  D. EnvironmentSystem
  E. EnvironmentUser
  F. NeutralSystem
  G. NeutralUser

Generation settings (identical to run_qwen25vl_pie_a100.py):
  temperature=1.0, top_p=1.0, repetition_penalty=1.05

Seed scheme (non-overlapping across conditions):
  A: 10000..10129
  B: 11000..11129
  C: 12000..12129
  D: 13000..13129
  E: 14000..14129
  F: 15000..15129
  G: 16000..16129

Prompt selection within a category:
  All 130 prompts of the category are used (the category contains exactly 130
  prompts: 33+32+33+32 across its four subcategories). Order: sorted by
  (subcategory_code, numeric index).

Usage (single command runs all 7 conditions sequentially):
  python run_qwen25vl_pie_oneshot_a100.py
  python run_qwen25vl_pie_oneshot_a100.py --pilot     (only 5 trials per condition)
  python run_qwen25vl_pie_oneshot_a100.py --condition A   (run only condition A)

Expected runtime: ~3 hours for all 7 conditions on A100.
"""

import os
import csv
import json
import time
import argparse
import traceback
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# -------- Paths --------
STIMULUS_PATH = "/workspace/stimuli/104.jpg"
PROMPTS_CSV = "/workspace/csv/pie_prompts_v2.csv"
OUT_BASE = Path("/workspace/results")

# -------- Model --------
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_MEMORY = {0: "70GiB", "cpu": "0GiB"}

# -------- Generation (identical to run_qwen25vl_pie_a100.py) --------
TEMPERATURE = 1.0
TOP_P = 1.0
REPETITION_PENALTY = 1.05
MAX_NEW_TOKENS = 512

# -------- Experiment --------
N_PER_CONDITION = 130
N_DOTS_RIGHT = 11

# -------- Q1 text (identical to run_qwen25vl_n60_v2.py and pie v1) --------
PREAMBLE_FIXED = (
    "Look at the image below straight on from the front. "
    "Please answer based on your visual impression alone — "
    "do not use any tools such as Python, image processing, "
    "a ruler, or a protractor to measure."
)
Q1_BODY = (
    "Estimate which dot on the right edge the line in the circle would hit "
    "if extended. Answer like 'the Xth dot from the top.'"
)
Q1_USER_TEXT = f"{PREAMBLE_FIXED}\n\n{Q1_BODY}"

# -------- Condition definitions --------
# Each condition: (id, label, category_filter, placement, seed_base)
CONDITIONS = [
    ("A", "baseline",          None,           "baseline", 10000),
    ("B", "character_system",  "character",    "system",   11000),
    ("C", "character_user",    "character",    "user",     12000),
    ("D", "environment_system","environment",  "system",   13000),
    ("E", "environment_user",  "environment",  "user",     14000),
    ("F", "neutral_system",    "neutral",      "system",   15000),
    ("G", "neutral_user",      "neutral",      "user",     16000),
]


# ==========================================================================
# Prompt loading
# ==========================================================================

def load_prompts(csv_path: Path):
    """Load all 390 prompts from v2 CSV."""
    prompts = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts


def select_for_condition(all_prompts, category):
    """Select 130 prompts for a category, sorted by (subcategory_code, numeric idx)."""
    if category is None:
        # Baseline: no prompts needed (we return None placeholders)
        return [None] * N_PER_CONDITION
    cat_prompts = [p for p in all_prompts if p["category"] == category]
    def sort_key(p):
        parts = p["prompt_id"].split("-")
        return (parts[0], parts[1], int(parts[2]))
    cat_prompts.sort(key=sort_key)
    assert len(cat_prompts) == N_PER_CONDITION, (
        f"Category {category} has {len(cat_prompts)} prompts (expected {N_PER_CONDITION})"
    )
    return cat_prompts


# ==========================================================================
# Message construction
# ==========================================================================

def build_messages(image, preamble_or_none, placement):
    """Build messages for one trial.

    If preamble_or_none is None, use baseline (no preamble).
    Otherwise place preamble per placement.
    """
    if placement == "baseline" or preamble_or_none is None:
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": Q1_USER_TEXT},
            ],
        }]
    elif placement == "system":
        return [
            {"role": "system", "content": preamble_or_none},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": Q1_USER_TEXT},
            ]},
        ]
    elif placement == "user":
        combined = f"{preamble_or_none}\n\n{Q1_USER_TEXT}"
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": combined},
            ],
        }]
    else:
        raise ValueError(f"Unknown placement: {placement}")


def build_inputs(processor, image, preamble_or_none, placement):
    messages = build_messages(image, preamble_or_none, placement)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    )
    return inputs, text


# ==========================================================================
# Inference
# ==========================================================================

def generate_response(model, processor, inputs, device, seed):
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_len:]
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


# ==========================================================================
# Response parsing (identical to run_qwen25vl_pie_a100.py)
# ==========================================================================

DOT_PATTERN_TOP = re.compile(
    r"(?:the\s+)?"
    r"(\d+)(?:st|nd|rd|th)?"
    r"\s*(?:dot|circle|point)"
    r"[^.]{0,30}?"
    r"from\s+the\s+top",
    re.IGNORECASE,
)
DOT_PATTERN_BOTTOM = re.compile(
    r"(?:the\s+)?"
    r"(\d+)(?:st|nd|rd|th)?"
    r"\s*(?:dot|circle|point)"
    r"[^.]{0,30}?"
    r"from\s+the\s+bottom",
    re.IGNORECASE,
)
BOXED_PATTERN = re.compile(r"\\boxed\{(\d+)\}")


def parse_q1_answer(response: str):
    matches_top = list(DOT_PATTERN_TOP.finditer(response))
    matches_bot = list(DOT_PATTERN_BOTTOM.finditer(response))
    if matches_top and not matches_bot:
        return {"value": int(matches_top[-1].group(1)), "direction": "top",
                "parse_status": "ok"}
    elif matches_bot and not matches_top:
        return {"value": int(matches_bot[-1].group(1)), "direction": "bottom",
                "parse_status": "ok"}
    elif matches_top and matches_bot:
        last_top = matches_top[-1]
        last_bot = matches_bot[-1]
        if last_top.start() > last_bot.start():
            return {"value": int(last_top.group(1)), "direction": "top",
                    "parse_status": "ambiguous"}
        else:
            return {"value": int(last_bot.group(1)), "direction": "bottom",
                    "parse_status": "ambiguous"}
    else:
        boxed = BOXED_PATTERN.findall(response)
        if boxed:
            return {"value": int(boxed[-1]), "direction": "unknown",
                    "parse_status": "boxed_only"}
        return {"value": None, "direction": "unknown", "parse_status": "no_match"}


def normalize_to_top(value, direction, n_dots=N_DOTS_RIGHT):
    if value is None:
        return None
    if value < 1 or value > n_dots:
        return None
    if direction == "bottom":
        return n_dots - value + 1
    elif direction == "top":
        return value
    else:
        return None


# ==========================================================================
# Model loading
# ==========================================================================

def load_model_and_processor():
    print(f"[load] Loading {MODEL_ID} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=MAX_MEMORY,
    )
    model.eval()
    t1 = time.time()
    print(f"[load] Loaded in {t1 - t0:.1f}s")
    device = next(model.parameters()).device
    print(f"[load] Device: {device}")
    return model, processor, device


# ==========================================================================
# Run one condition
# ==========================================================================

def run_condition(condition, all_prompts, image, model, processor, device,
                  out_dir, pilot_n=None):
    cond_id, cond_label, category, placement, seed_base = condition
    n_target = pilot_n if pilot_n else N_PER_CONDITION

    print(f"\n{'=' * 60}")
    print(f"Condition {cond_id}: {cond_label}")
    print(f"  category: {category}, placement: {placement}")
    print(f"  N: {n_target}, seed range: {seed_base}..{seed_base + n_target - 1}")
    print(f"{'=' * 60}")

    selected = select_for_condition(all_prompts, category)
    selected = selected[:n_target]

    results_jsonl = out_dir / f"results_{cond_id}_{cond_label}.jsonl"
    # Save chat template sample
    if selected[0] is not None or placement == "baseline":
        sample_preamble = selected[0]["preamble"] if selected[0] else None
        _, sample_text = build_inputs(processor, image, sample_preamble, placement)
        sample_path = out_dir / f"chat_template_sample_{cond_id}.txt"
        with sample_path.open("w", encoding="utf-8") as f:
            f.write(f"=== condition: {cond_id} ({cond_label}) ===\n")
            f.write(f"=== placement: {placement} ===\n")
            f.write(f"=== preamble: {sample_preamble!r} ===\n\n")
            f.write(sample_text)

    t_start = time.time()
    n_done = 0
    n_failed = 0
    valid_values = []

    for trial_idx, prompt_obj in enumerate(selected):
        seed = seed_base + trial_idx
        preamble = prompt_obj["preamble"] if prompt_obj else None
        prompt_id = prompt_obj["prompt_id"] if prompt_obj else "(baseline)"
        label = prompt_obj["label"] if prompt_obj else "(no preamble)"

        try:
            inputs, _ = build_inputs(processor, image, preamble, placement)
            t_trial = time.time()
            response = generate_response(model, processor, inputs, device, seed)
            t_elapsed = time.time() - t_trial

            parsed = parse_q1_answer(response)
            normalized = normalize_to_top(parsed["value"], parsed["direction"])

            if normalized is not None:
                valid_values.append(normalized)
            else:
                n_failed += 1

            record = {
                "condition_id": cond_id,
                "condition_label": cond_label,
                "category": category,
                "placement": placement,
                "trial_idx": trial_idx,
                "seed": seed,
                "prompt_id": prompt_id,
                "label": label,
                "response": response,
                "parsed_value": parsed["value"],
                "parsed_direction": parsed["direction"],
                "parse_status": parsed["parse_status"],
                "normalized_top": normalized,
                "elapsed_sec": round(t_elapsed, 2),
                "error": None,
            }
            with results_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_done += 1

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"  [ERROR] trial {trial_idx} ({prompt_id}): {err_msg}")
            traceback.print_exc()
            n_failed += 1
            err_record = {
                "condition_id": cond_id,
                "trial_idx": trial_idx,
                "seed": seed,
                "prompt_id": prompt_id,
                "label": label,
                "response": None,
                "parsed_value": None,
                "parsed_direction": None,
                "parse_status": "error",
                "normalized_top": None,
                "elapsed_sec": None,
                "error": err_msg,
            }
            with results_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(err_record, ensure_ascii=False) + "\n")

        if (trial_idx + 1) % 10 == 0 or trial_idx == n_target - 1:
            elapsed = time.time() - t_start
            rate = (trial_idx + 1) / elapsed if elapsed > 0 else 0.0
            mean_str = f"{np.mean(valid_values):.2f}" if valid_values else "N/A"
            sd_str = (f"{np.std(valid_values, ddof=1):.2f}"
                      if len(valid_values) > 1 else "N/A")
            print(f"  [{trial_idx + 1:>3}/{n_target}] mean={mean_str} SD={sd_str}  "
                  f"rate={rate * 60:.1f}/min  valid={len(valid_values)}",
                  flush=True)

    t_total = time.time() - t_start
    print(f"\n  Condition {cond_id} done: {n_done} trials in {t_total/60:.1f}min")
    print(f"  Parse failures: {n_failed}")
    if valid_values:
        print(f"  Valid: N={len(valid_values)}, "
              f"mean={np.mean(valid_values):.3f}, "
              f"SD={np.std(valid_values, ddof=1):.3f}")

    return {
        "condition_id": cond_id,
        "condition_label": cond_label,
        "n_target": n_target,
        "n_done": n_done,
        "n_failed": n_failed,
        "n_valid": len(valid_values),
        "mean": float(np.mean(valid_values)) if valid_values else None,
        "sd": float(np.std(valid_values, ddof=1)) if len(valid_values) > 1 else None,
        "elapsed_sec": t_total,
    }


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: 5 trials per condition (35 total).")
    parser.add_argument("--condition", choices=[c[0] for c in CONDITIONS], default=None,
                        help="Run only one condition (default: all).")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_pilot" if args.pilot else ""
    out_dir = OUT_BASE / f"pie_oneshot_{timestamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"[config] Output dir: {out_dir}")
    print(f"[config] Prompts CSV: {PROMPTS_CSV}")
    all_prompts = load_prompts(Path(PROMPTS_CSV))
    print(f"[config] Total prompts loaded: {len(all_prompts)}")
    assert len(all_prompts) == 390, "Expected 390 prompts in v2 CSV"

    image = Image.open(STIMULUS_PATH).convert("RGB")
    print(f"[stim] Loaded image: {image.size}")

    # Save config
    config = {
        "script": "run_qwen25vl_pie_oneshot_a100.py",
        "model_id": MODEL_ID,
        "n_per_condition": N_PER_CONDITION,
        "pilot": args.pilot,
        "single_condition": args.condition,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stimulus_path": STIMULUS_PATH,
        "prompts_csv": PROMPTS_CSV,
        "timestamp": timestamp,
        "conditions": [{"id": c[0], "label": c[1], "category": c[2],
                        "placement": c[3], "seed_base": c[4]}
                       for c in CONDITIONS],
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Model
    model, processor, device = load_model_and_processor()

    # Filter conditions
    if args.condition:
        run_conds = [c for c in CONDITIONS if c[0] == args.condition]
    else:
        run_conds = CONDITIONS

    pilot_n = 5 if args.pilot else None

    # Run each condition
    summary_rows = []
    for cond in run_conds:
        summary = run_condition(cond, all_prompts, image, model, processor, device,
                                out_dir, pilot_n=pilot_n)
        summary_rows.append(summary)

    # Write summary CSV
    summary_csv = out_dir / "condition_summary.csv"
    if summary_rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            fields = list(summary_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    print(f"\n{'=' * 60}")
    print("ALL CONDITIONS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output dir: {out_dir}")
    print(f"\nFinal summary:")
    print(f"{'cond':<5} {'label':<22} {'N_valid':>8} {'mean':>7} {'SD':>6}")
    for s in summary_rows:
        sd_str = f"{s['sd']:.3f}" if s['sd'] is not None else "N/A"
        m_str = f"{s['mean']:.3f}" if s['mean'] is not None else "N/A"
        print(f"{s['condition_id']:<5} {s['condition_label']:<22} "
              f"{s['n_valid']:>8} {m_str:>7} {sd_str:>6}")


if __name__ == "__main__":
    main()
