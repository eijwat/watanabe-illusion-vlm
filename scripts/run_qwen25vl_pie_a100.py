"""
run_qwen25vl_pie_a100.py
========================
Prompt Intervention Experiment (PIE) — Qwen2.5-VL-32B BF16 on A100.

Tests whether preamble prompts (300 prompts across Character / Environment / Neutral
categories) can cancel the mental extrapolation bias on the Watanabe Illusion.

Design:
- 300 preamble prompts from pie_prompts_v1.csv
- N=5 trials per prompt -> 1,500 trials per placement
- Two placement modes: --placement system  or  --placement user
- Q1 only (Q2/Q3 omitted for cost reasons)
- Q1 text matches run_qwen25vl_n60_v2.py exactly (PREAMBLE + Q1_BODY)

Inherited from run_qwen25vl_n60_v2.py:
- temperature = 1.0
- top_p = 1.0
- repetition_penalty = 1.05
- Single-turn inference (no conversation history)
- pixel_values reconstruction for input verification (1 trial only)

PIE-specific:
- Seed scheme: prompt_idx * 1000 + trial_idx (collision-free, reproducible)
- Per-prompt JSONL append (crash-resilient without explicit resume)
- summary.csv aggregates Q1 numeric answers per prompt for downstream analysis

Usage:
  # System message placement (recommended for primary analysis)
  python run_qwen25vl_pie_a100.py --placement system

  # User message placement (secondary, for comparison)
  python run_qwen25vl_pie_a100.py --placement user

  # Pilot: only first 3 prompts of each subcategory (36 prompts, ~10-15 min)
  python run_qwen25vl_pie_a100.py --placement system --pilot

Overnight execution (recommended):
  caffeinate python run_qwen25vl_pie_a100.py --placement system 2>&1 | tee pie_system.log
  caffeinate python run_qwen25vl_pie_a100.py --placement user   2>&1 | tee pie_user.log
"""

import os
import sys
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
PROMPTS_CSV = "/workspace/pie_prompts_v1.csv"
OUT_BASE = Path("/workspace/results")

# -------- Model --------
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_MEMORY = {0: "70GiB", "cpu": "0GiB"}  # A100 80GB headroom

# -------- Generation (must match run_qwen25vl_n60_v2.py) --------
TEMPERATURE = 1.0
TOP_P = 1.0
REPETITION_PENALTY = 1.05
MAX_NEW_TOKENS = 512

# -------- Experiment --------
N_TRIALS_PER_PROMPT = 5
PROMPT_SET_NAME = "human_G1G2G3_aligned_v1_with_PIE"

# -------- Q1 text (identical to run_qwen25vl_n60_v2.py) --------
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

# Max value for Q1 (number of dots on right edge, from stimulus_metadata.csv)
N_DOTS_RIGHT = 11


# ==========================================================================
# Prompt loading
# ==========================================================================

def load_prompts(csv_path: Path, pilot: bool = False):
    """Load PIE prompts from CSV. Returns list of dicts with keys:
    prompt_id, category, subcategory, subcategory_code, label, preamble, full_prompt.
    """
    prompts = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)

    if pilot:
        # Take first 3 prompts of each subcategory (12 subcategories x 3 = 36)
        from collections import defaultdict
        grouped = defaultdict(list)
        for p in prompts:
            grouped[p["subcategory_code"]].append(p)
        pilot_prompts = []
        for subcat in sorted(grouped.keys()):
            pilot_prompts.extend(grouped[subcat][:3])
        return pilot_prompts

    return prompts


# ==========================================================================
# Message construction (placement-aware)
# ==========================================================================

def build_messages(image, preamble: str, placement: str):
    """Build messages for Qwen2.5-VL given preamble and placement.

    Args:
        image: PIL.Image (the stimulus)
        preamble: the preamble text from pie_prompts_v1.csv
        placement: 'system' or 'user'

    Returns:
        messages list ready for apply_chat_template.
    """
    if placement == "system":
        return [
            {"role": "system", "content": preamble},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": Q1_USER_TEXT},
                ],
            },
        ]
    elif placement == "user":
        combined_text = f"{preamble}\n\n{Q1_USER_TEXT}"
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": combined_text},
            ],
        }]
    elif placement == "baseline":
        # No preamble — for sanity check that matches run_qwen25vl_n60_v2.py
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": Q1_USER_TEXT},
            ],
        }]
    else:
        raise ValueError(f"Unknown placement: {placement}")


def build_inputs(processor, image, preamble: str, placement: str):
    """Build model inputs (text + image) for a given preamble and placement."""
    messages = build_messages(image, preamble, placement)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return inputs, text


# ==========================================================================
# Inference
# ==========================================================================

def generate_response(model, processor, inputs, device, seed):
    """Stochastic generation with given seed (temperature=1.0, top_p=1.0)."""
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
# Response parsing
# ==========================================================================

# Regex patterns for extracting Q1 numeric answer.
# Strategy: scan for "Xth dot from the top/bottom" patterns, then extract X.
# Direction is recorded but normalization happens in post-analysis.

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

# Boxed answer (sometimes used)
BOXED_PATTERN = re.compile(r"\\boxed\{(\d+)\}")


def parse_q1_answer(response: str):
    """Extract Q1 numeric answer and direction from a response string.

    Returns:
        dict with keys: value (int or None), direction (str), parse_status (str)
        direction is one of: 'top', 'bottom', 'unknown'
        parse_status is one of: 'ok', 'boxed_only', 'ambiguous', 'no_match'
    """
    # Try "from the top" first
    matches_top = list(DOT_PATTERN_TOP.finditer(response))
    matches_bot = list(DOT_PATTERN_BOTTOM.finditer(response))

    if matches_top and not matches_bot:
        # Take last match (typically the conclusion)
        return {
            "value": int(matches_top[-1].group(1)),
            "direction": "top",
            "parse_status": "ok",
        }
    elif matches_bot and not matches_top:
        return {
            "value": int(matches_bot[-1].group(1)),
            "direction": "bottom",
            "parse_status": "ok",
        }
    elif matches_top and matches_bot:
        # Both found — take the last one overall
        last_top = matches_top[-1]
        last_bot = matches_bot[-1]
        if last_top.start() > last_bot.start():
            return {
                "value": int(last_top.group(1)),
                "direction": "top",
                "parse_status": "ambiguous",
            }
        else:
            return {
                "value": int(last_bot.group(1)),
                "direction": "bottom",
                "parse_status": "ambiguous",
            }
    else:
        # Fallback: boxed answer (direction unknown)
        boxed = BOXED_PATTERN.findall(response)
        if boxed:
            return {
                "value": int(boxed[-1]),
                "direction": "unknown",
                "parse_status": "boxed_only",
            }
        return {
            "value": None,
            "direction": "unknown",
            "parse_status": "no_match",
        }


def normalize_to_top(value, direction, n_dots=N_DOTS_RIGHT):
    """Normalize a Q1 answer to 'from the top' direction.
    If direction is 'bottom', convert. If 'top', leave as-is.
    Returns None if value is None or out of range.
    """
    if value is None:
        return None
    if value < 1 or value > n_dots:
        return None  # out of range
    if direction == "bottom":
        return n_dots - value + 1
    elif direction == "top":
        return value
    else:
        # direction unknown — return None for safety
        return None


# ==========================================================================
# Model loading
# ==========================================================================

def load_model_and_processor():
    """Load Qwen2.5-VL-32B in BF16 with explicit memory limits."""
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
    print(f"[load] Model device: {device}")
    return model, processor, device


# ==========================================================================
# Verification (called once at trial 0)
# ==========================================================================

def save_chat_template_sample(processor, image, preamble, placement, out_dir):
    """Save the resolved chat template text for one example, for sanity-checking."""
    _, text = build_inputs(processor, image, preamble, placement)
    with (out_dir / "chat_template_sample.txt").open("w", encoding="utf-8") as f:
        f.write(f"=== placement: {placement} ===\n")
        f.write(f"=== preamble: {preamble!r} ===\n\n")
        f.write(text)
    print(f"  [verify] saved chat_template_sample.txt")


# ==========================================================================
# Main experiment loop
# ==========================================================================

def run_experiment(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / f"pie_{args.placement}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup output files
    results_jsonl = out_dir / "results.jsonl"
    summary_csv = out_dir / "summary.csv"
    config_json = out_dir / "config.json"

    # Load prompts
    prompts = load_prompts(Path(PROMPTS_CSV), pilot=args.pilot)
    n_prompts = len(prompts)
    total_trials = n_prompts * N_TRIALS_PER_PROMPT

    print(f"[config] placement: {args.placement}")
    print(f"[config] n_prompts: {n_prompts} ({'PILOT' if args.pilot else 'FULL'})")
    print(f"[config] n_trials_per_prompt: {N_TRIALS_PER_PROMPT}")
    print(f"[config] total_trials: {total_trials}")
    print(f"[config] temperature: {TEMPERATURE}")
    print(f"[config] top_p: {TOP_P}")
    print(f"[config] repetition_penalty: {REPETITION_PENALTY}")
    print(f"[config] output: {out_dir}")

    # Save config
    config = {
        "script": "run_qwen25vl_pie_a100.py",
        "model_id": MODEL_ID,
        "placement": args.placement,
        "n_prompts": n_prompts,
        "n_trials_per_prompt": N_TRIALS_PER_PROMPT,
        "total_trials": total_trials,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stimulus_path": STIMULUS_PATH,
        "prompts_csv": PROMPTS_CSV,
        "q1_user_text": Q1_USER_TEXT,
        "pilot": args.pilot,
        "timestamp": timestamp,
        "prompt_set_name": PROMPT_SET_NAME,
    }
    with config_json.open("w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Load image
    image = Image.open(STIMULUS_PATH).convert("RGB")
    print(f"[stim] Loaded image: size={image.size}")

    # Load model
    model, processor, device = load_model_and_processor()

    # Save chat template sample for sanity check
    save_chat_template_sample(processor, image, prompts[0]["preamble"],
                              args.placement, out_dir)

    # Init summary CSV
    summary_fields = [
        "prompt_id", "category", "subcategory", "subcategory_code", "label",
        "n_trials", "n_parsed_top", "n_parsed_bottom", "n_parsed_unknown",
        "n_parse_failed", "mean_normalized", "std_normalized",
        "raw_values_top_str",  # comma-separated normalized values
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()

    # Main loop
    t_start = time.time()
    total_done = 0

    for p_idx, prompt in enumerate(prompts):
        prompt_id = prompt["prompt_id"]
        preamble = prompt["preamble"]

        prompt_results = []
        normalized_values = []
        n_parsed_top = 0
        n_parsed_bottom = 0
        n_parsed_unknown = 0
        n_parse_failed = 0

        for trial_idx in range(N_TRIALS_PER_PROMPT):
            seed = p_idx * 1000 + trial_idx

            try:
                inputs, _ = build_inputs(processor, image, preamble, args.placement)
                t_trial = time.time()
                response = generate_response(model, processor, inputs, device, seed)
                t_elapsed = time.time() - t_trial

                parsed = parse_q1_answer(response)
                normalized = normalize_to_top(parsed["value"], parsed["direction"])

                if parsed["parse_status"] == "no_match":
                    n_parse_failed += 1
                elif parsed["direction"] == "top":
                    n_parsed_top += 1
                elif parsed["direction"] == "bottom":
                    n_parsed_bottom += 1
                else:
                    n_parsed_unknown += 1

                if normalized is not None:
                    normalized_values.append(normalized)

                trial_record = {
                    "prompt_id": prompt_id,
                    "trial_idx": trial_idx,
                    "seed": seed,
                    "response": response,
                    "parsed_value": parsed["value"],
                    "parsed_direction": parsed["direction"],
                    "parse_status": parsed["parse_status"],
                    "normalized_top": normalized,
                    "elapsed_sec": round(t_elapsed, 2),
                    "error": None,
                }
                prompt_results.append(trial_record)

                # Append to JSONL immediately (crash resilience)
                with results_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(trial_record, ensure_ascii=False) + "\n")

                total_done += 1

            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                print(f"  [ERROR] {prompt_id} trial {trial_idx}: {err_msg}")
                print(tb)
                n_parse_failed += 1
                err_record = {
                    "prompt_id": prompt_id,
                    "trial_idx": trial_idx,
                    "seed": seed,
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

        # Per-prompt summary
        if normalized_values:
            mean_norm = float(np.mean(normalized_values))
            std_norm = float(np.std(normalized_values, ddof=1)) if len(normalized_values) > 1 else 0.0
        else:
            mean_norm = None
            std_norm = None

        summary_row = {
            "prompt_id": prompt_id,
            "category": prompt["category"],
            "subcategory": prompt["subcategory"],
            "subcategory_code": prompt["subcategory_code"],
            "label": prompt["label"],
            "n_trials": N_TRIALS_PER_PROMPT,
            "n_parsed_top": n_parsed_top,
            "n_parsed_bottom": n_parsed_bottom,
            "n_parsed_unknown": n_parsed_unknown,
            "n_parse_failed": n_parse_failed,
            "mean_normalized": mean_norm,
            "std_normalized": std_norm,
            "raw_values_top_str": ",".join(str(v) for v in normalized_values),
        }
        with summary_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields, quoting=csv.QUOTE_ALL)
            writer.writerow(summary_row)

        # Progress log
        elapsed = time.time() - t_start
        rate = total_done / elapsed if elapsed > 0 else 0.0
        eta_sec = (total_trials - total_done) / rate if rate > 0 else 0.0
        eta_h = eta_sec / 3600
        mean_str = f"{mean_norm:.2f}" if mean_norm is not None else "N/A"
        print(f"[{p_idx + 1:>3}/{n_prompts}] {prompt_id} "
              f"({prompt['label'][:30]:30s}) "
              f"mean={mean_str}  rate={rate*60:.1f}/min  ETA={eta_h:.2f}h",
              flush=True)

    t_total = time.time() - t_start
    print(f"\n[done] Total trials: {total_done}/{total_trials}")
    print(f"[done] Total time: {t_total/3600:.2f}h ({t_total:.0f}s)")
    print(f"[done] Output dir: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement", choices=["system", "user"], required=True,
                        help="Where to place the preamble: system message or user message.")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: 3 prompts per subcategory (36 total).")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
