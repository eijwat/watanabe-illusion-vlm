"""
run_qwen25vl_cancellation_a100_q1.py
=====================================
Phase 1: prompt cancellation experiment to ask *what* Qwen2.5-VL-32B knows
that produces the mental-extension bias on the Watanabe Illusion.

Design:
- 8 prompt conditions, N = 20 stochastic trials each = 160 total turns
- Q1 only (from-top); the from-bottom direction (Q2) is reserved for a
  follow-up if Q1 shows clear effects
- Independent single-turn structure (no conversation history)
- seed = prompt_id * 1000 + trial_idx (no collision across prompts)
- machine: A100 80GB (x86_64)

Prompt set (decided 2026-05-21 with Eiji-san):
- P0: Control. Matches run_qwen25vl_n60_a100_q1q2.py Q1.
- P1: "Watanabe Illusion" named explicitly (probe specific knowledge)
- P2: "visual illusion" named (probe generic illusion class knowledge)
- P3: Knowledge suppression ("ignore prior knowledge, pixel geometry only")
- P4: Counter-hint ("typically hits a dot near the top")
- P5: CoT-style ("first describe the slope, then answer")
- P6: Ground-truth disclosure ("the correct answer is the 1st dot... what is YOUR estimate?")
- P7: Random distractor (unrelated noise tokens around the question)

Expected runtime on A100: ~14 s / turn x 160 turns = ~37 min.

Usage (A100, inside Docker):
    python scripts/run_qwen25vl_cancellation_a100_q1.py --n-trials 20
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ========== 設定 ==========
STIMULUS_PATH = "/workspace/stimuli/104.jpg"
OUT_DIR = Path("/workspace/results/qwen25vl_cancellation_v1_a100")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_MEMORY = {0: "75GiB", "cpu": "0GiB"}

TEMPERATURE = 1.0
TOP_P = 1.0
REPETITION_PENALTY = 1.05
MAX_NEW_TOKENS = 512


# ========== プロンプト定義 ==========
PREAMBLE = (
    "Look at the image below straight on from the front. "
    "Please answer based on your visual impression alone — "
    "do not use any tools such as Python, image processing, "
    "a ruler, or a protractor to measure."
)

Q1_TASK = (
    "Estimate which dot on the right edge the line in the circle would hit "
    "if extended. Answer like 'the Xth dot from the top.'"
)

PROMPT_SET = [
    {
        "id": "P0",
        "category": "control",
        "body": Q1_TASK,
        "comment": "Baseline. Matches run_qwen25vl_n60_a100_q1q2.py Q1.",
    },
    {
        "id": "P1",
        "category": "name_specific",
        "body": (
            "The image below shows the Watanabe Illusion. "
            + Q1_TASK
        ),
        "comment": "Specific illusion name. Tests whether the model has "
                   "knowledge of this particular illusion.",
    },
    {
        "id": "P2",
        "category": "name_generic",
        "body": (
            "The image below contains a visual illusion. "
            + Q1_TASK
        ),
        "comment": "Abstract illusion class. Tests generic 'illusion' "
                   "knowledge.",
    },
    {
        "id": "P3",
        "category": "knowledge_suppress",
        "body": (
            "Ignore any prior knowledge about visual illusions or typical "
            "geometric figures. Answer based only on the line's pixel-level "
            "geometry. "
            + Q1_TASK
        ),
        "comment": "Knowledge suppression. If bias shrinks here, "
                   "language-knowledge pathway is implicated.",
    },
    {
        "id": "P4",
        "category": "counter_hint",
        "body": (
            "For figures like this one, the line typically hits a dot near "
            "the top of the right edge. "
            + Q1_TASK
        ),
        "comment": "Counter-prior in the upward direction. Tests whether "
                   "linguistic priors can override the bias.",
    },
    {
        "id": "P5",
        "category": "force_visual_cot",
        "body": (
            "First describe the line's slope in your own words. Then "
            + Q1_TASK
        ),
        "comment": "Force the model to verbalize the visual percept before "
                   "committing to an answer.",
    },
    {
        "id": "P6",
        "category": "ground_truth_disclose",
        "body": (
            "The correct answer is the 1st dot from the top. Independently, "
            "what is your own visual estimate? "
            + Q1_TASK
        ),
        "comment": "Ground-truth disclosure. Tests whether the model can "
                   "report a visual estimate that diverges from disclosed "
                   "knowledge.",
    },
    {
        "id": "P7",
        "category": "random_distractor",
        "body": (
            "While listening to instrumental music in a quiet library on a "
            "Tuesday afternoon, "
            + Q1_TASK
        ),
        "comment": "Unrelated context tokens. Should match P0 if the noise "
                   "tokens carry no relevant signal.",
    },
]

assert len({p["id"] for p in PROMPT_SET}) == len(PROMPT_SET), "duplicate IDs"


def assemble_prompt(prompt_body):
    """Final prompt = preamble + double-newline + body."""
    return f"{PREAMBLE}\n\n{prompt_body}"


# ========== 1 ターン (Q1 only, no history) ==========
def build_single_turn_inputs(processor, image, prompt_text):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    )
    return inputs


def generate_response(model, processor, inputs, device, seed):
    inputs = {k: v.to(device) if hasattr(v, 'to') else v
              for k, v in inputs.items()}
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
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ========== 永続化 ==========
def save_progress(out_path, header, entries):
    payload = {**header, "entries": entries}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_existing_progress(out_path):
    if not out_path.exists():
        return None
    try:
        with open(out_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] failed to load existing progress: {e}")
        return None


# ========== main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20,
                        help="trials per prompt (default 20)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing progress JSON if present")
    parser.add_argument("--prompts", type=str, default=None,
                        help="comma-separated subset of prompt IDs to run "
                             "(default: all)")
    args = parser.parse_args()

    # フィルタ
    if args.prompts:
        wanted = set(p.strip() for p in args.prompts.split(","))
        selected = [p for p in PROMPT_SET if p["id"] in wanted]
        if not selected:
            print(f"[ERROR] no prompt IDs match: {wanted}")
            sys.exit(1)
    else:
        selected = PROMPT_SET

    print("=" * 70)
    print("Phase 1 — Prompt cancellation experiment (Q1 only)")
    print("=" * 70)
    print(f"  model:      {MODEL_ID}")
    print(f"  stimulus:   {STIMULUS_PATH}")
    print(f"  out_dir:    {OUT_DIR}")
    print(f"  n_trials per prompt: {args.n_trials}")
    print(f"  prompts: {[p['id'] for p in selected]}")
    print(f"  total turns: {len(selected) * args.n_trials}")
    print()
    print(f"  T={TEMPERATURE}, top_p={TOP_P}, rep_penalty={REPETITION_PENALTY}, "
          f"max_new_tokens={MAX_NEW_TOKENS}")
    print(f"  seed strategy: seed = prompt_id_index * 1000 + trial_idx")
    print()
    for p in selected:
        full = assemble_prompt(p["body"])
        snippet = (full[:160] + "…") if len(full) > 160 else full
        print(f"  [{p['id']}] {p['category']}:")
        print(f"     {snippet!r}")
    print()

    if not os.path.isfile(STIMULUS_PATH):
        print(f"[ERROR] stimulus not found: {STIMULUS_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_json = OUT_DIR / "cancellation_progress.json"
    existing = load_existing_progress(out_json) if args.resume else None
    if existing is not None:
        completed = existing.get("entries", [])
        completed_keys = {(e["prompt_id"], e["trial_idx"]) for e in completed}
        print(f"[resume] {len(completed)} entries already completed")
    else:
        completed = []
        completed_keys = set()

    header = {
        "started_at": existing.get("started_at") if existing else datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "stimulus": STIMULUS_PATH,
        "preprocessing": "none (Qwen2.5-VL native dynamic resolution)",
        "machine": "A100 80GB PCIe (x86_64)",
        "structure": "single-turn independent (Q1 only)",
        "n_trials_per_prompt": args.n_trials,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_strategy": "seed = prompt_index * 1000 + trial_idx",
        "preamble": PREAMBLE,
        "prompts": [
            {
                "id": p["id"],
                "category": p["category"],
                "body": p["body"],
                "full": assemble_prompt(p["body"]),
                "comment": p["comment"],
            }
            for p in PROMPT_SET
        ],
    }

    img = Image.open(STIMULUS_PATH).convert("RGB")
    print(f"stimulus loaded: {img.size}")

    print(f"\nLoading {MODEL_ID}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=MAX_MEMORY,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    param_devices = {}
    for _, param in model.named_parameters():
        dev = str(param.device)
        param_devices[dev] = param_devices.get(dev, 0) + 1
    print(f"  param device summary: {param_devices}")
    if any("cpu" in d for d in param_devices):
        print("  [WARN] some parameters on CPU!")
    header["param_device_summary"] = param_devices

    overall_start = time.time()

    # 全プロンプト内で安定した seed 配分にするため、
    # PROMPT_SET 上の index を使う (filtered list ではない)
    prompt_index_by_id = {p["id"]: i for i, p in enumerate(PROMPT_SET)}

    for prompt_spec in selected:
        pid = prompt_spec["id"]
        prompt_full = assemble_prompt(prompt_spec["body"])
        idx_in_full = prompt_index_by_id[pid]
        print(f"\n{'='*70}\n[{pid}] {prompt_spec['category']}  "
              f"(seed_base = {idx_in_full*1000})\n{'='*70}")

        for trial_idx in range(args.n_trials):
            key = (pid, trial_idx)
            if key in completed_keys:
                print(f"  trial {trial_idx+1}/{args.n_trials}: (skipped)")
                continue

            seed = idx_in_full * 1000 + trial_idx
            trial_start = time.time()
            try:
                inputs = build_single_turn_inputs(processor, img, prompt_full)
                resp = generate_response(model, processor, inputs, device, seed)
                entry = {
                    "prompt_id": pid,
                    "category": prompt_spec["category"],
                    "trial_idx": trial_idx,
                    "seed": seed,
                    "prompt": prompt_full,
                    "response": resp,
                    "elapsed_seconds": time.time() - trial_start,
                    "completed_at": datetime.now().isoformat(),
                }
                short = resp[:80].replace("\n", " ")
                print(f"  trial {trial_idx+1}/{args.n_trials}: "
                      f"{entry['elapsed_seconds']:.1f}s | {short}...")
            except Exception as e:
                print(f"  trial {trial_idx+1}/{args.n_trials}: [ERROR] {e}")
                traceback.print_exc()
                entry = {
                    "prompt_id": pid,
                    "category": prompt_spec["category"],
                    "trial_idx": trial_idx,
                    "seed": seed,
                    "prompt": prompt_full,
                    "error": str(e),
                    "elapsed_seconds": time.time() - trial_start,
                    "completed_at": datetime.now().isoformat(),
                }
            completed.append(entry)
            save_progress(out_json, header, completed)

    total_min = (time.time() - overall_start) / 60
    print(f"\n=== all done in {total_min:.1f} min ===")
    print(f"output: {out_json}")


if __name__ == "__main__":
    main()
