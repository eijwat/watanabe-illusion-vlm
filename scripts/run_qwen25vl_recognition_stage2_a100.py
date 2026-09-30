"""
run_qwen25vl_recognition_stage2_a100.py
========================================
Stage 2 recognition probe: does Qwen2.5-VL-32B identify the 104.jpg
stimulus by name? Classical illusions serve as positive controls.

Background (paper_v11 claim under test):
    "本論文で初出の新規錯視であり、訓練データに直接的記述を含むとは
     考えにくい" (Watanabe Illusion is a new illusion first reported in
     this paper; the training data is unlikely to contain a direct
     description of it.)

This script provides empirical support for that claim by asking the
model whether it recognizes 104.jpg, with three classical illusions
(Müller-Lyer, Poggendorff, Kanizsa) as positive controls. If the model
correctly names the controls but cannot name 104.jpg, this supports the
"unfamiliar to current VLMs" framing.

Design:
- 4 stimuli x 3 questions x N=10 trials = 120 total turns
- Stimuli: 104.jpg (Watanabe) + 3 control illusions (Müller-Lyer, Poggendorff,
           Kanizsa) downloaded from Wikipedia by Eiji-san
- Questions:
    Q-V1: "Have you seen this figure before?" (yes/no/unsure)
    Q-V2: "Does this figure correspond to any named visual illusion?"
          (forced naming)
    Q-V3: "Describe what you see... Recognition:" (description + naming)
- Independent single-turn structure (no conversation history across turns)
- seed = stimulus_idx * 10000 + question_idx * 1000 + trial_idx
  (collision-free across all 120 trials)
- machine: A100 80GB (x86_64)

Output:
- /workspace/results/qwen25vl_recognition_stage2_a100/
    progress.json         (resume-safe: header + all entries so far)
    responses.jsonl       (append-only one line per turn)
    summary.csv           (parsed key columns, regenerated each run)

Expected runtime on A100: ~5-8 s / turn x 120 turns = ~10-15 min
(shorter max_new_tokens than cancellation experiment's 512)

Usage (A100, inside Docker):
    python scripts/run_qwen25vl_recognition_stage2_a100.py
    python scripts/run_qwen25vl_recognition_stage2_a100.py --n-trials 10 --resume
    python scripts/run_qwen25vl_recognition_stage2_a100.py --stimuli watanabe,poggendorff
"""

import os
import sys
import csv
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


# ========== Paths ==========
STIMULI_DIR = Path("/workspace/stimuli")
OUT_DIR = Path("/workspace/results/qwen25vl_recognition_stage2_a100")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ========== Model ==========
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_MEMORY = {0: "75GiB", "cpu": "0GiB"}


# ========== Generation ==========
TEMPERATURE = 1.0
TOP_P = 1.0
REPETITION_PENALTY = 1.05
MAX_NEW_TOKENS = 256  # shorter than cancellation (512); recognition answers are brief


# ========== Stimuli ==========
# Eiji-san manually downloads Wikipedia images for Müller-Lyer / Poggendorff /
# Kanizsa and places them at the following paths. 104.jpg already exists.
STIMULI = [
    {
        "id": "watanabe",
        "path": str(STIMULI_DIR / "104.jpg"),
        "true_name": "Watanabe Illusion",
        "name_aliases": ["watanabe"],  # lowercase keyword(s) for parser
        "is_control": False,
    },
    {
        "id": "muller_lyer",
        "path": str(STIMULI_DIR / "control_muller_lyer.png"),
        "true_name": "Müller-Lyer Illusion",
        "name_aliases": ["müller-lyer", "muller-lyer", "müller lyer",
                         "muller lyer", "mueller-lyer", "mueller lyer"],
        "is_control": True,
    },
    {
        "id": "poggendorff",
        "path": str(STIMULI_DIR / "control_poggendorff.png"),
        "true_name": "Poggendorff Illusion",
        "name_aliases": ["poggendorff"],
        "is_control": True,
    },
    {
        "id": "kanizsa",
        "path": str(STIMULI_DIR / "control_kanizsa.png"),
        "true_name": "Kanizsa Triangle / Kanizsa Illusion",
        "name_aliases": ["kanizsa"],
        "is_control": True,
    },
]
STIMULUS_BY_ID = {s["id"]: s for s in STIMULI}


# ========== Questions ==========
# Finalized prompts (no preamble — recognition is independent of the
# "visual impression alone" framing used in Q1/Q2/Q3 behavioral runs).
QUESTIONS = [
    {
        "id": "QV1",
        "type": "yes_no",
        "body": (
            "Have you seen this figure before? Answer 'yes', 'no', or "
            "'unsure' on the first line, then briefly explain in one "
            "sentence. Do not search the web. Answer only from your own "
            "knowledge."
        ),
    },
    {
        "id": "QV2",
        "type": "naming",
        "body": (
            "Does this figure correspond to any named visual illusion that "
            "you know? If yes, state its name on the first line. If no, "
            "write 'no'. Do not search the web."
        ),
    },
    {
        "id": "QV3",
        "type": "describe",
        "body": (
            "Describe what you see in this image in one sentence. Then, on "
            "a new line starting with 'Recognition:', state whether you "
            "recognize this as a specific published or named visual "
            "stimulus, and if so, name it. Do not search the web."
        ),
    },
]
QUESTION_BY_ID = {q["id"]: q for q in QUESTIONS}


# ========== Seed strategy ==========
def make_seed(stimulus_idx: int, question_idx: int, trial_idx: int) -> int:
    """Deterministic seed; non-overlapping across all 120 trials.

    With max trial_idx < 1000 and max question_idx < 10, no collision occurs
    even if N is increased substantially later.
    """
    return stimulus_idx * 10000 + question_idx * 1000 + trial_idx


# ========== 1 turn (no conversation history) ==========
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
    return processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()


# ========== Parsing ==========
def parse_yes_no(text: str) -> str:
    """Extract first-line yes/no/unsure label. Returns 'yes', 'no',
    'unsure', or 'unknown' if parsing fails."""
    if not text:
        return "unknown"
    first_line = text.strip().split("\n")[0].lower()
    # check at word boundary to avoid matching "no_match" etc
    if "yes" in first_line and "no" not in first_line:
        return "yes"
    if first_line.startswith("yes") or first_line == "yes.":
        return "yes"
    if first_line.startswith("no") or first_line == "no.":
        return "no"
    if "unsure" in first_line:
        return "unsure"
    # secondary: check for standalone tokens
    tokens = first_line.replace(".", " ").replace(",", " ").split()
    if "yes" in tokens:
        return "yes"
    if "no" in tokens:
        return "no"
    if "unsure" in tokens or "unknown" in tokens:
        return "unsure"
    return "unknown"


def detect_illusion_names(text: str) -> dict:
    """Detect mentions of target illusion names in the response.

    Returns dict mapping stimulus_id to bool (True if any of its
    name_aliases appears in the lowercased text).
    """
    if not text:
        return {s["id"]: False for s in STIMULI}
    lower = text.lower()
    detections = {}
    for s in STIMULI:
        detections[s["id"]] = any(alias in lower for alias in s["name_aliases"])
    return detections


# ========== Persistence ==========
def save_progress(out_path: Path, header: dict, entries: list):
    payload = {**header, "entries": entries}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_existing_progress(out_path: Path):
    if not out_path.exists():
        return None
    try:
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] failed to load existing progress: {e}")
        return None


def append_jsonl(out_path: Path, entry: dict):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_summary_csv(out_path: Path, entries: list):
    cols = [
        "stimulus_id", "true_name", "is_control",
        "question_id", "question_type",
        "trial_idx", "seed",
        "yes_no_label",
        "named_watanabe", "named_muller_lyer", "named_poggendorff",
        "named_kanizsa",
        "named_target",       # bool: response named the true illusion
        "named_wrong",        # bool: response named ANY OTHER target
        "response_len_chars",
        "elapsed_sec",
        "response_full",      # full text (kept for manual review)
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for e in entries:
            s = STIMULUS_BY_ID[e["stimulus_id"]]
            q = QUESTION_BY_ID[e["question_id"]]
            detections = e.get("detections", {})
            named_target = detections.get(e["stimulus_id"], False)
            named_wrong = any(
                v for k, v in detections.items() if k != e["stimulus_id"]
            )
            writer.writerow({
                "stimulus_id": e["stimulus_id"],
                "true_name": s["true_name"],
                "is_control": s["is_control"],
                "question_id": e["question_id"],
                "question_type": q["type"],
                "trial_idx": e["trial_idx"],
                "seed": e["seed"],
                "yes_no_label": e.get("yes_no_label", ""),
                "named_watanabe": detections.get("watanabe", False),
                "named_muller_lyer": detections.get("muller_lyer", False),
                "named_poggendorff": detections.get("poggendorff", False),
                "named_kanizsa": detections.get("kanizsa", False),
                "named_target": named_target,
                "named_wrong": named_wrong,
                "response_len_chars": len(e.get("response_full", "")),
                "elapsed_sec": round(e.get("elapsed_sec", 0.0), 2),
                "response_full": e.get("response_full", ""),
            })


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10,
                        help="trials per (stimulus, question) pair (default 10)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing progress.json if present")
    parser.add_argument("--stimuli", type=str, default=None,
                        help="comma-separated subset of stimulus IDs to run "
                             "(default: all 4)")
    parser.add_argument("--questions", type=str, default=None,
                        help="comma-separated subset of question IDs to run "
                             "(default: all 3)")
    args = parser.parse_args()

    # Filter stimuli
    if args.stimuli:
        wanted = set(s.strip() for s in args.stimuli.split(","))
        selected_stimuli = [s for s in STIMULI if s["id"] in wanted]
        if not selected_stimuli:
            print(f"[ERROR] no stimulus IDs match: {wanted}")
            sys.exit(1)
    else:
        selected_stimuli = STIMULI

    # Filter questions
    if args.questions:
        wanted = set(q.strip() for q in args.questions.split(","))
        selected_questions = [q for q in QUESTIONS if q["id"] in wanted]
        if not selected_questions:
            print(f"[ERROR] no question IDs match: {wanted}")
            sys.exit(1)
    else:
        selected_questions = QUESTIONS

    total = len(selected_stimuli) * len(selected_questions) * args.n_trials

    print("=" * 70)
    print("Stage 2 — Recognition probe (Qwen2.5-VL-32B on A100)")
    print("=" * 70)
    print(f"  model:      {MODEL_ID}")
    print(f"  out_dir:    {OUT_DIR}")
    print(f"  n_trials:   {args.n_trials}")
    print(f"  stimuli:    {[s['id'] for s in selected_stimuli]}")
    print(f"  questions:  {[q['id'] for q in selected_questions]}")
    print(f"  total turns: {total}")
    print()
    print(f"  T={TEMPERATURE}, top_p={TOP_P}, rep_penalty={REPETITION_PENALTY}, "
          f"max_new_tokens={MAX_NEW_TOKENS}")
    print(f"  seed strategy: stimulus_idx*10000 + question_idx*1000 + trial_idx")
    print()
    for q in selected_questions:
        snippet = (q["body"][:140] + "…") if len(q["body"]) > 140 else q["body"]
        print(f"  [{q['id']}] {q['type']}:")
        print(f"     {snippet!r}")
    print()

    # Check stimulus files
    missing = [s for s in selected_stimuli if not os.path.isfile(s["path"])]
    if missing:
        print("[ERROR] missing stimulus files:")
        for s in missing:
            print(f"  - {s['id']}: {s['path']}")
        print("\nEiji-san: please place Wikipedia-downloaded control images at:")
        for s in missing:
            if s["is_control"]:
                print(f"  {s['path']}")
        sys.exit(1)

    # Load (resume)
    out_json = OUT_DIR / "progress.json"
    out_jsonl = OUT_DIR / "responses.jsonl"
    out_csv = OUT_DIR / "summary.csv"

    existing = load_existing_progress(out_json) if args.resume else None
    if existing is not None:
        completed = existing.get("entries", [])
        completed_keys = {
            (e["stimulus_id"], e["question_id"], e["trial_idx"])
            for e in completed
        }
        print(f"[resume] {len(completed)} entries already completed")
    else:
        # Fresh run: clear any previous JSONL to keep file in sync
        completed = []
        completed_keys = set()
        if out_jsonl.exists():
            out_jsonl.unlink()
            print(f"  [fresh] cleared {out_jsonl}")

    # Build header
    header = {
        "started_at": existing.get("started_at") if existing
                      else datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "task": "Watanabe Illusion recognition probe (Stage 2)",
        "preprocessing": "none (Qwen2.5-VL native dynamic resolution)",
        "machine": "A100 80GB PCIe (x86_64)",
        "structure": "single-turn independent (no conversation history)",
        "n_trials_per_stim_x_question": args.n_trials,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_strategy": "seed = stimulus_idx * 10000 + question_idx * 1000 + trial_idx",
        "stimuli": [
            {"id": s["id"], "true_name": s["true_name"],
             "is_control": s["is_control"], "path": s["path"]}
            for s in STIMULI
        ],
        "questions": [
            {"id": q["id"], "type": q["type"], "body": q["body"]}
            for q in QUESTIONS
        ],
    }

    # Load model
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overall_start = time.time()

    # Pre-load images (each only once, reused across trials)
    images = {}
    for s in selected_stimuli:
        img = Image.open(s["path"]).convert("RGB")
        images[s["id"]] = img
        print(f"  loaded {s['id']}: {img.size}  ({s['path']})")

    # Main loop
    # Order: stimulus_idx outer, question_idx middle, trial_idx inner
    # (each trial fresh, no history; this order is convenient for monitoring)
    done = len(completed)
    for s_idx, s in enumerate(selected_stimuli):
        # Use *global* stimulus index (matches STIMULI list order) for seed
        global_s_idx = next(i for i, x in enumerate(STIMULI) if x["id"] == s["id"])
        for q_idx, q in enumerate(selected_questions):
            global_q_idx = next(i for i, x in enumerate(QUESTIONS) if x["id"] == q["id"])
            for trial_idx in range(args.n_trials):
                key = (s["id"], q["id"], trial_idx)
                if key in completed_keys:
                    continue

                seed = make_seed(global_s_idx, global_q_idx, trial_idx)
                t0 = time.time()
                try:
                    inputs = build_single_turn_inputs(
                        processor, images[s["id"]], q["body"]
                    )
                    response = generate_response(
                        model, processor, inputs, device, seed
                    )
                    elapsed = time.time() - t0
                    yes_no = parse_yes_no(response) if q["type"] == "yes_no" else ""
                    detections = detect_illusion_names(response)

                    entry = {
                        "stimulus_id": s["id"],
                        "question_id": q["id"],
                        "trial_idx": trial_idx,
                        "seed": seed,
                        "response_full": response,
                        "yes_no_label": yes_no,
                        "detections": detections,
                        "elapsed_sec": elapsed,
                        "timestamp": datetime.now().isoformat(),
                    }
                except Exception as e:
                    elapsed = time.time() - t0
                    entry = {
                        "stimulus_id": s["id"],
                        "question_id": q["id"],
                        "trial_idx": trial_idx,
                        "seed": seed,
                        "response_full": "",
                        "yes_no_label": "",
                        "detections": {sid: False for sid in
                                       (x["id"] for x in STIMULI)},
                        "elapsed_sec": elapsed,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat(),
                    }
                    print(f"  [ERROR] {s['id']} / {q['id']} / {trial_idx}: {e}")

                completed.append(entry)
                completed_keys.add(key)
                done += 1
                append_jsonl(out_jsonl, entry)
                save_progress(out_json, header, completed)
                write_summary_csv(out_csv, completed)

                # Compact progress line
                preview = (entry["response_full"][:60] + "…").replace("\n", " ")
                print(f"  [{done:>3d}/{total}]  {s['id']:>12s} / {q['id']} / "
                      f"trial {trial_idx:>2d}  ({elapsed:.1f}s)  "
                      f"yn={yes_no:<8s}  → {preview!r}")

    overall_elapsed = time.time() - overall_start
    print()
    print("=" * 70)
    print(f"Done. {done}/{total} entries.  "
          f"total elapsed: {overall_elapsed/60:.1f} min")
    print(f"  progress: {out_json}")
    print(f"  jsonl:    {out_jsonl}")
    print(f"  summary:  {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
