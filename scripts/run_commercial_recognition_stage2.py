"""
run_commercial_recognition_stage2.py
=====================================
Stage 2 recognition probe via commercial VLM APIs (Phase B counterpart to
run_qwen25vl_recognition_stage2_a100.py).

Background:
    The author's personal Web UI accounts (Claude.ai, ChatGPT.com,
    Gemini) are contaminated by prior conversations about the Watanabe
    Illusion. To probe recognition independently of that contamination,
    we query the latest API snapshots directly. Each API call is a fresh
    request with no conversation history.

    Note: this departs from paper_v11's behavioral protocol (which used
    Web UI in default Instant/Flash modes precisely because users
    experience those modes). The recognition probe is independent of
    that argument: here we want maximum opportunity for the model to
    name the figure, so we run with non-trivial reasoning effort.

Models (latest as of 2026-05):
    Anthropic:  claude-opus-4-7              (extended thinking default)
    OpenAI:     gpt-5.5  with reasoning.effort=medium
    Google:     gemini-3.5-flash  with thinking_level=medium

Design:
    - 3 models x 4 stimuli x 3 questions x N=10 = 360 API calls
    - Identical questions to Phase A
    - Independent single calls (no conversation history per call)
    - temperature=1.0 where settable
    - Resume-safe (progress.json + JSONL append + summary.csv regen)

Stimuli (same as Phase A):
    /stimuli/104.jpg
    /stimuli/control_muller_lyer.png
    /stimuli/control_poggendorff.png
    /stimuli/control_kanizsa.png

API keys (environment):
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GOOGLE_API_KEY  (or GEMINI_API_KEY)

Expected cost: ~$30-$50 total (3 providers).

Usage (MacBook, with caffeinate):
    caffeinate -i python run_commercial_recognition_stage2.py
    python run_commercial_recognition_stage2.py --models anthropic
    python run_commercial_recognition_stage2.py --n-trials 10 --resume
    python run_commercial_recognition_stage2.py --pilot   # 1 trial each

Required packages:
    pip install anthropic openai google-genai pillow
"""

import os
import sys
import csv
import json
import time
import base64
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# ========== Paths ==========
# On the MacBook, mirror the Phase A layout under a project-local stimuli dir.
# Override via --stimuli-dir if the layout is different.
DEFAULT_STIMULI_DIR = Path.home() / "watanabe-illusion" / "stimuli"
DEFAULT_OUT_DIR = Path.home() / "watanabe-illusion" / "results" / \
                  "commercial_recognition_stage2"


# ========== Generation (where settable) ==========
TEMPERATURE = 1.0
MAX_TOKENS = 1024  # commercial models with reasoning can produce more


# ========== Stimuli (same as Phase A) ==========
def make_stimuli_list(stimuli_dir: Path):
    return [
        {"id": "watanabe",
         "path": str(stimuli_dir / "104.jpg"),
         "true_name": "Watanabe Illusion",
         "name_aliases": ["watanabe"],
         "is_control": False},
        {"id": "muller_lyer",
         "path": str(stimuli_dir / "control_muller_lyer.png"),
         "true_name": "Müller-Lyer Illusion",
         "name_aliases": ["müller-lyer", "muller-lyer", "müller lyer",
                          "muller lyer", "mueller-lyer", "mueller lyer"],
         "is_control": True},
        {"id": "poggendorff",
         "path": str(stimuli_dir / "control_poggendorff.png"),
         "true_name": "Poggendorff Illusion",
         "name_aliases": ["poggendorff"],
         "is_control": True},
        {"id": "kanizsa",
         "path": str(stimuli_dir / "control_kanizsa.png"),
         "true_name": "Kanizsa Triangle / Kanizsa Illusion",
         "name_aliases": ["kanizsa"],
         "is_control": True},
    ]


# ========== Questions (identical to Phase A) ==========
QUESTIONS = [
    {"id": "QV1", "type": "yes_no",
     "body": (
         "Have you seen this figure before? Answer 'yes', 'no', or "
         "'unsure' on the first line, then briefly explain in one "
         "sentence. Do not search the web. Answer only from your own "
         "knowledge."
     )},
    {"id": "QV2", "type": "naming",
     "body": (
         "Does this figure correspond to any named visual illusion that "
         "you know? If yes, state its name on the first line. If no, "
         "write 'no'. Do not search the web."
     )},
    {"id": "QV3", "type": "describe",
     "body": (
         "Describe what you see in this image in one sentence. Then, on "
         "a new line starting with 'Recognition:', state whether you "
         "recognize this as a specific published or named visual "
         "stimulus, and if so, name it. Do not search the web."
     )},
]


# ========== Model registry ==========
MODELS = [
    {
        "key": "anthropic",
        "model_id": "claude-opus-4-7",
        "label": "Claude Opus 4.7",
        "env_var": "ANTHROPIC_API_KEY",
        "reasoning_setting": "extended_thinking_default",
    },
    {
        "key": "openai",
        "model_id": "gpt-5.5",
        "label": "GPT-5.5",
        "env_var": "OPENAI_API_KEY",
        "reasoning_setting": "reasoning.effort=medium",
    },
    {
        "key": "google",
        "model_id": "gemini-3.5-flash",
        "label": "Gemini 3.5 Flash",
        "env_var": "GOOGLE_API_KEY",  # also accepts GEMINI_API_KEY
        "reasoning_setting": "thinking_level=medium",
    },
]
MODEL_BY_KEY = {m["key"]: m for m in MODELS}


# ========== Seed strategy (record-only; APIs don't all support seeds) ==========
def make_seed(stimulus_idx: int, question_idx: int, trial_idx: int) -> int:
    return stimulus_idx * 10000 + question_idx * 1000 + trial_idx


# ========== Image encoding ==========
def load_image_b64(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a stimulus file."""
    with open(path, "rb") as f:
        data = f.read()
    if path.lower().endswith(".png"):
        media_type = "image/png"
    elif path.lower().endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    else:
        # default to jpeg if unknown — most are jpg/png
        media_type = "image/jpeg"
    return base64.b64encode(data).decode("ascii"), media_type


# ========== Provider-specific call wrappers ==========

def call_anthropic(model_id: str, image_b64: str, media_type: str,
                   prompt_text: str, max_tokens: int = MAX_TOKENS,
                   temperature: float = TEMPERATURE) -> dict:
    """Returns {"text": str, "raw": dict, "thinking": str|None}."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": media_type,
                            "data": image_b64}},
                {"type": "text", "text": prompt_text},
            ],
        }],
    )
    text_parts = []
    thinking_parts = []
    for block in msg.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "thinking":
            thinking_parts.append(getattr(block, "thinking", ""))
    return {
        "text": "\n".join(text_parts).strip(),
        "thinking": "\n".join(thinking_parts).strip() if thinking_parts else None,
        "raw": msg.model_dump() if hasattr(msg, "model_dump") else str(msg),
    }


def call_openai(model_id: str, image_b64: str, media_type: str,
                prompt_text: str, max_tokens: int = MAX_TOKENS,
                temperature: float = TEMPERATURE) -> dict:
    """Returns {"text": str, "raw": dict, "reasoning": str|None}.

    Uses the Responses API with reasoning.effort=medium.
    """
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY
    data_url = f"data:{media_type};base64,{image_b64}"
    resp = client.responses.create(
        model=model_id,
        reasoning={"effort": "medium"},
        max_output_tokens=max_tokens,
        # Note: GPT-5.5 reasoning models may ignore temperature; we set it
        # but don't rely on it.
        temperature=temperature,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    # Prefer output_text helper, fall back to manual concat
    text = getattr(resp, "output_text", None)
    if not text:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        chunks.append(c.text)
        text = "\n".join(chunks).strip()
    return {
        "text": (text or "").strip(),
        "reasoning": None,  # reasoning tokens not exposed as text by default
        "raw": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
    }


def call_google(model_id: str, image_b64: str, media_type: str,
                prompt_text: str, max_tokens: int = MAX_TOKENS,
                temperature: float = TEMPERATURE) -> dict:
    """Returns {"text": str, "raw": dict, "thinking": str|None}.

    Uses google-genai with thinking_level=medium.
    """
    from google import genai
    from google.genai import types as gtypes

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    image_part = gtypes.Part.from_bytes(
        data=base64.b64decode(image_b64), mime_type=media_type
    )
    config = gtypes.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=gtypes.ThinkingConfig(thinking_level="medium"),
    )
    resp = client.models.generate_content(
        model=model_id,
        contents=[image_part, prompt_text],
        config=config,
    )
    text = getattr(resp, "text", "") or ""
    return {
        "text": text.strip(),
        "thinking": None,
        "raw": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
    }


PROVIDER_DISPATCH = {
    "anthropic": call_anthropic,
    "openai":    call_openai,
    "google":    call_google,
}


# ========== Retry wrapper ==========
def call_with_retry(provider_key: str, *args, max_attempts: int = 4,
                    base_delay: float = 2.0, **kwargs) -> dict:
    """Exponential backoff retry around the provider-specific call.

    Returns the call result on success, raises after max_attempts.
    """
    fn = PROVIDER_DISPATCH[provider_key]
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt + 1 == max_attempts:
                break
            delay = base_delay * (2 ** attempt)
            print(f"    [retry] attempt {attempt+1} failed: "
                  f"{type(e).__name__}: {e}.  sleeping {delay:.1f}s")
            time.sleep(delay)
    raise last_exc


# ========== Parsing (same logic as Phase A) ==========
def parse_yes_no(text: str) -> str:
    if not text:
        return "unknown"
    first_line = text.strip().split("\n")[0].lower()
    if first_line.startswith("yes") or first_line == "yes.":
        return "yes"
    if first_line.startswith("no") or first_line == "no.":
        return "no"
    if "unsure" in first_line:
        return "unsure"
    tokens = first_line.replace(".", " ").replace(",", " ").split()
    if "yes" in tokens:
        return "yes"
    if "no" in tokens:
        return "no"
    if "unsure" in tokens or "unknown" in tokens:
        return "unsure"
    if "yes" in first_line and "no" not in first_line:
        return "yes"
    return "unknown"


def detect_illusion_names(text: str, stimuli: list) -> dict:
    if not text:
        return {s["id"]: False for s in stimuli}
    lower = text.lower()
    return {
        s["id"]: any(alias in lower for alias in s["name_aliases"])
        for s in stimuli
    }


# ========== Persistence (same pattern as Phase A) ==========
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


def write_summary_csv(out_path: Path, entries: list, stimuli: list):
    stimulus_by_id = {s["id"]: s for s in stimuli}
    question_by_id = {q["id"]: q for q in QUESTIONS}
    cols = [
        "model_key", "model_id", "model_label",
        "stimulus_id", "true_name", "is_control",
        "question_id", "question_type",
        "trial_idx", "seed",
        "yes_no_label",
        "named_watanabe", "named_muller_lyer", "named_poggendorff",
        "named_kanizsa",
        "named_target", "named_wrong",
        "response_len_chars",
        "elapsed_sec",
        "error",
        "response_full",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for e in entries:
            s = stimulus_by_id[e["stimulus_id"]]
            q = question_by_id[e["question_id"]]
            m = MODEL_BY_KEY[e["model_key"]]
            detections = e.get("detections", {})
            named_target = detections.get(e["stimulus_id"], False)
            named_wrong = any(
                v for k, v in detections.items() if k != e["stimulus_id"]
            )
            writer.writerow({
                "model_key": e["model_key"],
                "model_id": m["model_id"],
                "model_label": m["label"],
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
                "error": e.get("error", ""),
                "response_full": e.get("response_full", ""),
            })


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli-dir", type=str,
                        default=str(DEFAULT_STIMULI_DIR),
                        help=f"Directory containing 104.jpg and "
                             f"control_*.png files "
                             f"(default: {DEFAULT_STIMULI_DIR})")
    parser.add_argument("--out-dir", type=str,
                        default=str(DEFAULT_OUT_DIR),
                        help=f"Output directory "
                             f"(default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="trials per (model, stimulus, question) "
                             "(default 10)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing progress.json if present")
    parser.add_argument("--models", type=str, default=None,
                        help="comma-separated subset of model keys: "
                             "anthropic, openai, google")
    parser.add_argument("--stimuli", type=str, default=None,
                        help="comma-separated subset of stimulus IDs")
    parser.add_argument("--questions", type=str, default=None,
                        help="comma-separated subset of question IDs")
    parser.add_argument("--pilot", action="store_true",
                        help="pilot mode: N=1 trial per (model, stim, Q)")
    parser.add_argument("--sleep-sec", type=float, default=0.5,
                        help="sleep between calls to avoid rate limits "
                             "(default 0.5s)")
    args = parser.parse_args()

    if args.pilot:
        args.n_trials = 1

    stimuli_dir = Path(args.stimuli_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build stimuli list
    stimuli = make_stimuli_list(stimuli_dir)
    stimulus_by_id = {s["id"]: s for s in stimuli}

    # Filter
    if args.models:
        wanted = set(s.strip() for s in args.models.split(","))
        selected_models = [m for m in MODELS if m["key"] in wanted]
        if not selected_models:
            print(f"[ERROR] no model keys match: {wanted}")
            sys.exit(1)
    else:
        selected_models = MODELS

    if args.stimuli:
        wanted = set(s.strip() for s in args.stimuli.split(","))
        selected_stimuli = [s for s in stimuli if s["id"] in wanted]
        if not selected_stimuli:
            print(f"[ERROR] no stimulus IDs match: {wanted}")
            sys.exit(1)
    else:
        selected_stimuli = stimuli

    if args.questions:
        wanted = set(s.strip() for s in args.questions.split(","))
        selected_questions = [q for q in QUESTIONS if q["id"] in wanted]
        if not selected_questions:
            print(f"[ERROR] no question IDs match: {wanted}")
            sys.exit(1)
    else:
        selected_questions = QUESTIONS

    total = (len(selected_models) * len(selected_stimuli)
             * len(selected_questions) * args.n_trials)

    print("=" * 70)
    print("Stage 2 — Recognition probe (commercial VLM APIs, Phase B)")
    print("=" * 70)
    print(f"  out_dir:    {out_dir}")
    print(f"  stimuli_dir: {stimuli_dir}")
    print(f"  n_trials:   {args.n_trials}")
    print(f"  models:     {[m['key'] for m in selected_models]}")
    print(f"  stimuli:    {[s['id'] for s in selected_stimuli]}")
    print(f"  questions:  {[q['id'] for q in selected_questions]}")
    print(f"  total API calls: {total}")
    print()
    print(f"  T={TEMPERATURE} (where settable), max_tokens={MAX_TOKENS}")
    print(f"  reasoning: Anthropic=extended_thinking_default, "
          f"OpenAI=effort=medium, Google=thinking_level=medium")
    print(f"  sleep between calls: {args.sleep_sec}s")
    print()

    # Check stimulus files
    missing = [s for s in selected_stimuli if not os.path.isfile(s["path"])]
    if missing:
        print("[ERROR] missing stimulus files:")
        for s in missing:
            print(f"  - {s['id']}: {s['path']}")
        sys.exit(1)

    # Check API keys
    missing_keys = []
    for m in selected_models:
        env_var = m["env_var"]
        if not os.environ.get(env_var):
            # google special case: also accept GEMINI_API_KEY
            if m["key"] == "google" and os.environ.get("GEMINI_API_KEY"):
                continue
            missing_keys.append((m["key"], env_var))
    if missing_keys:
        print("[ERROR] missing API keys:")
        for k, env in missing_keys:
            print(f"  - {k}: set ${env}")
        sys.exit(1)

    # Pre-load and encode stimulus images
    print("Loading stimulus images...")
    image_b64 = {}  # stim_id -> (b64, media_type)
    for s in selected_stimuli:
        b64, mt = load_image_b64(s["path"])
        image_b64[s["id"]] = (b64, mt)
        print(f"  {s['id']:>12s}: {len(b64)//1024} KB b64, {mt}, "
              f"{s['path']}")
    print()

    # Resume
    out_json = out_dir / "progress.json"
    out_jsonl = out_dir / "responses.jsonl"
    out_csv = out_dir / "summary.csv"

    existing = load_existing_progress(out_json) if args.resume else None
    if existing is not None:
        completed = existing.get("entries", [])
        completed_keys = {
            (e["model_key"], e["stimulus_id"], e["question_id"], e["trial_idx"])
            for e in completed
        }
        print(f"[resume] {len(completed)} entries already completed")
    else:
        completed = []
        completed_keys = set()
        if out_jsonl.exists():
            out_jsonl.unlink()
            print(f"  [fresh] cleared {out_jsonl}")

    # Header
    header = {
        "started_at": existing.get("started_at") if existing
                      else datetime.now().isoformat(),
        "task": "Watanabe Illusion recognition probe (Stage 2, Phase B commercial)",
        "structure": "single API call independent (no conversation history)",
        "n_trials_per_combo": args.n_trials,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed_strategy": "stimulus_idx*10000 + question_idx*1000 + trial_idx (recorded only; not all APIs accept seed)",
        "models": MODELS,
        "stimuli": [
            {"id": s["id"], "true_name": s["true_name"],
             "is_control": s["is_control"], "path": s["path"]}
            for s in stimuli
        ],
        "questions": QUESTIONS,
    }

    overall_start = time.time()
    done = len(completed)

    # Main loop: model outer (to amortize SDK init), then stim, then question, trial
    for m in selected_models:
        print(f"\n--- {m['label']} ({m['model_id']}) ---")
        for s in selected_stimuli:
            global_s_idx = next(i for i, x in enumerate(stimuli)
                                if x["id"] == s["id"])
            b64, media_type = image_b64[s["id"]]
            for q in selected_questions:
                global_q_idx = next(i for i, x in enumerate(QUESTIONS)
                                    if x["id"] == q["id"])
                for trial_idx in range(args.n_trials):
                    key = (m["key"], s["id"], q["id"], trial_idx)
                    if key in completed_keys:
                        continue

                    seed = make_seed(global_s_idx, global_q_idx, trial_idx)
                    t0 = time.time()
                    try:
                        result = call_with_retry(
                            m["key"], m["model_id"], b64, media_type, q["body"],
                        )
                        elapsed = time.time() - t0
                        text = result["text"]
                        yes_no = parse_yes_no(text) if q["type"] == "yes_no" else ""
                        detections = detect_illusion_names(text, stimuli)

                        entry = {
                            "model_key": m["key"],
                            "model_id": m["model_id"],
                            "stimulus_id": s["id"],
                            "question_id": q["id"],
                            "trial_idx": trial_idx,
                            "seed": seed,
                            "response_full": text,
                            "yes_no_label": yes_no,
                            "detections": detections,
                            "elapsed_sec": elapsed,
                            "thinking": result.get("thinking"),
                            "timestamp": datetime.now().isoformat(),
                            "error": "",
                        }
                    except Exception as e:
                        elapsed = time.time() - t0
                        entry = {
                            "model_key": m["key"],
                            "model_id": m["model_id"],
                            "stimulus_id": s["id"],
                            "question_id": q["id"],
                            "trial_idx": trial_idx,
                            "seed": seed,
                            "response_full": "",
                            "yes_no_label": "",
                            "detections": {sid: False for sid in
                                           (x["id"] for x in stimuli)},
                            "elapsed_sec": elapsed,
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc(),
                            "timestamp": datetime.now().isoformat(),
                        }
                        print(f"  [ERROR] {m['key']}/{s['id']}/{q['id']}/"
                              f"trial {trial_idx}: {e}")

                    completed.append(entry)
                    completed_keys.add(key)
                    done += 1
                    append_jsonl(out_jsonl, entry)
                    save_progress(out_json, header, completed)
                    write_summary_csv(out_csv, completed, stimuli)

                    preview = (entry["response_full"][:60] + "…")\
                              .replace("\n", " ")
                    print(f"  [{done:>3d}/{total}]  {m['key']:>9s} / "
                          f"{s['id']:>12s} / {q['id']} / "
                          f"trial {trial_idx:>2d}  ({elapsed:.1f}s)  "
                          f"yn={yes_no:<8s}  → {preview!r}")

                    if args.sleep_sec > 0:
                        time.sleep(args.sleep_sec)

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
