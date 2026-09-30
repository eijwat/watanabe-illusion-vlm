"""
run_qwen25vl_n60_a100_q1q2.py
==============================
Qwen2.5-VL-32B BF16 で N=60 確率的試行を走らせる本実験スクリプト。
A100 80GB (x86_64) 上で実行する Q1/Q2 のみのプロトコル。

背景:
- DGX Spark で N=60 v2 (3 ターン Q1/Q2/Q3) を実行済み。
- Qwen3-VL Instruct は推論ループ型の応答スタイルで、N=60 統計実験には
  不向きであることが判明 (パイロット観察)。Qwen3-VL の推論ループの
  解析は別実験として独立に扱う。
- 一方、本実験では Q3 が Watanabe Illusion の本質的現象ではなく
  副次的な「45° 量子化」を測るだけと判明しているため、Q1/Q2 に集中する。
- A100 で Qwen2.5-VL Q1/Q2 のみを N=60 で走らせ、DGX Spark 3 ターン版の
  Q1/Q2 部分との一貫性を確認する (マシン違いのみの追試)。

run_qwen25vl_n60_v2.py (DGX Spark 用) からの変更点:
- プロトコル: 3 ターン (Q1/Q2/Q3) -> 2 ターン (Q1/Q2 のみ)
- prompt_set: "human_G1G2G3_aligned_v1" -> "human_G1G2_aligned_v1"
- seed 戦略: trial_idx*3 + qidx -> trial_idx*2 + qidx (Q1=0, Q2=1)
- 出力先: results/qwen25vl_n60_v2/ -> results/qwen25vl_n60_a100_q1q2/
- max_memory: {0: "100GiB"} (DGX Spark 統合メモリ) -> {0: "75GiB"} (A100 80GB VRAM)

変えていないもの (DGX Spark 結果との一貫性検証のため):
- モデル: Qwen/Qwen2.5-VL-32B-Instruct
- ローダクラス: Qwen2_5_VLForConditionalGeneration
- processor 呼び出し: Qwen2.5-VL 流 (apply_chat_template tokenize=False
  + processor(text, images) の 2 段階)
- preamble (視覚印象のみ、ツール禁止)
- Q1/Q2 のプロンプト本文 (人間実験 G1/G2 と完全一致)
- temperature=1.0, top_p=1.0, repetition_penalty=1.05, max_new_tokens=512
- ターン独立構造 (会話履歴の引き継ぎなし)
- pixel_values 逆構築による入力検証 PNG 保存 (trial 0 の Q1 で 1 枚)
- 逐次 JSON 追記によるクラッシュ耐性
- 1 試行失敗でも次へ進むエラーハンドリング

Usage (A100 上の Docker 内):
  # パイロット動作確認 (N=3)
  python run_qwen25vl_n60_a100_q1q2.py --n-trials 3

  # 本番 (N=60、再開可能)
  python run_qwen25vl_n60_a100_q1q2.py --n-trials 60 --resume
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

# -------- 設定 --------
STIMULUS_PATH = "/workspace/stimuli/104.jpg"
OUT_DIR = Path("/workspace/results/qwen25vl_n60_a100_q1q2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_MEMORY = {0: "75GiB", "cpu": "0GiB"}
# A100 80GB VRAM 専用なので 75GiB に制限 (Qwen2.5-VL-32B BF16 ~65GB +
# KV cache + activation)。"cpu": "0GiB" で CPU offload を完全禁止。
# これを怠ると accelerate が保守的に CPU offload を選び、生成速度が
# 50-100x 落ちる罠がある (過去の test_llava_illusion_repeated.py の教訓)。

# 実験設定 (Qwen2.5-VL N=60 v2 と完全一致)
N_TRIALS_DEFAULT = 60
TEMPERATURE = 1.0
TOP_P = 1.0
REPETITION_PENALTY = 1.05
# DGX Spark N=60 v2 と完全同一の 512。Qwen2.5-VL は簡潔に答えるので十分。
MAX_NEW_TOKENS = 512

# プロンプト定義 (human_G1G2_aligned_v1) — Q3 廃止
PROMPT_SET_NAME = "human_G1G2_aligned_v1"

PREAMBLE = (
    "Look at the image below straight on from the front. "
    "Please answer based on your visual impression alone — "
    "do not use any tools such as Python, image processing, "
    "a ruler, or a protractor to measure."
)

Q1_BODY = (
    "Estimate which dot on the right edge the line in the circle would hit "
    "if extended. Answer like 'the Xth dot from the top.'"
)

Q2_BODY = (
    "Estimate which dot on the right edge the line in the circle would hit "
    "if extended. Answer like 'the Xth dot from the bottom.'"
)

Q1 = f"{PREAMBLE}\n\n{Q1_BODY}"
Q2 = f"{PREAMBLE}\n\n{Q2_BODY}"


# -------- pixel_values 逆構築 (v2 と同一) --------
def reconstruct_qwen25vl_image(pv, image_grid_thw, image_processor):
    patch_size = getattr(image_processor, 'patch_size', 14)
    temporal_patch_size = getattr(image_processor, 'temporal_patch_size', 2)
    merge_size = getattr(image_processor, 'merge_size', 2)

    if hasattr(image_grid_thw, 'tolist'):
        thw = image_grid_thw.tolist()
    else:
        thw = image_grid_thw
    if isinstance(thw[0], list):
        thw = thw[0]
    grid_t, grid_h, grid_w = thw

    if pv.shape[0] != grid_t * grid_h * grid_w:
        return None
    if pv.shape[1] != 3 * temporal_patch_size * patch_size * patch_size:
        return None
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        return None

    pv = pv.detach().cpu().float()
    pv = pv.unsqueeze(0)

    gh_m = grid_h // merge_size
    gw_m = grid_w // merge_size
    ms = merge_size
    C, tp, ph, pw = 3, temporal_patch_size, patch_size, patch_size

    pv_10d = pv.view(1, grid_t, gh_m, gw_m, ms, ms, C, tp, ph, pw)
    forward_perm = [0, 1, 4, 7, 5, 8, 3, 2, 6, 9]
    inverse_perm = [forward_perm.index(i) for i in range(10)]
    pv_view_order = pv_10d.permute(*inverse_perm)

    pv_image = pv_view_order[:, :, 0, :, :, :, :, :, :, :]
    pv_image = pv_image[:, 0][0]

    H_full = gh_m * ms * ph
    W_full = gw_m * ms * pw
    arr = pv_image.contiguous().view(C, H_full, W_full)

    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    arr = (arr * std + mean).clamp(0, 1).numpy()
    arr_uint8 = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr_uint8)


def save_input_verification(inputs, processor, out_dir):
    pv = inputs["pixel_values"]
    image_grid_thw = inputs.get("image_grid_thw", None)

    meta = {
        "pixel_values_shape": list(pv.shape),
        "pixel_values_dtype": str(pv.dtype),
        "image_grid_thw": image_grid_thw.tolist() if image_grid_thw is not None else None,
        "image_mean": list(processor.image_processor.image_mean),
        "image_std": list(processor.image_processor.image_std),
        "patch_size": getattr(processor.image_processor, 'patch_size', None),
        "temporal_patch_size": getattr(processor.image_processor, 'temporal_patch_size', None),
        "merge_size": getattr(processor.image_processor, 'merge_size', None),
    }
    with open(out_dir / "input_verification_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if pv.dim() == 2 and image_grid_thw is not None:
        recon = reconstruct_qwen25vl_image(pv, image_grid_thw, processor.image_processor)
        if recon is not None:
            recon.save(out_dir / "input_to_model_full.png")
            print(f"  [verify] saved input_to_model_full.png  size={recon.size}")
            print(f"  [verify] image_grid_thw: {image_grid_thw.tolist()}")
        else:
            print(f"  [verify] reconstruction returned None (pv shape mismatch). "
                  f"meta saved but PNG skipped.")


# -------- 単発質問 (画像 + テキスト、会話履歴なし) --------
def build_single_turn_inputs(processor, image, prompt_text):
    """1ターンのみ: user に (画像, テキスト) を渡す。会話履歴なし。
    DGX Spark の run_qwen25vl_n60_v2.py と完全同一の流儀
    (apply_chat_template tokenize=False + processor(text, images) の 2 段階)。
    """
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
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return inputs


def generate_response(model, processor, inputs, device, seed):
    """seed を設定して確率的に生成。"""
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

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


# -------- 1 試行を実行 (2ターン独立、Q1/Q2 のみ) --------
def run_single_trial(model, processor, image, device, trial_idx, save_pv_dir=None):
    """2ターン独立構造で 1 試行を実行。
    Q1/Q2 はそれぞれ独立した user メッセージとして送信され、
    会話履歴は引き継がない。
    """
    # --- Q1 ---
    inputs_q1 = build_single_turn_inputs(processor, image, Q1)
    if save_pv_dir is not None:
        save_input_verification(inputs_q1, processor, save_pv_dir)
    a1 = generate_response(model, processor, inputs_q1, device,
                            seed=trial_idx * 2 + 0)

    # --- Q2 ---
    inputs_q2 = build_single_turn_inputs(processor, image, Q2)
    a2 = generate_response(model, processor, inputs_q2, device,
                            seed=trial_idx * 2 + 1)

    return {
        "trial_idx": trial_idx,
        "Q1": {"prompt": Q1, "response": a1},
        "Q2": {"prompt": Q2, "response": a2},
    }


# -------- 結果の永続化 (逐次追記) --------
def save_progress(out_path, header, trials):
    payload = {**header, "trials": trials}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_existing_progress(out_path):
    if not out_path.exists():
        return None
    try:
        with open(out_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  [WARN] failed to load existing progress: {e}")
        return None


# -------- main --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                        help=f"number of trials (default: {N_TRIALS_DEFAULT})")
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing progress JSON if present")
    args = parser.parse_args()

    if not os.path.isfile(STIMULUS_PATH):
        print(f"[ERROR] stimulus not found: {STIMULUS_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"model:  {MODEL_ID}")
    print(f"dtype:  bfloat16")
    print(f"prompt_set: {PROMPT_SET_NAME}")
    print(f"n_trials: {args.n_trials}")
    print(f"temperature: {TEMPERATURE}")
    print(f"repetition_penalty: {REPETITION_PENALTY}")
    print(f"top_p: {TOP_P}")
    print(f"max_memory: {MAX_MEMORY}")
    print(f"\nstructure: 2-turn INDEPENDENT (no conversation carry-over, Q1/Q2 only)")
    print(f"\nQ1 prompt:\n  {Q1!r}")
    print(f"\nQ2 prompt:\n  {Q2!r}")

    out_json = OUT_DIR / "n60_progress.json"
    print(f"\noutput: {out_json}")

    existing = load_existing_progress(out_json) if args.resume else None
    if existing is not None:
        completed_trials = existing.get("trials", [])
        completed_idx = {t["trial_idx"] for t in completed_trials}
        print(f"[resume] found {len(completed_trials)} completed trials")

        # 同じ JSON ファイルでプロンプトセットが違ったら警告 (混在を防ぐ)
        existing_prompt_set = existing.get("prompt_set")
        if existing_prompt_set and existing_prompt_set != PROMPT_SET_NAME:
            print(f"[ERROR] existing prompt_set='{existing_prompt_set}' "
                  f"differs from current='{PROMPT_SET_NAME}'")
            print("  Refusing to mix prompt sets in one file. "
                  "Move/rename the existing file or change OUT_DIR.")
            sys.exit(1)
    else:
        completed_trials = []
        completed_idx = set()

    header = {
        "started_at": existing.get("started_at") if existing else datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "stimulus": STIMULUS_PATH,
        "preprocessing": "none (Qwen2.5-VL native dynamic resolution)",
        "prompt_set": PROMPT_SET_NAME,
        "structure": "2-turn independent (no conversation carry-over, Q1/Q2 only)",
        "machine": "A100 80GB PCIe (x86_64)",
        "n_trials_target": args.n_trials,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_strategy": "seed = trial_idx * 2 + question_idx (Q1=0, Q2=1)",
        "prompts": {"Q1": Q1, "Q2": Q2},
    }

    img = Image.open(STIMULUS_PATH).convert("RGB")
    print(f"\nstimulus loaded: {img.size}")

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

    print(f"\n=== running {args.n_trials} trials ===")
    save_pv_done = len(completed_trials) > 0
    overall_start = time.time()

    for trial_idx in range(args.n_trials):
        if trial_idx in completed_idx:
            print(f"\n[trial {trial_idx+1}/{args.n_trials}] (skipped, already completed)")
            continue

        trial_start = time.time()
        print(f"\n[trial {trial_idx+1}/{args.n_trials}] (seed base = {trial_idx*2})")

        save_pv_dir = OUT_DIR if not save_pv_done else None

        try:
            trial_result = run_single_trial(
                model, processor, img, device, trial_idx,
                save_pv_dir=save_pv_dir,
            )
            trial_result["elapsed_seconds"] = time.time() - trial_start
            trial_result["completed_at"] = datetime.now().isoformat()
            completed_trials.append(trial_result)
            save_pv_done = True

            a1_short = trial_result["Q1"]["response"][:80].replace("\n", " ")
            a2_short = trial_result["Q2"]["response"][:80].replace("\n", " ")
            print(f"  Q1: {a1_short}...")
            print(f"  Q2: {a2_short}...")
            print(f"  elapsed: {trial_result['elapsed_seconds']:.1f}s")

        except Exception as e:
            print(f"  [ERROR] trial {trial_idx} failed: {e}")
            traceback.print_exc()
            err_trial = {
                "trial_idx": trial_idx,
                "error": str(e),
                "elapsed_seconds": time.time() - trial_start,
                "completed_at": datetime.now().isoformat(),
            }
            completed_trials.append(err_trial)

        save_progress(out_json, header, completed_trials)

    overall_elapsed = time.time() - overall_start
    print(f"\n=== all trials done in {overall_elapsed/60:.1f} minutes ===")
    print(f"output: {out_json}")
    print(f"input verification PNG: {OUT_DIR}/input_to_model_full.png")


if __name__ == "__main__":
    main()
