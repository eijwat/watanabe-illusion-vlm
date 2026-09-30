"""
run_qwen25vl_logit_lens_v3_a100.py
===================================
Phase 2 supplementary experiment v3: irrelevant-prior control.

v1 で観察された「中盤層 ~39 で '1' が立つ」現象が
本当に視覚判断由来か (v2 で検証中) を、もう一段別のコントロールで補強。

v1: CONTINUATION = "The line would hit the "       → 視覚判断 + コンテキスト
v2: CONTINUATION = "The topmost dot is the "       → 概念リコール ("1" 直接誘導)
v3: CONTINUATION = "The number five is the "       → 完全に無関係なコンテキスト
                                                       で "5" を直接誘導

仮説:
- v3 で中盤層 ~39 に '5' のピークが出る → モデルはコンテキストに完全に
  追随して数字を出している。v1 の '1' ピークも同じ機構 (= 言語駆動)
- v3 で '5' ピークが出ない → モデルは無関係なコンテキストには引きずられず、
  v1 の '1' ピークは視覚判断由来である可能性が強まる

Usage:
    python scripts/run_qwen25vl_logit_lens_v3_a100.py --n-trials 10
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ========== 設定 ==========
STIMULUS_PATH = "/workspace/stimuli/104.jpg"
OUT_DIR = Path("/workspace/results/qwen25vl_logit_lens_v3_a100")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_MEMORY = {0: "75GiB", "cpu": "0GiB"}

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

# Phase 2 v3: irrelevant-prior control。
# "The number five is the " で "5" を強く誘導しても、モデルが
# その誘導に乗らずに視覚由来の数字を出すかを見る。
CONTINUATION = "The number five is the "

# 数字単体 1-9 を追う方針 (10th/11th は先頭トークンが '1' で 1st と衝突するため除外)。
# 行動データの分布で 1-9 が大部分を占めるため、論文の主張には十分。
DIGITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
GT_DIGIT = "1"
BIASED_DIGIT_BASELINE = "5"  # 過去 N=60 で頻出


# ========== トークン ID 抽出 ==========
def find_digit_token_ids(processor, digits):
    """各数字単体のトークン ID を取得。
    Qwen のトークナイザでは "1" は普通単一の ID にマップされる
    (inspect 段階で確認済み: '1'=16, '2'=17, ..., '9'=24)。
    """
    tokenizer = processor.tokenizer
    found = {}
    for d in digits:
        ids = tokenizer.encode(d, add_special_tokens=False)
        if len(ids) != 1:
            print(f"  [warn] digit '{d}' tokenized to multiple ids: {ids}")
        found[d] = {
            'token_id': ids[0],
            'token_str': tokenizer.decode([ids[0]]),
        }
    return found


# ========== Hook 機構 ==========
class LayerCapture:
    """各 transformer ブロック出力 (hidden state) を最終トークン位置で捕捉。

    LM の各層と、ViT の各ブロックの両方を扱う。
    capture[name] = hidden state at final seq position (1D vector)
    """
    def __init__(self):
        self.lm_hiddens = []   # list[tensor (hidden_dim,)] in layer order
        self.vit_hiddens = []  # list[tensor (vit_hidden_dim,)] in layer order
        self.handles = []

    def attach_lm(self, lm_layers):
        """LM の decoder layer 群に hook を attach。"""
        for i, layer in enumerate(lm_layers):
            def make_hook(idx):
                def hook(module, inp, out):
                    # out can be a tuple (hidden_state, ...) or a tensor
                    h = out[0] if isinstance(out, tuple) else out
                    # h.shape = (B, T, D), 最終トークン位置を取る
                    self.lm_hiddens.append(
                        h[:, -1, :].detach().to(torch.float32).cpu()
                    )
                return hook
            self.handles.append(layer.register_forward_hook(make_hook(i)))

    def attach_vit(self, vit_blocks):
        """ViT の block 群に hook を attach。
        ViT 内部のトークンは (n_patches, D) なので、最終位置は merger に
        渡る直前の patch 列。今回は「平均」を取って 1 ベクトルにする
        (=ViT probe の A.* 相当)。
        """
        for i, block in enumerate(vit_blocks):
            def make_hook(idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    # h.shape は (n_patches, D)
                    if h.dim() == 2:
                        self.vit_hiddens.append(
                            h.mean(dim=0).detach().to(torch.float32).cpu()
                        )
                    elif h.dim() == 3:
                        # (B, n_patches, D)
                        self.vit_hiddens.append(
                            h.mean(dim=1)[0].detach().to(torch.float32).cpu()
                        )
                return hook
            self.handles.append(block.register_forward_hook(make_hook(i)))

    def reset(self):
        self.lm_hiddens.clear()
        self.vit_hiddens.clear()

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ========== 1 試行の forward + logit lens ==========
def find_final_norm(lm):
    """LM の最終 RMSNorm モジュールを見つける。
    Qwen2 系は通常 lm.norm (Qwen2RMSNorm) を持つ。
    layer-wise logit lens の hidden state は pre-norm なので、
    lm_head に通す前に必ずこの norm を適用しないと最終ロジットと
    一致しない。
    """
    candidates = ['norm', 'final_layernorm', 'layer_norm', 'ln_f']
    for attr in candidates:
        if hasattr(lm, attr):
            mod = getattr(lm, attr)
            print(f"  found final_norm: lm.{attr} ({type(mod).__name__})")
            return mod
    # 階層下も探す
    if hasattr(lm, 'model'):
        for attr in candidates:
            if hasattr(lm.model, attr):
                mod = getattr(lm.model, attr)
                print(f"  found final_norm: lm.model.{attr} ({type(mod).__name__})")
                return mod
    print("  [WARN] final_norm not found by name; listing lm children:")
    for name, _ in lm.named_children():
        print(f"    lm child: {name}")
    return None


def run_one_trial(model, processor, image, prompt_text, capture,
                  digit_token_ids, final_norm, seed):
    """1 試行分: prompt を forward して、各層の hidden state を捕捉、
    最終トークン位置で (final_norm → lm_head) を介して digit token の
    ロジットを取得。

    final_norm は必ず lm_head の前に適用する。これを省略すると
    最終層 logit lens が output.logits と一致しない (Qwen2 系の構造)。

    戻り値: dict
      - 'lm_layer_logits': np.ndarray shape (n_lm_layers, n_digits)
      - 'lm_final_logits': np.ndarray shape (n_digits,)  最終層 logit
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
    text_with_continuation = text + CONTINUATION

    inputs = processor(
        text=[text_with_continuation],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    capture.reset()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False)

    # lm_head の重みを CPU の float32 で取り出す
    lm_head = model.lm_head
    lm_head_w = lm_head.weight.detach().to(torch.float32).cpu()  # (V, D)

    # final_norm の重みも CPU の float32 で取り出す
    # Qwen2RMSNorm: y = x * weight / sqrt(mean(x^2) + eps)
    # ここでは torch の forward を CPU で再現する代わりに、
    # モジュール自体を CPU 上で適用する
    if final_norm is None:
        print("  [WARN] no final_norm, applying lm_head directly")
        apply_norm = lambda h: h
    else:
        # CPU 上の final_norm を構築 (重みコピー)
        # モジュールごと .cpu() してから .float() で fp32 化
        # 元のモデルを動かさず、deep copy 相当の clone を作る
        import copy
        norm_cpu = copy.deepcopy(final_norm).to('cpu').to(torch.float32)
        norm_cpu.eval()
        def apply_norm(h):
            with torch.no_grad():
                return norm_cpu(h)

    # LM 各層 (capture.lm_hiddens は最終トークン位置の hidden, shape (1, D))
    lm_logits_per_layer = []
    for h in capture.lm_hiddens:
        # h shape: (1, D), float32, CPU
        h_normed = apply_norm(h)
        logits = h_normed @ lm_head_w.T  # (1, V)
        digit_logits = []
        for d in DIGITS:
            tid = digit_token_ids[d]['token_id']
            digit_logits.append(logits[0, tid].item())
        lm_logits_per_layer.append(digit_logits)
    lm_logits_per_layer = np.array(lm_logits_per_layer)  # (n_layers, n_digits)

    # 最終層出力 (model 直接の出力 logits) — これは「真の logit」
    final_logits = outputs.logits[0, -1, :].detach().to(torch.float32).cpu()
    final_digit_logits = []
    for d in DIGITS:
        tid = digit_token_ids[d]['token_id']
        final_digit_logits.append(final_logits[tid].item())
    final_digit_logits = np.array(final_digit_logits)

    # 検証: 最終層 lens (apply_norm(layer 63 hidden) @ lm_head.T) が
    # output.logits と一致するか
    if len(capture.lm_hiddens) >= 1:
        h_last = capture.lm_hiddens[-1]
        h_last_normed = apply_norm(h_last)
        verify_logits = (h_last_normed @ lm_head_w.T)[0]
        # 9 digits 部分だけ比較
        verify_digit_logits = np.array([
            verify_logits[digit_token_ids[d]['token_id']].item() for d in DIGITS
        ])
        # final_digit_logits と一致しているはず
        max_abs_diff = float(np.max(np.abs(verify_digit_logits - final_digit_logits)))
    else:
        max_abs_diff = None

    return {
        'lm_layer_logits': lm_logits_per_layer,
        'lm_final_logits': final_digit_logits,
        'verify_max_abs_diff': max_abs_diff,
    }


# ========== 可視化 ==========
def plot_lm_heatmap(lm_layer_logits_avg, digit_strs, save_path,
                    title_suffix=""):
    """LM 層 × digit のヒートマップを描く。

    lm_layer_logits_avg: shape (n_layers, n_digits)
    各層で softmax をかけて確率に変換し、対数スケールで heatmap 表示。
    """
    logits = lm_layer_logits_avg
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)

    n_layers, n_d = probs.shape
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Panel A: probability heatmap
    ax = axes[0]
    im = ax.imshow(probs.T, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('LM layer (early → late)', fontsize=11)
    ax.set_ylabel('digit token', fontsize=11)
    ax.set_yticks(range(n_d))
    ax.set_yticklabels(digit_strs)
    ax.set_title(f'Layer-wise probability over digit tokens (1-9){title_suffix}',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='softmax probability (over 9 digits)')

    # 各層の argmax を白丸で重ねる
    argmax_per_layer = probs.argmax(axis=1)
    ax.plot(range(n_layers), argmax_per_layer, 'wo', markersize=4,
            markeredgecolor='black', label='argmax per layer')
    ax.legend(loc='upper left')

    # Panel B: 全層の logit trajectory (line plot)
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_d))
    for i, d in enumerate(digit_strs):
        # GT '1' と biased baseline '5'/'6' を強調
        if d in ('1',):
            lw, alpha = 3.0, 1.0
            label = f"{d} (ground truth)"
        elif d in ('5', '6'):
            lw, alpha = 2.5, 1.0
            label = f"{d} (biased baseline)"
        else:
            lw, alpha = 1.0, 0.5
            label = d
        ax.plot(probs[:, i], color=colors[i], linewidth=lw, alpha=alpha,
                label=label)
    ax.set_xlabel('LM layer', fontsize=11)
    ax.set_ylabel('softmax probability', fontsize=11)
    ax.set_title(f'Per-digit trajectory across LM layers{title_suffix}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


def plot_argmax_trajectory(lm_layer_logits_per_trial, digit_strs, save_path):
    """全試行の per-layer argmax を 1 枚に重ねる。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    n_trials = len(lm_layer_logits_per_trial)
    for trial_idx, logits in enumerate(lm_layer_logits_per_trial):
        argmax_per_layer = logits.argmax(axis=1)
        ax.plot(argmax_per_layer, alpha=0.5, linewidth=1.5,
                label=f'trial {trial_idx}' if trial_idx < 5 else None)

    ax.set_xlabel('LM layer', fontsize=11)
    ax.set_ylabel('argmax digit', fontsize=11)
    ax.set_yticks(range(len(digit_strs)))
    ax.set_yticklabels(digit_strs)
    ax.set_title('Per-trial argmax trajectory across LM layers '
                 f'(N={n_trials})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if n_trials <= 5:
        ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


# ========== main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--inspect-only", action="store_true",
                        help="トークン ID と LM/ViT 構造を確認して終了")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 2 — Logit lens on Qwen2.5-VL-32B (Watanabe Illusion)")
    print("=" * 70)
    print(f"  model:      {MODEL_ID}")
    print(f"  stimulus:   {STIMULUS_PATH}")
    print(f"  out_dir:    {OUT_DIR}")
    print(f"  n_trials:   {args.n_trials}")
    print()
    full_prompt = PREAMBLE + "\n\n" + Q1_TASK
    print(f"Prompt (will be followed by CONTINUATION):")
    print(f"  {full_prompt!r}")
    print(f"Continuation appended to assistant turn:")
    print(f"  {CONTINUATION!r}")
    print()

    if not os.path.isfile(STIMULUS_PATH):
        print(f"[ERROR] stimulus not found: {STIMULUS_PATH}")
        sys.exit(1)

    print(f"Loading {MODEL_ID}...")
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

    # トークン ID を確認
    print(f"\n=== Digit token IDs ===")
    digit_ids = find_digit_token_ids(processor, DIGITS)
    for d in DIGITS:
        info = digit_ids[d]
        print(f"  '{d}': token_id={info['token_id']:>7d}  "
              f"token_str={info['token_str']!r}")

    # LM 構造を確認
    lm = model.model.language_model
    if hasattr(lm, 'layers'):
        lm_layers = lm.layers
    elif hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
        lm_layers = lm.model.layers
    else:
        print("[ERROR] LM layers not found")
        for name, _ in lm.named_children():
            print(f"  lm child: {name}")
        sys.exit(1)
    n_lm_layers = len(lm_layers)
    print(f"\n  LM layers: {n_lm_layers}")

    vit = model.model.visual
    n_vit_blocks = len(vit.blocks)
    print(f"  ViT blocks: {n_vit_blocks}")

    lm_head = model.lm_head
    print(f"  lm_head: {type(lm_head).__name__}  "
          f"weight shape: {tuple(lm_head.weight.shape)}")

    # final_norm を取得 (lm_head に通す前に必要)
    print(f"\n  Looking for final_norm in language_model...")
    final_norm = find_final_norm(lm)

    if args.inspect_only:
        print("\n[inspect-only] done.")
        return

    # 試行
    print(f"\n=== Running {args.n_trials} trials ===")
    img = Image.open(STIMULUS_PATH).convert("RGB")
    capture = LayerCapture()
    capture.attach_lm(lm_layers)

    per_trial_lm_logits = []
    per_trial_final = []
    per_trial_verify_diff = []
    t0 = time.time()
    for trial_idx in range(args.n_trials):
        trial_t = time.time()
        result = run_one_trial(
            model, processor, img, full_prompt, capture,
            digit_ids, final_norm, seed=trial_idx,
        )
        per_trial_lm_logits.append(result['lm_layer_logits'])
        per_trial_final.append(result['lm_final_logits'])
        per_trial_verify_diff.append(result['verify_max_abs_diff'])
        final = result['lm_final_logits']
        top_idx = int(np.argmax(final))
        # logit lens の最終層 (= final layer の hidden を normed lens に通したもの)
        last_layer_lens = result['lm_layer_logits'][-1]
        lens_top_idx = int(np.argmax(last_layer_lens))
        verify = result['verify_max_abs_diff']
        verify_str = f"verify_diff={verify:.4f}" if verify is not None else "no_verify"
        print(f"  trial {trial_idx+1}/{args.n_trials}: "
              f"output.logits top='{DIGITS[top_idx]}' "
              f"(logit={final[top_idx]:.3f}), "
              f"layer-63 lens top='{DIGITS[lens_top_idx]}' "
              f"(logit={last_layer_lens[lens_top_idx]:.3f}), "
              f"{verify_str}, "
              f"elapsed={time.time()-trial_t:.1f}s")

    capture.detach()
    print(f"\nall trials in {(time.time()-t0)/60:.1f} min")

    lm_logits_stack = np.stack(per_trial_lm_logits, axis=0)
    lm_logits_avg = lm_logits_stack.mean(axis=0)

    np.savez_compressed(
        OUT_DIR / 'layer_logits.npz',
        per_trial_lm_logits=lm_logits_stack,
        lm_logits_avg=lm_logits_avg,
        per_trial_final=np.stack(per_trial_final, axis=0),
    )
    print(f"  saved: {OUT_DIR/'layer_logits.npz'}")

    meta = {
        'model_id': MODEL_ID,
        'machine': 'A100 80GB PCIe (x86_64)',
        'n_trials': args.n_trials,
        'preamble': PREAMBLE,
        'q1_task': Q1_TASK,
        'continuation': CONTINUATION,
        'digits': DIGITS,
        'digit_token_ids': {
            d: {
                'token_id': digit_ids[d]['token_id'],
                'token_str': digit_ids[d]['token_str'],
            }
            for d in DIGITS
        },
        'n_lm_layers': n_lm_layers,
        'n_vit_blocks': n_vit_blocks,
        'final_norm_applied': final_norm is not None,
        'final_norm_type': type(final_norm).__name__ if final_norm else None,
        'verify_max_abs_diff_per_trial': [
            float(d) if d is not None else None
            for d in per_trial_verify_diff
        ],
        'started_at': datetime.now().isoformat(),
        'note_10th_11th_excluded': (
            "Tokens '10' and '11' share their first subword '1' with '1st', "
            "so they cannot be disambiguated at the digit-token level. "
            "We restrict the analysis to digits 1-9, which cover the bulk "
            "of the response distribution in our N=60 behavioral runs."
        ),
    }
    with open(OUT_DIR / 'layer_logits_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  saved: {OUT_DIR/'layer_logits_meta.json'}")

    plot_lm_heatmap(lm_logits_avg, DIGITS,
                    OUT_DIR / 'fig_lm_heatmap.png',
                    title_suffix=f"  (N={args.n_trials} trials, mean)")
    plot_argmax_trajectory(per_trial_lm_logits, DIGITS,
                           OUT_DIR / 'fig_argmax_per_layer.png')

    print(f"\n=== Final-layer logits (averaged over {args.n_trials} trials) ===")
    final_avg = np.stack(per_trial_final, axis=0).mean(axis=0)
    final_logits_shifted = final_avg - final_avg.max()
    final_probs = np.exp(final_logits_shifted)
    final_probs = final_probs / final_probs.sum()
    for d, lg, pr in zip(DIGITS, final_avg, final_probs):
        marker = "  <-- BIASED expectation" if d in ('5', '6') else \
                 "  <-- GROUND TRUTH" if d == '1' else ""
        print(f"  '{d}': logit={lg:8.3f}  prob={pr:.4f}{marker}")


if __name__ == "__main__":
    main()
