"""
run_qwen25vl_logit_lens_a100.py
================================
Phase 2: layer-wise logit lens analysis to localize the Watanabe Illusion
bias within Qwen2.5-VL-32B's language model.

Method:
- Append "The line would hit the " to the standard Q1 prompt. This places
  the next-token slot at a position where the model is committing to a
  specific dot ordinal ("1st", "2nd", ..., "11th").
- For each layer of the LM (and optionally each block of the vision tower),
  apply lm_head (the unembedding) to the hidden state at the final position
  and read off the logit (or probability) for each of the ordinals
  "1st", "2nd", ..., "11th".
- Average over N=10 stochastic trials (T=1.0 generation) to get a layer-by-
  ordinal heatmap. Same image (104.jpg), same prompt across trials.

This directly addresses the Phase 1 finding: the bias survives declarative
cancellation, so it must be sitting in some internal representation. Where?
The heatmap answers:
  - Is the correct ordinal "1st" preferred at any layer?
  - At which layer does the biased ordinal ("5th"/"6th") take over?
  - Is the transition sharp (one layer flip) or gradual?
  - Does the vision-tower output already encode the biased preference, or
    only the LM does?

Usage (A100, inside Docker):
    python scripts/run_qwen25vl_logit_lens_a100.py --n-trials 10

Output:
  results/qwen25vl_logit_lens_v1_a100/
    layer_logits.npz       per-trial (or averaged) layer x ordinal logits
    layer_logits_meta.json full setup, token IDs, ordinal list, etc.
    fig_lm_heatmap.png     LM layer x ordinal heatmap (averaged)
    fig_vit_logits.png     ViT layer logits for the 11 ordinals
    fig_argmax_per_layer.png  per-layer argmax ordinal trajectory
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
OUT_DIR = Path("/workspace/results/qwen25vl_logit_lens_v1_a100")
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

# Phase 2: 末尾に commit-style continuation を付加
# 「next token は dot ordinal」という強い制約を与える
CONTINUATION = "The line would hit the "

ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th",
            "8th", "9th", "10th", "11th"]
GT_ORDINAL = "1st"
BIASED_ORDINAL_BASELINE = "5th"  # 過去 N=60 で頻出


# ========== トークン ID 抽出 ==========
def find_ordinal_token_ids(processor, ordinals):
    """各 ordinal の '先頭トークン' (= 直後に来るときに参照する ID) を見つける。

    Qwen のトークナイザでは ' 1st' のような空白先頭が標準のはず。
    直前の文脈 'hit the ' があるので、参照すべきは ' 1st' のような形態。
    """
    tokenizer = processor.tokenizer
    found = {}
    for ord_str in ordinals:
        # 試行: 先行空白 + ordinal を encode して最初のトークン ID を取得
        # CONTINUATION の末尾が空白なので、続く ordinal は最初の subword に
        # 引き継がれる形になる
        candidates = [
            ord_str,              # "1st"
            " " + ord_str,        # " 1st"
        ]
        for c in candidates:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if ids:
                found[ord_str] = {
                    'candidate': c,
                    'token_id_first': ids[0],
                    'token_str_first': tokenizer.decode([ids[0]]),
                    'all_ids': ids,
                    'all_strs': [tokenizer.decode([i]) for i in ids],
                }
                break
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
def run_one_trial(model, processor, image, prompt_text, capture,
                  ordinal_token_ids, seed):
    """1 試行分: prompt を forward して、各層の hidden state を捕捉、
    最終トークン位置で lm_head を介して ordinal token のロジットを取得。

    戻り値: dict
      - 'lm_layer_logits': np.ndarray shape (n_lm_layers, n_ordinals)
      - 'vit_layer_logits': np.ndarray shape (n_vit_blocks, n_ordinals)
      - 'lm_final_logits': np.ndarray shape (n_ordinals,)  最終層 logit
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
    # ここで assistant 側の出力に CONTINUATION を直接書く
    # apply_chat_template は最後に <|im_start|>assistant\n を付加するので、
    # その続きに CONTINUATION を append する。
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

    # 各層の hidden state を lm_head に通す
    lm_head = model.lm_head  # nn.Linear(hidden_dim, vocab_size)
    lm_head_w = lm_head.weight.detach().to(torch.float32).cpu()  # (V, D)

    # LM 各層
    lm_logits_per_layer = []
    for h in capture.lm_hiddens:
        # h shape (1, D) → lm_head 適用 → (1, V)
        logits = h @ lm_head_w.T  # (1, V)
        # ordinal の最初のトークンの logit を取る
        ord_logits = []
        for ord_str in ORDINALS:
            tid = ordinal_token_ids[ord_str]['token_id_first']
            ord_logits.append(logits[0, tid].item())
        lm_logits_per_layer.append(ord_logits)
    lm_logits_per_layer = np.array(lm_logits_per_layer)  # (n_layers, n_ordinals)

    # ViT 各 block
    # 注意: ViT の hidden_dim と lm_head の入力次元が違うので、
    # 直接 lm_head には通せない。merger を経由した後の表現は LM hidden_dim
    # と整合するが、それは LM 内部の話。
    # ここでは ViT 各層の hidden を ViT 最終層相当として、merger を通した
    # 結果のみ「LM 空間に投影された」と扱える。
    # シンプルにするため、ViT 各層は lm_head に直接は通さず、ViT 最終層の
    # hidden を merger 経由で投影した最終結果のみを別途扱う。
    # → ViT 層別 logit lens は今回スキップし、別途追加。
    vit_logits_per_layer = None  # (placeholder)

    # 最終層出力の ordinal logit (もっとも「実出力に近い」もの)
    final_logits = outputs.logits[0, -1, :].detach().to(torch.float32).cpu()
    final_ord_logits = []
    for ord_str in ORDINALS:
        tid = ordinal_token_ids[ord_str]['token_id_first']
        final_ord_logits.append(final_logits[tid].item())
    final_ord_logits = np.array(final_ord_logits)

    return {
        'lm_layer_logits': lm_logits_per_layer,
        'vit_layer_logits': vit_logits_per_layer,
        'lm_final_logits': final_ord_logits,
    }


# ========== 可視化 ==========
def plot_lm_heatmap(lm_layer_logits_avg, ordinal_strs, save_path,
                    title_suffix=""):
    """LM 層 × ordinal のヒートマップを描く。

    lm_layer_logits_avg: shape (n_layers, n_ordinals)
    各層で softmax をかけて確率に変換し、対数スケールで heatmap 表示。
    """
    # softmax 確率 (各層内で 11 ordinal の中での相対)
    logits = lm_layer_logits_avg
    # 数値安定化
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)

    n_layers, n_ord = probs.shape
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Panel A: probability heatmap
    ax = axes[0]
    im = ax.imshow(probs.T, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('LM layer (early → late)', fontsize=11)
    ax.set_ylabel('ordinal token', fontsize=11)
    ax.set_yticks(range(n_ord))
    ax.set_yticklabels(ordinal_strs)
    ax.set_title(f'Layer-wise probability over dot ordinals{title_suffix}',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='softmax probability (over 11 ordinals)')

    # 各層の argmax を黒丸で重ねる
    argmax_per_layer = probs.argmax(axis=1)
    ax.plot(range(n_layers), argmax_per_layer, 'wo', markersize=4,
            markeredgecolor='black', label='argmax per layer')
    ax.legend(loc='upper left')

    # Panel B: 全層の logit trajectory (line plot)
    ax = axes[1]
    colors = plt.cm.tab20(np.linspace(0, 1, n_ord))
    for i, ord_str in enumerate(ordinal_strs):
        lw = 2.5 if ord_str in ('1st', '5th', '6th') else 1.0
        alpha = 1.0 if ord_str in ('1st', '5th', '6th') else 0.4
        ax.plot(probs[:, i], color=colors[i], linewidth=lw, alpha=alpha,
                label=ord_str)
    ax.set_xlabel('LM layer', fontsize=11)
    ax.set_ylabel('softmax probability', fontsize=11)
    ax.set_title(f'Per-ordinal trajectory across LM layers{title_suffix}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


def plot_argmax_trajectory(lm_layer_logits_per_trial, ordinal_strs, save_path):
    """全試行の per-layer argmax を 1 枚に重ねる。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    n_trials = len(lm_layer_logits_per_trial)
    for trial_idx, logits in enumerate(lm_layer_logits_per_trial):
        argmax_per_layer = logits.argmax(axis=1)
        ax.plot(argmax_per_layer, alpha=0.5, linewidth=1.5,
                label=f'trial {trial_idx}' if trial_idx < 5 else None)

    ax.set_xlabel('LM layer', fontsize=11)
    ax.set_ylabel('argmax ordinal index (0=1st, …, 10=11th)', fontsize=11)
    ax.set_yticks(range(len(ordinal_strs)))
    ax.set_yticklabels(ordinal_strs)
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
    print(f"\n=== Ordinal token IDs ===")
    ord_ids = find_ordinal_token_ids(processor, ORDINALS)
    for ord_str in ORDINALS:
        info = ord_ids[ord_str]
        print(f"  {ord_str:>4s}: candidate={info['candidate']!r:>10s}  "
              f"first_id={info['token_id_first']:>7d}  "
              f"first_str={info['token_str_first']!r:>8s}  "
              f"all_strs={info['all_strs']}")

    # LM 構造を確認
    # Qwen2.5-VL の LM 部分は model.model.language_model
    lm = model.model.language_model
    # 内部の decoder layers を取得
    if hasattr(lm, 'layers'):
        lm_layers = lm.layers
    elif hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
        lm_layers = lm.model.layers
    else:
        print("[ERROR] LM layers not found")
        # 構造をダンプ
        for name, _ in lm.named_children():
            print(f"  lm child: {name}")
        sys.exit(1)
    n_lm_layers = len(lm_layers)
    print(f"\n  LM layers: {n_lm_layers}")

    # ViT 構造 (参考)
    vit = model.model.visual
    n_vit_blocks = len(vit.blocks)
    print(f"  ViT blocks: {n_vit_blocks}")

    # lm_head 確認
    lm_head = model.lm_head
    print(f"  lm_head: {type(lm_head).__name__}  "
          f"weight shape: {tuple(lm_head.weight.shape)}")

    if args.inspect_only:
        print("\n[inspect-only] done.")
        return

    # 試行
    print(f"\n=== Running {args.n_trials} trials ===")
    img = Image.open(STIMULUS_PATH).convert("RGB")
    capture = LayerCapture()
    capture.attach_lm(lm_layers)
    # ViT は今回 layer-wise lens は省略 (lm_head に直接通せないため)
    # capture.attach_vit(vit.blocks)

    per_trial_lm_logits = []
    per_trial_final = []
    t0 = time.time()
    for trial_idx in range(args.n_trials):
        trial_t = time.time()
        result = run_one_trial(
            model, processor, img, full_prompt, capture,
            ord_ids, seed=trial_idx,
        )
        per_trial_lm_logits.append(result['lm_layer_logits'])
        per_trial_final.append(result['lm_final_logits'])
        # ロジット top
        final = result['lm_final_logits']
        top_idx = int(np.argmax(final))
        print(f"  trial {trial_idx+1}/{args.n_trials}: "
              f"final-layer top ordinal = {ORDINALS[top_idx]}  "
              f"(logit={final[top_idx]:.3f})  "
              f"elapsed={time.time()-trial_t:.1f}s")

    capture.detach()
    print(f"\nall trials in {(time.time()-t0)/60:.1f} min")

    # 平均
    lm_logits_stack = np.stack(per_trial_lm_logits, axis=0)  # (N, n_layers, n_ord)
    lm_logits_avg = lm_logits_stack.mean(axis=0)

    # 保存
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
        'ordinals': ORDINALS,
        'ordinal_token_ids': {
            o: {
                'candidate': ord_ids[o]['candidate'],
                'token_id_first': ord_ids[o]['token_id_first'],
                'token_str_first': ord_ids[o]['token_str_first'],
                'all_strs': ord_ids[o]['all_strs'],
            }
            for o in ORDINALS
        },
        'n_lm_layers': n_lm_layers,
        'n_vit_blocks': n_vit_blocks,
        'started_at': datetime.now().isoformat(),
    }
    with open(OUT_DIR / 'layer_logits_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  saved: {OUT_DIR/'layer_logits_meta.json'}")

    # 可視化
    plot_lm_heatmap(lm_logits_avg, ORDINALS,
                    OUT_DIR / 'fig_lm_heatmap.png',
                    title_suffix=f"  (N={args.n_trials} trials, mean)")
    plot_argmax_trajectory(per_trial_lm_logits, ORDINALS,
                           OUT_DIR / 'fig_argmax_per_layer.png')

    # 最終層の dot ordinal logit を表示
    print(f"\n=== Final-layer logits (averaged over {args.n_trials} trials) ===")
    final_avg = np.stack(per_trial_final, axis=0).mean(axis=0)
    final_logits_shifted = final_avg - final_avg.max()
    final_probs = np.exp(final_logits_shifted)
    final_probs = final_probs / final_probs.sum()
    for ord_str, lg, pr in zip(ORDINALS, final_avg, final_probs):
        marker = "  <-- BIASED expectation" if ord_str in ('5th', '6th') else \
                 "  <-- GROUND TRUTH" if ord_str == '1st' else ""
        print(f"  {ord_str:>4s}: logit={lg:8.3f}  prob={pr:.4f}{marker}")


if __name__ == "__main__":
    main()
