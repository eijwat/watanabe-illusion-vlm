"""
plot_logit_lens_4panel.py
==========================
Phase 2 logit lens の v1/v2/v3/v4 結果を 2x2 grid の合成図にまとめる。

入力:
  results/qwen25vl_logit_lens_v1_a100/layer_logits.npz
  results/qwen25vl_logit_lens_v2_a100/layer_logits.npz
  results/qwen25vl_logit_lens_v3_a100/layer_logits.npz
  results/qwen25vl_logit_lens_v4_a100/layer_logits.npz

出力:
  fig_logit_lens_4panel.png      (paper_v6 の Figure 10)
  fig_logit_lens_4panel.pdf      (vector 版)

Usage:
  # Docker 外でも (numpy + matplotlib のみで動く)
  python scripts/plot_logit_lens_4panel.py
  # オプションで結果ディレクトリの親パスを指定
  python scripts/plot_logit_lens_4panel.py --results-dir ~/watanabe-illusion/results
  python scripts/plot_logit_lens_4panel.py --out-dir ~/watanabe-illusion/figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ========== 設定 ==========
DIGITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 4 つの run の定義 (順序が出力 grid の順序)
RUNS = [
    {
        'key':   'v1',
        'dir':   'qwen25vl_logit_lens_v1_a100',
        'continuation': '"The line would hit the "',
        'label': 'v1: visual judgment',
        'subtitle': 'CONTINUATION = "The line would hit the "',
    },
    {
        'key':   'v2',
        'dir':   'qwen25vl_logit_lens_v2_a100',
        'continuation': '"The topmost dot is the "',
        'label': 'v2: concept recall',
        'subtitle': 'CONTINUATION = "The topmost dot is the "',
    },
    {
        'key':   'v3',
        'dir':   'qwen25vl_logit_lens_v3_a100',
        'continuation': '"The number five is the "',
        'label': 'v3: irrelevant prior',
        'subtitle': 'CONTINUATION = "The number five is the "',
    },
    {
        'key':   'v4',
        'dir':   'qwen25vl_logit_lens_v4_a100',
        'continuation': '"The line would hit dot number "',
        'label': 'v4: visual judgment + numerical cue',
        'subtitle': 'CONTINUATION = "The line would hit dot number "',
    },
]


def load_run(results_dir: Path, run_dir: str):
    """1 つの run の layer_logits.npz と meta.json を読み込む。

    Returns:
      probs: shape (n_layers, n_digits) — softmax 確率 (each layer)
      meta:  dict
    """
    npz_path = results_dir / run_dir / 'layer_logits.npz'
    meta_path = results_dir / run_dir / 'layer_logits_meta.json'

    data = np.load(npz_path)
    # data 内の重要キー: 'per_trial_lm_logits', 'lm_logits_avg', 'per_trial_final'
    logits_avg = data['lm_logits_avg']  # (n_layers, n_digits)

    # softmax (各層内で 9 digit の中での相対)
    logits_shifted = logits_avg - logits_avg.max(axis=1, keepdims=True)
    probs = np.exp(logits_shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)

    with open(meta_path) as f:
        meta = json.load(f)

    return probs, meta


def plot_4panel(runs_data, out_path_png, out_path_pdf):
    """2x2 grid に 4 つの heatmap を描く。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=False, sharey=True)
    axes = axes.flatten()

    # 全 4 パネルの最大確率を計算して共通スケール用
    all_probs = np.concatenate([d['probs'].flatten() for d in runs_data])
    vmax = float(all_probs.max())

    ims = []
    for i, run_data in enumerate(runs_data):
        ax = axes[i]
        probs = run_data['probs']  # (n_layers, n_digits)
        n_layers, n_d = probs.shape

        # heatmap: x=layer, y=digit (probs.T)
        im = ax.imshow(
            probs.T, aspect='auto', origin='lower',
            cmap='viridis', vmin=0.0, vmax=vmax,
            interpolation='nearest',
        )
        ims.append(im)

        # 各層の argmax を白丸で重ねる
        argmax_per_layer = probs.argmax(axis=1)
        ax.plot(range(n_layers), argmax_per_layer,
                'o', markersize=3.5,
                markerfacecolor='white', markeredgecolor='black',
                markeredgewidth=0.5, label='argmax')

        # GT 候補 ('1' = digit_index 0) を強調する横線
        ax.axhline(0, color='red', linewidth=1.2, alpha=0.4,
                   linestyle=':', label='ground truth (1)')

        # x 軸
        ax.set_xlabel('LM layer (early → late)', fontsize=11)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 63])
        ax.set_xlim(-0.5, n_layers - 0.5)

        # y 軸
        if i % 2 == 0:
            ax.set_ylabel('digit token', fontsize=11)
        ax.set_yticks(range(n_d))
        ax.set_yticklabels(DIGITS)

        # パネルタイトル
        ax.set_title(
            f"{run_data['label']}\n{run_data['subtitle']}",
            fontsize=11, fontweight='bold', loc='left',
        )

        # 最終層の top digit を右下にアノテート
        final_top = int(argmax_per_layer[-1])
        final_top_prob = probs[-1, final_top]
        ax.text(
            0.98, 0.02,
            f"final-layer argmax: '{DIGITS[final_top]}' "
            f"(P = {final_top_prob:.2f})",
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor='white', alpha=0.85, edgecolor='gray'),
        )

        # 中盤層ハイライト (35-45) — v1/v2/v4 で '1' ピークの帯
        if run_data['key'] in ('v1', 'v2', 'v4'):
            rect = Rectangle(
                (35 - 0.5, -0.5), 11, n_d,
                linewidth=1.5, edgecolor='cyan', facecolor='none',
                linestyle='--', alpha=0.6,
            )
            ax.add_patch(rect)

        # legend (左上だけに表示、煩雑回避)
        if i == 0:
            ax.legend(loc='upper left', fontsize=9,
                      framealpha=0.9, edgecolor='gray')

    # 全体タイトル
    fig.suptitle(
        "Layer-wise logit lens on Qwen2.5-VL-32B's 64 LM layers "
        "(N = 10 trials per panel, mean)",
        fontsize=14, fontweight='bold', y=0.995,
    )

    # 共通 colorbar (右側)
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.10, 0.014, 0.78])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label('softmax probability (over 9 digits, per layer)',
                   fontsize=10)

    # 注釈テキスト (図全体下部)
    fig.text(
        0.5, 0.005,
        "Cyan dashed boxes (v1, v2, v4): mid-layer band (~35-45) where digit '1' rises to argmax. "
        "Compare v1/v4 (overwritten in late layers) vs v2 (persists to final layer).",
        ha='center', fontsize=9.5, style='italic',
    )

    plt.tight_layout(rect=[0, 0.02, 0.92, 0.98])
    plt.savefig(out_path_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_path_pdf, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out_path_png}")
    print(f"  saved: {out_path_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results-dir', type=str,
        default=str(Path.home() / 'watanabe-illusion' / 'results'),
        help='Parent directory containing qwen25vl_logit_lens_v{1..4}_a100/',
    )
    parser.add_argument(
        '--out-dir', type=str,
        default=str(Path.home() / 'watanabe-illusion' / 'figures'),
        help='Output directory for the composite figure',
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading from: {results_dir}")
    print(f"Output to:    {out_dir}")
    print()

    runs_data = []
    for run in RUNS:
        npz_path = results_dir / run['dir'] / 'layer_logits.npz'
        if not npz_path.exists():
            print(f"[ERROR] Missing: {npz_path}")
            print(f"        Did you run all of v1/v2/v3/v4?")
            return
        probs, meta = load_run(results_dir, run['dir'])
        final_top_idx = int(probs[-1].argmax())
        print(f"  {run['key']} {run['continuation']}: "
              f"loaded probs.shape={probs.shape}, "
              f"final-layer top='{DIGITS[final_top_idx]}' "
              f"(P={probs[-1, final_top_idx]:.3f}), "
              f"n_trials={meta.get('n_trials', '?')}")
        runs_data.append({
            **run,
            'probs': probs,
            'meta': meta,
        })

    print()
    plot_4panel(
        runs_data,
        out_path_png=out_dir / 'fig_logit_lens_4panel.png',
        out_path_pdf=out_dir / 'fig_logit_lens_4panel.pdf',
    )


if __name__ == "__main__":
    main()
