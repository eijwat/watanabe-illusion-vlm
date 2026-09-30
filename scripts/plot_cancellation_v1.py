"""
plot_cancellation_v1.py
========================
Phase 1 prompt cancellation experiment の可視化。

入力:
  /mnt/user-data/uploads/cancellation_progress.json (8 prompts × N=20)

出力:
  fig_cancellation_means.png        : 棒グラフ (mean ± SD per prompt, P0 参照線)
  fig_cancellation_distributions.png : 各プロンプトの分布を縦に並べたもの
"""

import json
import re
import sys
import statistics
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ========== 設定 ==========
INPUT_JSON = Path("/mnt/user-data/uploads/cancellation_progress.json")
OUT_DIR = Path("/home/claude")
N_DOTS = 11
GT_Q1 = 1  # Q1 from-top の正解 = 1

# 色 (8 プロンプトに対する categorical palette)
PROMPT_COLORS = {
    'P0': '#444444',  # control = dark gray (基準)
    'P1': '#1f77b4',  # name_specific = blue
    'P2': '#17becf',  # name_generic = cyan
    'P3': '#2ca02c',  # knowledge_suppress = green
    'P4': '#ff7f0e',  # counter_hint = orange
    'P5': '#9467bd',  # force_visual_cot = purple
    'P6': '#d62728',  # ground_truth_disclose = red
    'P7': '#8c564b',  # random_distractor = brown
}

PROMPT_SHORT_LABELS = {
    'P0': 'P0 Control',
    'P1': 'P1 "Watanabe Illusion"',
    'P2': 'P2 "visual illusion"',
    'P3': 'P3 Suppress prior',
    'P4': 'P4 Counter-hint (top)',
    'P5': 'P5 Force visual CoT',
    'P6': 'P6 GT disclosure',
    'P7': 'P7 Random distractor',
}

PROMPT_ORDER = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']


# ========== 抽出 ==========
SPELLED = {
    'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
    'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,
    'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6,
    'seventh':7,'eighth':8,'ninth':9,'tenth':10,'eleventh':11,'twelfth':12,
}
SPELLED_RE = '|'.join(SPELLED.keys())


def extract_q1(text):
    """Q1 response から (num, direction) を抽出して正規化 (from-top)."""
    m = re.search(r'\bthe\s+(\d+)(?:st|nd|rd|th)\s+dot\s+from\s+the\s+(top|bottom)\b',
                  text, re.I)
    if m:
        return int(m.group(1)), m.group(2).lower()
    m = re.search(r'\bthe\s+(' + SPELLED_RE + r')\s+dot\s+from\s+the\s+(top|bottom)\b',
                  text, re.I)
    if m:
        return SPELLED[m.group(1).lower()], m.group(2).lower()
    m = re.search(r'\\boxed\{[^}]*?(\d+)[^}]*?(top|bottom)', text, re.I)
    if m:
        return int(m.group(1)), m.group(2).lower()
    m = re.search(r'\\boxed\{\s*(\d+)\s*\}', text)
    if m:
        n = int(m.group(1))
        dm = re.search(r'(\d+|' + SPELLED_RE + r')(?:st|nd|rd|th)?\s+dot\s+from\s+the\s+(top|bottom)',
                       text, re.I)
        return n, (dm.group(2).lower() if dm else None)
    m = re.findall(r'(\d+)(?:st|nd|rd|th)\s+dot', text, re.I)
    if m:
        return int(m[-1]), None
    return None, None


def normalize_q1(num, direction):
    """direction が 'bottom' なら from-top に変換。
    None (unspecified) は top と扱う(Q1 で要求した方向)."""
    if num is None:
        return None
    if direction == 'bottom':
        return N_DOTS - num + 1
    return num


# ========== データ読み込み・集計 ==========
def main():
    data = json.load(open(INPUT_JSON))
    entries = data['entries']
    print(f"Loaded {len(entries)} entries")

    by_prompt = {pid: [] for pid in PROMPT_ORDER}
    for e in entries:
        n, d = extract_q1(e['response'])
        norm = normalize_q1(n, d)
        if norm is not None:
            by_prompt[e['prompt_id']].append(norm)

    # サマリ統計
    stats = {}
    for pid in PROMPT_ORDER:
        vals = by_prompt[pid]
        if len(vals) >= 2:
            stats[pid] = {
                'n': len(vals),
                'mean': statistics.mean(vals),
                'median': statistics.median(vals),
                'sd': statistics.stdev(vals),
                'min': min(vals),
                'max': max(vals),
                'vals': vals,
            }

    p0_mean = stats['P0']['mean']
    p0_sd = stats['P0']['sd']
    print(f"\nP0 baseline: mean = {p0_mean:.2f}, sd = {p0_sd:.2f}")
    for pid in PROMPT_ORDER:
        s = stats[pid]
        diff = s['mean'] - p0_mean
        print(f"  {pid}: mean={s['mean']:.2f}, sd={s['sd']:.2f}, "
              f"diff_from_P0={diff:+.2f}")

    # ========== 図 1: 棒グラフ ==========
    fig, ax = plt.subplots(figsize=(13, 6))

    xs = np.arange(len(PROMPT_ORDER))
    means = [stats[p]['mean'] for p in PROMPT_ORDER]
    sds = [stats[p]['sd'] for p in PROMPT_ORDER]
    colors = [PROMPT_COLORS[p] for p in PROMPT_ORDER]

    bars = ax.bar(xs, means, yerr=sds, capsize=6, color=colors,
                  edgecolor='black', linewidth=0.7,
                  alpha=0.85, error_kw={'linewidth': 1.4})

    # P0 参照線 (mean ± SD のバンド)
    ax.axhline(p0_mean, color=PROMPT_COLORS['P0'], linewidth=1.3,
               linestyle='--', alpha=0.7,
               label=f'P0 baseline mean = {p0_mean:.2f}')
    ax.axhspan(p0_mean - p0_sd, p0_mean + p0_sd, color=PROMPT_COLORS['P0'],
               alpha=0.10, label='P0 ± 1 SD')

    # Ground truth ライン
    ax.axhline(GT_Q1, color='red', linewidth=2.0, linestyle='-',
               alpha=0.65, label=f'Ground truth = {GT_Q1} (topmost dot)')

    # 各バーの上に mean ± sd 値
    for i, (m, s, p) in enumerate(zip(means, sds, PROMPT_ORDER)):
        ax.text(i, m + s + 0.15, f'{m:.2f}', ha='center',
                fontsize=10, fontweight='bold')

    # 個別 trial 点を上に重ねる
    rng = np.random.default_rng(0)
    for i, pid in enumerate(PROMPT_ORDER):
        vals = stats[pid]['vals']
        x_jit = i + rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(x_jit, vals, s=22, color='black', alpha=0.45,
                   edgecolor='none', zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([PROMPT_SHORT_LABELS[p] for p in PROMPT_ORDER],
                       rotation=15, ha='right', fontsize=10)
    ax.set_ylabel("Q1 response (dot # from top)", fontsize=12)
    ax.set_title("Prompt cancellation experiment — "
                 "Qwen2.5-VL-32B Q1 (A100, N=20 per prompt)\n"
                 "bars: mean ± SD  •  dots: individual trials",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(means) + max(sds) + 1.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig_cancellation_means.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  saved: fig_cancellation_means.png")

    # ========== 図 2: 分布の縦並び ==========
    fig, axes = plt.subplots(8, 1, figsize=(11, 11), sharex=True)
    bin_edges = np.arange(0.5, 12.5, 1.0)

    for ax, pid in zip(axes, PROMPT_ORDER):
        vals = stats[pid]['vals']
        ax.hist(vals, bins=bin_edges, color=PROMPT_COLORS[pid],
                alpha=0.75, edgecolor='black', linewidth=0.6)

        # P0 mean (基準線)
        ax.axvline(p0_mean, color='black', linewidth=1.4, linestyle='--',
                   alpha=0.6)
        # この prompt の mean
        m = stats[pid]['mean']
        ax.axvline(m, color=PROMPT_COLORS[pid], linewidth=2.5, linestyle='-')
        # Ground truth
        ax.axvline(GT_Q1, color='red', linewidth=1.5, alpha=0.55)

        # アノテート
        diff = m - p0_mean
        label_text = (f"{PROMPT_SHORT_LABELS[pid]}\n"
                      f"mean = {m:.2f}  (vs P0: {diff:+.2f})  "
                      f"sd = {stats[pid]['sd']:.2f}")
        ax.text(0.02, 0.95, label_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.85, edgecolor='none'))

        ax.set_ylabel('count', fontsize=9)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 16)
        ax.set_yticks([0, 5, 10, 15])
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel('Q1 response (dot # from top)', fontsize=12)
    axes[0].set_title('Prompt cancellation — Q1 distribution per prompt '
                      '(N=20 each)\n'
                      'red line: ground truth (1)  •  '
                      'black dashed: P0 baseline mean  •  '
                      'colored solid: this-prompt mean',
                      fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig_cancellation_distributions.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  saved: fig_cancellation_distributions.png")

    # JSON サマリ保存
    summary = {
        'P0_baseline': {'mean': p0_mean, 'sd': p0_sd},
        'ground_truth': GT_Q1,
        'per_prompt': {
            pid: {
                'category': data['prompts'][i]['category'],
                'n': stats[pid]['n'],
                'mean': stats[pid]['mean'],
                'sd': stats[pid]['sd'],
                'median': stats[pid]['median'],
                'min': stats[pid]['min'],
                'max': stats[pid]['max'],
                'diff_from_P0': stats[pid]['mean'] - p0_mean,
                'distribution': dict(sorted(Counter(stats[pid]['vals']).items())),
            }
            for i, pid in enumerate(PROMPT_ORDER)
        },
    }
    with open(OUT_DIR / 'cancellation_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: cancellation_stats.json")


if __name__ == "__main__":
    main()
