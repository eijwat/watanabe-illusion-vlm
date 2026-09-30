"""
plot_qwen25vl_n60_v2_vs_human.py
================================
v2 N=60 結果と人間 N=130 を並べる論文用 3 パネル図。

- Q1 (from top): 人間 4.6489±2.6310 / VLM 60試行
- Q2 (from bottom): 人間 6.3588±2.8557 / VLM 60試行
- Q3 (degrees): 人間 37.443±22.803 / VLM 60試行
- Ground truth: Q1=1, Q2=11, Q3=23.5°

人間データは個別値が手元にないので mean±SD から正規近似曲線を描く。
VLM は実測値のヒストグラム。
"""

import json
import re
import statistics
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats


# ===== 抽出関数 (analyze スクリプトと同じロジック) =====
ORDINAL_WORDS = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10, 'eleventh': 11,
}
N_DOTS = 11


def extract_dot_answer(response):
    num_pat = re.compile(r'(\d+)(?:st|nd|rd|th)?\s*dot\s+from\s+the\s+(top|bottom)', re.IGNORECASE)
    word_pat = re.compile(r'(' + '|'.join(ORDINAL_WORDS.keys()) + r')\s+dot\s+from\s+the\s+(top|bottom)', re.IGNORECASE)
    matches = []
    for m in num_pat.finditer(response):
        matches.append((m.start(), int(m.group(1)), m.group(2).lower()))
    for m in word_pat.finditer(response):
        matches.append((m.start(), ORDINAL_WORDS[m.group(1).lower()], m.group(2).lower()))
    if not matches:
        return None, None
    matches.sort()
    _, num, direction = matches[-1]
    return num, direction


def normalize_to_top(num, direction, n=N_DOTS):
    if direction == 'top':
        return num
    elif direction == 'bottom':
        return n - num + 1
    return None


def normalize_to_bottom(num, direction, n=N_DOTS):
    if direction == 'bottom':
        return num
    elif direction == 'top':
        return n - num + 1
    return None


def extract_angle(response):
    boxed = list(re.finditer(r'\\?boxed\{(\d+(?:\.\d+)?)\}', response))
    if boxed:
        return float(boxed[-1].group(1))
    candidates = []
    for pat in [
        r'\\?\(?\s*(\d+(?:\.\d+)?)\s*\^\\?circ',
        r'(\d+(?:\.\d+)?)\s*°',
        r'(\d+(?:\.\d+)?)[\s-]*degree',
    ]:
        for m in re.finditer(pat, response, re.IGNORECASE):
            candidates.append((m.start(), float(m.group(1))))
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    return None


# ===== パラメータ =====
HUMAN = {
    'Q1': {'mean': 4.6489, 'sd': 2.6310, 'label': 'Q1: dot from top', 'unit': '', 'truth': 1},
    'Q2': {'mean': 6.3588, 'sd': 2.8557, 'label': 'Q2: dot from bottom', 'unit': '', 'truth': 11},
    'Q3': {'mean': 37.443, 'sd': 22.803, 'label': 'Q3: degrees above horizontal', 'unit': '°', 'truth': 23.5},
}
HUMAN_N = 130


# ===== カラー設計 =====
COLOR_VLM = '#2E5C8A'       # 落ち着いた青 (VLM)
COLOR_HUMAN = '#A0A0A0'     # 中グレー (人間正規近似)
COLOR_HUMAN_FILL = '#D5D5D5'  # 薄いグレー (人間ヒストグラム背景)
COLOR_TRUTH = '#2E8B57'     # シー グリーン (ground truth)
COLOR_HUMAN_MEAN = '#606060'
COLOR_VLM_MEAN = '#1A3D5C'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='/mnt/user-data/uploads/n60_progress.json')
    ap.add_argument('--out-dir', default='/home/claude/work')
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    trials = data['trials']

    q1_top = []
    q2_bot = []
    q3 = []
    for t in trials:
        n, d = extract_dot_answer(t['Q1']['response'])
        if n is not None:
            q1_top.append(normalize_to_top(n, d))
        n, d = extract_dot_answer(t['Q2']['response'])
        if n is not None:
            q2_bot.append(normalize_to_bottom(n, d))
        a = extract_angle(t['Q3']['response'])
        if a is not None:
            q3.append(a)
    
    print(f"Extracted: Q1={len(q1_top)}, Q2={len(q2_bot)}, Q3={len(q3)}")

    vlm_data = {'Q1': q1_top, 'Q2': q2_bot, 'Q3': q3}
    
    # ===== 図の作成 =====
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.labelsize': 11,
        'axes.titlesize': 11.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 8.5,
        'legend.frameon': False,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    
    panel_configs = [
        ('Q1', axes[0], np.arange(0.5, 12, 1), 1, 11, 'Dot position (from top)'),
        ('Q2', axes[1], np.arange(0.5, 12, 1), 1, 11, 'Dot position (from bottom)'),
        ('Q3', axes[2], np.arange(-2.5, 92.5, 5), 0, 90, 'Degrees above horizontal'),
    ]
    
    summary_text = {
        'Q1': f'Human: μ={HUMAN["Q1"]["mean"]:.2f}, σ={HUMAN["Q1"]["sd"]:.2f}\nVLM:    μ={np.mean(q1_top):.2f}, σ={np.std(q1_top, ddof=1):.2f}',
        'Q2': f'Human: μ={HUMAN["Q2"]["mean"]:.2f}, σ={HUMAN["Q2"]["sd"]:.2f}\nVLM:    μ={np.mean(q2_bot):.2f}, σ={np.std(q2_bot, ddof=1):.2f}',
        'Q3': f'Human: μ={HUMAN["Q3"]["mean"]:.1f}°, σ={HUMAN["Q3"]["sd"]:.1f}°\nVLM:    μ={np.mean(q3):.1f}°, σ={np.std(q3, ddof=1):.1f}°',
    }
    
    for q_key, ax, bins, x_min, x_max, xlabel in panel_configs:
        h_mean = HUMAN[q_key]['mean']
        h_sd = HUMAN[q_key]['sd']
        truth = HUMAN[q_key]['truth']
        vlm_values = vlm_data[q_key]
        
        # --- 人間: 正規近似曲線 (mean±SD) ---
        x_smooth = np.linspace(x_min, x_max, 500)
        if q_key in ('Q1', 'Q2'):
            human_density = (stats.norm.cdf(np.arange(1.5, 12.5), h_mean, h_sd)
                             - stats.norm.cdf(np.arange(0.5, 11.5), h_mean, h_sd))
            human_x = np.arange(1, 12)
            ax.bar(human_x, human_density, width=0.85, color=COLOR_HUMAN_FILL,
                   edgecolor=COLOR_HUMAN, linewidth=0.8, alpha=0.9, zorder=2,
                   label='Human N=130')
        else:
            human_density = (stats.norm.cdf(np.arange(2.5, 95, 5), h_mean, h_sd)
                             - stats.norm.cdf(np.arange(-2.5, 90, 5), h_mean, h_sd))
            human_x = np.arange(0, 91, 5)
            ax.bar(human_x, human_density, width=4.5, color=COLOR_HUMAN_FILL,
                   edgecolor=COLOR_HUMAN, linewidth=0.8, alpha=0.9, zorder=2,
                   label='Human N=130')
        
        # --- VLM: 実測値のヒストグラム ---
        if q_key in ('Q1', 'Q2'):
            counter = Counter(vlm_values)
            vlm_x = np.arange(1, 12)
            vlm_counts = np.array([counter.get(i, 0) for i in vlm_x])
            vlm_props = vlm_counts / len(vlm_values)
            ax.bar(vlm_x, vlm_props, width=0.45, color=COLOR_VLM,
                   edgecolor='none', alpha=0.95, zorder=3,
                   label=f'Qwen2.5-VL-32B N={len(vlm_values)}')
        else:
            vlm_counter = Counter(vlm_values)
            vlm_x_unique = sorted(vlm_counter.keys())
            vlm_counts = np.array([vlm_counter[v] for v in vlm_x_unique])
            vlm_props = vlm_counts / len(vlm_values)
            ax.bar(vlm_x_unique, vlm_props, width=2.5, color=COLOR_VLM,
                   edgecolor='none', alpha=0.95, zorder=3,
                   label=f'Qwen2.5-VL-32B N={len(vlm_values)}')
        
        # --- Ground truth ---
        ax.axvline(truth, color=COLOR_TRUTH, linestyle='--', linewidth=1.5,
                   zorder=4, label=f'Ground truth')
        
        # --- 平均位置の縦線 ---
        ax.axvline(h_mean, color=COLOR_HUMAN_MEAN, linestyle=':', linewidth=1.0,
                   alpha=0.7, zorder=2.5)
        ax.axvline(np.mean(vlm_values), color=COLOR_VLM_MEAN, linestyle=':',
                   linewidth=1.0, alpha=0.9, zorder=3.5)
        
        # --- 統計サマリーをテキストボックスで ---
        ax.text(0.97, 0.97, summary_text[q_key], transform=ax.transAxes,
                fontsize=8.5, verticalalignment='top', horizontalalignment='right',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#BBBBBB', linewidth=0.6, alpha=0.92))
        
        # --- 軸装飾 ---
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Proportion of trials' if q_key == 'Q1' else '')
        ax.set_title(HUMAN[q_key]['label'], fontweight='bold')
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if q_key in ('Q1', 'Q2'):
            ax.set_xticks(range(1, 12))
        else:
            ax.set_xticks(range(0, 91, 15))
    
    # ===== 共通凡例を最下部に =====
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_HUMAN_FILL,
                      edgecolor=COLOR_HUMAN, label='Human N=130 (normal-approx from μ, σ)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_VLM,
                      edgecolor='none', label='Qwen2.5-VL-32B N=60'),
        Line2D([0], [0], color=COLOR_TRUTH, linestyle='--', linewidth=1.5,
               label='Ground truth'),
        Line2D([0], [0], color=COLOR_HUMAN_MEAN, linestyle=':', linewidth=1.2,
               label='Human mean'),
        Line2D([0], [0], color=COLOR_VLM_MEAN, linestyle=':', linewidth=1.2,
               label='VLM mean'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=9.5)
    
    fig.suptitle('Watanabe Illusion: Human (N=130) vs Qwen2.5-VL-32B (N=60, 3-turn independent)',
                 fontsize=12.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = out_dir / 'qwen25vl_n60_v2_vs_human.png'
    pdf_path = out_dir / 'qwen25vl_n60_v2_vs_human.pdf'
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
