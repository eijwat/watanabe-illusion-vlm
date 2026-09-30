"""
analyze_n60_q1q2.py
====================
A100 と DGX Spark の Qwen2.5-VL-32B N=60 Q1/Q2 結果を統合解析し、
論文 v4 用の図を 3 枚生成する。

入力:
  /mnt/user-data/uploads/n60_progress_a100.json    (or local path)
  /mnt/user-data/uploads/n60_progress_spark.json
  /mnt/project/human_2nd_test.csv                  (人間 N=130 wide format)

出力:
  fig_q1q2_distributions.png   - メイン図: Q1, Q2, Q1+Q2 sum の重ね描き
                                 (A100 / Spark / 人間 N=130 の 3 系列)
  fig_machine_consistency.png  - trial-by-trial 散布図 (A100 vs Spark)
                                 マシン違いの確率的非決定性を可視化
  fig_n60_v2_vs_human.png      - 既存図 (qwen25vl_n60_v2_vs_human.png) の更新版
                                 (A100 / Spark のいずれかを Qwen line として使う)
  stats_summary.json           - 全統計値の構造化サマリ

抽出ロジック:
  - 数字: \d+ + ordinal suffix (st/nd/rd/th)
  - 綴り: first/second/third/.../twelfth (A100 trial 7, 54 で観察)
  - \boxed{N}, \boxed{\text{Xth dot from ...}} 形式
  - 方向 (top/bottom) も同時抽出 (compliance チェック用)
"""

import json
import re
import sys
import csv
from pathlib import Path
from collections import Counter
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ========== 設定 ==========
A100_JSON = Path("/mnt/user-data/uploads/n60_progress_a100.json")
SPARK_JSON = Path("/mnt/user-data/uploads/n60_progress_spark.json")
HUMAN_CSV = Path("/mnt/project/human_2nd_test.csv")
OUT_DIR = Path("/home/claude")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 11 ドットなので Q1+Q2 sum=12 が「ground truth (絶対整合)」
N_DOTS = 11
GT_Q1 = 1   # 104.jpg: 線は dot 1 (top から)
GT_Q2 = 11  # 104.jpg: 線は dot 11 (bottom から) と同じ位置 → これは線が dot 1 から見て top 1 番目 = bottom 11 番目
GT_SUM = N_DOTS + 1  # 12

# 色 (3 系列、人間中心の配色)
COLOR_A100 = "#1f77b4"      # blue
COLOR_SPARK = "#ff7f0e"     # orange
COLOR_HUMAN = "#2ca02c"     # green
COLOR_GT = "#d62728"        # red (ground truth マーカー)


# ========== 数値抽出 (綴り対応版) ==========
SPELLED_MAP = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12,
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
    'eleventh': 11, 'twelfth': 12,
}
SPELLED_REGEX = '|'.join(SPELLED_MAP.keys())


def extract_number_and_direction(text):
    """response から (number:int|None, direction:'top'|'bottom'|None) を抽出。
    優先順位:
      1. "the (Xth|spelled) dot from the (top|bottom)"
      2. \boxed{...top|bottom...} with embedded number
      3. \boxed{N} + 周辺から direction
      4. fallback: 最後の "Xth dot" (direction unspecified)
    """
    # Rule 1a: 数字 ordinal
    m = re.search(
        r'\bthe\s+(\d+)(?:st|nd|rd|th)\s+dot\s+from\s+the\s+(top|bottom)\b',
        text, re.IGNORECASE
    )
    if m:
        return int(m.group(1)), m.group(2).lower()

    # Rule 1b: 綴り ordinal/cardinal
    m = re.search(
        r'\bthe\s+(' + SPELLED_REGEX + r')\s+dot\s+from\s+the\s+(top|bottom)\b',
        text, re.IGNORECASE
    )
    if m:
        return SPELLED_MAP[m.group(1).lower()], m.group(2).lower()

    # Rule 2: \boxed{...} に数値と方向の両方
    m = re.search(
        r'\\boxed\{[^}]*?(\d+)(?:st|nd|rd|th)?\s*dot[^}]*?(top|bottom)',
        text, re.IGNORECASE
    )
    if m:
        return int(m.group(1)), m.group(2).lower()

    # Rule 3: \boxed{N} だけ
    m = re.search(r'\\boxed\{\s*(\d+)\s*\}', text)
    if m:
        n = int(m.group(1))
        # 本文から direction を探す
        dir_m = re.search(
            r'(\d+|' + SPELLED_REGEX + r')(?:st|nd|rd|th)?\s+dot\s+from\s+the\s+(top|bottom)',
            text, re.IGNORECASE
        )
        if dir_m:
            return n, dir_m.group(2).lower()
        return n, None

    # Rule 4: 数字だけの "Xth dot"
    m = re.findall(r'(\d+)(?:st|nd|rd|th)\s+dot', text, re.IGNORECASE)
    if m:
        return int(m[-1]), None

    return None, None


def extract_trials(data, expected_q1_dir='top', expected_q2_dir='bottom'):
    """JSON から trial の list of dicts に変換。
    各 dict は: trial, q1_num, q1_dir, q2_num, q2_dir, elapsed
    """
    rows = []
    for t in data['trials']:
        q1_text = t['Q1']['response']
        q2_text = t['Q2']['response']
        q1_num, q1_dir = extract_number_and_direction(q1_text)
        q2_num, q2_dir = extract_number_and_direction(q2_text)
        rows.append({
            'trial': t['trial_idx'],
            'q1_num': q1_num, 'q1_dir': q1_dir,
            'q2_num': q2_num, 'q2_dir': q2_dir,
            'elapsed': t.get('elapsed_seconds', 0),
        })
    return rows


def normalize_q1(num, direction, n_dots=N_DOTS):
    """Q1 は 'from top' が正規。'bottom' で答えた場合は補正。"""
    if num is None:
        return None
    if direction == 'bottom':
        return n_dots - num + 1
    return num


def normalize_q2(num, direction, n_dots=N_DOTS):
    """Q2 は 'from bottom' が正規。'top' で答えた場合は補正。"""
    if num is None:
        return None
    if direction == 'top':
        return n_dots - num + 1
    return num


# ========== 人間 N=130 データ読み込み ==========
def load_human_data(value_max=20.0):
    """human_2nd_test.csv (wide format: 4 columns) から Q1, Q2, Q3 を listwise で抽出。
    Q1/Q2 は dot 番号なので、合理的な範囲外 (value_max を超える) は誤入力として除外。
    記憶では N=130 が正規 (project memory に記載)。
    """
    q1_list, q2_list, q3_list = [], [], []
    n_excluded = 0
    with open(HUMAN_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            try:
                q1 = float(row[0]) if row[0] != '' else None
                q2 = float(row[1]) if row[1] != '' else None
                q3 = float(row[2]) if row[2] != '' else None
            except (ValueError, IndexError):
                continue
            if q1 is None or q2 is None or q3 is None:
                continue
            # 異常値除外: dot 番号は [0, value_max] 内のはず (Q3 角度は別範囲なので除外せず)
            if q1 < 0 or q1 > value_max or q2 < 0 or q2 > value_max:
                n_excluded += 1
                continue
            q1_list.append(q1)
            q2_list.append(q2)
            q3_list.append(q3)
    if n_excluded > 0:
        print(f"  [human data] excluded {n_excluded} rows (Q1/Q2 outside [0, {value_max}])")
    return q1_list, q2_list, q3_list


# ========== 統計サマリ ==========
def compute_stats(values):
    """list of numbers から mean, median, stdev, n を返す。"""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return {'n': len(vals), 'mean': None, 'median': None, 'stdev': None,
                'min': None, 'max': None}
    return {
        'n': len(vals),
        'mean': statistics.mean(vals),
        'median': statistics.median(vals),
        'stdev': statistics.stdev(vals),
        'min': min(vals),
        'max': max(vals),
    }


# ========== Plot 1: distributions ==========
def plot_distributions(a100_q1, a100_q2, a100_sum,
                       spark_q1, spark_q2, spark_sum,
                       human_q1, human_q2, human_sum,
                       save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 共通スタイル
    def plot_panel(ax, a, s, h, title, xlabel, gt_value=None,
                   xlim=None, bin_edges=None):
        if bin_edges is None:
            all_vals = list(a) + list(s) + list(h)
            mn, mx = min(all_vals), max(all_vals)
            # 0.5 刻み (整数バイン: −0.5, 0.5, 1.5, …)
            mn_int = int(np.floor(mn))
            mx_int = int(np.ceil(mx))
            bin_edges = np.arange(mn_int - 0.5, mx_int + 1.5, 1.0)

        # 人間 (background, 半透明)
        ax.hist(h, bins=bin_edges, density=True, alpha=0.35,
                color=COLOR_HUMAN, label=f'Human N={len(h)}',
                edgecolor='none')
        # A100 (step outline)
        ax.hist(a, bins=bin_edges, density=True, histtype='step',
                color=COLOR_A100, linewidth=2.5,
                label=f'Qwen2.5-VL (A100) N={len(a)}')
        # Spark (step outline, dashed)
        ax.hist(s, bins=bin_edges, density=True, histtype='step',
                color=COLOR_SPARK, linewidth=2.5, linestyle='--',
                label=f'Qwen2.5-VL (DGX Spark) N={len(s)}')

        # mean を点で重ね描き
        ax.axvline(np.mean(h), color=COLOR_HUMAN, linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(np.mean(a), color=COLOR_A100, linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(np.mean(s), color=COLOR_SPARK, linestyle=':', linewidth=1.5, alpha=0.7)

        # Ground truth マーカー
        if gt_value is not None:
            ax.axvline(gt_value, color=COLOR_GT, linewidth=2.5, alpha=0.85,
                       label=f'Ground truth = {gt_value}')

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plot_panel(axes[0], a100_q1, spark_q1, human_q1,
               "Q1: dot number from top", "dot # (from top)",
               gt_value=GT_Q1, xlim=(0, 12),
               bin_edges=np.arange(-0.5, 13.5, 1.0))
    plot_panel(axes[1], a100_q2, spark_q2, human_q2,
               "Q2: dot number from bottom", "dot # (from bottom)",
               gt_value=GT_Q2, xlim=(0, 12),
               bin_edges=np.arange(-0.5, 13.5, 1.0))
    plot_panel(axes[2], a100_sum, spark_sum, human_sum,
               "Q1 + Q2 sum (consistency check)", "Q1 + Q2",
               gt_value=GT_SUM, xlim=(0, 24),
               bin_edges=np.arange(-0.5, 25.5, 1.0))

    fig.suptitle(
        "Qwen2.5-VL-32B vs Human (N=130) on 104.jpg (mental extension bias)",
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


# ========== Plot 2: machine consistency (trial-by-trial scatter) ==========
def plot_machine_consistency(a100_rows, spark_rows, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Q1, Q2, Q1+Q2 sum をそれぞれ散布図
    panels = [
        ('q1_norm', 'Q1 (from top, normalized)', GT_Q1),
        ('q2_norm', 'Q2 (from bottom, normalized)', GT_Q2),
        ('sum', 'Q1 + Q2', GT_SUM),
    ]

    for ax, (key, title, gt) in zip(axes, panels):
        xs, ys = [], []
        for a, s in zip(a100_rows, spark_rows):
            if a[key] is not None and s[key] is not None:
                xs.append(a[key])
                ys.append(s[key])
        xs = np.array(xs)
        ys = np.array(ys)

        # ジッターを少しだけ加える (重なり可視化のため)
        rng = np.random.default_rng(42)
        xs_j = xs + rng.uniform(-0.15, 0.15, size=len(xs))
        ys_j = ys + rng.uniform(-0.15, 0.15, size=len(ys))

        ax.scatter(xs_j, ys_j, alpha=0.6, s=40,
                   edgecolor='black', linewidth=0.5,
                   color='#4a4a8a')

        # 一致率
        n_match = int(np.sum(xs == ys))
        n_total = len(xs)
        ax.text(0.05, 0.95,
                f'exact match: {n_match}/{n_total} ({100*n_match/n_total:.1f}%)',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

        # y=x 対角線
        lo = min(xs.min(), ys.min()) - 0.5
        hi = max(xs.max(), ys.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--',
                linewidth=1.5, alpha=0.6, label='y = x')
        # GT マーカー
        ax.axvline(gt, color=COLOR_GT, linewidth=1.5, alpha=0.5)
        ax.axhline(gt, color=COLOR_GT, linewidth=1.5, alpha=0.5)

        ax.set_xlabel('A100 (x86_64, HBM2e)', fontsize=11)
        ax.set_ylabel('DGX Spark (aarch64, LPDDR5x)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Trial-by-trial: same model, same seed, different machine "
        "(numerical non-determinism across architectures)",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


# ========== Plot 3: updated qwen25vl_n60_v2_vs_human (Spark or A100 line) ==========
def plot_n60_vs_human_updated(a100_q1, a100_q2, a100_sum,
                              spark_q1, spark_q2, spark_sum,
                              human_q1, human_q2, human_sum,
                              save_path):
    """既存図 qwen25vl_n60_v2_vs_human.png のレイアウトを踏襲し、
    両マシンを並べた更新版。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [
        ('Q1 (from top)', human_q1, a100_q1, spark_q1, GT_Q1),
        ('Q2 (from bottom)', human_q2, a100_q2, spark_q2, GT_Q2),
        ('Q1+Q2 sum', human_sum, a100_sum, spark_sum, GT_SUM),
    ]

    for ax, (title, h, a, s, gt) in zip(axes, pairs):
        all_vals = list(h) + list(a) + list(s)
        bin_edges = np.arange(int(min(all_vals))-0.5, int(max(all_vals))+1.5, 1.0)

        # 人間ヒストグラム (背景)
        ax.hist(h, bins=bin_edges, density=True, alpha=0.4,
                color=COLOR_HUMAN, label=f'Human (N={len(h)})',
                edgecolor=COLOR_HUMAN, linewidth=0.7)

        # A100 + Spark を一つの "Qwen2.5-VL" line として step plot
        # (両機平均的なシグナルとして見せる: 別系列としても見せる)
        ax.hist(a, bins=bin_edges, density=True, histtype='step',
                color=COLOR_A100, linewidth=2.0,
                label=f'Qwen2.5-VL A100 (N={len(a)})')
        ax.hist(s, bins=bin_edges, density=True, histtype='step',
                color=COLOR_SPARK, linewidth=2.0, linestyle='--',
                label=f'Qwen2.5-VL Spark (N={len(s)})')

        # ground truth
        ax.axvline(gt, color=COLOR_GT, linewidth=2.5, alpha=0.85,
                   label=f'Ground truth = {gt}')

        # mean 線
        ax.axvline(np.mean(h), color=COLOR_HUMAN, linestyle=':', linewidth=1.3)
        ax.axvline(np.mean(a), color=COLOR_A100, linestyle=':', linewidth=1.3)
        ax.axvline(np.mean(s), color=COLOR_SPARK, linestyle=':', linewidth=1.3)

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(all_vals)-1, max(all_vals)+1)

    fig.suptitle(
        "Qwen2.5-VL-32B (Q1/Q2 only) vs Human (N=130) — both machines",
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {save_path}")


# ========== Main ==========
def main():
    print("Loading data...")
    a100_data = json.load(open(A100_JSON))
    spark_data = json.load(open(SPARK_JSON))
    human_q1, human_q2, human_q3 = load_human_data()
    human_sum = [q1 + q2 for q1, q2 in zip(human_q1, human_q2)]

    print(f"  A100  trials: {len(a100_data['trials'])}")
    print(f"  Spark trials: {len(spark_data['trials'])}")
    print(f"  Human listwise N: {len(human_q1)}")

    # 抽出 (綴り対応版)
    a100_rows = extract_trials(a100_data)
    spark_rows = extract_trials(spark_data)

    # 正規化 (方向違反を補正)
    for r in a100_rows + spark_rows:
        r['q1_norm'] = normalize_q1(r['q1_num'], r['q1_dir'])
        r['q2_norm'] = normalize_q2(r['q2_num'], r['q2_dir'])
        if r['q1_norm'] is not None and r['q2_norm'] is not None:
            r['sum'] = r['q1_norm'] + r['q2_norm']
        else:
            r['sum'] = None

    # 数値リスト (抽出成功分のみ)
    a100_q1 = [r['q1_norm'] for r in a100_rows if r['q1_norm'] is not None]
    a100_q2 = [r['q2_norm'] for r in a100_rows if r['q2_norm'] is not None]
    a100_sum = [r['sum'] for r in a100_rows if r['sum'] is not None]
    spark_q1 = [r['q1_norm'] for r in spark_rows if r['q1_norm'] is not None]
    spark_q2 = [r['q2_norm'] for r in spark_rows if r['q2_norm'] is not None]
    spark_sum = [r['sum'] for r in spark_rows if r['sum'] is not None]

    # 統計
    stats = {
        'A100': {
            'machine': a100_data.get('machine'),
            'Q1': compute_stats(a100_q1),
            'Q2': compute_stats(a100_q2),
            'Q1+Q2_sum': compute_stats(a100_sum),
            'n_q1_extracted': len(a100_q1),
            'n_q2_extracted': len(a100_q2),
            'n_trials_total': len(a100_rows),
            'q1_direction_compliance_top': sum(1 for r in a100_rows if r['q1_dir'] == 'top'),
            'q1_direction_violation_bottom': sum(1 for r in a100_rows if r['q1_dir'] == 'bottom'),
            'q2_direction_compliance_bottom': sum(1 for r in a100_rows if r['q2_dir'] == 'bottom'),
            'q2_direction_violation_top': sum(1 for r in a100_rows if r['q2_dir'] == 'top'),
        },
        'Spark': {
            'machine': spark_data.get('machine'),
            'Q1': compute_stats(spark_q1),
            'Q2': compute_stats(spark_q2),
            'Q1+Q2_sum': compute_stats(spark_sum),
            'n_q1_extracted': len(spark_q1),
            'n_q2_extracted': len(spark_q2),
            'n_trials_total': len(spark_rows),
            'q1_direction_compliance_top': sum(1 for r in spark_rows if r['q1_dir'] == 'top'),
            'q1_direction_violation_bottom': sum(1 for r in spark_rows if r['q1_dir'] == 'bottom'),
            'q2_direction_compliance_bottom': sum(1 for r in spark_rows if r['q2_dir'] == 'bottom'),
            'q2_direction_violation_top': sum(1 for r in spark_rows if r['q2_dir'] == 'top'),
        },
        'Human_N130_listwise': {
            'Q1': compute_stats(human_q1),
            'Q2': compute_stats(human_q2),
            'Q1+Q2_sum': compute_stats(human_sum),
            'Q3': compute_stats(human_q3),
            'N_listwise': len(human_q1),
        },
        'A100_vs_Spark_consistency': {
            'q1_exact_match': sum(
                1 for a, s in zip(a100_rows, spark_rows)
                if a['q1_norm'] is not None and s['q1_norm'] is not None
                and a['q1_norm'] == s['q1_norm']
            ),
            'q2_exact_match': sum(
                1 for a, s in zip(a100_rows, spark_rows)
                if a['q2_norm'] is not None and s['q2_norm'] is not None
                and a['q2_norm'] == s['q2_norm']
            ),
            'sum_exact_match': sum(
                1 for a, s in zip(a100_rows, spark_rows)
                if a['sum'] is not None and s['sum'] is not None
                and a['sum'] == s['sum']
            ),
        },
    }

    with open(OUT_DIR / 'stats_summary.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  saved: {OUT_DIR/'stats_summary.json'}")

    print("\nGenerating plots...")
    plot_distributions(
        a100_q1, a100_q2, a100_sum,
        spark_q1, spark_q2, spark_sum,
        human_q1, human_q2, human_sum,
        OUT_DIR / 'fig_q1q2_distributions.png'
    )
    plot_machine_consistency(
        a100_rows, spark_rows,
        OUT_DIR / 'fig_machine_consistency.png'
    )
    plot_n60_vs_human_updated(
        a100_q1, a100_q2, a100_sum,
        spark_q1, spark_q2, spark_sum,
        human_q1, human_q2, human_sum,
        OUT_DIR / 'fig_n60_v2_vs_human.png'
    )

    # コンソールサマリ
    print("\n" + "=" * 70)
    print("STATS SUMMARY (key numbers)")
    print("=" * 70)
    for label in ['A100', 'Spark', 'Human_N130_listwise']:
        s = stats[label]
        print(f"\n[{label}]")
        for k in ['Q1', 'Q2', 'Q1+Q2_sum']:
            v = s[k]
            print(f"  {k:>12s}: n={v['n']:3d}  mean={v['mean']:.3f}  "
                  f"median={v['median']:.1f}  stdev={v['stdev']:.3f}")
    print(f"\nGround truth: Q1={GT_Q1}, Q2={GT_Q2}, sum={GT_SUM}  (104.jpg, 11 dots)")
    print(f"\nA100 vs Spark trial match:")
    c = stats['A100_vs_Spark_consistency']
    print(f"  Q1:  {c['q1_exact_match']}/60")
    print(f"  Q2:  {c['q2_exact_match']}/60")
    print(f"  Sum: {c['sum_exact_match']}/60")


if __name__ == "__main__":
    main()
