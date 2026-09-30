"""
analyze_qwen25vl_n60_v2.py
N=60, 3-turn independent, human_G1G2G3_aligned_v1 の集計。
"""
import json
import re
import statistics
from collections import Counter
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/n60_progress.json'
N_DOTS = 11

ORDINAL_WORDS = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10, 'eleventh': 11,
}

def extract_dot_answer(response):
    """応答から (数字, 'top'|'bottom') を抽出。出現順最後を採用。"""
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
    """\boxed{N}, \(N^\circ\), N°, N degrees の最終出現を抽出。
    boxed があれば最優先（モデルの「最終回答」マーカー）。"""
    boxed = list(re.finditer(r'\\?boxed\{(\d+(?:\.\d+)?)\}', response))
    if boxed:
        # 最後の boxed
        return float(boxed[-1].group(1)), 'boxed'
    candidates = []
    for pat, label in [
        (r'\\?\(?\s*(\d+(?:\.\d+)?)\s*\^\\?circ', 'latex'),
        (r'(\d+(?:\.\d+)?)\s*°', 'unicode'),
        (r'(\d+(?:\.\d+)?)[\s-]*degree', 'word'),
    ]:
        for m in re.finditer(pat, response, re.IGNORECASE):
            candidates.append((m.start(), float(m.group(1)), label))
    if candidates:
        candidates.sort()
        _, val, src = candidates[-1]
        return val, src
    return None, None

with open(PATH) as f:
    data = json.load(f)

trials = data['trials']
print(f"Source: {PATH}")
print(f"Run: {data['started_at']}, model={data['model_id']}")
print(f"Structure: {data['structure']}")
print(f"Prompt set: {data['prompt_set']}")
print(f"N trials: {len(trials)}, T={data['temperature']}, rep_pen={data['repetition_penalty']}")

# === 抽出 ===
q1_data = []  # (trial_idx, raw_num, raw_dir, top_norm, bottom_norm)
q2_data = []
q3_data = []

q1_failed, q2_failed, q3_failed = [], [], []

for t in trials:
    idx = t['trial_idx']
    
    # Q1
    n, d = extract_dot_answer(t['Q1']['response'])
    if n is None:
        q1_failed.append(idx)
    else:
        q1_data.append((idx, n, d, normalize_to_top(n, d), normalize_to_bottom(n, d)))
    
    # Q2
    n, d = extract_dot_answer(t['Q2']['response'])
    if n is None:
        q2_failed.append(idx)
    else:
        q2_data.append((idx, n, d, normalize_to_top(n, d), normalize_to_bottom(n, d)))
    
    # Q3
    a, src = extract_angle(t['Q3']['response'])
    if a is None:
        q3_failed.append(idx)
    else:
        q3_data.append((idx, a, src))

print(f"\nExtraction success: Q1={len(q1_data)}/60, Q2={len(q2_data)}/60, Q3={len(q3_data)}/60")
if q1_failed: print(f"  Q1 failed: {q1_failed}")
if q2_failed: print(f"  Q2 failed: {q2_failed}")
if q3_failed: print(f"  Q3 failed: {q3_failed}")

# === 方向遵守チェック ===
q1_dir_count = Counter(d for _, _, d, _, _ in q1_data)
q2_dir_count = Counter(d for _, _, d, _, _ in q2_data)
print(f"\nDirection compliance:")
print(f"  Q1 direction (asked: top): top={q1_dir_count.get('top',0)}, bottom={q1_dir_count.get('bottom',0)}")
print(f"  Q2 direction (asked: bottom): top={q2_dir_count.get('top',0)}, bottom={q2_dir_count.get('bottom',0)}")

# === Q1: from-top 基準で分布 ===
def show_dist(values, label, n_dots=N_DOTS, header_unit='from top'):
    print(f"\n=== {label} ===")
    counter = Counter(values)
    for pos in range(1, n_dots+1):
        cnt = counter.get(pos, 0)
        bar = '█' * cnt
        print(f"  Dot {pos:2d} {header_unit}: {cnt:3d}  {bar}")
    if values:
        sorted_v = sorted(values)
        print(f"  N: {len(values)}")
        print(f"  Median: {statistics.median(values):.1f}")
        print(f"  Mean:   {statistics.mean(values):.2f}")
        print(f"  Stdev:  {statistics.stdev(values):.2f}" if len(values) > 1 else "  (N=1)")
        q25 = sorted_v[len(sorted_v)//4]
        q75 = sorted_v[3*len(sorted_v)//4]
        print(f"  IQR:    [{q25}, {q75}]")
        print(f"  Min/Max: {min(values)} / {max(values)}")

# Q1 を「from top」に正規化（人間 Q1 と直接比較するため）
q1_top_norm = [v for _, _, _, v, _ in q1_data]
show_dist(q1_top_norm, "Q1: dot from top (normalized, n={}/60)".format(len(q1_top_norm)),
          header_unit='from top')

# Q2 を「from bottom」に正規化（人間 Q2 と直接比較）
q2_bot_norm = [v for _, _, _, _, v in q2_data]
show_dist(q2_bot_norm, "Q2: dot from bottom (normalized, n={}/60)".format(len(q2_bot_norm)),
          header_unit='from bottom')

# === Q3 角度分布 ===
print(f"\n=== Q3: degrees above horizontal (n={len(q3_data)}/60) ===")
angles = [a for _, a, _ in q3_data]
counter3 = Counter(angles)
for angle in sorted(counter3.keys()):
    cnt = counter3[angle]
    bar = '█' * cnt
    print(f"  {angle:5.1f}°: {cnt:3d}  {bar}")
print(f"  Median: {statistics.median(angles):.1f}°")
print(f"  Mean:   {statistics.mean(angles):.2f}°")
print(f"  Stdev:  {statistics.stdev(angles):.2f}°")
sorted_a = sorted(angles)
print(f"  IQR:    [{sorted_a[len(sorted_a)//4]}, {sorted_a[3*len(sorted_a)//4]}]°")
print(f"  Min/Max: {min(angles)}° / {max(angles)}°")
src_count = Counter(s for _, _, s in q3_data)
print(f"  Source: {dict(src_count)}")

# === Q1 + Q2 内的整合性 (同 trial 内) ===
print(f"\n=== Q1↔Q2 internal consistency (per-trial) ===")
# trial_idx でマージ
q1_by_idx = {idx: top for idx, _, _, top, _ in q1_data}
q2_by_idx = {idx: bot for idx, _, _, _, bot in q2_data}
common = sorted(set(q1_by_idx.keys()) & set(q2_by_idx.keys()))
print(f"  Trials with both Q1 and Q2: {len(common)}")
sums = []
for idx in common:
    q1_top = q1_by_idx[idx]
    q2_bot = q2_by_idx[idx]
    # Q1 (from top) と Q2 (from bottom) が同じ dot を指していれば q1 + q2 = 12 (= n_dots+1)
    s = q1_top + q2_bot
    sums.append(s)

if sums:
    sum_counter = Counter(sums)
    print(f"  Distribution of (Q1_from_top + Q2_from_bottom):")
    for s in sorted(sum_counter.keys()):
        cnt = sum_counter[s]
        marker = ' <-- consistent (=12)' if s == 12 else ''
        print(f"    sum={s:2d}: {cnt:3d}{marker}")
    print(f"  Mean sum: {statistics.mean(sums):.2f} (perfect consistency = 12)")
    print(f"  Consistent trials (sum=12): {sum_counter.get(12, 0)}/{len(sums)}")

# === 人間データとの比較 ===
print(f"\n=== Human N=130 vs Qwen2.5-VL-32B N=60 ===")
human = {
    'Q1': (4.6489, 2.6310),
    'Q2': (6.3588, 2.8557),
    'Q3': (37.443, 22.803),
}
print(f"          Human (N=130)         | Qwen2.5-VL-32B v2 (N=60)")
print(f"  Q1:   {human['Q1'][0]:.3f} ± {human['Q1'][1]:.3f}    | {statistics.mean(q1_top_norm):.3f} ± {statistics.stdev(q1_top_norm):.3f}  (median={statistics.median(q1_top_norm):.1f})")
print(f"  Q2:   {human['Q2'][0]:.3f} ± {human['Q2'][1]:.3f}    | {statistics.mean(q2_bot_norm):.3f} ± {statistics.stdev(q2_bot_norm):.3f}  (median={statistics.median(q2_bot_norm):.1f})")
print(f"  Q3:   {human['Q3'][0]:.3f}° ± {human['Q3'][1]:.3f}°  | {statistics.mean(angles):.3f}° ± {statistics.stdev(angles):.3f}°  (median={statistics.median(angles):.1f}°)")

# Ground truth との誤差
print(f"\nGround truth: Q1=1, Q2=11, Q3=23.5°")
print(f"  Human   Q1 error:  {human['Q1'][0]-1:+.2f} dots from truth")
print(f"  Qwen v2 Q1 error:  {statistics.mean(q1_top_norm)-1:+.2f} dots from truth")
print(f"  Human   Q3 error:  {human['Q3'][0]-23.5:+.2f}° from truth")
print(f"  Qwen v2 Q3 error:  {statistics.mean(angles)-23.5:+.2f}° from truth")
