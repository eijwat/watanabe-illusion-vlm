"""
渡辺錯視（Watanabe Illusion）刺激画像生成スクリプト
=====================================================
AI（Vision Encoder解析）と人間（オンライン実験）の両方で使用する
刺激画像セットを生成する。

構図:
  - 横長の長方形（800×400px）
  - 左下に円（長方形の左辺・下辺に内接）、円内に斜めの線分
  - 右辺に縦の点列（11個）、上辺に横の点列（等間隔）
  - 右上角の点は右辺と上辺で共有

条件:
  - 角度: 5°〜85°（5°刻み、17段階）
  - standard:    円あり、標準線分長（円の直径）
  - no_circle:   円なし（線分のみ）
  - short_line:  円あり、線分長0.5倍
  - long_line:   円あり、線分長1.5倍

アンチエイリアス:
  - 4倍のスーパーサンプリングで描画後、LANCZOS で縮小

出力:
  - stimuli/ ディレクトリに PNG 画像（68枚）
  - stimuli/stimulus_metadata.csv に正解情報

使い方:
  python generate_stimuli.py

依存ライブラリ:
  numpy, Pillow
"""

import numpy as np
from PIL import Image, ImageDraw
import csv
import os


def generate_watanabe_stimulus(
    angle_deg,
    condition='standard',
    img_width=800,
    img_height=400,
    line_width=2,
    dot_radius=6,
    bg_color=255,
    fg_color=0,
    supersample=4,
):
    """
    渡辺錯視の刺激画像を1枚生成する。

    Parameters
    ----------
    angle_deg : float
        線分の角度（水平からの度数、5-85）
    condition : str
        'standard'    - 円あり、標準線分長
        'no_circle'   - 円なし（線分のみ）
        'short_line'  - 円あり、線分長0.5倍
        'long_line'   - 円あり、線分長1.5倍
    img_width, img_height : int
        出力画像サイズ（ピクセル）
    line_width : int
        線の太さ（出力スケール）
    dot_radius : int
        点の半径（出力スケール）
    bg_color, fg_color : int
        背景色と前景色（0-255）
    supersample : int
        アンチエイリアス用スーパーサンプリング倍率

    Returns
    -------
    img : PIL.Image
        生成された刺激画像
    info : dict
        正解情報を含むメタデータ
    """
    ss = supersample
    W, H = img_width * ss, img_height * ss
    lw = line_width * ss
    dr = dot_radius * ss

    img = Image.new('RGB', (W, H), (bg_color,) * 3)
    draw = ImageDraw.Draw(img)

    # --- 長方形 ---
    margin = 15 * ss
    rect_left = margin
    rect_top = margin
    rect_right = W - margin
    rect_bottom = H - margin
    rect_w = rect_right - rect_left
    rect_h = rect_bottom - rect_top

    draw.rectangle(
        [rect_left, rect_top, rect_right, rect_bottom],
        outline=(fg_color,) * 3, width=lw,
    )

    # --- 円（左辺・下辺に内接）---
    circle_r = rect_h * 0.137
    circle_cx = rect_left + circle_r
    circle_cy = rect_bottom - circle_r

    # --- 点列（等間隔）---
    n_dots_vertical = 11
    dot_spacing = rect_h / (n_dots_vertical - 1)

    # 右辺の縦点列（上から下）
    dots_right = [
        (rect_right, rect_top + i * dot_spacing)
        for i in range(n_dots_vertical)
    ]

    # 上辺の横点列（角の点は右辺と共有、その左から等間隔）
    dots_top = []
    x = rect_right - dot_spacing
    while x >= rect_left:
        dots_top.append((x, rect_top))
        x -= dot_spacing
    n_dots_horizontal = len(dots_top)

    # 点描画
    for dx, dy in dots_right + dots_top:
        draw.ellipse(
            [dx - dr, dy - dr, dx + dr, dy + dr],
            fill=(fg_color,) * 3,
        )

    # --- 線分 ---
    angle_rad = np.radians(angle_deg)
    line_length_base = circle_r * 2  # 円の直径
    length_multiplier = {'short_line': 0.5, 'long_line': 1.5}.get(condition, 1.0)
    line_length = line_length_base * length_multiplier

    lsx = circle_cx - (line_length / 2) * np.cos(angle_rad)
    lsy = circle_cy + (line_length / 2) * np.sin(angle_rad)
    lex = circle_cx + (line_length / 2) * np.cos(angle_rad)
    ley = circle_cy - (line_length / 2) * np.sin(angle_rad)

    draw.line(
        [(lsx, lsy), (lex, ley)],
        fill=(fg_color,) * 3, width=lw,
    )

    # --- 円描画（no_circle 条件以外）---
    if condition != 'no_circle':
        draw.ellipse(
            [circle_cx - circle_r, circle_cy - circle_r,
             circle_cx + circle_r, circle_cy + circle_r],
            outline=(fg_color,) * 3, width=lw,
        )

    # --- スーパーサンプリング縮小（アンチエイリアス）---
    img = img.resize((img_width, img_height), Image.LANCZOS)

    # --- 正解計算（出力スケール）---
    circle_cx_o = circle_cx / ss
    circle_cy_o = circle_cy / ss
    rect_right_o = rect_right / ss
    rect_top_o = rect_top / ss

    target_y_right = circle_cy_o - np.tan(angle_rad) * (rect_right_o - circle_cx_o)
    target_x_top = (
        circle_cx_o + (circle_cy_o - rect_top_o) / np.tan(angle_rad)
        if angle_deg > 0 else float('inf')
    )

    dots_right_o = [(dx / ss, dy / ss) for dx, dy in dots_right]
    dots_top_o = [(dx / ss, dy / ss) for dx, dy in dots_top]

    if target_y_right >= rect_top_o:
        hit_edge = 'right'
        distances = [abs(dy - target_y_right) for _, dy in dots_right_o]
        correct_idx = int(np.argmin(distances))
        correct_dot_label = f"right_{correct_idx + 1}"
        correct_dot_number = correct_idx + 1
    else:
        hit_edge = 'top'
        distances = [abs(dx - target_x_top) for dx, _ in dots_top_o]
        correct_idx = int(np.argmin(distances))
        correct_dot_label = f"top_{correct_idx + 1}"
        correct_dot_number = correct_idx + 1

    return img, {
        'angle_deg': angle_deg,
        'condition': condition,
        'hit_edge': hit_edge,
        'correct_dot_label': correct_dot_label,
        'correct_dot_number': correct_dot_number,
        'target_y_right': round(target_y_right, 2),
        'target_x_top': round(target_x_top, 2),
        'n_dots_vertical': n_dots_vertical,
        'n_dots_horizontal': n_dots_horizontal,
        'dot_spacing': round(dot_spacing / ss, 2),
        'circle_cx': round(circle_cx_o, 2),
        'circle_cy': round(circle_cy_o, 2),
        'circle_r': round(circle_r / ss, 2),
        'img_width': img_width,
        'img_height': img_height,
    }


def generate_all_stimuli(output_dir='stimuli'):
    """
    全条件の刺激画像を生成し、メタデータを CSV に保存する。

    Parameters
    ----------
    output_dir : str
        出力ディレクトリ

    Returns
    -------
    metadata : list of dict
        各画像のメタデータ
    """
    os.makedirs(output_dir, exist_ok=True)

    angles = list(range(5, 90, 5))  # 5, 10, 15, ..., 85
    conditions = ['standard', 'no_circle', 'short_line', 'long_line']

    metadata = []

    for angle in angles:
        for condition in conditions:
            filename = f"watanabe_{angle:02d}deg_{condition}.png"
            filepath = os.path.join(output_dir, filename)

            img, info = generate_watanabe_stimulus(angle, condition)
            img.save(filepath)

            info['filename'] = filename
            metadata.append(info)

    # メタデータ CSV 保存
    csv_path = os.path.join(output_dir, 'stimulus_metadata.csv')
    fieldnames = [
        'filename', 'angle_deg', 'condition', 'hit_edge',
        'correct_dot_label', 'correct_dot_number',
        'target_y_right', 'target_x_top',
        'n_dots_vertical', 'n_dots_horizontal', 'dot_spacing',
        'circle_cx', 'circle_cy', 'circle_r',
        'img_width', 'img_height',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Generated {len(metadata)} stimulus images in '{output_dir}/'")
    print(f"Metadata saved to '{csv_path}'")

    return metadata


if __name__ == '__main__':
    metadata = generate_all_stimuli()

    # サマリ出力
    print(f"\nAngles: {metadata[0]['angle_deg']}° - {metadata[-1]['angle_deg']}° (5° steps)")
    print(f"Conditions: {sorted(set(m['condition'] for m in metadata))}")
    print(f"Dots: {metadata[0]['n_dots_vertical']} (right) + {metadata[0]['n_dots_horizontal']} (top)")
    print(f"Dot spacing: {metadata[0]['dot_spacing']}px")
    print()
    print("Correct answers (standard condition):")
    for m in metadata:
        if m['condition'] == 'standard':
            print(f"  {m['angle_deg']:5.1f}° -> {m['hit_edge']:5s} edge, dot {m['correct_dot_label']}")
