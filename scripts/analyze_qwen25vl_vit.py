"""
analyze_qwen25vl_vit.py
=========================
Phase 4: Qwen2.5-VL-32B の vision encoder (ViT + merger) が
Watanabe Illusion 系刺激の角度・空間情報をどれだけ正確に持っているかを
直接 probe する解析スクリプト。

論文での位置づけ:
- LLaVA + 別 ViT による間接的な internal analysis (CLIP/DINOv2/ViT-ImageNet) は既に終了
- 本スクリプトは Qwen2.5-VL-32B 自身の vision encoder を probe することで、
  N=60 行動データの直接の補完証拠を得る
- 仮説: Vision encoder は角度・空間情報を保持している (R² ≈ 高い)
        → 行動の bias は LM 段で発生 (knowledge dominance) を直接示す

Probe 構成 (5 種類):
  A.angle    : visual.blocks[-1] mean-pooled → angle_deg
  B.angle    : merger output mean-pooled    → angle_deg
  A.spatial  : visual.blocks[-1] mean-pooled → target_y_right
  B.spatial  : merger output mean-pooled    → target_y_right
  C.spatial  : merger output flattened patches → target_y_right (空間情報保持)

各 probe について:
  - 全 68 サンプル (4 条件 × 17 角度) で Ridge + Leave-One-Out CV
  - 4 条件別 (各 17 サンプル) で同様の解析

評価指標: R², MAE
誤差プロット: oblique effect (45° 付近で誤差が増加するか) の確認

Usage:
  # 動作確認モード (5 画像のみで A.angle のみ走らせる)
  python scripts/analyze_qwen25vl_vit.py --smoke-test

  # 本番
  python scripts/analyze_qwen25vl_vit.py

出力:
  /workspace/results/qwen25vl_vit_probe/
    ├── input_to_model_full.png      # 1 枚目の刺激の preprocessing 検証
    ├── representations.npz          # 全画像の各 probe 用表現 (再解析用キャッシュ)
    ├── probe_results.csv            # R²/MAE のサマリー
    ├── probe_results.json           # 詳細 (per-sample予測値含む)
    ├── fig_angle_probe.png          # 予測 vs 真の角度
    ├── fig_angle_error.png          # 角度誤差 vs 真の角度 (oblique effect 検出用)
    ├── fig_spatial_probe.png        # 予測 vs 真の target_y_right
    └── fig_summary_bars.png         # R²/MAE のバープロット (probe × 条件)
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ========== 設定 ==========
STIMULI_DIR = Path("/workspace/stimuli")
METADATA_CSV = Path("/workspace/stimuli/stimulus_metadata.csv")
OUT_DIR = Path("/workspace/results/qwen25vl_vit_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_MEMORY = {0: "100GiB", "cpu": "0GiB"}

CONDITIONS = ["standard", "no_circle", "short_line", "long_line"]
RIDGE_ALPHA = 1.0  # 既存スクリプトと整合


# ========== メタデータ読み込み ==========
def load_metadata(csv_path):
    """stimulus_metadata.csv から (filename, angle_deg, condition, target_y_right) を読み込む。
    pandas に依存しないように csv モジュールで実装。"""
    import csv
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'filename': row['filename'],
                'angle_deg': float(row['angle_deg']),
                'condition': row['condition'],
                'target_y_right': float(row['target_y_right']),
                'correct_dot_label': row['correct_dot_label'],
                'correct_dot_number': int(row['correct_dot_number']),
            })
    return rows


# ========== pixel_values 逆構築 (run_qwen25vl_n60_v2.py と同一ロジック) ==========
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


# ========== Vision tower の取得 (transformers バージョン差を吸収) ==========
def get_vision_tower(model):
    """Qwen2.5-VL の vision tower (visual encoder) を返す。
    transformers のバージョンによって配置が違うので、複数の候補を順に試す。
    Returns: (vision_tower_module, attribute_path_str)
    """
    candidates = [
        ('visual',                  lambda m: m.visual),
        ('model.visual',            lambda m: m.model.visual),
        ('vision_model',            lambda m: m.vision_model),
        ('model.vision_model',      lambda m: m.model.vision_model),
        ('vision_tower',            lambda m: m.vision_tower),
        ('model.vision_tower',      lambda m: m.model.vision_tower),
    ]
    for path, getter in candidates:
        try:
            vt = getter(model)
            # blocks 属性 (ViT layer list) の存在確認
            if hasattr(vt, 'blocks') and len(vt.blocks) > 0:
                return vt, path
        except (AttributeError, IndexError):
            continue
    
    # フォールバック: model 全体を walk して 'blocks' を持つ最大の module を探す
    print("[WARN] standard paths failed, scanning module tree for vision tower...")
    candidates_found = []
    for name, mod in model.named_modules():
        if hasattr(mod, 'blocks') and isinstance(mod.blocks, torch.nn.ModuleList) and len(mod.blocks) >= 8:
            # ViT-like の depth (>=8 layers) を持つもの
            candidates_found.append((name, mod, len(mod.blocks)))
    
    if not candidates_found:
        return None, None
    
    # blocks 数が多い (典型的に ViT は 24-32 層) ものを優先
    candidates_found.sort(key=lambda x: -x[2])
    name, vt, n_blocks = candidates_found[0]
    print(f"[INFO] auto-detected vision tower: '{name}' with {n_blocks} blocks")
    return vt, name


def call_vision_tower(vision_tower, pixel_values, image_grid_thw):
    """vision tower を forward する。引数名のバージョン差を吸収。
    Returns: merger 後の token sequence。
    """
    # 候補1: (pixel_values, grid_thw=image_grid_thw)
    # 候補2: (pixel_values, image_grid_thw)
    # 候補3: (pixel_values, image_grid_thw=image_grid_thw)
    last_err = None
    for kwargs in [
        dict(grid_thw=image_grid_thw),
        dict(image_grid_thw=image_grid_thw),
    ]:
        try:
            return vision_tower(pixel_values, **kwargs)
        except (TypeError, RuntimeError) as e:
            last_err = e
    # 位置引数のみ
    try:
        return vision_tower(pixel_values, image_grid_thw)
    except (TypeError, RuntimeError) as e:
        last_err = e
    raise RuntimeError(f"All vision tower forward variants failed. Last error: {last_err}")


# ========== 画像 1 枚から内部表現を抽出 ==========
def extract_representations(model, processor, image_pil, vision_tower, save_input_png_path=None):
    """1 枚の画像から probe 用の表現を 3 種類抽出して返す:
       A_pool: visual.blocks[-1] 出力を mean-pool (1D ベクトル)
       B_pool: merger output を mean-pool (1D ベクトル)
       C_flat: merger output を flatten (1D ベクトル, 空間情報保持)
       
    Qwen2.5-VL の vision tower forward は:
        model.visual(pixel_values, grid_thw) -> merger 後の token sequence
    途中の visual.blocks[-1] 出力は forward hook で取得する。
    """
    # processor で画像をテンソル化
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": "describe."},  # ダミー (vision tower のみ実行)
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_pil], padding=True, return_tensors="pt")
    
    pixel_values = inputs['pixel_values'].to(model.device, dtype=model.dtype)
    image_grid_thw = inputs['image_grid_thw'].to(model.device)
    
    if save_input_png_path is not None:
        recon = reconstruct_qwen25vl_image(inputs['pixel_values'], inputs['image_grid_thw'],
                                            processor.image_processor)
        if recon is not None:
            recon.save(save_input_png_path)
            print(f"  [verify] saved {save_input_png_path}  size={recon.size}  grid_thw={inputs['image_grid_thw'].tolist()}")
    
    # forward hook で blocks[-1] 出力を捕獲
    captured = {}
    
    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            captured['block_last'] = out.detach().clone()
        elif isinstance(out, tuple):
            captured['block_last'] = out[0].detach().clone()
    
    last_block = vision_tower.blocks[-1]
    handle = last_block.register_forward_hook(hook)
    
    try:
        with torch.no_grad():
            visual_output = call_vision_tower(vision_tower, pixel_values, image_grid_thw)
    finally:
        handle.remove()
    
    # 取得できる表現:
    #   - block_last: ViT 最終層出力 (merger 前) → A 用
    #   - merger 後: LM への入力相当 → B/C 用
    #
    # transformers 5.6.2 で `model.model.visual.forward()` は merger を適用した状態で
    # 返すことを前提にしていたが、smoke test で確認した shape 一致から、merger 前の
    # tensor がそのまま返ってきている事が判明した。よって merger を別途適用する。
    
    if isinstance(visual_output, torch.Tensor):
        vision_out_tensor = visual_output
    elif hasattr(visual_output, 'last_hidden_state'):
        vision_out_tensor = visual_output.last_hidden_state
    elif isinstance(visual_output, (tuple, list)):
        vision_out_tensor = visual_output[0]
    else:
        for attr in ('hidden_states', 'pooler_output', 'image_embeds', 'features'):
            if hasattr(visual_output, attr):
                cand = getattr(visual_output, attr)
                if isinstance(cand, torch.Tensor):
                    vision_out_tensor = cand
                    break
        else:
            raise RuntimeError(
                f"Could not extract tensor from vision_tower output of type "
                f"{type(visual_output).__name__}. Attributes: {dir(visual_output)}"
            )
    
    block_last = captured['block_last']  # (n_patches, hidden_dim_visual)
    
    # block_last から余分な batch 次元があれば落とす
    if block_last.dim() == 3 and block_last.shape[0] == 1:
        block_last = block_last.squeeze(0)
    if vision_out_tensor.dim() == 3 and vision_out_tensor.shape[0] == 1:
        vision_out_tensor = vision_out_tensor.squeeze(0)
    
    # merger を別途適用して "LM への入力" 表現を取得
    # vision_tower.merger は (n_patches, hidden_visual) -> (n_patches // sms**2, hidden_lm)
    merger_module = getattr(vision_tower, 'merger', None)
    if merger_module is None:
        raise RuntimeError("vision_tower.merger not found; cannot compute LM-input representation")
    with torch.no_grad():
        merger_out = merger_module(vision_out_tensor)
    if merger_out.dim() == 3 and merger_out.shape[0] == 1:
        merger_out = merger_out.squeeze(0)
    
    # mean pool
    A_pool = block_last.mean(dim=0).to(torch.float32).cpu().numpy()       # ViT final layer
    B_pool = merger_out.mean(dim=0).to(torch.float32).cpu().numpy()        # merger 後 (LM入力)
    # C_flat: merger 後をそのまま flatten (空間情報保持)
    C_flat = merger_out.to(torch.float32).cpu().numpy().flatten()
    
    grid_thw = inputs['image_grid_thw'][0].tolist()
    
    return {
        'A_pool': A_pool,
        'B_pool': B_pool,
        'C_flat': C_flat,
        'block_last_shape': list(block_last.shape),
        'merger_out_shape': list(merger_out.shape),
        'grid_thw': grid_thw,
    }


# ========== 全画像の表現を抽出 ==========
def extract_all_representations(model, processor, vision_tower, metadata, smoke_test=False):
    """全画像 (smoke_test なら最初の 5 枚のみ) から表現を抽出。
    結果を numpy 配列として返す。"""
    if smoke_test:
        metadata = metadata[:5]
    
    print(f"\nExtracting representations from {len(metadata)} stimuli...")
    
    A_list, B_list, C_list = [], [], []
    grid_thw_list = []
    angle_list, condition_list, target_list, dotnum_list = [], [], [], []
    filenames = []
    c_flat_lengths = set()
    
    for i, meta in enumerate(tqdm(metadata)):
        img_path = STIMULI_DIR / meta['filename']
        if not img_path.is_file():
            print(f"  [WARN] missing: {img_path}, skipping")
            continue
        img = Image.open(img_path).convert("RGB")
        
        save_path = OUT_DIR / "input_to_model_full.png" if i == 0 else None
        rep = extract_representations(model, processor, img, vision_tower,
                                       save_input_png_path=save_path)
        
        if i == 0:
            print(f"  block_last shape (ViT final, A):    {rep['block_last_shape']}")
            print(f"  merger_out shape (after merger, B): {rep['merger_out_shape']}")
            print(f"  A_pool dim:        {len(rep['A_pool'])}")
            print(f"  B_pool dim:        {len(rep['B_pool'])}")
            print(f"  C_flat dim:        {len(rep['C_flat'])}")
            print(f"  grid_thw:          {rep['grid_thw']}")
            # 期待される関係を確認
            n_patch_vit = rep['block_last_shape'][0]
            n_token_lm  = rep['merger_out_shape'][0]
            if n_patch_vit > 0 and n_token_lm > 0:
                ratio = n_patch_vit / n_token_lm
                print(f"  patch reduction ratio: {n_patch_vit} -> {n_token_lm}  ({ratio:.1f}x, expect 4x for spatial_merge_size=2)")
        
        A_list.append(rep['A_pool'])
        B_list.append(rep['B_pool'])
        C_list.append(rep['C_flat'])
        grid_thw_list.append(rep['grid_thw'])
        c_flat_lengths.add(len(rep['C_flat']))
        
        angle_list.append(meta['angle_deg'])
        condition_list.append(meta['condition'])
        target_list.append(meta['target_y_right'])
        dotnum_list.append(meta['correct_dot_number'])
        filenames.append(meta['filename'])
    
    # C_flat は画像サイズが共通であれば同じ長さになるはず
    if len(c_flat_lengths) > 1:
        print(f"  [WARN] C_flat lengths vary across stimuli: {c_flat_lengths}")
        print(f"  C.* probes will be skipped (variable feature dim)")
        C_arr = None
    else:
        C_arr = np.stack(C_list, axis=0)
    
    return {
        'A': np.stack(A_list, axis=0),
        'B': np.stack(B_list, axis=0),
        'C': C_arr,
        'angle': np.array(angle_list),
        'condition': np.array(condition_list),
        'target': np.array(target_list),
        'dot_number': np.array(dotnum_list),
        'filenames': filenames,
        'grid_thw_list': grid_thw_list,
    }


# ========== Ridge + LOO probe ==========
def ridge_loo_probe(X, y, alpha=RIDGE_ALPHA):
    """Ridge regression with leave-one-out CV.
    Returns: dict with R2, MAE, predictions, residuals."""
    if len(y) < 3:
        return None  # too few samples
    
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=np.float64)
    
    for train_idx, test_idx in loo.split(X):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X[train_idx], y[train_idx])
        preds[test_idx] = ridge.predict(X[test_idx])
    
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    return {
        'R2': float(r2),
        'MAE': float(mae),
        'predictions': preds.tolist(),
        'targets': y.tolist(),
        'n_samples': len(y),
    }


# ========== probe を全条件で実行 ==========
def run_all_probes(reps, smoke_test=False):
    """5 種類の probe (A.angle, B.angle, A.spatial, B.spatial, C.spatial) を
    全データ・条件別の各設定で実行。"""
    angle = reps['angle']
    target = reps['target']
    dotnum = reps['dot_number']
    condition = reps['condition']
    
    probe_specs = [
        ('A.angle',     reps['A'], angle,  'angle_deg',          'visual.blocks[-1] mean-pool'),
        ('B.angle',     reps['B'], angle,  'angle_deg',          'merger output mean-pool'),
        ('A.spatial_y', reps['A'], target, 'target_y_right',     'visual.blocks[-1] mean-pool'),
        ('B.spatial_y', reps['B'], target, 'target_y_right',     'merger output mean-pool'),
        ('A.dot_num',   reps['A'], dotnum, 'correct_dot_number', 'visual.blocks[-1] mean-pool'),
        ('B.dot_num',   reps['B'], dotnum, 'correct_dot_number', 'merger output mean-pool'),
    ]
    if reps['C'] is not None:
        probe_specs.append(
            ('C.spatial_y', reps['C'], target, 'target_y_right',     'merger output flattened (spatial)')
        )
        probe_specs.append(
            ('C.dot_num',   reps['C'], dotnum, 'correct_dot_number', 'merger output flattened (spatial)')
        )
    
    if smoke_test:
        # smoke test ではデータ数も probe 数も少ないので A.angle のみ
        probe_specs = [probe_specs[0]]
    
    results = []
    
    for probe_id, X, y, target_name, repr_desc in probe_specs:
        # 全データ
        full = ridge_loo_probe(X, y)
        if full is not None:
            results.append({
                'probe_id': probe_id,
                'condition': 'all',
                'target': target_name,
                'representation': repr_desc,
                'n_samples': full['n_samples'],
                'R2': full['R2'],
                'MAE': full['MAE'],
                'predictions': full['predictions'],
                'targets': full['targets'],
            })
            print(f"  {probe_id:12s} all       n={full['n_samples']:3d}  R²={full['R2']:+.4f}  MAE={full['MAE']:.3f}")
        
        # 条件別
        if not smoke_test:
            for cond in CONDITIONS:
                mask = condition == cond
                if mask.sum() < 3:
                    continue
                Xc = X[mask]
                yc = y[mask]
                cond_result = ridge_loo_probe(Xc, yc)
                if cond_result is not None:
                    results.append({
                        'probe_id': probe_id,
                        'condition': cond,
                        'target': target_name,
                        'representation': repr_desc,
                        'n_samples': cond_result['n_samples'],
                        'R2': cond_result['R2'],
                        'MAE': cond_result['MAE'],
                        'predictions': cond_result['predictions'],
                        'targets': cond_result['targets'],
                    })
                    print(f"  {probe_id:12s} {cond:11s} n={cond_result['n_samples']:3d}  R²={cond_result['R2']:+.4f}  MAE={cond_result['MAE']:.3f}")
    
    return results


# ========== 図 1: 予測 vs 真値 (角度 probe) ==========
def plot_angle_probe(results, out_path):
    angle_results = [r for r in results if 'angle' in r['probe_id'] and r['condition'] == 'all']
    if not angle_results:
        return
    
    fig, axes = plt.subplots(1, len(angle_results), figsize=(5.5 * len(angle_results), 5),
                              squeeze=False)
    axes = axes[0]
    
    for ax, res in zip(axes, angle_results):
        preds = np.array(res['predictions'])
        targets = np.array(res['targets'])
        ax.scatter(targets, preds, alpha=0.7, s=40, color='#2E5C8A', edgecolor='white', linewidth=0.5)
        # y=x 線
        lims = [min(targets.min(), preds.min()) - 5,
                max(targets.max(), preds.max()) + 5]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('True angle (°)')
        ax.set_ylabel('Predicted angle (°)')
        ax.set_title(f"{res['probe_id']}\nR²={res['R2']:.3f}, MAE={res['MAE']:.2f}°")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Angle probe (Qwen2.5-VL-32B vision encoder, all 4 conditions, n={angle_results[0]["n_samples"]})',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


# ========== 図 2: 角度誤差プロット (oblique effect 検出) ==========
def plot_angle_error(results, out_path):
    angle_results = [r for r in results if 'angle' in r['probe_id'] and r['condition'] == 'all']
    if not angle_results:
        return
    
    fig, axes = plt.subplots(1, len(angle_results), figsize=(5.5 * len(angle_results), 4.5),
                              squeeze=False)
    axes = axes[0]
    
    for ax, res in zip(axes, angle_results):
        preds = np.array(res['predictions'])
        targets = np.array(res['targets'])
        errors = preds - targets  # signed error
        abs_errors = np.abs(errors)
        
        ax.scatter(targets, errors, alpha=0.6, s=35, color='#2E5C8A',
                   edgecolor='white', linewidth=0.5, label='Signed error')
        ax.axhline(0, color='black', linewidth=0.6, linestyle='-', alpha=0.4)
        ax.axvline(45, color='#A02020', linewidth=0.8, linestyle=':', alpha=0.6,
                   label='45° (oblique)')
        ax.axvline(0, color='#A0A0A0', linewidth=0.6, linestyle=':', alpha=0.5)
        ax.axvline(90, color='#A0A0A0', linewidth=0.6, linestyle=':', alpha=0.5)
        ax.set_xlabel('True angle (°)')
        ax.set_ylabel('Prediction error (pred − true, °)')
        ax.set_title(f"{res['probe_id']} error pattern\nMAE={res['MAE']:.2f}°")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle('Prediction error vs true angle (check for oblique effect at 45°)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


# ========== 図 3: 空間 probe (target_y_right と correct_dot_number) ==========
def plot_spatial_probe(results, out_path):
    """空間 probe の予測 vs 真値プロット。
    target_y_right (連続値, px) と correct_dot_number (1-19) を別々の行で表示。"""
    spatial_y_results = [r for r in results if r['target'] == 'target_y_right' and r['condition'] == 'all']
    dotnum_results    = [r for r in results if r['target'] == 'correct_dot_number' and r['condition'] == 'all']
    
    if not spatial_y_results and not dotnum_results:
        return
    
    n_cols = max(len(spatial_y_results), len(dotnum_results))
    n_rows = (1 if spatial_y_results else 0) + (1 if dotnum_results else 0)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.8 * n_rows),
                              squeeze=False)
    
    row_idx = 0
    for row_results, ylabel, unit in [
        (spatial_y_results, 'target_y_right', 'px'),
        (dotnum_results, 'correct_dot_number', ''),
    ]:
        if not row_results:
            continue
        for col_idx, res in enumerate(row_results):
            ax = axes[row_idx, col_idx]
            preds = np.array(res['predictions'])
            targets = np.array(res['targets'])
            ax.scatter(targets, preds, alpha=0.7, s=40, color='#2E8B57',
                       edgecolor='white', linewidth=0.5)
            lims = [min(targets.min(), preds.min()),
                    max(targets.max(), preds.max())]
            margin = 0.05 * (lims[1] - lims[0]) if lims[1] != lims[0] else 1
            lims = [lims[0] - margin, lims[1] + margin]
            ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(f'True {ylabel}{(" ("+unit+")") if unit else ""}')
            ax.set_ylabel(f'Predicted {ylabel}{(" ("+unit+")") if unit else ""}')
            mae_unit = f"{res['MAE']:.1f}{(' '+unit) if unit else ''}"
            ax.set_title(f"{res['probe_id']}\nR²={res['R2']:.3f}, MAE={mae_unit}")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3)
        # 余りの軸を隠す
        for col_idx in range(len(row_results), n_cols):
            axes[row_idx, col_idx].axis('off')
        row_idx += 1
    
    fig.suptitle(f'Spatial probes (target_y_right and correct_dot_number)',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


# ========== 図 4: R²/MAE バープロット (probe × 条件) ==========
def plot_summary_bars(results, out_path):
    angle_results = [r for r in results if r['target'] == 'angle_deg']
    spatial_y_results = [r for r in results if r['target'] == 'target_y_right']
    dotnum_results = [r for r in results if r['target'] == 'correct_dot_number']
    
    n_rows = sum(1 for x in [angle_results, spatial_y_results, dotnum_results] if x)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4 * n_rows), squeeze=False)
    
    def plot_grouped_bars(ax, results_subset, metric_key, ylabel, title):
        if not results_subset:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center')
            return
        probe_ids = sorted(set(r['probe_id'] for r in results_subset))
        conditions = ['all'] + CONDITIONS
        n_probes = len(probe_ids)
        n_conds = len(conditions)
        x = np.arange(n_probes)
        width = 0.8 / n_conds
        cmap = plt.get_cmap('tab10')
        
        for i, cond in enumerate(conditions):
            vals = []
            for pid in probe_ids:
                rs = [r for r in results_subset if r['probe_id'] == pid and r['condition'] == cond]
                vals.append(rs[0][metric_key] if rs else np.nan)
            ax.bar(x + i * width, vals, width, label=cond, color=cmap(i))
        
        ax.set_xticks(x + width * (n_conds - 1) / 2)
        ax.set_xticklabels(probe_ids, rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc='best')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
    
    row = 0
    if angle_results:
        plot_grouped_bars(axes[row, 0], angle_results, 'R2',  'R²',     'Angle probe — R² (higher = better)')
        plot_grouped_bars(axes[row, 1], angle_results, 'MAE', 'MAE (°)', 'Angle probe — MAE (lower = better)')
        row += 1
    if spatial_y_results:
        plot_grouped_bars(axes[row, 0], spatial_y_results, 'R2',  'R²',      'target_y_right probe — R² (higher = better)')
        plot_grouped_bars(axes[row, 1], spatial_y_results, 'MAE', 'MAE (px)', 'target_y_right probe — MAE (lower = better)')
        row += 1
    if dotnum_results:
        plot_grouped_bars(axes[row, 0], dotnum_results, 'R2',  'R²',  'correct_dot_number probe — R² (higher = better)')
        plot_grouped_bars(axes[row, 1], dotnum_results, 'MAE', 'MAE', 'correct_dot_number probe — MAE (lower = better)')
        row += 1
    
    fig.suptitle('Qwen2.5-VL-32B vision encoder probes: R² and MAE',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


# ========== 結果保存 ==========
def save_results(results, reps, header_info, smoke_test=False):
    # CSV (predictions/targets を除く)
    import csv
    csv_path = OUT_DIR / 'probe_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'probe_id', 'condition', 'target', 'representation', 'n_samples', 'R2', 'MAE'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"  saved: {csv_path}")
    
    # JSON (predictions 含むフル)
    json_path = OUT_DIR / 'probe_results.json'
    out = {
        **header_info,
        'results': results,
    }
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  saved: {json_path}")
    
    # 表現キャッシュ (NPZ)
    if not smoke_test:
        npz_path = OUT_DIR / 'representations.npz'
        save_dict = {
            'A': reps['A'],
            'B': reps['B'],
            'angle': reps['angle'],
            'condition': reps['condition'],
            'target': reps['target'],
            'dot_number': reps['dot_number'],
            'filenames': np.array(reps['filenames']),
        }
        if reps['C'] is not None:
            save_dict['C'] = reps['C']
        np.savez(npz_path, **save_dict)
        print(f"  saved: {npz_path}")


# ========== main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run with only 5 stimuli + A.angle probe (verification mode)')
    parser.add_argument('--inspect', action='store_true',
                        help='Load model, print structure (top-level + vision-tower candidates) and exit')
    args = parser.parse_args()
    
    if not METADATA_CSV.is_file():
        print(f"[ERROR] metadata not found: {METADATA_CSV}")
        sys.exit(1)
    
    metadata = load_metadata(METADATA_CSV)
    print(f"loaded metadata: {len(metadata)} entries")
    
    # 整合性確認
    angles = sorted(set(m['angle_deg'] for m in metadata))
    conds = sorted(set(m['condition'] for m in metadata))
    print(f"  angles ({len(angles)}): {angles}")
    print(f"  conditions: {conds}")
    
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
    for _, p in model.named_parameters():
        d = str(p.device)
        param_devices[d] = param_devices.get(d, 0) + 1
    print(f"  param device summary: {param_devices}")
    if any('cpu' in d for d in param_devices):
        print("  [WARN] some parameters on CPU!")
    
    # --- inspect モード ---
    if args.inspect:
        print("\n=== model top-level attributes ===")
        for name in dir(model):
            if name.startswith('_'):
                continue
            try:
                obj = getattr(model, name)
                if isinstance(obj, torch.nn.Module):
                    print(f"  model.{name}  -> {type(obj).__name__}")
            except Exception:
                continue
        if hasattr(model, 'model'):
            print("\n=== model.model (sub-model) top-level attributes ===")
            for name in dir(model.model):
                if name.startswith('_'):
                    continue
                try:
                    obj = getattr(model.model, name)
                    if isinstance(obj, torch.nn.Module):
                        print(f"  model.model.{name}  -> {type(obj).__name__}")
                except Exception:
                    continue
        print("\n=== modules with `blocks` attribute (ViT-like) ===")
        for name, mod in model.named_modules():
            if hasattr(mod, 'blocks') and isinstance(mod.blocks, torch.nn.ModuleList):
                print(f"  {name}  ->  {type(mod).__name__}  (blocks: {len(mod.blocks)})")
        print("\n[inspect] Done. Exiting.")
        return
    
    # vision tower の取得
    vision_tower, vt_path = get_vision_tower(model)
    if vision_tower is None:
        print("[ERROR] could not locate vision tower in the model. Run with --inspect to debug.")
        sys.exit(1)
    print(f"\nvision tower: model.{vt_path}  ({type(vision_tower).__name__}, {len(vision_tower.blocks)} blocks)")
    
    # 表現抽出
    t1 = time.time()
    reps = extract_all_representations(model, processor, vision_tower, metadata, smoke_test=args.smoke_test)
    print(f"\nextraction done in {time.time()-t1:.1f}s")
    print(f"  A shape: {reps['A'].shape}")
    print(f"  B shape: {reps['B'].shape}")
    if reps['C'] is not None:
        print(f"  C shape: {reps['C'].shape}")
    
    # probe 実行
    print(f"\n=== Running probes ===")
    t2 = time.time()
    results = run_all_probes(reps, smoke_test=args.smoke_test)
    print(f"\nprobing done in {time.time()-t2:.1f}s")
    
    # 結果保存
    header_info = {
        'started_at': datetime.now().isoformat(),
        'model_id': MODEL_ID,
        'smoke_test': args.smoke_test,
        'n_stimuli_processed': int(reps['A'].shape[0]),
        'ridge_alpha': RIDGE_ALPHA,
        'A_dim': int(reps['A'].shape[1]),
        'B_dim': int(reps['B'].shape[1]),
        'C_dim': int(reps['C'].shape[1]) if reps['C'] is not None else None,
    }
    save_results(results, reps, header_info, smoke_test=args.smoke_test)
    
    # 図
    print(f"\n=== Generating figures ===")
    plot_angle_probe(results, OUT_DIR / 'fig_angle_probe.png')
    plot_angle_error(results, OUT_DIR / 'fig_angle_error.png')
    if not args.smoke_test:
        plot_spatial_probe(results, OUT_DIR / 'fig_spatial_probe.png')
        plot_summary_bars(results, OUT_DIR / 'fig_summary_bars.png')
    
    print(f"\n=== Done ===")
    print(f"output: {OUT_DIR}")


if __name__ == '__main__':
    main()
