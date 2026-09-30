"""
渡辺錯視 × Vision Encoder 内部解析スクリプト
=============================================
3系統のVision Encoder（CLIP, DINOv2, ViT-ImageNet）に刺激画像を入力し、
方位バイアス（oblique effect）の存在を検証する。

解析内容:
  1. Attention Map の可視化（線分領域への注意分布）
  2. 内部表現のコサイン類似度行列（角度間の弁別精度）
  3. 線形プローブによる角度推定（予測誤差の角度依存性）

使い方:
  # 先に刺激画像を生成しておく
  python generate_stimuli.py

  # 解析実行
  python analyze_vision_encoders.py

  # standard条件のみ解析する場合
  python analyze_vision_encoders.py --conditions standard

  # 特定のモデルのみ
  python analyze_vision_encoders.py --models clip dinov2

出力:
  figures/ ディレクトリに解析結果の図を保存

依存ライブラリ:
  torch, torchvision, transformers, numpy, Pillow, matplotlib, scikit-learn, tqdm

推奨環境:
  NVIDIA GPU（A6000等）、CUDA対応PyTorch
"""

import os
import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut


# =========================================================================
#  1. モデル定義
# =========================================================================

class VisionEncoderBase:
    """Vision Encoder の共通インターフェース"""

    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.model = None
        self.transform = None

    def load(self):
        raise NotImplementedError

    def extract_features(self, img_tensor):
        """
        画像テンソルから特徴ベクトルを抽出する。

        Returns
        -------
        cls_token : torch.Tensor, shape (D,)
            CLSトークンの特徴ベクトル
        patch_tokens : torch.Tensor, shape (N_patches, D)
            パッチトークンの特徴ベクトル
        attn_weights : torch.Tensor, shape (n_heads, N_patches+1, N_patches+1)
            最終層の自己注意重み
        """
        raise NotImplementedError

    def preprocess(self, pil_img):
        """PIL画像をモデル入力テンソルに変換する"""
        return self.transform(pil_img).unsqueeze(0).to(self.device)


class CLIPEncoder(VisionEncoderBase):
    """CLIP ViT-L/14"""

    def __init__(self, device):
        super().__init__('CLIP_ViT-L/14', device)

    def load(self):
        from transformers import CLIPVisionModel
        from torchvision import transforms

        self.model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            attn_implementation="eager",
        ).to(self.device).eval()

        # Resize to exact input size (no CenterCrop) to preserve
        # the full stimulus layout including the circle in the lower-left.
        # The standard CLIP preprocessing uses Resize(224)+CenterCrop(224),
        # but this crops out the lower-left circle from our 800x400 stimuli.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def preprocess(self, pil_img):
        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def extract_features(self, img_tensor):
        with torch.no_grad():
            outputs = self.model(
                pixel_values=img_tensor,
                output_attentions=True,
            )
        cls_token = outputs.last_hidden_state[0, 0]  # CLS token
        patch_tokens = outputs.last_hidden_state[0, 1:]  # patch tokens
        # attentions: tuple of (batch, heads, seq, seq)
        attn_weights = outputs.attentions[-1][0]  # last layer
        return cls_token, patch_tokens, attn_weights


class DINOv2Encoder(VisionEncoderBase):
    """DINOv2 ViT-B/14"""

    def __init__(self, device):
        super().__init__('DINOv2_ViT-B/14', device)

    def load(self):
        self.model = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14',
        ).to(self.device).eval()

        # DINOv2の前処理
        # Resize to exact input size (no CenterCrop) to preserve
        # the full stimulus layout including the circle in the lower-left.
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Attention hookで最終ブロックの入力をキャプチャし、
        # qkvからattention weightを再計算する
        self._attn_weights = None
        last_attn = self.model.blocks[-1].attn

        def attn_hook(module, input, output):
            # input は tuple で、input[0] が (B, N, C) のテンソル
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = (C // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self._attn_weights = attn[0].detach()  # (heads, N, N)

        self._hook = last_attn.register_forward_hook(attn_hook)

    def extract_features(self, img_tensor):
        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
            cls_token = features['x_norm_clstoken'][0]
            patch_tokens = features['x_norm_patchtokens'][0]
        attn_weights = self._attn_weights if self._attn_weights is not None else None
        return cls_token, patch_tokens, attn_weights


class ViTImageNetEncoder(VisionEncoderBase):
    """ViT-B/16 (ImageNet supervised)"""

    def __init__(self, device):
        super().__init__('ViT-B/16_ImageNet', device)

    def load(self):
        from transformers import ViTModel, ViTImageProcessor
        self.processor = ViTImageProcessor.from_pretrained(
            'google/vit-base-patch16-224'
        )
        self.model = ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
            attn_implementation="eager",
        ).to(self.device).eval()

    def preprocess(self, pil_img):
        inputs = self.processor(images=pil_img, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)

    def extract_features(self, img_tensor):
        with torch.no_grad():
            outputs = self.model(
                pixel_values=img_tensor,
                output_attentions=True,
            )
        cls_token = outputs.last_hidden_state[0, 0]
        patch_tokens = outputs.last_hidden_state[0, 1:]
        attn_weights = outputs.attentions[-1][0]  # last layer
        return cls_token, patch_tokens, attn_weights


# =========================================================================
#  2. データ読み込み
# =========================================================================

def load_stimuli(stimuli_dir, conditions=None):
    """
    刺激画像とメタデータを読み込む。

    Parameters
    ----------
    stimuli_dir : str
        stimuli/ ディレクトリのパス
    conditions : list of str or None
        読み込む条件（Noneなら全条件）

    Returns
    -------
    stimuli : list of dict
        各刺激の情報（'image', 'angle_deg', 'condition', ...）
    """
    csv_path = os.path.join(stimuli_dir, 'stimulus_metadata.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metadata CSV not found: {csv_path}\n"
            f"Run 'python generate_stimuli.py' first."
        )

    stimuli = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if conditions and row['condition'] not in conditions:
                continue
            img_path = os.path.join(stimuli_dir, row['filename'])
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found, skipping")
                continue
            pil_img = Image.open(img_path).convert('RGB')
            stimuli.append({
                'image': pil_img,
                'angle_deg': float(row['angle_deg']),
                'condition': row['condition'],
                'hit_edge': row['hit_edge'],
                'correct_dot_label': row['correct_dot_label'],
                'correct_dot_number': int(row['correct_dot_number']),
                'filename': row['filename'],
            })

    print(f"Loaded {len(stimuli)} stimuli from {stimuli_dir}")
    return stimuli


# =========================================================================
#  3. 特徴抽出
# =========================================================================

def extract_all_features(encoder, stimuli):
    """
    全刺激画像からエンコーダの特徴を抽出する。

    Returns
    -------
    results : list of dict
        各刺激の特徴情報
    """
    results = []
    for stim in tqdm(stimuli, desc=f"  {encoder.name}"):
        img_tensor = encoder.preprocess(stim['image'])
        cls_token, patch_tokens, attn_weights = encoder.extract_features(img_tensor)

        results.append({
            'angle_deg': stim['angle_deg'],
            'condition': stim['condition'],
            'filename': stim['filename'],
            'cls_token': cls_token.cpu(),
            'patch_tokens': patch_tokens.cpu(),
            'attn_weights': attn_weights.cpu() if attn_weights is not None else None,
        })

    return results


# =========================================================================
#  4. 解析 1: Attention Map 可視化
# =========================================================================

def analyze_attention_maps(results, encoder_name, output_dir, condition='standard'):
    """
    CLS トークンから各パッチへの注意重みを可視化する。
    """
    fig_dir = os.path.join(output_dir, 'attention_maps')
    os.makedirs(fig_dir, exist_ok=True)

    # standard条件だけフィルタ
    cond_results = [r for r in results if r['condition'] == condition]
    if not cond_results:
        print(f"  No results for condition '{condition}', skipping attention maps")
        return

    n_angles = len(cond_results)
    cols = 6
    rows = (n_angles + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).flatten()

    for i, r in enumerate(cond_results):
        ax = axes[i]
        attn = r['attn_weights']  # (heads, seq, seq)

        if attn is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
            ax.set_title(f"{r['angle_deg']:.0f}°")
            continue

        # CLS→パッチ の注意を全ヘッドで平均
        # attn shape: (heads, seq_len, seq_len)
        # CLS は index 0
        cls_to_patch = attn[:, 0, 1:].mean(dim=0)  # (n_patches,)

        # パッチ数から空間サイズを推定
        n_patches = cls_to_patch.shape[0]
        h = w = int(np.sqrt(n_patches))
        if h * w != n_patches:
            # 非正方形の場合（CLIP ViT-L/14: 16x16=256）
            # DINOv2: 37x37=1369 for 518x518 input
            h = w = int(np.round(np.sqrt(n_patches)))
            # 端数は切り捨て
            cls_to_patch = cls_to_patch[:h * w]

        attn_map = cls_to_patch.reshape(h, w).numpy()
        ax.imshow(attn_map, cmap='hot', interpolation='bilinear')
        ax.set_title(f"{r['angle_deg']:.0f}°", fontsize=10)
        ax.axis('off')

    # 余ったaxesを非表示
    for i in range(n_angles, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"Attention Maps (CLS→patches) — {encoder_name}", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, f'attention_{encoder_name.replace("/", "_")}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =========================================================================
#  5. 解析 2: コサイン類似度行列
# =========================================================================

def analyze_cosine_similarity(results, encoder_name, output_dir, condition='standard'):
    """
    異なる角度間のCLSトークンのコサイン類似度行列を計算し、可視化する。
    方位バイアスがあれば、cardinal方位（0°/90°付近）で類似度が高く、
    oblique方位（45°付近）で弁別精度が低下する。
    """
    fig_dir = os.path.join(output_dir, 'cosine_similarity')
    os.makedirs(fig_dir, exist_ok=True)

    cond_results = [r for r in results if r['condition'] == condition]
    if not cond_results:
        return

    angles = [r['angle_deg'] for r in cond_results]
    cls_features = torch.stack([r['cls_token'] for r in cond_results])

    # 正規化
    cls_normed = F.normalize(cls_features, dim=1)
    sim_matrix = (cls_normed @ cls_normed.T).numpy()

    # --- 類似度行列のヒートマップ ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=sim_matrix.min(), vmax=1.0)
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels([f"{a:.0f}°" for a in angles], rotation=45, fontsize=8)
    ax.set_yticks(range(len(angles)))
    ax.set_yticklabels([f"{a:.0f}°" for a in angles], fontsize=8)
    ax.set_xlabel("Angle")
    ax.set_ylabel("Angle")
    ax.set_title(f"Cosine Similarity (CLS tokens) — {encoder_name}")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, f'cosine_sim_{encoder_name.replace("/", "_")}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # --- 隣接角度間の類似度（弁別精度の指標）---
    neighbor_sims = []
    for i in range(len(angles) - 1):
        neighbor_sims.append(sim_matrix[i, i + 1])

    mid_angles = [(angles[i] + angles[i + 1]) / 2 for i in range(len(angles) - 1)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(mid_angles, neighbor_sims, 'o-', color='steelblue', linewidth=2)
    ax.set_xlabel("Angle (midpoint between neighbors)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Adjacent Angle Similarity — {encoder_name}\n"
                 f"(higher = harder to discriminate)")
    ax.axvline(45, color='gray', linestyle='--', alpha=0.5, label='45° (max oblique)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, f'neighbor_sim_{encoder_name.replace("/", "_")}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =========================================================================
#  6. 解析 3: 線形プローブ（角度推定）
# =========================================================================

def analyze_linear_probe(results, encoder_name, output_dir, condition='standard'):
    """
    CLSトークンの特徴ベクトルから線形回帰で角度を予測し、
    予測誤差の角度依存性を調べる。

    Leave-one-out交差検証を使用（サンプル数が少ないため）。
    """
    fig_dir = os.path.join(output_dir, 'linear_probe')
    os.makedirs(fig_dir, exist_ok=True)

    cond_results = [r for r in results if r['condition'] == condition]
    if not cond_results:
        return

    angles = np.array([r['angle_deg'] for r in cond_results])
    features = torch.stack([r['cls_token'] for r in cond_results]).numpy()

    # Leave-one-out 交差検証
    loo = LeaveOneOut()
    predictions = np.zeros(len(angles))
    errors = np.zeros(len(angles))

    for train_idx, test_idx in loo.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = angles[train_idx], angles[test_idx]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        predictions[test_idx[0]] = pred
        errors[test_idx[0]] = pred - y_test[0]

    # --- 予測 vs 正解 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(angles, predictions, c='steelblue', s=60, zorder=3)
    ax.plot([0, 90], [0, 90], 'k--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel("True angle (°)")
    ax.set_ylabel("Predicted angle (°)")
    ax.set_title(f"Linear Probe: Predicted vs True — {encoder_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)

    # --- 誤差の角度依存性 ---
    ax = axes[1]
    ax.bar(angles, np.abs(errors), width=4, color='steelblue', alpha=0.7)
    ax.set_xlabel("True angle (°)")
    ax.set_ylabel("|Prediction error| (°)")
    ax.set_title(f"Prediction Error by Angle — {encoder_name}\n"
                 f"(oblique effect → larger error near 45°)")
    ax.axvline(45, color='red', linestyle='--', alpha=0.5, label='45° (max oblique)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, f'linear_probe_{encoder_name.replace("/", "_")}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # 数値結果をCSVに保存
    csv_path = os.path.join(fig_dir, f'linear_probe_{encoder_name.replace("/", "_")}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['angle_deg', 'predicted_deg', 'error_deg', 'abs_error_deg'])
        for a, p, e in zip(angles, predictions, errors):
            writer.writerow([a, round(p, 2), round(e, 2), round(abs(e), 2)])
    print(f"  Saved: {csv_path}")

    return {
        'angles': angles,
        'predictions': predictions,
        'errors': errors,
    }


# =========================================================================
#  7. 条件間比較（円あり vs 円なし etc.）
# =========================================================================

def get_patch_grid_size(n_patches):
    """パッチ数から空間グリッドサイズを推定する"""
    h = w = int(np.round(np.sqrt(n_patches)))
    # 完全な正方形でない場合の調整
    while h * w < n_patches:
        w += 1
    return h, w


def extract_local_patch_features(results, condition='standard'):
    """
    線分がある左下領域のパッチトークンを抽出し、
    平均プーリングで1つの特徴ベクトルにまとめる。

    刺激画像の構図:
      - 左下に円と線分がある
      - パッチグリッドの左下 1/4 領域を「線分領域」として使用

    Returns
    -------
    angles : list of float
    local_features : torch.Tensor, shape (n_stimuli, D)
    """
    cond_results = [r for r in results if r['condition'] == condition]
    if not cond_results:
        return None, None

    angles = [r['angle_deg'] for r in cond_results]
    local_features = []

    for r in cond_results:
        patch_tokens = r['patch_tokens']  # (n_patches, D)
        n_patches = patch_tokens.shape[0]
        h, w = get_patch_grid_size(n_patches)

        # パッチをグリッドに並べる
        # 使えるパッチ数に合わせてトリミング
        usable = h * w
        patches_grid = patch_tokens[:usable].reshape(h, w, -1)

        # 左下 1/4 領域（下半分 × 左半分）
        h_mid = h // 2
        w_mid = w // 2
        local_patches = patches_grid[h_mid:, :w_mid, :]  # (h/2, w/2, D)

        # 平均プーリング
        local_feat = local_patches.reshape(-1, local_patches.shape[-1]).mean(dim=0)
        local_features.append(local_feat)

    return angles, torch.stack(local_features)


def analyze_patch_cosine_similarity(results, encoder_name, output_dir, condition='standard'):
    """
    左下領域のパッチトークンを使ったコサイン類似度解析。
    CLSトークンで見えなかった方位情報がパッチレベルに残っているか検証する。
    """
    fig_dir = os.path.join(output_dir, 'patch_cosine_similarity')
    os.makedirs(fig_dir, exist_ok=True)

    angles, local_features = extract_local_patch_features(results, condition)
    if angles is None:
        return

    # 正規化してコサイン類似度
    feat_normed = F.normalize(local_features, dim=1)
    sim_matrix = (feat_normed @ feat_normed.T).numpy()

    # --- 類似度行列のヒートマップ ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=sim_matrix.min(), vmax=1.0)
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels([f"{a:.0f}°" for a in angles], rotation=45, fontsize=8)
    ax.set_yticks(range(len(angles)))
    ax.set_yticklabels([f"{a:.0f}°" for a in angles], fontsize=8)
    ax.set_xlabel("Angle")
    ax.set_ylabel("Angle")
    ax.set_title(f"Cosine Similarity (local patch tokens) — {encoder_name}")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, f'patch_cosine_sim_{encoder_name.replace("/", "_")}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # --- 隣接角度間の類似度 ---
    neighbor_sims = [sim_matrix[i, i + 1] for i in range(len(angles) - 1)]
    mid_angles = [(angles[i] + angles[i + 1]) / 2 for i in range(len(angles) - 1)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(mid_angles, neighbor_sims, 'o-', color='steelblue', linewidth=2)
    ax.set_xlabel("Angle (midpoint between neighbors)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Adjacent Angle Similarity (local patches) — {encoder_name}\n"
                 f"(higher = harder to discriminate)")
    ax.axvline(45, color='gray', linestyle='--', alpha=0.5, label='45° (max oblique)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, f'patch_neighbor_sim_{encoder_name.replace("/", "_")}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def analyze_patch_linear_probe(results, encoder_name, output_dir, condition='standard'):
    """
    左下領域のパッチトークンから線形回帰で角度を予測する。
    CLSトークンで予測できなかったモデルでも、
    パッチレベルでは方位情報が残っているかを検証する。
    """
    fig_dir = os.path.join(output_dir, 'patch_linear_probe')
    os.makedirs(fig_dir, exist_ok=True)

    angles_list, local_features = extract_local_patch_features(results, condition)
    if angles_list is None:
        return None

    angles = np.array(angles_list)
    features = local_features.numpy()

    # Leave-one-out 交差検証
    loo = LeaveOneOut()
    predictions = np.zeros(len(angles))
    errors = np.zeros(len(angles))

    for train_idx, test_idx in loo.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = angles[train_idx], angles[test_idx]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        predictions[test_idx[0]] = pred
        errors[test_idx[0]] = pred - y_test[0]

    # --- 予測 vs 正解 ---
    fig, axes_fig = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes_fig[0]
    ax.scatter(angles, predictions, c='steelblue', s=60, zorder=3)
    ax.plot([0, 90], [0, 90], 'k--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel("True angle (°)")
    ax.set_ylabel("Predicted angle (°)")
    ax.set_title(f"Patch Linear Probe: Predicted vs True — {encoder_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)

    ax = axes_fig[1]
    ax.bar(angles, np.abs(errors), width=4, color='steelblue', alpha=0.7)
    ax.set_xlabel("True angle (°)")
    ax.set_ylabel("|Prediction error| (°)")
    ax.set_title(f"Patch Prediction Error — {encoder_name}")
    ax.axvline(45, color='red', linestyle='--', alpha=0.5, label='45° (max oblique)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, f'patch_probe_{encoder_name.replace("/", "_")}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # CSV保存
    csv_path = os.path.join(fig_dir, f'patch_probe_{encoder_name.replace("/", "_")}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['angle_deg', 'predicted_deg', 'error_deg', 'abs_error_deg'])
        for a, p, e in zip(angles, predictions, errors):
            writer.writerow([a, round(p, 2), round(e, 2), round(abs(e), 2)])
    print(f"  Saved: {csv_path}")

    return {
        'angles': angles,
        'predictions': predictions,
        'errors': errors,
    }


def plot_cross_model_patch_summary(all_patch_probe_results, output_dir):
    """
    全モデルのパッチトークン線形プローブ結果を1つの図にまとめる。
    """
    if not all_patch_probe_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['steelblue', 'coral', 'seagreen']
    markers = ['o', '^', 's']

    n_models = len(all_patch_probe_results)
    x_offsets = np.linspace(-1.0, 1.0, n_models)

    # --- 予測 vs 正解 ---
    ax = axes[0]
    for i, (model_name, result) in enumerate(all_patch_probe_results.items()):
        ax.scatter(result['angles'] + x_offsets[i], result['predictions'],
                   c=colors[i % len(colors)], s=50, alpha=0.8,
                   marker=markers[i % len(markers)], label=model_name,
                   edgecolors='white', linewidths=0.5, zorder=3)
    ax.plot([0, 90], [0, 90], 'k--', alpha=0.5)
    ax.set_xlabel("True angle (°)")
    ax.set_ylabel("Predicted angle (°)")
    ax.set_title("Patch Linear Probe: All Models")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)

    # --- 絶対誤差 ---
    ax = axes[1]
    bar_width = 1.2
    offsets = np.linspace(-bar_width, bar_width, n_models)
    for i, (model_name, result) in enumerate(all_patch_probe_results.items()):
        ax.bar(result['angles'] + offsets[i], np.abs(result['errors']),
               width=bar_width, color=colors[i % len(colors)],
               alpha=0.7, label=model_name)
    ax.set_xlabel("True angle (°)")
    ax.set_ylabel("|Prediction error| (°)")
    ax.set_title("Patch Prediction Error: All Models")
    ax.axvline(45, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'cross_model_patch_summary.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def analyze_condition_comparison(all_results, encoder_name, output_dir):
    """
    条件間（standard, no_circle, short_line, long_line）で
    CLS特徴ベクトルのコサイン類似度を比較する。
    """
    fig_dir = os.path.join(output_dir, 'condition_comparison')
    os.makedirs(fig_dir, exist_ok=True)

    conditions = sorted(set(r['condition'] for r in all_results))
    if len(conditions) < 2:
        print(f"  Only one condition, skipping condition comparison")
        return

    angles = sorted(set(r['angle_deg'] for r in all_results))

    # 各条件・各角度のCLSトークンを集める
    cond_features = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r['condition'] == cond]
        cond_results.sort(key=lambda r: r['angle_deg'])
        cond_features[cond] = {
            r['angle_deg']: r['cls_token'] for r in cond_results
        }

    # standard vs 各条件のコサイン類似度
    if 'standard' not in cond_features:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = {'no_circle': 'red', 'short_line': 'green', 'long_line': 'purple'}

    for cond in conditions:
        if cond == 'standard':
            continue
        sims = []
        valid_angles = []
        for angle in angles:
            if angle in cond_features['standard'] and angle in cond_features[cond]:
                f1 = cond_features['standard'][angle]
                f2 = cond_features[cond][angle]
                sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
                sims.append(sim)
                valid_angles.append(angle)

        color = colors.get(cond, 'gray')
        ax.plot(valid_angles, sims, 'o-', color=color, label=f'standard vs {cond}',
                linewidth=2, markersize=6)

    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Cosine similarity (standard vs condition)")
    ax.set_title(f"Condition Comparison — {encoder_name}\n"
                 f"(lower similarity = larger effect of the manipulation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, f'condition_cmp_{encoder_name.replace("/", "_")}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =========================================================================
#  8. モデル間比較サマリ
# =========================================================================

def plot_cross_model_summary(all_probe_results, output_dir):
    """
    全モデルの線形プローブ結果を1つの図にまとめる。
    """
    if not all_probe_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['steelblue', 'coral', 'seagreen']
    markers = ['o', '^', 's']  # 丸、三角、四角

    # モデル数に応じたx軸オフセット（±1°程度）
    n_models = len(all_probe_results)
    x_offsets = np.linspace(-1.0, 1.0, n_models)

    # --- 予測 vs 正解 ---
    ax = axes[0]
    for i, (model_name, result) in enumerate(all_probe_results.items()):
        ax.scatter(result['angles'] + x_offsets[i], result['predictions'],
                   c=colors[i % len(colors)], s=50, alpha=0.8,
                   marker=markers[i % len(markers)], label=model_name,
                   edgecolors='white', linewidths=0.5, zorder=3)
    ax.plot([0, 90], [0, 90], 'k--', alpha=0.5)
    ax.set_xlabel("True angle (°)")
    ax.set_ylabel("Predicted angle (°)")
    ax.set_title("Linear Probe: All Models")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)

    # --- 絶対誤差 ---
    ax = axes[1]
    bar_width = 1.2
    offsets = np.linspace(-bar_width, bar_width, n_models)
    for i, (model_name, result) in enumerate(all_probe_results.items()):
        ax.bar(result['angles'] + offsets[i], np.abs(result['errors']),
               width=bar_width, color=colors[i % len(colors)],
               alpha=0.7, label=model_name)
    ax.set_xlabel("True angle (°)")
    ax.set_ylabel("|Prediction error| (°)")
    ax.set_title("Prediction Error: All Models")
    ax.axvline(45, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'cross_model_summary.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================================
#  9. メイン
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Watanabe Illusion × Vision Encoder Analysis'
    )
    parser.add_argument(
        '--stimuli_dir', type=str, default='stimuli',
        help='Path to stimuli directory (default: stimuli/)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='figures',
        help='Path to output directory (default: figures/)',
    )
    parser.add_argument(
        '--models', type=str, nargs='+',
        default=['clip', 'dinov2', 'vit_imagenet'],
        choices=['clip', 'dinov2', 'vit_imagenet'],
        help='Models to analyze (default: all three)',
    )
    parser.add_argument(
        '--conditions', type=str, nargs='+', default=None,
        help='Conditions to analyze (default: all)',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 刺激画像読み込み
    stimuli = load_stimuli(args.stimuli_dir, conditions=args.conditions)
    if not stimuli:
        print("No stimuli found. Exiting.")
        return

    # エンコーダの定義
    encoder_map = {
        'clip': CLIPEncoder,
        'dinov2': DINOv2Encoder,
        'vit_imagenet': ViTImageNetEncoder,
    }

    all_probe_results = {}
    all_patch_probe_results = {}

    for model_key in args.models:
        print(f"\n{'='*60}")
        encoder = encoder_map[model_key](device)
        print(f"Loading {encoder.name}...")
        try:
            encoder.load()
        except Exception as e:
            print(f"  Failed to load {encoder.name}: {e}")
            print(f"  Skipping...")
            continue
        print(f"  Loaded successfully")

        # 特徴抽出
        print(f"Extracting features...")
        all_results = extract_all_features(encoder, stimuli)

        # 解析1: Attention Map
        print(f"Analyzing attention maps...")
        analyze_attention_maps(all_results, encoder.name, args.output_dir)

        # 解析2: コサイン類似度（CLSトークン）
        print(f"Analyzing cosine similarity (CLS)...")
        analyze_cosine_similarity(all_results, encoder.name, args.output_dir)

        # 解析3: 線形プローブ（CLSトークン）
        print(f"Analyzing linear probe (CLS)...")
        probe_result = analyze_linear_probe(all_results, encoder.name, args.output_dir)
        if probe_result:
            all_probe_results[encoder.name] = probe_result

        # 解析4: コサイン類似度（パッチトークン）
        print(f"Analyzing cosine similarity (local patches)...")
        analyze_patch_cosine_similarity(all_results, encoder.name, args.output_dir)

        # 解析5: 線形プローブ（パッチトークン）
        print(f"Analyzing linear probe (local patches)...")
        patch_probe_result = analyze_patch_linear_probe(
            all_results, encoder.name, args.output_dir
        )
        if patch_probe_result:
            all_patch_probe_results[encoder.name] = patch_probe_result

        # 解析6: 条件間比較
        conditions_present = set(r['condition'] for r in all_results)
        if len(conditions_present) > 1:
            print(f"Analyzing condition comparison...")
            analyze_condition_comparison(all_results, encoder.name, args.output_dir)

        # メモリ解放
        del encoder.model
        del all_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # モデル間比較サマリ（CLSトークン）
    if len(all_probe_results) > 1:
        print(f"\nGenerating cross-model summary (CLS)...")
        plot_cross_model_summary(all_probe_results, args.output_dir)

    # モデル間比較サマリ（パッチトークン）
    if len(all_patch_probe_results) > 1:
        print(f"Generating cross-model summary (patches)...")
        plot_cross_model_patch_summary(all_patch_probe_results, args.output_dir)

    print(f"\nAll analyses complete. Results saved to '{args.output_dir}/'")


if __name__ == '__main__':
    main()
