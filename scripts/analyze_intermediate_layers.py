"""
渡辺錯視 × Vision Encoder 中間層解析スクリプト
================================================
3系統のVision Encoder（CLIP, DINOv2, ViT-ImageNet）の全Transformerブロック
出力を解析し、方位情報がどの層で保持・消失するかを特定する。

既存の analyze_vision_encoders.py が最終層のみを対象としていたのに対し、
本スクリプトは全中間層を網羅的に解析する。

解析内容:
  1. 層別線形プローブ（CLS / パッチトークン）
     - 各層からRidge回帰で角度予測、R²とMAEを層番号でプロット
  2. 層別コサイン類似度（CLS / パッチトークン）
     - 各層の隣接角度間類似度を層番号でプロット
  3. 層別コサイン類似度行列（主要な層のヒートマップ）
  4. モデル間比較サマリ

使い方:
  # 先に刺激画像を生成しておく
  python generate_stimuli.py

  # 全モデル・全条件で解析
  python analyze_intermediate_layers.py

  # standard条件のみ
  python analyze_intermediate_layers.py --conditions standard

  # 特定のモデルのみ
  python analyze_intermediate_layers.py --models clip dinov2

出力:
  figures/intermediate_layers/ に解析結果の図とCSVを保存

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
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut


# =========================================================================
#  1. データ読み込み（analyze_vision_encoders.py と同一）
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
                'filename': row['filename'],
            })

    print(f"Loaded {len(stimuli)} stimuli from {stimuli_dir}")
    return stimuli


# =========================================================================
#  2. モデル定義（中間層出力対応版）
# =========================================================================

class IntermediateEncoderBase:
    """中間層出力に対応したVision Encoderの基底クラス"""

    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.model = None
        self.transform = None
        self.n_layers = 0  # Transformerブロック数

    def load(self):
        raise NotImplementedError

    def preprocess(self, pil_img):
        """PIL画像をモデル入力テンソルに変換する"""
        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def extract_all_layers(self, img_tensor):
        """
        全Transformerブロックの出力を抽出する。

        Returns
        -------
        layer_outputs : list of dict
            各層の出力。各dictは以下のキーを持つ:
            - 'cls_token': torch.Tensor, shape (D,)
            - 'patch_tokens': torch.Tensor, shape (N_patches, D)
            層0 = 埋め込み層の出力（Transformerブロック適用前）
            層1〜N = 各Transformerブロックの出力
        """
        raise NotImplementedError


class CLIPIntermediateEncoder(IntermediateEncoderBase):
    """CLIP ViT-L/14 — 中間層出力対応"""

    def __init__(self, device):
        super().__init__('CLIP_ViT-L/14', device)

    def load(self):
        from transformers import CLIPVisionModel
        from torchvision import transforms

        self.model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            attn_implementation="eager",
        ).to(self.device).eval()

        self.n_layers = self.model.config.num_hidden_layers  # 24
        # Resize to exact input size (no CenterCrop) to preserve
        # the full stimulus layout including the circle in the lower-left
        self.transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def extract_all_layers(self, img_tensor):
        with torch.no_grad():
            outputs = self.model(
                pixel_values=img_tensor,
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of (n_layers+1) tensors
        # [0] = embedding output, [1]〜[N] = each transformer block output
        # shape: (batch, seq_len, hidden_dim)
        layer_outputs = []
        for hidden_state in outputs.hidden_states:
            hs = hidden_state[0]  # remove batch dim
            layer_outputs.append({
                'cls_token': hs[0].cpu(),
                'patch_tokens': hs[1:].cpu(),
            })

        return layer_outputs


class DINOv2IntermediateEncoder(IntermediateEncoderBase):
    """DINOv2 ViT-B/14 — 中間層出力対応"""

    def __init__(self, device):
        super().__init__('DINOv2_ViT-B/14', device)

    def load(self):
        from torchvision import transforms

        self.model = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14',
        ).to(self.device).eval()

        self.n_layers = len(self.model.blocks)  # 12

        # Resize to exact input size (no CenterCrop) to preserve
        # the full stimulus layout including the circle in the lower-left
        self.transform = transforms.Compose([
            transforms.Resize(
                (518, 518),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # 全ブロックの出力をフックでキャプチャ
        self._block_outputs = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                self._block_outputs.append(output.detach())
            return hook_fn

        self._hooks = []
        for i, block in enumerate(self.model.blocks):
            h = block.register_forward_hook(make_hook(i))
            self._hooks.append(h)

    def extract_all_layers(self, img_tensor):
        self._block_outputs = []

        with torch.no_grad():
            # 埋め込み層 → 各ブロックを手動で順に通す（1回の推論で完了）
            x = self.model.prepare_tokens_with_masks(img_tensor)
            embedding_output = x.detach()

            # 各ブロックを手動で実行（フックが自動的にキャプチャ）
            for block in self.model.blocks:
                x = block(x)

        # 層0: 埋め込み層の出力
        layer_outputs = [{
            'cls_token': embedding_output[0, 0].cpu(),
            'patch_tokens': embedding_output[0, 1:].cpu(),
        }]

        # 層1〜N: 各Transformerブロックの出力
        for block_out in self._block_outputs:
            bo = block_out[0]  # remove batch dim
            layer_outputs.append({
                'cls_token': bo[0].cpu(),
                'patch_tokens': bo[1:].cpu(),
            })

        return layer_outputs


class ViTImageNetIntermediateEncoder(IntermediateEncoderBase):
    """ViT-B/16 ImageNet — 中間層出力対応"""

    def __init__(self, device):
        super().__init__('ViT-B/16_ImageNet', device)

    def load(self):
        from transformers import ViTModel, ViTImageProcessor

        self.processor = ViTImageProcessor.from_pretrained(
            'google/vit-base-patch16-224',
        )
        self.model = ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
            attn_implementation="eager",
        ).to(self.device).eval()

        self.n_layers = self.model.config.num_hidden_layers  # 12

    def preprocess(self, pil_img):
        inputs = self.processor(images=pil_img, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)

    def extract_all_layers(self, img_tensor):
        with torch.no_grad():
            outputs = self.model(
                pixel_values=img_tensor,
                output_hidden_states=True,
            )

        layer_outputs = []
        for hidden_state in outputs.hidden_states:
            hs = hidden_state[0]
            layer_outputs.append({
                'cls_token': hs[0].cpu(),
                'patch_tokens': hs[1:].cpu(),
            })

        return layer_outputs


# =========================================================================
#  3. 特徴抽出（全層）
# =========================================================================

def extract_all_layer_features(encoder, stimuli):
    """
    全刺激画像から全層の特徴を抽出する。

    Returns
    -------
    results : list of dict
        各刺激について:
        - 'angle_deg': float
        - 'condition': str
        - 'filename': str
        - 'layer_outputs': list of dict (各層の cls_token, patch_tokens)
    """
    results = []
    for stim in tqdm(stimuli, desc=f"  {encoder.name}"):
        img_tensor = encoder.preprocess(stim['image'])
        layer_outputs = encoder.extract_all_layers(img_tensor)

        results.append({
            'angle_deg': stim['angle_deg'],
            'condition': stim['condition'],
            'filename': stim['filename'],
            'layer_outputs': layer_outputs,
        })

    return results


# =========================================================================
#  4. ユーティリティ
# =========================================================================

def get_patch_grid_size(n_patches):
    """パッチ数から空間グリッドサイズを推定する"""
    h = w = int(np.round(np.sqrt(n_patches)))
    while h * w < n_patches:
        w += 1
    return h, w


def extract_local_patch_mean(patch_tokens):
    """
    左下1/4領域のパッチトークンを平均プーリングする。

    Parameters
    ----------
    patch_tokens : torch.Tensor, shape (n_patches, D)

    Returns
    -------
    local_feat : torch.Tensor, shape (D,)
    """
    n_patches = patch_tokens.shape[0]
    h, w = get_patch_grid_size(n_patches)
    usable = h * w
    patches_grid = patch_tokens[:usable].reshape(h, w, -1)

    h_mid = h // 2
    w_mid = w // 2
    local_patches = patches_grid[h_mid:, :w_mid, :]  # 左下1/4
    return local_patches.reshape(-1, local_patches.shape[-1]).mean(dim=0)


def run_loo_linear_probe(features, angles):
    """
    Leave-one-out 交差検証による線形プローブ。

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, D)
    angles : np.ndarray, shape (n_samples,)

    Returns
    -------
    dict with keys: 'r2', 'mae', 'predictions', 'errors'
    """
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

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((angles - angles.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(errors))

    return {
        'r2': r2,
        'mae': mae,
        'predictions': predictions,
        'errors': errors,
    }


def compute_neighbor_similarity(features):
    """
    隣接角度間のコサイン類似度の平均を計算する。

    Parameters
    ----------
    features : torch.Tensor, shape (n_angles, D)

    Returns
    -------
    mean_sim : float
        隣接角度間のコサイン類似度の平均
    neighbor_sims : list of float
        各隣接ペアの類似度
    """
    feat_normed = F.normalize(features, dim=1)
    sim_matrix = (feat_normed @ feat_normed.T).numpy()

    neighbor_sims = [sim_matrix[i, i + 1] for i in range(len(sim_matrix) - 1)]
    return float(np.mean(neighbor_sims)), neighbor_sims


def compute_sim_matrix(features):
    """
    コサイン類似度行列を計算する。

    Parameters
    ----------
    features : torch.Tensor, shape (n_angles, D)

    Returns
    -------
    sim_matrix : np.ndarray, shape (n_angles, n_angles)
    """
    feat_normed = F.normalize(features, dim=1)
    return (feat_normed @ feat_normed.T).numpy()


# =========================================================================
#  5. 層別解析の実行
# =========================================================================

def analyze_per_layer(results, encoder_name, n_layers, condition='standard'):
    """
    全層についてCLS/パッチトークンの線形プローブとコサイン類似度を計算する。

    Parameters
    ----------
    results : list of dict
        extract_all_layer_features() の出力
    encoder_name : str
    n_layers : int
        Transformerブロック数
    condition : str
        解析対象の条件

    Returns
    -------
    layer_metrics : dict
        層ごとのメトリクス:
        - 'cls_r2': list of float (n_layers+1)
        - 'cls_mae': list of float
        - 'cls_neighbor_sim': list of float
        - 'patch_r2': list of float
        - 'patch_mae': list of float
        - 'patch_neighbor_sim': list of float
        - 'cls_sim_matrices': list of np.ndarray
        - 'patch_sim_matrices': list of np.ndarray
        - 'cls_probe_details': list of dict (predictions, errors for each layer)
        - 'patch_probe_details': list of dict
        - 'angles': np.ndarray
    """
    cond_results = [r for r in results if r['condition'] == condition]
    if not cond_results:
        print(f"  No results for condition '{condition}'")
        return None

    cond_results.sort(key=lambda r: r['angle_deg'])
    angles = np.array([r['angle_deg'] for r in cond_results])
    total_layers = n_layers + 1  # 埋め込み層 + Transformerブロック

    # 初期化
    metrics = {
        'cls_r2': [], 'cls_mae': [], 'cls_neighbor_sim': [],
        'patch_r2': [], 'patch_mae': [], 'patch_neighbor_sim': [],
        'cls_sim_matrices': [], 'patch_sim_matrices': [],
        'cls_probe_details': [], 'patch_probe_details': [],
        'cls_neighbor_sims_per_angle': [], 'patch_neighbor_sims_per_angle': [],
        'angles': angles,
    }

    for layer_idx in tqdm(range(total_layers), desc=f"  Layers ({condition})"):
        # --- CLSトークン ---
        cls_features = torch.stack([
            r['layer_outputs'][layer_idx]['cls_token'] for r in cond_results
        ])
        cls_np = cls_features.numpy()

        cls_probe = run_loo_linear_probe(cls_np, angles)
        cls_mean_sim, cls_nsims = compute_neighbor_similarity(cls_features)
        cls_sim_mat = compute_sim_matrix(cls_features)

        metrics['cls_r2'].append(cls_probe['r2'])
        metrics['cls_mae'].append(cls_probe['mae'])
        metrics['cls_neighbor_sim'].append(cls_mean_sim)
        metrics['cls_sim_matrices'].append(cls_sim_mat)
        metrics['cls_probe_details'].append(cls_probe)
        metrics['cls_neighbor_sims_per_angle'].append(cls_nsims)

        # --- パッチトークン（左下1/4の平均プーリング）---
        patch_features = torch.stack([
            extract_local_patch_mean(
                r['layer_outputs'][layer_idx]['patch_tokens']
            )
            for r in cond_results
        ])
        patch_np = patch_features.numpy()

        patch_probe = run_loo_linear_probe(patch_np, angles)
        patch_mean_sim, patch_nsims = compute_neighbor_similarity(patch_features)
        patch_sim_mat = compute_sim_matrix(patch_features)

        metrics['patch_r2'].append(patch_probe['r2'])
        metrics['patch_mae'].append(patch_probe['mae'])
        metrics['patch_neighbor_sim'].append(patch_mean_sim)
        metrics['patch_sim_matrices'].append(patch_sim_mat)
        metrics['patch_probe_details'].append(patch_probe)
        metrics['patch_neighbor_sims_per_angle'].append(patch_nsims)

    return metrics


# =========================================================================
#  6. 可視化: 個別モデルの層別プロット
# =========================================================================

def plot_layer_metrics(metrics, encoder_name, n_layers, output_dir):
    """
    1モデルの層別メトリクスをプロットする。

    生成される図:
      1. R² vs 層番号（CLS / パッチ）
      2. MAE vs 層番号（CLS / パッチ）
      3. 隣接角度間類似度の平均 vs 層番号（CLS / パッチ）
      4. 主要な層のコサイン類似度行列ヒートマップ（CLS）
      5. 主要な層のコサイン類似度行列ヒートマップ（パッチ）
      6. 主要な層の予測 vs 正解プロット（CLS）
      7. 主要な層の予測 vs 正解プロット（パッチ）
    """
    fig_dir = os.path.join(output_dir, 'intermediate_layers')
    os.makedirs(fig_dir, exist_ok=True)

    safe_name = encoder_name.replace('/', '_')
    layer_indices = list(range(n_layers + 1))
    layer_labels = ['emb'] + [str(i) for i in range(1, n_layers + 1)]

    # ------------------------------------------------------------------
    # 図1: R² vs 層番号
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(layer_indices, metrics['cls_r2'], 'o-', color='steelblue',
            linewidth=2, markersize=6, label='CLS token')
    ax.plot(layer_indices, metrics['patch_r2'], 's-', color='coral',
            linewidth=2, markersize=6, label='Local patches (lower-left 1/4)')
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (LOO linear probe)")
    ax.set_title(f"Orientation Information by Layer — {encoder_name}")
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layer_labels, fontsize=7)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'r2_by_layer_{safe_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # 図2: MAE vs 層番号
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(layer_indices, metrics['cls_mae'], 'o-', color='steelblue',
            linewidth=2, markersize=6, label='CLS token')
    ax.plot(layer_indices, metrics['patch_mae'], 's-', color='coral',
            linewidth=2, markersize=6, label='Local patches (lower-left 1/4)')
    ax.set_xlabel("Layer")
    ax.set_ylabel("MAE (°)")
    ax.set_title(f"Prediction Error by Layer — {encoder_name}")
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layer_labels, fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'mae_by_layer_{safe_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # 図3: 隣接角度間類似度の平均 vs 層番号
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(layer_indices, metrics['cls_neighbor_sim'], 'o-', color='steelblue',
            linewidth=2, markersize=6, label='CLS token')
    ax.plot(layer_indices, metrics['patch_neighbor_sim'], 's-', color='coral',
            linewidth=2, markersize=6, label='Local patches (lower-left 1/4)')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean neighbor cosine similarity")
    ax.set_title(
        f"Angle Discriminability by Layer — {encoder_name}\n"
        f"(lower = more discriminable)"
    )
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layer_labels, fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'neighbor_sim_by_layer_{safe_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # 図4 & 5: 主要な層のコサイン類似度行列ヒートマップ
    # ------------------------------------------------------------------
    # 等間隔で6層を選択（埋め込み、序盤、中盤、終盤、最終層）
    key_layers = _select_key_layers(n_layers, n_display=6)
    angles = metrics['angles']

    for token_type, sim_matrices in [
        ('CLS', metrics['cls_sim_matrices']),
        ('patch', metrics['patch_sim_matrices']),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes_flat = axes.flatten()

        for plot_idx, layer_idx in enumerate(key_layers):
            ax = axes_flat[plot_idx]
            sim_mat = sim_matrices[layer_idx]

            im = ax.imshow(
                sim_mat, cmap='RdYlBu_r',
                vmin=sim_mat.min(), vmax=1.0,
            )
            label = 'emb' if layer_idx == 0 else f'L{layer_idx}'
            ax.set_title(f"Layer {label}", fontsize=11)
            ax.set_xticks(range(0, len(angles), 2))
            ax.set_xticklabels(
                [f"{angles[i]:.0f}°" for i in range(0, len(angles), 2)],
                rotation=45, fontsize=7,
            )
            ax.set_yticks(range(0, len(angles), 2))
            ax.set_yticklabels(
                [f"{angles[i]:.0f}°" for i in range(0, len(angles), 2)],
                fontsize=7,
            )
            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(
            f"Cosine Similarity Matrices ({token_type} tokens) — {encoder_name}",
            fontsize=14,
        )
        plt.tight_layout()
        fig.savefig(
            os.path.join(fig_dir, f'sim_heatmaps_{token_type}_{safe_name}.png'),
            dpi=150, bbox_inches='tight',
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # 図6 & 7: 主要な層の予測 vs 正解プロット
    # ------------------------------------------------------------------
    for token_type, probe_details in [
        ('CLS', metrics['cls_probe_details']),
        ('patch', metrics['patch_probe_details']),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes_flat = axes.flatten()

        for plot_idx, layer_idx in enumerate(key_layers):
            ax = axes_flat[plot_idx]
            probe = probe_details[layer_idx]

            ax.scatter(angles, probe['predictions'], c='steelblue', s=40, zorder=3)
            ax.plot([0, 90], [0, 90], 'k--', alpha=0.5)
            label = 'emb' if layer_idx == 0 else f'L{layer_idx}'
            r2 = metrics[f'{token_type.lower()}_r2'][layer_idx]
            ax.set_title(f"Layer {label}  (R²={r2:.3f})", fontsize=10)
            ax.set_xlabel("True angle (°)", fontsize=8)
            ax.set_ylabel("Predicted angle (°)", fontsize=8)
            ax.set_xlim(0, 90)
            ax.set_ylim(0, 90)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        fig.suptitle(
            f"Linear Probe: Predicted vs True ({token_type} tokens) — {encoder_name}",
            fontsize=14,
        )
        plt.tight_layout()
        fig.savefig(
            os.path.join(fig_dir, f'probe_scatter_{token_type}_{safe_name}.png'),
            dpi=150, bbox_inches='tight',
        )
        plt.close(fig)

    print(f"  Saved all per-model figures to {fig_dir}/")


def _select_key_layers(n_layers, n_display=6):
    """
    表示用に等間隔でn_display個の層インデックスを選択する。
    必ず層0（埋め込み）と層n_layers（最終層）を含む。
    """
    total = n_layers + 1  # 埋め込み + Transformerブロック
    if total <= n_display:
        return list(range(total))

    indices = [0]  # 埋め込み層
    step = (n_layers) / (n_display - 1)
    for i in range(1, n_display - 1):
        indices.append(int(round(i * step)))
    indices.append(n_layers)  # 最終層

    # 重複除去（stepが小さい場合）
    return sorted(set(indices))


# =========================================================================
#  7. 可視化: モデル間比較
# =========================================================================

def plot_cross_model_comparison(all_metrics, output_dir):
    """
    全モデルのR² / MAE / 隣接類似度を1つの図に重ねてプロットする。
    各モデルの層数が異なるため、X軸は「相対的な深さ（0〜1）」に正規化する。
    """
    fig_dir = os.path.join(output_dir, 'intermediate_layers')
    os.makedirs(fig_dir, exist_ok=True)

    colors = {
        'CLIP_ViT-L/14': 'steelblue',
        'DINOv2_ViT-B/14': 'coral',
        'ViT-B/16_ImageNet': 'seagreen',
    }
    markers = {
        'CLIP_ViT-L/14': 'o',
        'DINOv2_ViT-B/14': '^',
        'ViT-B/16_ImageNet': 's',
    }

    metric_configs = [
        ('cls_r2', 'R² (CLS token)', 'R²', 'r2_comparison_CLS.png'),
        ('patch_r2', 'R² (Local patches)', 'R²', 'r2_comparison_patch.png'),
        ('cls_mae', 'MAE (CLS token)', 'MAE (°)', 'mae_comparison_CLS.png'),
        ('patch_mae', 'MAE (Local patches)', 'MAE (°)', 'mae_comparison_patch.png'),
        ('cls_neighbor_sim', 'Neighbor Similarity (CLS)',
         'Mean cosine similarity', 'neighbor_sim_comparison_CLS.png'),
        ('patch_neighbor_sim', 'Neighbor Similarity (Local patches)',
         'Mean cosine similarity', 'neighbor_sim_comparison_patch.png'),
    ]

    for metric_key, title, ylabel, filename in metric_configs:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        for model_name, (metrics, n_layers) in all_metrics.items():
            values = metrics[metric_key]
            # X軸: 相対的な深さ（0 = 埋め込み, 1 = 最終層）
            rel_depth = np.linspace(0, 1, len(values))
            color = colors.get(model_name, 'gray')
            marker = markers.get(model_name, 'o')

            ax.plot(
                rel_depth, values, f'{marker}-',
                color=color, linewidth=2, markersize=6,
                label=f"{model_name} ({n_layers} layers)",
            )

        ax.set_xlabel("Relative depth (0=embedding, 1=final layer)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Cross-Model Comparison: {title}")
        if 'r2' in metric_key:
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            os.path.join(fig_dir, filename), dpi=150, bbox_inches='tight',
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # 統合図: R² を CLS / パッチ 並べた 2パネル図
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for panel_idx, (metric_key, panel_title) in enumerate([
        ('cls_r2', 'CLS Token'),
        ('patch_r2', 'Local Patches (lower-left 1/4)'),
    ]):
        ax = axes[panel_idx]
        for model_name, (metrics, n_layers) in all_metrics.items():
            values = metrics[metric_key]
            rel_depth = np.linspace(0, 1, len(values))
            color = colors.get(model_name, 'gray')
            marker = markers.get(model_name, 'o')
            ax.plot(
                rel_depth, values, f'{marker}-',
                color=color, linewidth=2, markersize=5,
                label=f"{model_name} ({n_layers}L)",
            )
        ax.set_xlabel("Relative depth")
        ax.set_ylabel("R²")
        ax.set_title(panel_title)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Where Does Orientation Information Disappear?",
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, 'cross_model_r2_summary.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.close(fig)

    print(f"  Saved cross-model comparison figures to {fig_dir}/")


# =========================================================================
#  8. 可視化: 条件間比較（層別）
# =========================================================================

def plot_condition_comparison_by_layer(
    results, encoder_name, n_layers, output_dir,
):
    """
    全条件（standard, no_circle, short_line, long_line）について
    層別R²を1つの図に重ねてプロットする。
    """
    fig_dir = os.path.join(output_dir, 'intermediate_layers')
    os.makedirs(fig_dir, exist_ok=True)

    conditions = sorted(set(r['condition'] for r in results))
    if len(conditions) < 2:
        return

    safe_name = encoder_name.replace('/', '_')
    cond_colors = {
        'standard': 'steelblue',
        'no_circle': 'red',
        'short_line': 'green',
        'long_line': 'purple',
    }

    # 各条件でメトリクスを計算
    cond_metrics = {}
    for cond in conditions:
        m = analyze_per_layer(results, encoder_name, n_layers, condition=cond)
        if m is not None:
            cond_metrics[cond] = m

    if len(cond_metrics) < 2:
        return

    layer_indices = list(range(n_layers + 1))
    layer_labels = ['emb'] + [str(i) for i in range(1, n_layers + 1)]

    for token_type, r2_key in [('CLS', 'cls_r2'), ('patch', 'patch_r2')]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for cond, m in cond_metrics.items():
            color = cond_colors.get(cond, 'gray')
            ax.plot(
                layer_indices, m[r2_key], 'o-',
                color=color, linewidth=2, markersize=5,
                label=cond,
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel("R²")
        ax.set_title(
            f"Condition Comparison by Layer ({token_type}) — {encoder_name}"
        )
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_labels, fontsize=7)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            os.path.join(
                fig_dir,
                f'condition_r2_{token_type}_{safe_name}.png',
            ),
            dpi=150, bbox_inches='tight',
        )
        plt.close(fig)

    print(f"  Saved condition comparison figures for {encoder_name}")


# =========================================================================
#  9. CSV出力
# =========================================================================

def save_layer_metrics_csv(metrics, encoder_name, n_layers, output_dir):
    """
    層別メトリクスをCSVに保存する。
    """
    fig_dir = os.path.join(output_dir, 'intermediate_layers')
    os.makedirs(fig_dir, exist_ok=True)
    safe_name = encoder_name.replace('/', '_')

    csv_path = os.path.join(fig_dir, f'layer_metrics_{safe_name}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'layer', 'layer_label',
            'cls_r2', 'cls_mae', 'cls_neighbor_sim',
            'patch_r2', 'patch_mae', 'patch_neighbor_sim',
        ])
        for i in range(n_layers + 1):
            label = 'emb' if i == 0 else str(i)
            writer.writerow([
                i, label,
                round(metrics['cls_r2'][i], 4),
                round(metrics['cls_mae'][i], 2),
                round(metrics['cls_neighbor_sim'][i], 6),
                round(metrics['patch_r2'][i], 4),
                round(metrics['patch_mae'][i], 2),
                round(metrics['patch_neighbor_sim'][i], 6),
            ])
    print(f"  Saved: {csv_path}")

    # 各層の線形プローブ詳細（角度ごとの予測値・誤差）
    for token_type, probe_key in [
        ('CLS', 'cls_probe_details'),
        ('patch', 'patch_probe_details'),
    ]:
        csv_path = os.path.join(
            fig_dir, f'probe_details_{token_type}_{safe_name}.csv',
        )
        angles = metrics['angles']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['angle_deg']
            for i in range(n_layers + 1):
                label = 'emb' if i == 0 else f'L{i}'
                header.extend([
                    f'pred_{label}', f'error_{label}',
                ])
            writer.writerow(header)

            for angle_idx in range(len(angles)):
                row = [angles[angle_idx]]
                for layer_idx in range(n_layers + 1):
                    probe = metrics[probe_key][layer_idx]
                    row.extend([
                        round(probe['predictions'][angle_idx], 2),
                        round(probe['errors'][angle_idx], 2),
                    ])
                writer.writerow(row)
        print(f"  Saved: {csv_path}")


# =========================================================================
#  10. メイン
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Watanabe Illusion × Vision Encoder Intermediate Layer Analysis',
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
        print("No stimuli found. Run 'python generate_stimuli.py' first.")
        return

    # エンコーダ定義
    encoder_map = {
        'clip': CLIPIntermediateEncoder,
        'dinov2': DINOv2IntermediateEncoder,
        'vit_imagenet': ViTImageNetIntermediateEncoder,
    }

    # モデル間比較用: {model_name: (metrics, n_layers)}
    all_metrics = {}

    for model_key in args.models:
        print(f"\n{'=' * 60}")
        encoder = encoder_map[model_key](device)
        print(f"Loading {encoder.name}...")
        try:
            encoder.load()
        except Exception as e:
            print(f"  Failed to load {encoder.name}: {e}")
            print(f"  Skipping...")
            continue
        print(f"  Loaded successfully ({encoder.n_layers} transformer blocks)")

        # 特徴抽出（全層）
        print(f"Extracting features from all layers...")
        results = extract_all_layer_features(encoder, stimuli)

        # 層別解析（standard条件）
        print(f"Analyzing per-layer metrics (standard)...")
        metrics = analyze_per_layer(
            results, encoder.name, encoder.n_layers, condition='standard',
        )

        if metrics is not None:
            # CSVに保存
            save_layer_metrics_csv(
                metrics, encoder.name, encoder.n_layers, args.output_dir,
            )

            # 個別モデルの可視化
            print(f"Generating per-model figures...")
            plot_layer_metrics(
                metrics, encoder.name, encoder.n_layers, args.output_dir,
            )

            all_metrics[encoder.name] = (metrics, encoder.n_layers)

        # 条件間比較（全条件が読み込まれている場合）
        conditions_present = set(r['condition'] for r in results)
        if len(conditions_present) > 1:
            print(f"Analyzing condition comparison by layer...")
            plot_condition_comparison_by_layer(
                results, encoder.name, encoder.n_layers, args.output_dir,
            )

        # メモリ解放
        del encoder.model
        del results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # モデル間比較
    if len(all_metrics) > 1:
        print(f"\nGenerating cross-model comparison...")
        plot_cross_model_comparison(all_metrics, args.output_dir)

    print(f"\nAll intermediate layer analyses complete.")
    print(f"Results saved to '{args.output_dir}/intermediate_layers/'")


if __name__ == '__main__':
    main()
