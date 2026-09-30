"""
decoder_model.py

ViT block_last 表現から画像を再構成する decoder。

設計方針:
- Qwen2.5-VL-32B の ViT (凍結) の block_last 出力を入力とする
- Pixel Shuffle (×2) を 3 回で 8 倍 upsampling、最後 1.75 倍は F.interpolate
- チャネル: 1280 → 256 → 64 → 32 → 16 → 3 (RGB)
- 出力レンジ: tanh で [-1, 1] (Normalize された ImageNet 入力と整合)

入出力 (560×280 letterbox 入力時の想定):
- Input:  (B, hidden_dim=1280, grid_h=20, grid_w=40)
- Output: (B, 3, 280, 560)

実装ノート:
- hidden_dim はコンストラクタで指定可能 (Qwen2.5-VL-32B 以外への対応)
- grid_h, grid_w は forward 時に変動可能 (dynamic resolution 対応)
- 最終 F.interpolate でアスペクト比を厳密に 280/560 にする (1.75x = 14/8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(num_channels: int) -> nn.Module:
    """GroupNorm を返す。num_groups は num_channels の最大の約数で 32 以下のものを選ぶ。"""
    for ng in (32, 16, 8, 4, 2, 1):
        if num_channels % ng == 0:
            return nn.GroupNorm(num_groups=ng, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


class PixelShuffleUpBlock(nn.Module):
    """
    1段の Pixel Shuffle upsampling ブロック.

    in_ch チャネル → 4 * out_ch チャネル (1x1 conv) → PixelShuffle(2) で out_ch チャネルに、
    かつ spatial を 2 倍に拡大。続いて 3x3 conv で局所的な調整を入れる。
    checkerboard artifact が出にくく学習しやすい。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # PixelShuffle(2) は (B, 4*C, H, W) → (B, C, 2H, 2W)
        self.expand = nn.Conv2d(in_ch, 4 * out_ch, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        self.refine = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = _make_norm(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.shuffle(x)
        x = self.refine(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ViTPatchDecoder(nn.Module):
    """
    ViT block_last (spatial reshaped) から画像を再構成する decoder.

    Args:
        hidden_dim: ViT の block_last hidden dimension (Qwen2.5-VL-32B は 1280 を想定)
        out_channels: 出力チャネル数 (RGB = 3)
        target_h, target_w: 最終出力サイズ。学習・評価時の Watanabe 刺激は 280×560
        mid_channels: 中間チャネル数のリスト [c1, c2, c3, c4]
            (1x1 conv 出力 → PS×3 後 → 最終 conv 入力 へと段階的に削減)
    """

    def __init__(
        self,
        hidden_dim: int = 1280,
        out_channels: int = 3,
        target_h: int = 280,
        target_w: int = 560,
        mid_channels: tuple = (256, 64, 32, 16),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.target_h = target_h
        self.target_w = target_w

        c1, c2, c3, c4 = mid_channels

        # 初段: 1x1 conv で hidden_dim → c1 にチャネル削減
        self.stem = nn.Sequential(
            nn.Conv2d(hidden_dim, c1, kernel_size=1, bias=False),
            _make_norm(c1),
            nn.GELU(),
        )

        # Pixel Shuffle ×3 で 8 倍 upsampling
        self.up1 = PixelShuffleUpBlock(c1, c2)  # ×2
        self.up2 = PixelShuffleUpBlock(c2, c3)  # ×2
        self.up3 = PixelShuffleUpBlock(c3, c4)  # ×2

        # 最終出力 conv (3×3) → 3 チャネル
        self.head = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            _make_norm(c4),
            nn.GELU(),
            nn.Conv2d(c4, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, hidden_dim, grid_h, grid_w)
                ViT block_last を (B, hidden_dim, grid_h, grid_w) に spatial reshape したもの
        Returns:
            recon: (B, out_channels, target_h, target_w) ∈ [-1, 1]
        """
        x = self.stem(feat)        # (B, c1, gh, gw)
        x = self.up1(x)            # (B, c2, 2gh, 2gw)
        x = self.up2(x)            # (B, c3, 4gh, 4gw)
        x = self.up3(x)            # (B, c4, 8gh, 8gw)

        # 最終 spatial を target_h × target_w に bilinear interpolate
        # (例: gh=20, gw=40 なら 8gh=160, 8gw=320 → bilinear で 280×560 に拡大)
        x = F.interpolate(
            x, size=(self.target_h, self.target_w),
            mode='bilinear', align_corners=False,
        )

        x = self.head(x)            # (B, 3, target_h, target_w)
        x = torch.tanh(x)           # [-1, 1]
        return x


def patches_to_spatial(
    block_last: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """
    ViT block_last (n_patches, hidden_dim) を (1, hidden_dim, grid_h, grid_w) に整形する.

    Qwen2.5-VL の block_last は flatten された patch token 列 (N, D) で、
    N = grid_t * grid_h * grid_w (画像入力時 grid_t=1)。
    Pixel Shuffle decoder に入れるために 2D 特徴マップに整形する。

    Args:
        block_last: (n_patches, hidden_dim) または (1, n_patches, hidden_dim)
        grid_h, grid_w: ViT の patch grid サイズ
    Returns:
        feat: (1, hidden_dim, grid_h, grid_w)
    """
    if block_last.dim() == 3 and block_last.shape[0] == 1:
        block_last = block_last.squeeze(0)
    if block_last.dim() != 2:
        raise ValueError(
            f"Expected block_last shape (n_patches, hidden_dim) or "
            f"(1, n_patches, hidden_dim), got {tuple(block_last.shape)}"
        )

    n_patches, hidden_dim = block_last.shape
    expected = grid_h * grid_w
    if n_patches != expected:
        raise ValueError(
            f"n_patches={n_patches} does not match grid_h*grid_w={expected} "
            f"(grid_h={grid_h}, grid_w={grid_w})"
        )

    # (n_patches, D) → (grid_h, grid_w, D) → (D, grid_h, grid_w) → (1, D, grid_h, grid_w)
    feat = block_last.view(grid_h, grid_w, hidden_dim)
    feat = feat.permute(2, 0, 1).contiguous().unsqueeze(0)
    return feat


# ===================================================================
# Smoke test (このファイル単体で動作確認)
# ===================================================================
if __name__ == "__main__":
    import sys

    # 期待形状: Qwen2.5-VL-32B の ViT block_last
    # 560x280 letterbox 入力で patch_size=14 → grid = 20 × 40 = 800 patches
    # hidden_dim は 1280 を想定 (実機で要確認)
    hidden_dim = 1280
    grid_h, grid_w = 20, 40
    n_patches = grid_h * grid_w
    target_h, target_w = 280, 560

    print(f"=== ViTPatchDecoder smoke test ===")
    print(f"hidden_dim = {hidden_dim}")
    print(f"grid       = {grid_h} x {grid_w} = {n_patches} patches")
    print(f"target     = {target_h} x {target_w}")
    print()

    # 1. patches_to_spatial の確認
    block_last = torch.randn(n_patches, hidden_dim)
    feat = patches_to_spatial(block_last, grid_h, grid_w)
    print(f"patches_to_spatial:  {tuple(block_last.shape)} -> {tuple(feat.shape)}")
    assert feat.shape == (1, hidden_dim, grid_h, grid_w), f"unexpected: {feat.shape}"

    # 2. decoder の forward
    decoder = ViTPatchDecoder(
        hidden_dim=hidden_dim,
        out_channels=3,
        target_h=target_h,
        target_w=target_w,
    )
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"decoder parameters:  {n_params:,}  ({n_params/1e6:.2f} M)")

    with torch.no_grad():
        recon = decoder(feat)
    print(f"decoder forward:     {tuple(feat.shape)} -> {tuple(recon.shape)}")
    assert recon.shape == (1, 3, target_h, target_w), f"unexpected: {recon.shape}"

    # 3. 値域チェック
    print(f"recon range:         [{recon.min().item():.3f}, {recon.max().item():.3f}]")
    assert recon.min() >= -1.0 - 1e-5 and recon.max() <= 1.0 + 1e-5

    # 4. バッチ入力の動作確認
    batch_feat = torch.randn(4, hidden_dim, grid_h, grid_w)
    with torch.no_grad():
        batch_recon = decoder(batch_feat)
    print(f"batch forward:       {tuple(batch_feat.shape)} -> {tuple(batch_recon.shape)}")
    assert batch_recon.shape == (4, 3, target_h, target_w)

    # 5. 異なる grid サイズでの動作確認 (dynamic resolution 対応)
    feat2 = torch.randn(1, hidden_dim, 14, 28)  # 別のアスペクト比
    with torch.no_grad():
        recon2 = decoder(feat2)
    print(f"dynamic grid:        {tuple(feat2.shape)} -> {tuple(recon2.shape)}")
    assert recon2.shape == (1, 3, target_h, target_w), \
        "target サイズは固定のはず (forward 内で interpolate)"

    print()
    print("All smoke tests passed.")
