"""
train_decoder.py
=================
Qwen2.5-VL-32B ViT (凍結) の block_last 表現から画像を再構成する decoder の訓練。

設計 (paper_v4 v1, A100 80GB):
- ViT: 凍結、forward hook で blocks[-1] 出力を捕捉 (shape: (n_patches, 1280))
- Decoder: ViTPatchDecoder (decoder_model.py 参照)
- 入力画像: letterbox 560×280、ImageNet mean/std で normalize
- ViT 入力: processor で正規化された pixel_values (Qwen2.5-VL 標準)
- Decoder ターゲット: ImageNet normalize 空間で letterbox 画像、tanh と整合
- Loss: MSE のみ (シンプル、過学習を恐れない)
- 訓練データ: stimuli/ 68 画像、毎 epoch 全画像シャッフル
- ハードウェア: A100 80GB、ViT は 65GB 占有なので decoder のバッチを小さく

実験ゴール (paper_v4 中心仮説):
  訓練後、104.jpg (input 23.5° 線) を decoder で復元し、復元画像の線角度を
  Hough 変換で測定。入力 23.5° と差があれば、それが decoder 経由で出た
  angular bias = "ViT 内部の visual representation を視覚として外在化" の効果。
  人間/VLM の言語出力 bias と同じ方向か (over-estimate 45° 寄り) を確認する。

実装ノート:
- 1 epoch 約 30 秒想定 (68 画像、A100、batch_size=4)
- decoder パラメータ ~3M、訓練は軽い
- ViT は torch.no_grad で forward のみ、decoder のみ backward
- checkpoint と復元 PNG を 25 epoch ごとに保存

Usage (A100 上の Docker 内):
    python scripts/train_decoder.py --epochs 200 --batch-size 4
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import csv

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# decoder_model.py を import (同じ scripts/ 配下に置いてあると想定)
sys.path.insert(0, str(Path(__file__).parent))
from decoder_model import ViTPatchDecoder, patches_to_spatial


# ========== 設定 ==========
STIMULI_DIR = Path("/workspace/stimuli")
METADATA_CSV = Path("/workspace/stimuli/stimulus_metadata.csv")
OUT_DIR = Path("/workspace/results/decoder_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_MEMORY = {0: "75GiB", "cpu": "0GiB"}

# letterbox 設定 (decoder_model.py の想定と一致)
TARGET_H, TARGET_W = 280, 560
HIDDEN_DIM = 1280
GRID_H, GRID_W = 20, 40  # check_vit_letterbox_shape.py で実測確認済み

# 訓練ハイパーパラメータ
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE_DEFAULT = 4

# ImageNet normalize 定数 (decoder ターゲットの正規化に使う)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Letterbox + Normalize utility ==========
def letterbox_pil(img: Image.Image, target_w: int, target_h: int,
                  fill=(128, 128, 128)) -> Image.Image:
    """アスペクト比保持で resize して、足りない領域を fill 色で埋める。"""
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), fill)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    return canvas


def pil_to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → torch.Tensor (3, H, W) で ImageNet mean/std 正規化。
    tanh 出力空間と整合させるため、[0,1] -> normalize でなく、
    [-1,1] に近づく ImageNet 正規化を使う。
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) ∈ [0, 1]
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    t = torch.from_numpy(arr)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = (t - mean) / std
    return t


def denormalize_to_pil(t: torch.Tensor) -> Image.Image:
    """ImageNet 正規化空間の (3, H, W) Tensor → 表示用 PIL 画像。"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(t.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(t.device)
    img = t.detach().cpu() * std.cpu() + mean.cpu()
    img = img.clamp(0, 1)
    arr = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ========== Dataset (68 stimuli) ==========
class StimuliDataset(Dataset):
    """stimuli/ の画像を letterbox + normalize して返す。
    各 sample: (letterbox_pil_for_vit, target_tensor, filename)
      - letterbox_pil_for_vit: ViT processor に渡す PIL 画像 (560×280)
      - target_tensor: decoder ターゲット (3, 280, 560) ImageNet 正規化
      - filename: ログ用
    """

    def __init__(self, stimuli_dir: Path, metadata_csv: Path):
        self.stimuli_dir = stimuli_dir
        with open(metadata_csv) as f:
            rows = list(csv.DictReader(f))
        self.filenames = [r['filename'] for r in rows]
        print(f"  loaded {len(self.filenames)} stimuli filenames")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img = Image.open(self.stimuli_dir / fn).convert("RGB")
        img_lb = letterbox_pil(img, TARGET_W, TARGET_H)
        target = pil_to_normalized_tensor(img_lb)
        return img_lb, target, fn


def collate_for_vit(batch):
    """カスタム collate: ViT 用に PIL リスト、target は Tensor stack。"""
    pils, targets, fns = zip(*batch)
    targets = torch.stack(list(targets), dim=0)
    return list(pils), targets, list(fns)


# ========== ViT feature extraction (frozen) ==========
class FrozenViTExtractor:
    """Qwen2.5-VL の ViT を凍結状態で wrap し、blocks[-1] 出力を取り出す。"""

    def __init__(self, model, processor):
        self.processor = processor
        self.vision_tower = model.model.visual
        self.device = next(model.parameters()).device
        # block_last を捕捉する hook
        self._cache = {}
        self._handle = self.vision_tower.blocks[-1].register_forward_hook(
            self._hook
        )

    def _hook(self, module, inp, out):
        self._cache["x"] = out

    def __del__(self):
        try:
            self._handle.remove()
        except Exception:
            pass

    @torch.no_grad()
    def extract(self, pil_images):
        """PIL リスト → block_last のリスト [(n_patches, 1280) ...]。
        Qwen2.5-VL の vision_tower は batch を平坦化して 1 度で処理するので、
        画像ごとに grid_h, grid_w が同じ (今回は固定 20×40) ならば
        block_last は (B*800, 1280) で出てくる。
        """
        # processor で batch を作る (text 部分はダミー)
        messages_list = []
        for img in pil_images:
            messages_list.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "x"},
                ],
            }])
        texts = [
            self.processor.apply_chat_template(m, tokenize=False,
                                                add_generation_prompt=True)
            for m in messages_list
        ]
        inputs = self.processor(
            text=texts, images=pil_images, padding=True, return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].to(self.device).to(torch.bfloat16)
        image_grid_thw = inputs["image_grid_thw"].to(self.device)

        _ = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
        block_last = self._cache["x"]
        # (total_patches, hidden_dim) を画像ごとに切り分け
        # 全画像同じ grid なら 1 画像あたり grid_h*grid_w patches
        out_list = []
        offset = 0
        for thw in image_grid_thw:
            t, h, w = thw.tolist()
            n = t * h * w
            out_list.append(block_last[offset:offset + n])
            offset += n
        return out_list, image_grid_thw


# ========== Reconstruction & angle measurement ==========
def measure_line_angle_in_recon(recon_pil: Image.Image) -> float:
    """復元画像から線の角度 (deg above horizontal) を Hough で測る。
    104.jpg の入力 23.5° との差分を取るための後処理。
    現時点では実装の placeholder として None を返してもよい。
    """
    # 簡易実装: グレースケール化 → Canny edge → HoughLines
    try:
        import cv2
    except ImportError:
        return float('nan')
    img_arr = np.asarray(recon_pil.convert("L"))
    # 円の中のエッジを強調するため軽い Gaussian
    img_blur = cv2.GaussianBlur(img_arr, (3, 3), 0.5)
    edges = cv2.Canny(img_blur, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=20, minLineLength=15, maxLineGap=5)
    if lines is None or len(lines) == 0:
        return float('nan')
    # 円内 (左下) にある線分のみ採用 (画像左 30% かつ下 60%)
    H, W = img_arr.shape
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if max(x1, x2) > W * 0.35:
            continue
        if min(y1, y2) < H * 0.40:
            continue
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        # 画像座標系では y は下向きなので、horizontal 上向き角度は -dy/dx の atan
        angle = np.degrees(np.arctan2(-dy, abs(dx) if dx >= 0 else -abs(dx)))
        candidates.append(angle)
    if not candidates:
        return float('nan')
    # 中央値を取る (外れ値耐性)
    return float(np.median(candidates))


# ========== Train loop ==========
def train(args):
    print(f"=" * 70)
    print(f"train_decoder.py — paper_v4 v1")
    print(f"=" * 70)
    print(f"  stimuli:    {STIMULI_DIR}")
    print(f"  out_dir:    {OUT_DIR}")
    print(f"  model:      {MODEL_ID}")
    print(f"  target_h×w: {TARGET_H}×{TARGET_W}")
    print(f"  grid_h×w:   {GRID_H}×{GRID_W}  (= {GRID_H*GRID_W} patches)")
    print(f"  hidden_dim: {HIDDEN_DIM}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs:     {args.epochs}")
    print(f"  lr:         {LR}")
    print()

    # --- データ ---
    ds = StimuliDataset(STIMULI_DIR, METADATA_CSV)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_for_vit, drop_last=False,
    )
    print(f"  dataset: {len(ds)} samples, "
          f"batches/epoch: {len(loader)}")

    # --- ViT (凍結) ---
    print(f"\nLoading frozen ViT...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=MAX_MEMORY,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  loaded in {time.time()-t0:.1f}s")
    extractor = FrozenViTExtractor(model, processor)
    device = extractor.device
    print(f"  device: {device}")

    # --- Decoder (学習対象) ---
    decoder = ViTPatchDecoder(
        hidden_dim=HIDDEN_DIM,
        out_channels=3,
        target_h=TARGET_H,
        target_w=TARGET_W,
    ).to(device).to(torch.float32)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"\ndecoder params: {n_params:,}  ({n_params/1e6:.2f} M)")

    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    # --- 訓練ログ ---
    log_path = OUT_DIR / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,elapsed_seconds,datetime\n")

    header = {
        "started_at": datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "decoder_params": n_params,
        "target_h": TARGET_H,
        "target_w": TARGET_W,
        "hidden_dim": HIDDEN_DIM,
        "grid_h": GRID_H,
        "grid_w": GRID_W,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "n_train_samples": len(ds),
        "loss": "MSE in ImageNet-normalized space",
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
    }
    with open(OUT_DIR / "train_config.json", "w") as f:
        json.dump(header, f, indent=2)

    # --- 訓練ループ ---
    print(f"\n=== Training ===")
    overall_t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        epoch_loss = 0.0
        epoch_t0 = time.time()
        n_batches = 0

        for pil_batch, target_batch, fn_batch in loader:
            # 1. ViT で特徴抽出 (no_grad)
            block_last_list, image_grid_thw = extractor.extract(pil_batch)

            # 2. 各画像の block_last を spatial feat に整形 → batch stack
            feats = []
            for bl, thw in zip(block_last_list, image_grid_thw):
                _, gh, gw = thw.tolist()
                feat = patches_to_spatial(bl, gh, gw)  # (1, 1280, gh, gw)
                feats.append(feat)
            feat_batch = torch.cat(feats, dim=0).to(torch.float32)

            # 3. decoder forward
            recon = decoder(feat_batch)  # (B, 3, 280, 560), tanh ∈ [-1,1]
            target_batch_dev = target_batch.to(device)

            # 4. MSE loss (ImageNet normalize 空間)
            loss = loss_fn(recon, target_batch_dev)

            # 5. backward (decoder のみ)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - epoch_t0
        msg = (f"  epoch {epoch:4d}/{args.epochs} | "
               f"loss {avg_loss:.5f} | "
               f"{elapsed:.1f}s")
        print(msg)
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{elapsed:.2f},"
                    f"{datetime.now().isoformat()}\n")

        # checkpoint + 復元 PNG (25 epoch ごと、最終 epoch、初回)
        save_now = (
            epoch == 1
            or epoch % args.save_every == 0
            or epoch == args.epochs
        )
        if save_now:
            ckpt_path = OUT_DIR / f"decoder_epoch{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            # 104.jpg の復元を 1 枚保存
            save_recon_for_104(decoder, extractor, device,
                                OUT_DIR / f"recon_104_epoch{epoch:04d}.png")

    total_elapsed = time.time() - overall_t0
    print(f"\n=== Training done in {total_elapsed/60:.1f} min ===")

    # 最終: 104.jpg の復元 + 角度測定
    final_recon_path = OUT_DIR / "recon_104_final.png"
    save_recon_for_104(decoder, extractor, device, final_recon_path)
    recon_pil = Image.open(final_recon_path)
    angle = measure_line_angle_in_recon(recon_pil)
    print(f"\n=== Final 104.jpg recon angle measurement ===")
    print(f"  measured angle in recon: {angle:.2f}°")
    print(f"  ground truth angle:      23.5°")
    print(f"  bias (recon - gt):       {angle - 23.5:+.2f}°")
    with open(OUT_DIR / "final_angle_measurement.json", "w") as f:
        json.dump({
            "ground_truth_angle_deg": 23.5,
            "measured_recon_angle_deg": angle,
            "bias_deg": angle - 23.5,
        }, f, indent=2)


def save_recon_for_104(decoder, extractor, device, save_path):
    """104.jpg の復元 PNG を 1 枚保存する。訓練中の進捗確認用。"""
    decoder.eval()
    img = Image.open(STIMULI_DIR / "104.jpg").convert("RGB")
    img_lb = letterbox_pil(img, TARGET_W, TARGET_H)
    block_last_list, image_grid_thw = extractor.extract([img_lb])
    bl = block_last_list[0]
    _, gh, gw = image_grid_thw[0].tolist()
    feat = patches_to_spatial(bl, gh, gw).to(torch.float32)
    with torch.no_grad():
        recon = decoder(feat)
    recon_pil = denormalize_to_pil(recon[0])
    # 比較しやすいように、original (letterboxed) と並べた画像にする
    canvas = Image.new("RGB", (TARGET_W * 2 + 10, TARGET_H), (255, 255, 255))
    canvas.paste(img_lb, (0, 0))
    canvas.paste(recon_pil, (TARGET_W + 10, 0))
    canvas.save(save_path)
    decoder.train()


# ========== main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--save-every", type=int, default=25,
                        help="checkpoint + recon PNG を保存する epoch 間隔")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
