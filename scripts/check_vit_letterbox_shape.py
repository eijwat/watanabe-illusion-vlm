"""
check_vit_letterbox_shape.py
=============================
Qwen2.5-VL-32B の ViT に letterbox 560×280 画像を入れた時の
block_last.shape と grid_thw を実測する。

これが確認できると、decoder_model.py の想定値
  hidden_dim=1280, grid_h=20, grid_w=40
が実機で成立していることが分かり、train_decoder.py の実装に進める。

Usage (A100 上の Docker 内):
    python check_vit_letterbox_shape.py
"""

import sys
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_MEMORY = {0: "75GiB", "cpu": "0GiB"}
TARGET_H, TARGET_W = 280, 560


def letterbox(img: Image.Image, target_w: int, target_h: int,
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


def main():
    print(f"Loading {MODEL_ID} (A100)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=MAX_MEMORY,
    )
    model.eval()

    # vision tower (今日の ViT probe で確定: model.model.visual)
    vision_tower = model.model.visual
    print(f"vision tower: {type(vision_tower).__name__}, "
          f"blocks: {len(vision_tower.blocks)}")
    print(f"  patch_size:  {processor.image_processor.patch_size}")
    print(f"  merge_size:  {processor.image_processor.merge_size}")

    # 104.jpg を letterbox 560×280 に整形
    img = Image.open("/workspace/stimuli/104.jpg").convert("RGB")
    print(f"\noriginal 104.jpg size: {img.size}")
    img_lb = letterbox(img, TARGET_W, TARGET_H)
    print(f"letterboxed size:      {img_lb.size}")

    # processor に通す
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_lb},
            {"type": "text", "text": "test"},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[img_lb], padding=True, return_tensors="pt"
    )
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    print(f"\npixel_values.shape: {tuple(pixel_values.shape)}")
    print(f"image_grid_thw:     {image_grid_thw.tolist()}")

    # ViT に forward (hook で block_last を捕捉)
    block_last_capture = {}

    def hook_fn(module, inp, out):
        # out は (n_patches, hidden_dim) または同等
        block_last_capture["x"] = out

    handle = vision_tower.blocks[-1].register_forward_hook(hook_fn)

    pixel_values_gpu = pixel_values.to(model.device).to(torch.bfloat16)
    image_grid_thw_gpu = image_grid_thw.to(model.device)

    with torch.no_grad():
        merger_out = vision_tower(pixel_values_gpu, grid_thw=image_grid_thw_gpu)
    handle.remove()

    block_last = block_last_capture["x"]
    print(f"\n=== Captured shapes ===")
    print(f"  block_last (ViT final block out): {tuple(block_last.shape)}")
    print(f"  merger_out (after merger):        {tuple(merger_out.shape)}")

    # 期待値との照合
    expected_grid_h, expected_grid_w = TARGET_H // 14, TARGET_W // 14
    expected_n_patches = expected_grid_h * expected_grid_w
    expected_hidden = 1280
    print(f"\n=== Expected (decoder_model.py assumption) ===")
    print(f"  grid_h = {TARGET_H}/14 = {expected_grid_h}")
    print(f"  grid_w = {TARGET_W}/14 = {expected_grid_w}")
    print(f"  n_patches = {expected_n_patches}")
    print(f"  hidden_dim = {expected_hidden}")

    actual_grid = image_grid_thw[0].tolist()  # [t, h, w]
    actual_n = block_last.shape[0] if block_last.dim() == 2 else block_last.shape[1]
    actual_hidden = block_last.shape[-1]

    print(f"\n=== Match check ===")
    print(f"  grid_thw [t,h,w]:    actual {actual_grid}   "
          f"expected [1, {expected_grid_h}, {expected_grid_w}]")
    print(f"  n_patches:           actual {actual_n}   "
          f"expected {expected_n_patches}")
    print(f"  hidden_dim:          actual {actual_hidden}   "
          f"expected {expected_hidden}")

    grid_ok = (actual_grid == [1, expected_grid_h, expected_grid_w])
    n_ok = (actual_n == expected_n_patches)
    h_ok = (actual_hidden == expected_hidden)

    print()
    if grid_ok and n_ok and h_ok:
        print("[OK] All shapes match decoder_model.py assumptions.")
        print("     → Ready to write train_decoder.py.")
    else:
        print("[MISMATCH] At least one shape differs from assumption.")
        print("           decoder_model.py needs adjustment.")


if __name__ == "__main__":
    main()
