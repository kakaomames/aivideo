import os
import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import json

# --- 設定値 ---
PROMPT = os.getenv("PROMPT", "a cinematic shot of a running cat")
SECONDS = int(os.getenv("SECONDS", "2"))
IMAGE_URL = os.getenv("IMAGE_URL") # 必須：ここからスタート画像を取得
FPS = 5 # 30fpsを作る前の素材なので5で十分！
FRAMES = SECONDS * FPS
OUTPUT_DIR = "output_frames"
IMAGE_SIZE = 256 # CPUで爆速生成するためのサイズ（ブラウザ側で拡大します）

# ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. スタート画像のダウンロードと準備
print(f"Downloading start image from: {IMAGE_URL}")
try:
    response = requests.get(IMAGE_URL, timeout=10)
    init_image = Image.open(BytesIO(response.content)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
except Exception as e:
    print(f"Failed to download image: {e}")
    exit(1)

# 2. モデルの準備 (CPU環境向け最適化)
# Stable Diffusion 1.5はメモリ消費が比較的安定しています
model_id = "runwayml/stable-diffusion-v1-5"
print("Loading model...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    safety_checker=None
).to("cpu")

# メモリ削減の魔法（CPU生成では必須）
pipe.enable_attention_slicing()

# 3. 画像生成ループ
print(f"Generating {FRAMES} frames...")
current_image = init_image
current_image.save(f"{OUTPUT_DIR}/frame_000.png")

for i in range(1, FRAMES):
    # プロンプトを微調整（動きを出すため）
    current_prompt = f"{PROMPT}, frame {i} of {FRAMES}"
    
    # Img2Img生成 (strength 0.5-0.6 が連続性を保つコツ)
    # num_inference_steps=4 は爆速のため。もっと綺麗にしたければ8に変えてください
    current_image = pipe(
        prompt=current_prompt, 
        image=current_image, 
        strength=0.55, 
        num_inference_steps=4
    ).images[0]
    
    current_image.save(f"{OUTPUT_DIR}/frame_{i:03d}.png")
    print(f"Saved frame {i}/{FRAMES}")

# 4. ブラウザ用の設計図（メタデータ）保存
meta = {
    "width": IMAGE_SIZE,
    "height": IMAGE_SIZE,
    "fps": FPS,
    "total_frames": FRAMES,
    "prompt": PROMPT
}
with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f)

print("All done!")
