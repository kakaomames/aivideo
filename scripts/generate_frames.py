import os
import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import json

# --- 環境変数とパラメータ設定 ---
ACTION_PROMPT = os.getenv("PROMPT", "blinking, hair blowing in the wind")
CHAR_PROMPT = os.getenv("CHARACTER_PROMPT", "Koishi Komeiji, Touhou Project, green hair, black hat with ribbon, third eye, anime style")
SECONDS = int(os.getenv("SECONDS", "2"))
IMAGE_URL = os.getenv("IMAGE_URL")

FPS = 5
FRAMES = SECONDS * FPS
OUTPUT_DIR = "output_frames"
IMAGE_SIZE = 256

# プロンプトの合体とネガティブプロンプト
FULL_PROMPT = f"{CHAR_PROMPT}, {ACTION_PROMPT}, masterpiece, high quality"
NEGATIVE_PROMPT = "noise, blurry, low quality, distorted, bad anatomy, deformed, noise background"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 画像ダウンロード
print(f"Downloading start image from: {IMAGE_URL}")
try:
    response = requests.get(IMAGE_URL, timeout=10)
    init_image = Image.open(BytesIO(response.content)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
except Exception as e:
    print(f"Failed to download image: {e}")
    exit(1)

# 2. パイプライン準備
model_id = "runwayml/stable-diffusion-v1-5"
print("Loading model...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    safety_checker=None
).to("cpu")

pipe.enable_attention_slicing()

# 3. 生成ループ
print(f"Generating {FRAMES} frames...")
current_image = init_image
current_image.save(f"{OUTPUT_DIR}/frame_000.png")

for i in range(1, FRAMES):
    # strength=0.35, steps=12 でキャラクターを極力維持しつつ動かす
    current_image = pipe(
        prompt=FULL_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=current_image, 
        strength=0.35, 
        num_inference_steps=12
    ).images[0]
    
    current_image.save(f"{OUTPUT_DIR}/frame_{i:03d}.png")
    print(f"Saved frame {i}/{FRAMES}")

# 4. メタデータ保存
meta = {
    "width": IMAGE_SIZE,
    "height": IMAGE_SIZE,
    "fps": FPS,
    "total_frames": FRAMES,
    "prompt": FULL_PROMPT
}
with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f)

print("All done!")
