# scripts/generate_frames.py

import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import json
import zipfile

# 最小限の設定
PROMPT = os.getenv("PROMPT", "a cat running")
SECONDS = int(os.getenv("SECONDS", "2"))
FPS = 5  # とりあえず低めで試す（ブラウザで補間するため）
FRAMES = SECONDS * FPS
OUTPUT_DIR = "output_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. モデルロード (CPU設定)
# 軽量なモデルIDにするのが鍵です（今回は例としてSD 1.5の軽量版など）
model_id = "runwayml/stable-diffusion-v1-5" 
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32, safety_checker=None
).to("cpu")

# 2. 生成ループ (Img2Imgで連続性維持)
init_image = None
for i in range(FRAMES):
    # プロンプトを少しずつ変化させると動きが出る
    current_prompt = f"{PROMPT}, frame {i} of {FRAMES}"
    
    if init_image is None:
        # 最初の1枚はランダムノイズから（Text2Imgの代わり）
        init_image = pipe(current_prompt, strength=1.0, num_inference_steps=5).images[0]
    else:
        # 2枚目以降は前の画像をベースにする (Img2Img)
        init_image = pipe(current_prompt, image=init_image, strength=0.6, num_inference_steps=5).images[0]
    
    init_image.save(f"{OUTPUT_DIR}/frame_{i:03d}.png")

# 3. メタデータの作成
meta = {
    "width": 512, # 生成サイズに合わせて調整
    "height": 512,
    "fps": FPS,
    "total_frames": FRAMES
}
with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f)

print("Generation Complete!")
