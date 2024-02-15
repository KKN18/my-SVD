import torch
import os
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

import numpy as np


pipe = StableVideoDiffusionPipeline.from_pretrained(
    "/home/nas2_userG/junhahyung/kkn/stable-video-diffusion-img2vid", torch_dtype=torch.float32, safety_checker=None,
)

# pipe = pipe.to("cuda")
pipe.enable_sequential_cpu_offload()
# pipe.enable_model_cpu_offload()
# pipe.unet.enable_forward_chunking()


    
image_path = "/home/nas2_userG/junhahyung/kkn/workspace/duck/001.jpg"

image = load_image(image_path)
image = image.resize((512, 512), Image.LANCZOS)

generator = torch.manual_seed(42)
motion_bucket_id = 200
frames = pipe(image, decode_chunk_size=1, generator=generator, num_frames=4, motion_bucket_id=motion_bucket_id, height=H, width=W).frames[0]

save_path = f"/home/nas2_userG/junhahyung/kkn/my-svd/outputs/original/motion_{motion_bucket_id}"
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Folder created at {save_path}")
else:
    print(f"Folder already exists at {save_path}")

# frames에 있는 각 이미지를 순회하며 저장합니다.
for index, frame in enumerate(frames):
    
    # 이미지 파일명을 설정합니다. 예: "frame_0.png", "frame_1.png", ...
    filename = f"frame_{index}.png"

    # 이미지를 지정된 경로에 저장합니다.
    frame.save(os.path.join(save_path, filename))

# export_to_video(frames, save_path + "/generated.mp4", fps=7)