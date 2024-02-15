import torch
import os
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import tensor2vid
from diffusers.utils import load_image, export_to_video
from diffusers.utils.torch_utils import is_compiled_module

import numpy as np
import time
import inspect

class MyStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):
    @torch.no_grad()
    def test(self, images, num_videos_per_prompt=1, num_frames=4, decode_chunk_size=1):
        device = self._execution_device
        image_latents = []
        base_dtype = torch.float32
        print(f"device: {device}")
        for image in images:
            width, height = image.size
            image = self.image_processor.preprocess(image, height=height, width=width)
            image_latent = self._encode_vae_image(
                image,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=False,
            )
            print(f"image_latent: {image_latent.shape}")
            image_latent = image_latent.to(base_dtype)
            image_latent = image_latent.unsqueeze(1)
            image_latents.append(image_latent)
        image_latents = torch.cat(image_latents, dim=1)
        print(f"image_latents: {image_latents.shape}")
        output_type = "pil"
        
        frames = self.decode_latents(image_latents, num_frames=num_frames, decode_chunk_size=decode_chunk_size)
        print(f"frames: {frames.shape}")
        frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        self.maybe_free_model_hooks()
        
        return frames
    
    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        # latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

pipe = MyStableVideoDiffusionPipeline.from_pretrained(
    "/home/nas2_userG/junhahyung/kkn/stable-video-diffusion-img2vid", torch_dtype=torch.float32
)
pipe.enable_sequential_cpu_offload()

save_path = "/home/nas2_userG/junhahyung/kkn/my-svd/outputs"
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Folder created at {save_path}")
else:
    print(f"Folder already exists at {save_path}")
    
image_paths = [
    '/home/nas2_userG/junhahyung/kkn/workspace/duck/001.jpg',
    '/home/nas2_userG/junhahyung/kkn/workspace/duck/005.jpg',
    '/home/nas2_userG/junhahyung/kkn/workspace/duck/009.jpg',
    '/home/nas2_userG/junhahyung/kkn/workspace/duck/015.jpg'
]

images = []

for image_path in image_paths:
    image = load_image(image_path)
    image = image.resize((512, 512), Image.LANCZOS)
    images.append(image)

generator = torch.manual_seed(42)
frames = pipe.test(images)[0]

for index, frame in enumerate(frames):

    filename = f"frame_{index}.png"
    frame.save(os.path.join(save_path, filename))

# export_to_video(frames, save_path + "/generated.mp4", fps=7)