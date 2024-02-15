from PIL import Image

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipelineOutput, _append_dims, tensor2vid
from diffusers.utils import load_image, export_to_video

from typing import Callable, Dict, List, Optional, Union

import torch
import os

class MyStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image], torch.FloatTensor],
        height: int = 512,
        width: int = 512,
        num_frames: Optional[int] = 4,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = 1,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        start_timestep: int = 20,
    ):
        image = images[0]
        width, height = image.size
        self.check_inputs(image, height, width)
        batch_size = 1
        self._guidance_scale = max_guidance_scale
        device = self._execution_device
        
        base_image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        print(f"base_image_embeddings: {base_image_embeddings.shape}")
        fps = fps - 1
        start_timestep_idx = start_timestep

        base_image = self.image_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(base_image.shape, generator=generator, device=device, dtype=base_image.dtype)
        base_image = base_image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        print(f"need_upcasting: {needs_upcasting}")
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        print(f"base_image.shape: {base_image.shape}")
        base_image_latents = self._encode_vae_image(
            base_image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        base_image_latents = base_image_latents.to(base_image_embeddings.dtype)
        print(f"base_image_latents: {base_image_latents.shape}")
            
        base_image_latents = base_image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            base_image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        num_channels_latents = self.unet.config.in_channels
        
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            base_image_embeddings.dtype,
            device,
            generator,
            latents=None,
        )
        
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            progress_bar.update(start_timestep_idx)
            for i, t in enumerate(timesteps[start_timestep_idx:], start=start_timestep_idx):
                print(f"ㅡ"*20)
                print(f"[INFO] self.do_classifier_free_guidance {self.do_classifier_free_guidance}")
                print(f"[INFO] latents: {latents.shape}")
                
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                print(f"[INFO] latent_model_input: {latent_model_input.shape}")

                latent_model_input = torch.cat([latent_model_input, base_image_latents], dim=2)
                print(f"[INFO] latent_model_input: {latent_model_input.shape}")

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=base_image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]
                print(f"[INFO] noise_pred: {noise_pred.shape}")

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                print(f"ㅡ"*20)
        
        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)

pipe = MyStableVideoDiffusionPipeline.from_pretrained(
    "/home/nas2_userG/junhahyung/kkn/stable-video-diffusion-img2vid", torch_dtype=torch.float32
)
pipe.enable_sequential_cpu_offload()

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
frames = pipe(images, decode_chunk_size=1, generator = torch.manual_seed(42), num_frames=4, start_timestep=0).frames[0]

save_path = "/home/nas2_userG/junhahyung/kkn/my-svd/outputs"
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
print("Frame saved")
# export_to_video(frames, save_path + "/generated.mp4", fps=7)
