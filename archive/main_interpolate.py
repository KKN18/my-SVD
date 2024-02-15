from PIL import Image

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipelineOutput, _append_dims, tensor2vid
from diffusers.utils import load_image, export_to_video

from typing import Callable, Dict, List, Optional, Union

import torch
import os
import time

class MyStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):
    @property
    def do_classifier_free_guidance(self):
        return False
    
    @torch.no_grad()
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image], torch.FloatTensor],
        height: int = 512,
        width: int = 512,
        num_frames: Optional[int] = 7,
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
        
        def check_num_frames(num_frames, images):
            expected_num_frames = len(images) * 2 - 1
            if num_frames != expected_num_frames:
                raise ValueError(f"num_frames should be equal to the number of elements in images list * 2 - 1, expected {expected_num_frames}, got {num_frames}")
        
        check_num_frames(num_frames, images)
        
        image = images[0]
        width, height = image.size
        self.check_inputs(image, height, width)
        batch_size = 1
        self._guidance_scale = max_guidance_scale
        device = self._execution_device
        
        base_image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        print(f"base_image_embeddings: {base_image_embeddings.shape}")
        print(f"base_image_embeddings type: {base_image_embeddings.dtype}")
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
        
        latents = []
        first_iter = True
        print(f"device: {device}")
        for image in images:
            width, height = image.size
            image = self.image_processor.preprocess(image, height=height, width=width).to(device)
            # noise = randn_tensor(image.shape, generator=None, device=device, dtype=image.dtype)
            # image = image + noise_aug_strength * noise

            ele_image_latents = self._encode_vae_image(
                image,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            ele_image_latents = ele_image_latents.to(torch.float32)
            rand_noise = torch.rand_like(ele_image_latents)
            
            timestep_tensor = torch.FloatTensor([timesteps[start_timestep]]).to("cuda:0")
            
            add_noise = False
            if add_noise:
                ele_latents = self.scheduler.add_noise(ele_image_latents, rand_noise, timestep_tensor)
            else:
                ele_latents = ele_image_latents
            
            ele_latents = ele_latents.unsqueeze(1)
            if first_iter == True:
                print(f"image.shape: {image.shape}")
                print(f"ele_image_latents: {ele_image_latents.shape}")
                print(f"ele_latents: {ele_latents.shape}")
                first_iter = False
            
            latents.append(ele_latents)
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        latents = torch.cat(latents, dim=1)
        #-------------Optional------------#
        # latents = self.prepare_latents(
        #     batch_size * num_videos_per_prompt,
        #     num_frames,
        #     num_channels_latents,
        #     height,
        #     width,
        #     base_image_embeddings.dtype,
        #     device,
        #     generator,
        #     latents,
        # )
        #---------------------------------#
        
        alpha = 0.5
        new_latents = []
        for i in range(latents.size(1) - 1):
            prev_latent = latents[:, i]
            next_latent = latents[:, i+1]
            new_latents.append(prev_latent)
            interpolate_latent = alpha * prev_latent + (1 - alpha) * next_latent
            new_latents.append(interpolate_latent)
        new_latents.append(latents[:, -1])
        new_latents = torch.cat(new_latents, dim=1)
        new_latents = new_latents.reshape(latents.shape[0], num_frames, *latents.shape[2:])
        latents = new_latents
        
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                print(f"ㅡ"*20)
                print(f"[INFO] self.do_classifier_free_guidance {self.do_classifier_free_guidance}")
                print(f"[INFO] latents: {latents.shape}")
                
                latent_model_input = self.scheduler.scale_model_input(latents, t)
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

                # if self.do_classifier_free_guidance:
                #     noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                #     noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

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
        
        print(f"output_type: {output_type}")
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
    width, height = image.size
    pipe.check_inputs(image, height, width)
    images.append(image)

generator = torch.manual_seed(42)
start_timestep = 15
motion_bucket_id = 127
frames = pipe(images, decode_chunk_size=1, num_frames=7, generator=generator, start_timestep=start_timestep, motion_bucket_id=motion_bucket_id).frames[0]

save_path = f"/home/nas2_userG/junhahyung/kkn/my-svd/outputs/main_interpolate/timestep_{start_timestep}_motion_{motion_bucket_id}"
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
