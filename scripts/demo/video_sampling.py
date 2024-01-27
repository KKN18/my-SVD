import os

from pytorch_lightning import seed_everything

from scripts.demo.streamlit_helpers import *

SAVE_PATH = "outputs/demo/vid/"

VERSION2SPECS = {
    "svd": {
        "T": 14,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 25,
        },
    },
    "svd_image_decoder": {
        "T": 14,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image_decoder.yaml",
        "ckpt": "checkpoints/svd_image_decoder.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 25,
        },
    },
    "svd_xt": {
        "T": 25,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd_xt.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "min_cfg": 1.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 30,
            "decoding_t": 14,
        },
    },
    "svd_xt_image_decoder": {
        "T": 25,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image_decoder.yaml",
        "ckpt": "checkpoints/svd_xt_image_decoder.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "min_cfg": 1.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 30,
            "decoding_t": 14,
        },
    },
}

import numpy as np
import torch
import torchvision.transforms as TT
from PIL import Image
import os

# Define a function to get the resizing factor for maintaining aspect ratio
def get_resizing_factor(target_size, image_size):
    h, w = target_size
    th, tw = image_size
    return min(h / th, w / tw)

# Function to process an image for prediction
def load_img_for_prediction_custom(W, H, image, device="cuda"):
    w, h = image.size

    image = np.array(image).transpose(2, 0, 1)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)

    rfs = get_resizing_factor((H, W), (h, w))
    resize_size = [int(np.ceil(rfs * s)) for s in (h, w)]
    top = (resize_size[0] - H) // 2
    left = (resize_size[1] - W) // 2

    image = torch.nn.functional.interpolate(
        image, resize_size, mode="area", antialias=False
    )
    image = TT.functional.crop(image, top=top, left=left, height=H, width=W)
    return image.to(device) * 2.0 - 1.0

# Function to load images in order and process them for prediction
def load_and_process_images(dir_path, W, H, device="cuda"):
    # Load and process images from dir_path in sorted order
    img_names = sorted([img for img in os.listdir(dir_path) if img.endswith('.jpg')])
    img_list = []
    for img_name in img_names:
        img_path = os.path.join(dir_path, img_name)
        img = Image.open(img_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        processed_img = load_img_for_prediction_custom(W, H, img, device)
        img_list.append(processed_img)

    return img_list

def load_and_process_images_test(dir_path, W, H, device="cuda"):
    # Load and process images from dir_path in sorted order
    img_names = sorted([img for img in os.listdir(dir_path) if img.endswith('.jpg')])

    # 짝수 프레임만 선택
    even_frame_img_names = [img for img in img_names if int(img.split('_')[1].split('.')[0]) % 2 == 0]

    # img_list 초기화
    img_list = []
    count = 0

    for img_name in even_frame_img_names:
        if count >= 14:
            break

        img_path = os.path.join(dir_path, img_name)
        img = Image.open(img_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        processed_img = load_img_for_prediction_custom(W, H, img, device)
        
        # 각 이미지를 두 번씩 추가
        img_list.extend([processed_img, processed_img])
        count += 2

    return img_list


if __name__ == "__main__":
    st.title("Stable Video Diffusion")
    version = st.selectbox(
        "Model Version",
        [k for k in VERSION2SPECS.keys()],
        0,
    )
    version_dict = VERSION2SPECS[version]
    if st.checkbox("Load Model"):
        mode = "img2vid"
    else:
        mode = "skip"

    H = st.sidebar.number_input(
        "H", value=version_dict["H"], min_value=64, max_value=2048
    )
    W = st.sidebar.number_input(
        "W", value=version_dict["W"], min_value=64, max_value=2048
    )
    T = st.sidebar.number_input(
        "T", value=version_dict["T"], min_value=0, max_value=128
    )

    C = version_dict["C"]
    F = version_dict["f"]
    options = version_dict["options"]

    if mode != "skip":
        state = init_st(version_dict, load_filter=True)
        if state["msg"]:
            st.info(state["msg"])
        model = state["model"]

        ukeys = set(
            get_unique_embedder_keys_from_conditioner(state["model"].conditioner)
        )

        value_dict = init_embedder_options(
            ukeys,
            {},
        )

        value_dict["image_only_indicator"] = 0

        if mode == "img2vid":
            img = load_img_for_prediction(W, H)
            
            cond_aug = st.number_input(
                "Conditioning augmentation:", value=0.02, min_value=0.0
            )
            test_mode = st.radio('Choose test mode:', (True, False))
            target_dir = "../workspace/"
            directories = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
            category = st.selectbox('Choose a category:', directories)
            
            value_dict["cond_frames_without_noise"] = img
            value_dict["cond_frames"] = img + cond_aug * torch.randn_like(img)
            value_dict["cond_aug"] = cond_aug

            if test_mode:

                # Use the directory where all the images are stored
                dir_path = f"/home/nas2_userG/junhahyung/kkn/workspace/{category}"
                img_list = load_and_process_images(dir_path, W, H)
                print(f"img_list count: {len(img_list)}")

                value_dict["cond_frames_without_noise"] = img_list
                value_dict["cond_frames"] = [img + cond_aug * torch.randn_like(img) for img in img_list]


        seed = st.sidebar.number_input(
            "seed", value=23, min_value=0, max_value=int(1e9)
        )
        seed_everything(seed)

        save_locally, save_path = init_save_locally(
            os.path.join(SAVE_PATH, version), init_value=True
        )

        options["num_frames"] = T

        sampler, num_rows, num_cols = init_sampling(options=options)
        num_samples = num_rows * num_cols

        decoding_t = st.number_input(
            "Decode t frames at a time (set small if you are low on VRAM)",
            value=1,  # options.get("decoding_t", T),
            min_value=1,
            max_value=int(1e9),
        )

        if st.checkbox("Overwrite fps in mp4 generator", False):
            saving_fps = st.number_input(
                f"saving video at fps:", value=value_dict["fps"], min_value=1
            )
        else:
            saving_fps = value_dict["fps"]
        print(f"num_samples: {num_samples}")
        if st.button("Sample"):
            out = do_sample(
                model,
                sampler,
                value_dict,
                num_samples,
                H,
                W,
                C,
                F,
                T=T,
                batch2model_input=["num_video_frames", "image_only_indicator"],
                force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
                force_cond_zero_embeddings=options.get(
                    "force_cond_zero_embeddings", None
                ),
                return_latents=False,
                decoding_t=decoding_t,
                test_mode=test_mode,
            )

            if isinstance(out, (tuple, list)):
                samples, samples_z = out
            else:
                samples = out
                samples_z = None

            if save_locally:
                save_video_as_grid_and_mp4(samples, save_path, T, fps=saving_fps)
