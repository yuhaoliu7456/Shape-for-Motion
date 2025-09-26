import os
import sys
from glob import glob
from typing import List, Optional, Union

from tqdm import tqdm

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import torch, torchvision
from fire import Fire

from sgm.modules.encoders.modules import VideoPredictionEmbedderWithEncoder
from scripts.demo.sv4d_helpers import (
    decode_latents,
    load_model,
    initial_model_load,
    read_video,
    run_img2vid,
    prepare_sampling,
    prepare_inputs,
    do_sample_per_step,
    sample_sv3d,
    save_video,
    preprocess_video,
    save_numpy_img
)


def sample(
    input_path: str = "assets/test_video.mp4",  # Can either be image file or folder with image files
    output_folder: Optional[str] = "outputs/sv4d",
    num_steps: Optional[int] = 20,
    sv3d_version: str = "sv3d_u",  # sv3d_u or sv3d_p
    img_size: int = 576, # image resolution
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 1e-5,
    seed: int = 23,
    encoding_t: int = 8,  # Number of frames encoded at a time! This eats most VRAM. Reduce if necessary.
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    elevations_deg: Optional[Union[float, List[float]]] = 10.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = 0.917,
    verbose: Optional[bool] = False,
    remove_bg: bool = False,
    n_frames: int = 21,
    ckpt_path=None,
):
    """
    Simple script to generate multiple novel-view videos conditioned on a video `input_path` or multiple frames, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t` and `encoding_t`.
    """
    # Set model config
    T = 5  # number of frames per sample
    V = 8  # number of views per sample
    subsampled_views = np.array([0, 2, 5, 7, 9, 12, 14, 16, 19])  # subsample (V+1=)9 (uniform) views from 21 SV3D views
    F = 8  # vae factor to downsize image->latent
    C = 4
    H, W = img_size, img_size
    n_views = V + 1  # number of output video views (1 input view + 8 novel views)
    n_views_sv3d = 21


    torch.manual_seed(seed)
    os.makedirs(output_folder, exist_ok=True)

    # Read input video frames i.e. images at view 0
    print(f"Reading {input_path}")

    images_v0 = read_video(input_path, n_frames=len(os.listdir(input_path)), device=device)

    # Get camera viewpoints
    if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
        elevations_deg = [elevations_deg] * n_views_sv3d
    assert (
        len(elevations_deg) == n_views_sv3d
    ), f"Please provide 1 value, or a list of {n_views_sv3d} values for elevations_deg! Given {len(elevations_deg)}"
    
    if azimuths_deg is None:
        azimuths_deg = np.linspace(0, 360, n_views_sv3d + 1)[1:] % 360

    assert (
        len(azimuths_deg) == n_views_sv3d
    ), f"Please provide a list of {n_views_sv3d} values for azimuths_deg! Given {len(azimuths_deg)}"
    
    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])

    
    azimuths_rad = np.array(
        [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    )

    
    # Sample multi-view images of the first frame using SV3D i.e. images at time 0
    for index in range(len(images_v0)):
        # for index in range(60, 80):
        save_video_path = os.path.join(output_folder,  'novelviews', f"{index:05d}_multiview.mp4")
        if os.path.exists(save_video_path):
            continue
        print(f"Sampling for index {index}")
        images_t0 = sample_sv3d(
            images_v0[index],
            n_views_sv3d,
            num_steps,
            sv3d_version,
            fps_id,
            motion_bucket_id,
            cond_aug,
            decoding_t,
            device,
            polars_rad,
            azimuths_rad,
            verbose,
            ckpt_path=ckpt_path
        )
        images_t0 = torch.roll(images_t0, 1, 0)  # move conditioning image to first frame, and thus each view has the same first frame!!!!
        os.makedirs(os.path.join(output_folder, 'novelviews', f"{index:05d}", "rgbs/p_0"), exist_ok=True)
        # save_video(os.path.join(output_folder,  'novelviews', f"{index:05d}_multiview.mp4"), [images_t0[i].unsqueeze(0) for i in range(n_views_sv3d)])
        print(os.path.join(output_folder,  'novelviews', f"{index:05d}_multiview.mp4"))
        for i in range(n_views_sv3d):
            torchvision.utils.save_image((images_t0[i]+1)/2., os.path.join(output_folder,  'novelviews', f"{index:05d}", "rgbs/p_0", 'sv3d_'+str(i)+'.png'))

    exit()

    


if __name__ == "__main__":
    """
    This file is transformed from the simple_video_sample_4d.py script.
    """
    Fire(sample)
