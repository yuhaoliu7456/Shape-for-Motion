import os
import cv2
import random
import numpy as np
from PIL import Image
from einops import rearrange
import torch



def tensor_to_vae_latent(t, vae, scale=True):
    t = t.to(vae.dtype)
    if len(t.shape) == 5:
        video_length = t.shape[1]

        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    elif len(t.shape) == 4:
        latents = vae.encode(t).latent_dist.sample()
    if scale:
        latents = latents * vae.config.scaling_factor

    return latents

def random_dilate_masks(combined_mask, min_dilation=1, max_dilation=5):
 
    if isinstance(combined_mask, torch.Tensor):
        combined_mask_np = combined_mask.cpu().numpy()
    elif isinstance(combined_mask, np.ndarray):
        combined_mask_np = combined_mask
    f, c, h, w = combined_mask_np.shape
    dilated_masks = np.zeros_like(combined_mask_np)
    
    for i in range(f):
        mask = combined_mask_np[i]
        mask_binary = (mask > 0.5).astype(np.uint8)
        dilation_size = random.randint(min_dilation, max_dilation)
        kernel_size = 2 * dilation_size + 1  
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(mask_binary.squeeze(), kernel, iterations=1)

        dilated_mask = dilated_mask.astype(np.float32)
        dilated_mask = np.clip(dilated_mask, 0, 1)

        dilated_masks[i] = dilated_mask[None, ...]

    if isinstance(combined_mask, torch.Tensor):
        dilated_masks_tensor = torch.from_numpy(dilated_masks).to(combined_mask.device).type(combined_mask.dtype)
    else:
        dilated_masks_tensor = dilated_masks

    return dilated_masks_tensor

def binary_mask(mask):
    mask = mask.mean(axis=-1, keepdims=True) # [num_frames, h, w, 1]
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    mask = mask.astype(np.uint8)
    return mask

def apply_mask(values, mask_values, mask_type):
    """
    Inputs are all numpy arrays
    values (np.ndarray): Pixel values of the image, shape (H, W, C), range [0, 255]
    mask_values (np.ndarray): Mask values, shape (H, W), range [0, 1]
    """
    assert values.shape[:2] == mask_values.shape[:2], "Values and mask values must have the same shape"
    
    if mask_type == 'random':
        return values * mask_values + (1 - mask_values) * np.random.randint(0, 256, values.shape)
    elif mask_type == 'gray':
        return values * mask_values + (1 - mask_values) * 128
    elif mask_type == 'black':
        return values * mask_values
    elif mask_type == 'white':
        return values * mask_values + (1 - mask_values) * 255
    elif mask_type == 'none':
        return values
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

def load_images_from_video(video_path, target_size=(512, 512)):
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), width=target_size[0], height=target_size[1])
    images = []
    for i in range(len(vr)):
        img = vr[i].asnumpy()
        img = Image.fromarray(img)
        images.append(img)

    return images