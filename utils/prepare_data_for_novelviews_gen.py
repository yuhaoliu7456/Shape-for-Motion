
import os
import numpy as np
from PIL import Image
import cv2
import pickle
from pathlib import Path
import sys


def preprocess_video(input_path, mask_path, depth_path, save_root=None, num_frames=None, W=576, H=576, image_frame_ratio=0.917):
    img_list = os.listdir(input_path)
    if num_frames is None:
        num_frames = len(img_list)
    img_list = sorted(img_list)[:num_frames]
    all_img_paths = [os.path.join(input_path, img) for img in img_list]
    all_mask_paths = [os.path.join(mask_path, img.replace('.jpg', '.png')) for img in img_list]
    all_depth_paths = [os.path.join(depth_path, img.replace('.jpg', '.png')) for img in img_list]

    assert len(all_img_paths) == len(all_mask_paths) == len(all_depth_paths)

    images = [Image.open(img_path).convert('RGB') for img_path in all_img_paths]
    masks = [Image.open(mask_path).convert('L') for mask_path in all_mask_paths]
    depths = [Image.open(depth_path) for depth_path in all_depth_paths]

    name_list = [Path(img_path).stem for img_path in all_img_paths]

    per_frame_params = []
    processed_rgbs = []
    processed_masks = []
    processed_depths = []
    

    # Process each frame individually without scaling
    for idx, (rgb, mask, depth) in enumerate(zip(images, masks, depths)):
        rgb_arr = np.array(rgb)
        mask_arr = np.array(mask) /255
        mask_arr[mask_arr > 0.1] = 255
        mask_arr = mask_arr.astype(np.uint8)
        depth_arr = np.array(depth)

        orig_h, orig_w = rgb_arr.shape[:2]  # Original image dimensions

        # Compute bounding box for the current frame
        ret, thresh = cv2.threshold(mask_arr, 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)

        # Center coordinates of the object
        center_x = x + w // 2
        center_y = y + h // 2

        # Determine box_size using image_frame_ratio
        object_box_size = max(w, h)
        box_size = int(object_box_size / image_frame_ratio)

        # Ensure box_size is at least as big as the object
        box_size = max(box_size, object_box_size)

        # Create a blank canvas of size (box_size, box_size) to center the object
        canvas_rgb = np.zeros((box_size, box_size, 3), dtype=np.uint8)
        canvas_mask = np.zeros((box_size, box_size), dtype=np.uint8)
        canvas_depth = np.zeros((box_size, box_size, 3), dtype=np.uint8)

        # Compute the offset to center the object in the new box_size
        center = box_size // 2
        offset_x = center - center_x
        offset_y = center - center_y

        # Compute start and end positions on the canvas
        start_x = max(offset_x, 0)
        end_x = min(offset_x + orig_w, box_size)
        start_y = max(offset_y, 0)
        end_y = min(offset_y + orig_h, box_size)

        # Compute the corresponding region on the original images
        img_start_x = max(-offset_x, 0)
        img_end_x = img_start_x + (end_x - start_x)
        img_start_y = max(-offset_y, 0)
        img_end_y = img_start_y + (end_y - start_y)

        # Place the original images onto the canvas
        canvas_rgb[start_y:end_y, start_x:end_x] = rgb_arr[img_start_y:img_end_y, img_start_x:img_end_x]
        canvas_mask[start_y:end_y, start_x:end_x] = mask_arr[img_start_y:img_end_y, img_start_x:img_end_x]
        canvas_depth[start_y:end_y, start_x:end_x] = depth_arr[img_start_y:img_end_y, img_start_x:img_end_x]
        

        # Resize to output dimensions
        final_rgb = cv2.resize(canvas_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        final_mask = cv2.resize(canvas_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        final_depth = cv2.resize(canvas_depth, (W, H), interpolation=cv2.INTER_LINEAR)
        

        # Save processed images
        final_rgb_float = final_rgb.astype(np.float32)
        final_mask_float = final_mask.astype(np.float32) / 255.0
        masked_rgb_float = final_rgb_float * final_mask_float[:, :, None] + (1 - final_mask_float[:, :, None]) * 255.0
        masked_rgb = masked_rgb_float.astype(np.uint8)
        processed_rgbs.append(Image.fromarray(masked_rgb))
        processed_masks.append(Image.fromarray(final_mask))
        processed_depths.append(Image.fromarray(final_depth))
        

        # Store per-frame parameters
        per_frame_params.append({
            'orig_h': orig_h,
            'orig_w': orig_w,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'box_size': box_size,
            'center': center,
            'start_x': start_x,
            'end_x': end_x,
            'start_y': start_y,
            'end_y': end_y,
            'img_start_x': img_start_x,
            'img_end_x': img_end_x,
            'img_start_y': img_start_y,
            'img_end_y': img_end_y,
        })

        # Save images if required
        if save_root is not None:
            # Ensure directories exist
            os.makedirs(os.path.join(save_root, 'processed_rgbs'), exist_ok=True)
            os.makedirs(os.path.join(save_root, 'processed_masks'), exist_ok=True)
            os.makedirs(os.path.join(save_root, 'processed_depths'), exist_ok=True)

            # Save RGB image
            processed_rgbs[-1].save(os.path.join(save_root, 'processed_rgbs', name_list[idx] + '.png'))

            # Save mask image
            processed_masks[-1].save(os.path.join(save_root, 'processed_masks', name_list[idx] + '.png'))

            # Save depth image
            processed_depths[-1].save(os.path.join(save_root, 'processed_depths', name_list[idx] + '.png'))



    # Save transformation parameters
    if save_root is not None:
        with open(os.path.join(save_root, 'transform_params.pkl'), 'wb') as f:
            pickle.dump({'per_frame_params': per_frame_params}, f)


input_path =sys.argv[1]
mask_path = sys.argv[2]
depth_path = sys.argv[3]
save_root = sys.argv[4]
image_frame_ratio = float(sys.argv[5])

preprocess_video(
    input_path=input_path,
    mask_path=mask_path,
    depth_path=depth_path,
    save_root=save_root,
    num_frames=None,
    W=576,
    H=576,
    image_frame_ratio=image_frame_ratio
)
