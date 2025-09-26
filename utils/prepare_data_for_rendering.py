import os
import sys
import numpy as np
from PIL import Image
import cv2
import pickle
import imageio
import yaml

def reverse_preprocess_video(processed_rgb_path, processed_mask_path, processed_mask2_path, processed_depth_path, transform_params_path, 
                             raw_rgb_path, raw_mask_path, 
                             save_root, save_name, start_index=0):

    # Load transformation parameters
    with open(transform_params_path, 'rb') as f:
        data = pickle.load(f)
    per_frame_params = data['per_frame_params'][start_index:]

    # Get list of image names
    if os.path.exists(raw_rgb_path.replace('raw-rgbs', 'raw-rgbs2')):
        raw_rgb_path = raw_rgb_path.replace('raw-rgbs', 'raw-rgbs2')
    img_list = os.listdir(raw_rgb_path)
    img_list = sorted(img_list)[start_index:] if start_index<len(img_list) else print('start_index is out of range')
    name_list = [os.path.splitext(os.path.basename(img))[0] for img in img_list]

    # Paths to processed RGB, mask, depth images, and raw RGB images
    if os.path.isdir(processed_rgb_path):
        all_processed_rgb_paths = [os.path.join(processed_rgb_path, name + '.png') for name in name_list]
    elif os.path.isfile(processed_rgb_path) and '.mp4' in processed_rgb_path:
        from decord import VideoReader, cpu
        vr = VideoReader(processed_rgb_path, ctx=cpu(0))
        processed_rgbs = [Image.fromarray(vr[i].asnumpy()).convert('RGB') for i in range(len(vr))]

    if os.path.isdir(processed_mask_path):
        all_processed_mask_paths = [os.path.join(processed_mask_path, name + '.png') for name in name_list]
    elif os.path.isfile(processed_mask_path) and '.mp4' in processed_mask_path:
        from decord import VideoReader, cpu
        vr = VideoReader(processed_mask_path, ctx=cpu(0))
        processed_masks = [Image.fromarray(vr[i].asnumpy()).convert('L') for i in range(len(vr))]
    
    if os.path.isdir(processed_mask2_path):
        all_processed_mask2_paths = [os.path.join(processed_mask2_path, name + '.png') for name in name_list]
    elif os.path.isfile(processed_mask2_path) and '.mp4' in processed_mask2_path:
        from decord import VideoReader, cpu
        vr = VideoReader(processed_mask2_path, ctx=cpu(0))
        processed_masks2 = [Image.fromarray(vr[i].asnumpy()).convert('L') for i in range(len(vr))]
    
    if os.path.isdir(processed_depth_path):
        all_processed_depth_paths = [os.path.join(processed_depth_path, name + '.png') for name in name_list]
    elif os.path.isfile(processed_depth_path) and '.mp4' in processed_depth_path:
        from decord import VideoReader, cpu
        vr = VideoReader(processed_depth_path, ctx=cpu(0))
        processed_depths = [Image.fromarray(vr[i].asnumpy()) for i in range(len(vr))]

    # Load processed RGB images, masks, depths, and raw RGB images
    if os.path.isdir(processed_rgb_path):
        processed_rgbs = [Image.open(img_path).convert('RGB') for img_path in all_processed_rgb_paths]
    if os.path.isdir(processed_mask_path):
        processed_masks = [Image.open(mask_path).convert('L') for mask_path in all_processed_mask_paths]
    if os.path.isdir(processed_mask2_path):
        processed_masks2 = [Image.open(mask_path).convert('L') for mask_path in all_processed_mask2_paths]

    if os.path.isdir(processed_depth_path):
        processed_depths = [Image.open(depth_path) for depth_path in all_processed_depth_paths]

    all_raw_rgb_paths = [os.path.join(raw_rgb_path, name + '.jpg') for name in name_list]
    try:
        raw_rgbs = [Image.open(img_path.replace('.jpg', '.png')).convert('RGB') for img_path in all_raw_rgb_paths]
    except:
        raw_rgbs = [Image.open(img_path).convert('RGB') for img_path in all_raw_rgb_paths]
    all_raw_mask_paths = [os.path.join(raw_mask_path, name + '.png') for name in name_list]
    raw_masks = [Image.open(mask_path).convert('L') for mask_path in all_raw_mask_paths]
    
    reconstructed_masks = []
    reconstructed_masks2 = []
    reconstructed_depths = []
    reconstructed_rgbs = []

    for idx, (proc_rgb, proc_mask, proc_mask2, depth, raw_rgb, raw_mask) in enumerate(zip(processed_rgbs, processed_masks, processed_masks2, processed_depths, raw_rgbs, raw_masks)):
        frame_params = per_frame_params[idx]
        orig_h = frame_params['orig_h']
        orig_w = frame_params['orig_w']
        offset_x = frame_params['offset_x']
        offset_y = frame_params['offset_y']
        box_size = frame_params['box_size']
        center = frame_params['center']
        start_x = frame_params['start_x']
        end_x = frame_params['end_x']
        start_y = frame_params['start_y']
        end_y = frame_params['end_y']
        img_start_x = frame_params['img_start_x']
        img_end_x = frame_params['img_end_x']
        img_start_y = frame_params['img_start_y']
        img_end_y = frame_params['img_end_y']

        # Step 1: Resize processed images back to box_size
        proc_rgb_arr = np.array(proc_rgb.resize((box_size, box_size), Image.LANCZOS))
        proc_mask_arr = np.array(proc_mask.resize((box_size, box_size), Image.NEAREST))
        proc_mask2_arr = np.array(proc_mask2.resize((box_size, box_size), Image.NEAREST))
        depth_arr = np.array(depth.resize((box_size, box_size), Image.LANCZOS))
        
        # Load raw RGB image
        raw_rgb_arr = np.array(raw_rgb)
        if raw_rgb_arr.shape[:2] != (orig_h, orig_w):
            raw_rgb_arr = cv2.resize(raw_rgb_arr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Load raw mask image
        raw_mask_arr = np.array(raw_mask)
        raw_mask_arr = raw_mask_arr[:,:, np.newaxis].repeat(3, axis=2)
        if raw_mask_arr.shape[:2] != (orig_h, orig_w):
            raw_mask_arr = cv2.resize(raw_mask_arr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        raw_mask_arr[raw_mask_arr > 0] = 255
        # mask the raw image with the raw mask
        raw_rgb_arr = raw_rgb_arr * (1 - raw_mask_arr / 255.0) + raw_mask_arr
        raw_rgb_arr = raw_rgb_arr.astype(np.uint8)

        # Initialize reconstructed_rgb with raw RGB image
        reconstructed_rgb = raw_rgb_arr.copy()
        reconstructed_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        reconstructed_mask2 = np.zeros((orig_h, orig_w), dtype=np.uint8)
        reconstructed_depth = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        # Define slices for the reconstructed image
        img_region_y = slice(img_start_y, img_end_y)
        img_region_x = slice(img_start_x, img_end_x)
        
        # Define slices for the object region
        obj_region_y = slice(start_y, end_y)
        obj_region_x = slice(start_x, end_x)

        # Extract object regions
        object_region_rgb = proc_rgb_arr[obj_region_y, obj_region_x]
        object_region_mask = proc_mask_arr[obj_region_y, obj_region_x]
        object_region_mask2 = proc_mask2_arr[obj_region_y, obj_region_x]
        object_region_depth = depth_arr[obj_region_y, obj_region_x]

        # Normalize the mask to [0, 1]
        alpha_mask = object_region_mask / 255.0
        alpha_mask = np.expand_dims(alpha_mask, axis=2)  # Shape (H, W, 1)

        # Since the background in masked RGB is white, we need to recover the original RGB values
        # Adjust the object_region_rgb to remove the white background effect
        object_region_rgb = (object_region_rgb - (1 - alpha_mask) * 255.0) / (alpha_mask + 1e-6)
        object_region_rgb = np.clip(object_region_rgb, 0, 255).astype(np.uint8)

        # Alpha blending to overlay the object onto the background
        reconstructed_rgb[img_region_y, img_region_x] = (
            object_region_rgb * alpha_mask + reconstructed_rgb[img_region_y, img_region_x] * (1 - alpha_mask)
        ).astype(np.uint8)

        # Update the reconstructed mask and depth
        reconstructed_mask[img_region_y, img_region_x] = object_region_mask
        reconstructed_mask2[img_region_y, img_region_x] = object_region_mask2
        reconstructed_depth[img_region_y, img_region_x] = (object_region_depth * alpha_mask).astype(np.uint8)

        # Save reconstructed images
        reconstructed_depths.append(reconstructed_depth)
        reconstructed_masks.append(reconstructed_mask)
        reconstructed_masks2.append(reconstructed_mask2)
        reconstructed_rgbs.append(reconstructed_rgb)

    os.makedirs(os.path.join(save_root, "edited-normal"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "edited-mask"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "edited-rgb"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "excluded-mask"), exist_ok=True)
    

    with imageio.get_writer(os.path.join(save_root, "edited-normal", save_name), fps=24, codec="libx264") as writer:
        for img in reconstructed_depths:
            writer.append_data(img)

    with imageio.get_writer(os.path.join(save_root, "edited-mask", save_name), fps=24, codec="libx264") as writer:
        for img in reconstructed_masks:
            writer.append_data(img)
    
    with imageio.get_writer(os.path.join(save_root, "excluded-mask", save_name), fps=24, codec="libx264") as writer:
        for img in reconstructed_masks2:
            writer.append_data(img)
    
    with imageio.get_writer(os.path.join(save_root, "edited-rgb", save_name), fps=24, codec="libx264") as writer:
        for img in reconstructed_rgbs:
            writer.append_data(img)
    
    
    raw_rgbs = [np.array(img) for img in raw_rgbs]
    raw_masks_out = []
    for mask in raw_masks:
        mask = np.array(mask)
        mask = mask[:,:, np.newaxis].repeat(3, axis=2)
        mask[mask > 0] = 255
        raw_masks_out.append(mask)

    
    os.makedirs(os.path.join(save_root, "ori-rgb"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "ori-mask"), exist_ok=True)
    
    with imageio.get_writer(os.path.join(save_root, "ori-rgb", save_name.split('.')[0]+'.mp4'), fps=24, codec="libx264") as writer:
        for img in raw_rgbs[:len(reconstructed_rgbs)]:
            writer.append_data(img)
    
    with imageio.get_writer(os.path.join(save_root, "ori-mask", save_name.split('.')[0]+'.mp4'), fps=24, codec="libx264") as writer:
        for img in raw_masks_out[:len(reconstructed_rgbs)]:
            writer.append_data(img)
    


if __name__ == "__main__":
    ori_root=sys.argv[1]
    save_name=sys.argv[2]
    edited_root=sys.argv[3]
    save_root=sys.argv[4]
    config_path=sys.argv[5] # This is because that not all videos are started from the first frame.
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if 'time_index' in config:
        time_index = config['time_index']
        start_index = int(time_index.split('-')[0]) if '-' in time_index else 0
    else:
        start_index = 0
    print(f"start_index: {start_index}")

    save_name = save_name.split('.')[0]+'.mp4'


    ####
    os.makedirs(edited_root, exist_ok=True)
    ####

    reverse_preprocess_video(
        processed_rgb_path=os.path.join(edited_root, 'mesh_image.mp4'),
        processed_mask_path=os.path.join(edited_root, 'mask.mp4'),
        processed_mask2_path=os.path.join(edited_root, 'mask_of_new_obj.mp4'),
        processed_depth_path=os.path.join(edited_root, 'normal.mp4'),
        transform_params_path=os.path.join(ori_root, 'transform_params.pkl'),
        raw_rgb_path=os.path.join(ori_root, 'raw-rgbs'),  # Path to the raw RGB images
        raw_mask_path=os.path.join(ori_root, 'raw-masks'),  # Path to the raw mask images
        save_root=save_root,
        save_name=save_name,
        start_index=start_index
    )
