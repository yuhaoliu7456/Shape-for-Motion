import numpy as np
from PIL import Image

from s3.utils import load_images_from_video, apply_mask, binary_mask, random_dilate_masks




def data_preparation(args):    
    # read the original video
    ori_frame = load_images_from_video(args.ori_input_folder, target_size=(args.width, args.height))
    ori_mask = load_images_from_video(args.ori_mask_folder, target_size=(args.width, args.height))
    
    # read the edited video
    edited_geometry = load_images_from_video(args.edited_geometry_folder, target_size=(args.width, args.height))
    edited_texture = load_images_from_video(args.edited_texture_folder, target_size=(args.width, args.height))
    edited_mask = load_images_from_video(args.edited_mask_folder, target_size=(args.width, args.height))
    edited_excluded_mask = load_images_from_video(args.excluded_mask_folder, target_size=(args.width, args.height))
    
    length = len(edited_geometry)
    print("Length of the video: ", length, '----------------')

    ori_frame = ori_frame[:length]
    edited_mask = edited_mask[:length]
    edited_excluded_mask = edited_excluded_mask[:length]
    ori_mask = ori_mask[:length]
    edited_texture = edited_texture[:length]
    

    # convert to numpy and concatenate
    ori_frame =  np.concatenate([np.array(img)[np.newaxis,:,:] for img in ori_frame])         # [num_frames, h, w, c]
    ori_mask = np.concatenate([np.array(img)[np.newaxis,:,:] for img in ori_mask])/255.0    # [num_frames, h, w, c]
    edited_geometry = np.concatenate([np.array(img)[np.newaxis,:,:] for img in edited_geometry]) # [num_frames, h, w, c]
    edited_mask = np.concatenate([np.array(img)[np.newaxis,:,:] for img in edited_mask])/255.0    # [num_frames, h, w, c]
    edited_excluded_mask = np.concatenate([np.array(img)[np.newaxis,:,:] for img in edited_excluded_mask])/255.0    # [num_frames, h, w, c]

    ori_mask = binary_mask(ori_mask)
    edited_mask = binary_mask(edited_mask)
    edited_excluded_mask = binary_mask(edited_excluded_mask)
    

    # # update the edited_mask: dilate the edited_mask
    # 1. Dilation(ori_mask)
    # 2. obtain the union of Dilation(ori_mask) and edited_mask
    # 3. Dilation(ori_mask) - the union
    dilated_ori_mask = random_dilate_masks(ori_mask.transpose((0, 3, 1, 2)), min_dilation=args.min_dilation, max_dilation=args.max_dilation)[:,[0],...] # [f, 1, h, w]
    dilated_ori_mask = dilated_ori_mask.transpose((0,2,3,1))
    intersection = np.logical_and(dilated_ori_mask, edited_mask).astype(np.float32)
    inpaint_mask = dilated_ori_mask - intersection
    inpaint_mask[inpaint_mask > 0.5] = 1 # double check
    inpaint_mask[inpaint_mask <= 0.5] = 0
        


    # remove the excluded mask in inpaint mask and edited mask ----------some cases may need this, some may not, case-by-case
    if args.use_excluded_mask_for_local_edit and edited_excluded_mask.sum() == 0:
        # means no new object is added, so the object is just rotated or deformed, in this case, we concern about the object is not match, and the local region is thus gray
        edited_excluded_mask = edited_mask * ori_mask
        edited_excluded_mask = edited_mask - edited_excluded_mask

    
    inpaint_mask = inpaint_mask * (1 - edited_excluded_mask)
    edited_mask = edited_mask * (1 - edited_excluded_mask)
    
    
    # conduct mask operations
    ori_raw_rgb = ori_frame.copy()
    background_values = apply_mask(values=ori_frame, mask_values=(1 - edited_mask), mask_type=args.mask_bg_type)
    background_values = apply_mask(values=background_values, mask_values=(1 - inpaint_mask), mask_type=args.mask_inpaint_type)
    ori_frame = apply_mask(values=ori_frame, mask_values=ori_mask, mask_type=args.mask_ref_type)
    edited_geometry = apply_mask(values=edited_geometry, mask_values=edited_mask, mask_type=args.mask_guide_type)


    # comvert to pil image 
    ori_frame_pil = [Image.fromarray(img.astype(np.uint8)) for img in ori_frame]
    ori_mask_pil = [Image.fromarray(img.astype(np.uint8).squeeze(-1)* 255) for img in ori_mask]
    edited_geometry_pil = [Image.fromarray(img.astype(np.uint8)) for img in edited_geometry]
    edited_mask_pil = [Image.fromarray(img.astype(np.uint8).squeeze(-1)* 255) for img in edited_mask]
    inpaint_mask_pil = [Image.fromarray(img.astype(np.uint8).squeeze(-1)* 255) for img in inpaint_mask]
    bg_pil = [Image.fromarray(img.astype(np.uint8)) for img in background_values]
    ori_raw_rgb_pil = [Image.fromarray(img.astype(np.uint8)) for img in ori_raw_rgb]
    for i in range(length):
        # merge the coarse texture and the background
        blended_mask = ((edited_mask[i] + edited_excluded_mask[i]) == 1).astype(np.uint8)    
        blended = blended_mask * edited_texture[i] + (1 - blended_mask) * background_values[i]
        bg_pil[i] = Image.fromarray(blended.astype(np.uint8))
    
    
    return {'ori_frame': ori_frame_pil,
            'ori_mask': ori_mask_pil,
            'edited_geometry': edited_geometry_pil,
            'edited_mask': edited_mask_pil,
            'bg': bg_pil,
            'inpaint_mask': inpaint_mask_pil,
            'ori_rgb': ori_raw_rgb_pil}

