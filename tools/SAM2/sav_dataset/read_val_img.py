from megfile import smart_open,smart_listdir, smart_path_join
from PIL import Image
import cv2
import numpy as np
import os

data_root = 's3+AIGC_GENERAL://public-dataset-p2/en-public-sam-video/sav_val' # /sav_052661
rgb_prefix = "JPEGImages_24fps"
mask_prefix = "Annotations_6fps"
rgb_root = smart_path_join(data_root, rgb_prefix)
mask_root = smart_path_join(data_root, mask_prefix)
video_list = smart_listdir(rgb_root)

for video in video_list:
    video_path = smart_path_join(rgb_root, video)
    image_list = smart_listdir(video_path) # all are .jpg
    first_frame = Image.open(smart_open(smart_path_join(video_path, image_list[0]), 'rb'))
    first_frame = np.array(first_frame)
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)

    mask_path = smart_path_join(mask_root, video)
    mask_folder_list = smart_listdir(mask_path) # ['000', '001', ...]

    for mask_folder in mask_folder_list:
        mask_folder_path = smart_path_join(mask_path, mask_folder)
        mask_list = smart_listdir(mask_folder_path)

        # concate the 1st mask with the 1st frame
        first_mask = Image.open(smart_open(smart_path_join(mask_folder_path, mask_list[0]), 'rb'))
        first_mask = np.array(first_mask)
        first_mask = first_mask[:, :, None]
        first_mask = np.concatenate([first_mask, first_mask, first_mask], axis=2)
        first_mask = first_mask.astype(np.uint8) * 255
        first_frame = np.concatenate([first_frame, first_mask], axis=1)
    cv2.imwrite(f'{video}_val.jpg', first_frame)
    
    # import pdb; pdb.set_trace()
