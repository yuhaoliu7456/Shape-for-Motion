import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2_video_predictor
import sys




my_video = sys.argv[1]
save_root = sys.argv[2]
x1 = sys.argv[3] 
y1 = sys.argv[4]
x2 = sys.argv[5]
y2 = sys.argv[6]
checkpoint = sys.argv[7]

model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

    inference_state = predictor.init_state(video_path=my_video)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    """
    TODO:
    add a mask as the initial input to the model!
    """
    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array([[x1, y1]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )


    # sending all clicks (and their labels) to `add_new_points_or_box`
    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )


    # propagate the prompts to get masklets throughout the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

    os.makedirs(save_root, exist_ok=True)

    img_list = os.listdir(my_video)
    # img_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for i in range(len(video_segments)):
        for out_obj_id, out_mask in video_segments[i].items():
            out_mask = out_mask.transpose((1, 2, 0)).astype(np.uint8)
            cv2.imwrite(os.path.join(save_root, img_list[i]), out_mask*255)
            