
import numpy as np
import cv2
from PIL import Image
import os
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]
img_list = os.listdir(input_path)
white_color = np.array([255, 255, 255])
tolerance = 10  # Adjust as needed based on your images
os.makedirs(output_path, exist_ok=True)

for img in img_list:
    image_path = os.path.join(input_path, img)
    image = Image.open(image_path).convert('RGB')
    image_arr = np.array(image)
    
    # Compute color distance from white
    color_distance = np.linalg.norm(image_arr - white_color, axis=-1)
    
    # Initial mask: Pixels that differ from white beyond the tolerance
    initial_mask = (color_distance > tolerance).astype(np.uint8)
    
    # Morphological opening to remove small artifacts
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel)
    
    # Connected component analysis to keep the largest component
    num_labels, labels_im = cv2.connectedComponents(clean_mask)
    if num_labels > 1:
        # Find the label of the largest component (excluding background label 0)
        label_counts = np.bincount(labels_im.flatten())
        label_counts[0] = 0  # Exclude background
        largest_label = label_counts.argmax()
        mask = (labels_im == largest_label).astype(np.uint8) * 255
    else:
        mask = clean_mask * 255  # No components found, use the clean mask as is
    
    # Edge smoothing (optional)
    mask = cv2.medianBlur(mask, 5)
    
    # Save the mask
    mask_img = Image.fromarray(mask)
    mask_img.save(os.path.join(output_path, img))

print("Done!")