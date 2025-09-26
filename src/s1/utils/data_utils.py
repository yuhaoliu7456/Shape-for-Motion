import cv2
import numpy as np


def pad_to_equal(image, desired_size, pad_value=255):
    old_size = image.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [pad_value, pad_value, pad_value]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_image


def generate_random_points_in_sphere(num_pts):
    points = []
    while len(points) < num_pts:
        remaining_pts = num_pts - len(points)
        candidate_points = np.random.uniform(-1, 1, (remaining_pts * 2, 3))  # generate points in sphere
        inside_sphere = np.linalg.norm(candidate_points, axis=1) <= 1  # filter points in sphere
        points.extend(candidate_points[inside_sphere][:remaining_pts])  # add points in sphere
    return np.array(points)