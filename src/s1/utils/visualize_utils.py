
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

def apply_colormap(gray, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    Input:
        gray: gray image, tensor/numpy, (H, W)
    Output:
        depth: (3, H, W), tensor
    """
    if type(gray) is not np.ndarray:
        gray = gray.detach().cpu().numpy().astype(np.float32)
    gray = gray.squeeze()
    assert len(gray.shape) == 2
    x = np.nan_to_num(gray)  # change nan to 0
    if minmax is None:
        mi = np.min(x)  # get minimum positive value
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_

def get_normal(depth):
    a=np.pi * 2.0
    bg_th = 0.1 * 255
    # bg threshold is set to 0.1, by default
    x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    z = np.ones_like(x) * a
    # x[depth < bg_th] = 0
    # y[depth < bg_th] = 0
    normal = np.stack([x, y, z], axis=2)
    normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
    normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    return normal_image