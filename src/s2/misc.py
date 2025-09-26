import torch
import math
import numpy as np



def rotate_verts(verts, axis='z', angle=math.pi/5):

    verts = verts.detach()  
    
    cos_theta = torch.cos(torch.tensor(angle))
    sin_theta = torch.sin(torch.tensor(angle))
    
    # select the rotation matrix based on the axis
    if axis.lower() == 'x':
        rotation_matrix = torch.tensor([
            [1, 0, 0],
            [0, cos_theta, -sin_theta],
            [0, sin_theta, cos_theta]
        ], device=verts.device)
    elif axis.lower() == 'y':
        rotation_matrix = torch.tensor([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ], device=verts.device)
    elif axis.lower() == 'z':
        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], device=verts.device)
    else:
        raise ValueError(f"Unsupported rotation axis: {axis}. Please use 'x', 'y', or 'z'")
    
    # perform the rotation
    edited_verts = torch.matmul(verts, rotation_matrix.T)
    return edited_verts

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)
