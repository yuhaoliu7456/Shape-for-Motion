import torch
import torch.nn.functional as F
import numpy as np
import json, os


def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics


def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws




if __name__ == "__main__":
    # name="dancing_spiderman"
    import sys
    name=sys.argv[1]
    data_root=sys.argv[2]
    inp_path = os.path.join(data_root, name)

    inp_list = os.listdir(inp_path+'/processed_rgbs')
    inp_list.sort(key=lambda x: int(x.split('.')[0]))
    n_frames = len(inp_list)
    time_stamp_list = np.linspace(0, 1, n_frames)
    
    n_views = 21 
    fov=33.8
    radius = 1/np.tan(np.deg2rad(fov)/2)
    # strictly speaking, radius should be: np.sqrt(3)/np.tan(np.deg2rad(fov)/2)
    # Note that azimuths contains 21 elements, but the last one is useless.

    num_elevations = 1     # elevation=0
    # By default, the number of elevations is set to 1, i.e., only one view, i.e., elevation=0
    
    save_dict = {}
    # contents
    save_dict["camera_angle_x"] = np.deg2rad(fov)
    save_dict["camera_origin"] = [0,0,1]
    img_matrix = [[None] * n_frames for _ in range(n_views*num_elevations)]
    # each row in img_matrix represents all frames of a view, with N_views*num_elevations rows in total
    

    # ele: 0
    elevation=0
    azimuths = np.linspace(0, 360, n_views + 1)[:-1] % 360  
    elevations = np.array([elevation] * n_views).astype(float)
    c2ws = spherical_camera_pose(azimuths, elevations, radius)
    # Note that the orginal input videe are all with the same elevation, i.e., 0

    for row_idx, c2w in enumerate(c2ws):
        for i, name in enumerate(inp_list):
            if row_idx == 0:
                file_path = "./processed_rgbs/" + name.split('.')[0]
            else:
                file_path = "./novelviews/" + name.split('.')[0] + "/rgbs/p_0/sv3d_"+ str(row_idx)
            img_matrix[row_idx][i] = {"file_path": file_path,
                                      "time": time_stamp_list[i],
                                      "vid": row_idx,
                                      "tid": i,
                                      "transform_matrix": c2w.tolist()}


    save_dict["frames"] = img_matrix
    save_path = os.path.join(inp_path, f"{os.path.basename(inp_path)}.json")
    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=4)