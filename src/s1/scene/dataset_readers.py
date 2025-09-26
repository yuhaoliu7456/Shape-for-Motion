import os
from PIL import Image
from typing import NamedTuple, Optional

from s1.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json

from pathlib import Path
from plyfile import PlyData, PlyElement
from s1.utils.sh_utils import SH2RGB
from s1.scene.gaussian_model import BasicPointCloud

from s1.utils.data_utils import  generate_random_points_in_sphere

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    orig_transform: Optional[np.array] = None
    alpha_mask: Optional[np.array] = None
    depth: Optional[np.array] = None
    orig_img: Optional[np.array] = None
    K: Optional[np.array] = None
    H: Optional[int] = None
    W: Optional[int] = None
    # For fine-tuning nerf-based mesh, optional
    mesh_verts: Optional[np.array] = None
    mesh_faces: Optional[np.array] = None
    vid: Optional[int] = None
    tid: Optional[int] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str




def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readCameras(path, transformsfile, 
                        white_background, extension=".png", 
                        cam_scale=1.0, downsample=1.0,
                        selected_coordinates=None):
    cam_infos = []


    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        print('Loading camera info from', transformsfile)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        selected_samples = []
        for idx, coordinate in enumerate(selected_coordinates):
            v, t = coordinate

            try:
                if frames[v][t] is None:
                    pass
                else:
                    selected_samples.append(frames[v][t])
            except:
                import pdb; pdb.set_trace()

        for idx, frame in enumerate(selected_samples):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']

            c2w = np.array(frame["transform_matrix"])
            orig_cam = np.array(frame["transform_matrix"])
            
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Open-cv) (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            
            image_path = os.path.join(path, cam_name)

            mask_path = image_path.replace('rgbs', 'masks')
            depth_path = image_path.replace('rgbs', 'depths')
            mask = Image.open(mask_path)
            depth = Image.open(depth_path)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            
            assert os.path.exists(depth_path)
            assert os.path.exists(mask_path)
            assert os.path.exists(image_path)
            
            image = image.resize((int(image.size[0] / downsample), int(image.size[1] / downsample)), Image.Resampling.LANCZOS)

            mask = np.array(mask) / 255.0
            mask[mask > 0.1] = 1
            mask = mask[:,:] if mask.ndim == 2 else mask[:,:,0] # [H, W]
            depth = np.array(depth) / 255.0
            depth = depth[:,:,[0]] if depth.ndim ==3 else depth[:,:,np.newaxis]# [H, W, 1] 
            

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, 
                                        alpha_mask=mask.astype('int').reshape(mask.shape[0], mask.shape[1], 1),
                                        depth=depth,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1], fid=frame_time, orig_transform=orig_cam,
                                        vid=frame['vid'], tid=frame['tid']))
    return cam_infos

def readInfo(path, white_background, eval, extension=".png", downsample=1.0, 
            views_index=None, views_num=1, time_index=None, time_num=1, time_indexs_that_use_novel_views=None,
            train_json="camera_sv3d.json", test_json="camera_sv3d.json"):
    
    selected_views = views_index if views_index is not None else range(views_num)
    
    if time_index is not None:
        if '-' in time_index:
            # means a range
            start_index = int(time_index.split('-')[0])
            end_index = int(time_index.split('-')[1])
            selected_times = range(start_index, end_index)
        else:
            # means a list
            selected_times = time_index
    else:
        selected_times = range(time_num)

    selected_coordinates = [(v, t) for v in selected_views for t in selected_times]
    
    # --------------------------------------------------------------
    # Remove the coordinates of novel views (v != 0) that do not appear in time_indexs_that_use_novel_views.
    # 
    # the current version is that we use all novel views from the first frame to the last frame. 
    if time_indexs_that_use_novel_views is not None:
        if type(time_indexs_that_use_novel_views) == str and '-' in time_indexs_that_use_novel_views:
            # such as 10-20
            start_index = int(time_indexs_that_use_novel_views.split('-')[0])
            end_index = int(time_indexs_that_use_novel_views.split('-')[1])
            time_indexs_that_use_novel_views = range(start_index, end_index)
        # else, means a list
        selected_coordinates = [(v, t) for v, t in selected_coordinates if t in time_indexs_that_use_novel_views or v == 0]
    # --------------------------------------------------------------

    print("Reading Training Transforms")
    train_cam_infos = readCameras(path, 
                                train_json,
                                white_background, 
                                extension, 
                                downsample=downsample,
                                selected_coordinates=selected_coordinates)
    
    print("Reading Test Transforms")
    test_cam_infos = readCameras(path, 
                                test_json, 
                                white_background, 
                                extension, 
                                downsample=downsample,
                                selected_coordinates=selected_coordinates)
    train_cam_infos = test_cam_infos
    # they are the same in the current version

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)
    # manually set the 'radius' in case the input video affect it. Because all the input frames share the same front view.

    nerf_normalization['radius'] = 3.5655
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = generate_random_points_in_sphere(num_pts)
        
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


