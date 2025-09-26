import torch
import numpy as np
import trimesh

from plyfile import PlyData, PlyElement

from s1.scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from s1.scene import DeformModelNormal as deform
from s1.scene import AppearanceModel as appearance_net

from s2.misc import rotate_verts

import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

SMALL_NUMBER = 1e-6

from scipy.spatial import cKDTree

def tensor_to_numpy(input):
    if isinstance(input, torch.Tensor):
        input = input.detach().cpu().numpy()
    return input

def obtain_deformed_verts(ori_verts, deform, time_input):
    """
    Inputs:
        ori_verts: [N, 3], torch.Tensor, either mesh vertices or gs_points
        deform: deform network
        time_input: [N, 1], torch.Tensor
    """
    d_xyz, d_rotation, d_scaling, _ = deform.step(ori_verts.detach(), time_input)
    deformed_verts = ori_verts + d_xyz 
    # Note that d_xyz should multiply a rotation matrix, but here we just use the d_xyz directly

    return deformed_verts


def plydata_read(path):
    
    plydata = PlyData.read(path)
    vertex_data = plydata['vertex']
    verts = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    faces = np.vstack(plydata['face'].data['vertex_indices'])

    try:
        indexs = vertex_data['vertex_index'].astype(np.int32)
        indexs = np.array(indexs)
    except:
        indexs = None
 
    try:
        colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
        print("successfully found color information")
    except ValueError:
        print("no color information found")
        colors = None
    return verts, faces, indexs, colors

def build_indices(input1, input2):
    """ 
    Build indices from input2 to input1. Use input2 to query input1, so we should first build a tree for input1.
    input1/2 can be either mesh_vertices or gs_points. both are numpy array
    """
    input1 = tensor_to_numpy(input1)
    input2 = tensor_to_numpy(input2)
    kdtree = cKDTree(input1)
    distances, indices = kdtree.query(input2)
    return indices


def build_mesh_from_gs(gaussians, xyz, normals):
    """
    build the mesh, note that this func does not consider the color of the mesh.
    xyz: could be d_xyz, or gaussians.get_xyz, or gaussians.get_xyz + d_xyz.
    normals should be modified accordingly.
    """
    dpsr_points = (xyz - gaussians.gaussian_center) / gaussians.gaussian_scale  # [-1, 1]
    dpsr_points = dpsr_points / 2.0 + 0.5  # [0, 1]
    dpsr_points = torch.clamp(dpsr_points, SMALL_NUMBER, 1 - SMALL_NUMBER)

    # Query SDF
    psr = gaussians.dpsr(dpsr_points.unsqueeze(0), normals.unsqueeze(0))
    sign = psr[0, 0, 0, 0].detach()  # Sign for Diso is opposite to dpsr
    sign = -1 if sign < 0 else 1
    psr = psr * sign

    psr -= gaussians.density_thres_param
    # density_thres_param =0 by default
    psr = psr.squeeze(0)
    
    verts, faces = gaussians.diffmc(psr, deform=None, isovalue=0.0)
    verts = verts * 2.0 - 1.0  # [-1, 1]
    verts = verts * gaussians.gaussian_scale + gaussians.gaussian_center
    verts = verts.to(torch.float32)
    faces = faces.to(torch.int32)
    
    return verts, faces

def query_color(appearance, fid, verts, deform_back, is_canonical=False):
    """
    Note:
        1. verts must be the canonical vertices
    """
    # Deform mesh vertex back to canonical mesh and query vertex color
    N = verts.shape[0]
    time_input = fid.unsqueeze(0).expand(N, -1)
    
    mesh_deform_back_dxyz, _, _, _ = deform_back.step(verts.detach(), time_input)
    mesh_canonical_xyz = verts + mesh_deform_back_dxyz
    vtx_color = appearance.step(mesh_canonical_xyz, time_input)
    return vtx_color

def save_mesh_by_trimesh(verts, faces, colors, save_path):
    # directly save the mesh with color by trimesh
    new_mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=trimesh.visual.ColorVisuals(vertex_colors=colors))
    new_mesh.export(save_path)
    
def save_mesh_from_gs(gaussians, xyz, normals, save_path, start_index, save_vert_color=False, appearance=None, fid=None, deform=None, deform_back=None, is_canonical=False, enable_smooth_vert_color=False):
    verts, faces = build_mesh_from_gs(gaussians, xyz, normals)
    if save_vert_color:
        # deform the canonical verts to the mesh verts at t=0, and query the color
        verts = obtain_deformed_verts(verts, deform, fid.unsqueeze(0).expand(verts.shape[0], -1))
        colors = query_color(appearance=appearance, fid=fid, verts=verts, deform_back=deform_back, is_canonical=is_canonical)
        colors = tensor_to_numpy(colors)
    else:
        colors = None
    verts = tensor_to_numpy(verts)
    faces = tensor_to_numpy(faces)
    
    if enable_smooth_vert_color:
        print("smooth the vertex color")
        # means that the mesh contains ladder artifact, should smooth the mesh
        verts = smooth_vertices(vertices=verts, faces=faces)
    save_mesh(verts, faces, colors=colors, save_path=save_path, start_index=start_index)

def smooth_vertices(vertices, faces):
    smoothed_vertices = vertices
    
    if isinstance(vertices, torch.Tensor):
        smoothed_vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()    

    mesh = trimesh.Trimesh(vertices=smoothed_vertices, faces=faces)

    trimesh.smoothing.filter_laplacian(
        mesh,
        lamb=0.5,
        iterations=10,
        implicit_time_integration=False,
        volume_constraint=True
    )

    smoothed_vertices = mesh.vertices
    if isinstance(vertices, torch.Tensor):
        smoothed_vertices = torch.from_numpy(smoothed_vertices).to(vertices.device)
        smoothed_vertices = smoothed_vertices.to(torch.float32)
    
    return smoothed_vertices


def save_mesh(verts, faces, colors, save_path, start_index=1):
    """
    Save mesh with vertex colors in PLY format
    Inputs:
        verts: [N, 3], np.array
        faces: [M, 3], np.array
        colors: [N, 3] or [N, 4], np.array
        save_path: str
        start_index: int, default 1
    """


    if colors is not None:
        if colors.shape[1] not in [3, 4]:
            raise ValueError("Colors must have 3 (RGB) or 4 (RGBA) channels.")

    if colors is not None and colors.shape[1] == 4:
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'),
                        ('vertex_index', 'i4')]
    else:
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                        ('vertex_index', 'i4')]

    vertex_data = np.empty(len(verts), dtype=vertex_dtype)
    vertex_data['x'] = verts[:, 0]
    vertex_data['y'] = verts[:, 1]
    vertex_data['z'] = verts[:, 2]

    # add vertex color
    if colors is not None:
        vertex_data['red'] = (colors[:, 0] * 255).astype(np.uint8)
        vertex_data['green'] = (colors[:, 1] * 255).astype(np.uint8)
        vertex_data['blue'] = (colors[:, 2] * 255).astype(np.uint8)
        if colors.shape[1] == 4:
            vertex_data['alpha'] = (colors[:, 3] * 255).astype(np.uint8)
    else:
        # default color: white
        vertex_data['red'] = 255
        vertex_data['green'] = 255
        vertex_data['blue'] = 255
        if 'alpha' in vertex_dtype:
            vertex_data['alpha'] = 255

    # add vertex index
    vertex_data['vertex_index'] = np.arange(start_index, start_index + len(verts))

    # define face data
    face_dtype = [('vertex_indices', 'i4', (3,))]
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data['vertex_indices'] = faces

    # create PlyElement
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    face_element = PlyElement.describe(face_data, 'face')

    # save PLY
    ply_data = PlyData([vertex_element, face_element], text=True)  
    ply_data.write(save_path)
    print(f"Mesh with colors saved at {save_path}")



def obtain_offsets(gs_points, prev_mesh_path, edited_mesh_path, extra_param=None):
    """
    Given two mesh paths, first obtain the offset of edited mesh vertices to prev. mesh vertices.
    Then, build indices from gs_points to mesh_vertices, and obtain the offset at gs_points space.

    Inputs:
        gs_points: [N_gs, 3], torch.Tensor
        prev_mesh_path: path to the prev. mesh
        edited_mesh_path: path to the edited mesh
    
    Outputs:
        off_xyz_gs_points: [N_gs, 3], torch.Tensor
        off_xyz_mesh: [N_mesh, 3], torch.Tensor
    """
    prev_verts, prev_faces, prev_indexs, pred_colors =  plydata_read(prev_mesh_path)
    edited_verts, edited_faces, edited_indexs, edited_colors =  plydata_read(edited_mesh_path)    
    

    prev_indices_set = set(prev_indexs)
    edited_indices_set = set(edited_indexs)

    common_indices = prev_indices_set & edited_indices_set
    deleted_indices = prev_indices_set - edited_indices_set
    new_indices = edited_indices_set - prev_indices_set

    prev_index_to_vertex = {idx: pos for idx, pos in zip(prev_indexs, prev_verts)}
    edited_index_to_vertex = {idx: pos for idx, pos in zip(edited_indexs, edited_verts)}
    vertex_index_to_prev_idx = {idx: i for i, idx in enumerate(prev_indexs)}

    # initialize the vertex differences, and fill the common indices
    off_xyz_mesh = np.full((len(prev_verts), 3), np.inf, dtype=np.float32)
    
    for idx in common_indices:
        i = vertex_index_to_prev_idx[idx]
        prev_pos = prev_index_to_vertex[idx]
        edited_pos = edited_index_to_vertex[idx]
        off_xyz_mesh[i] = edited_pos - prev_pos
        
    
    # build indices from gs_points to mesh_vertices
    indices = build_indices(prev_verts, gs_points.detach().cpu().numpy())
    off_xyz_gs_points = off_xyz_mesh[indices]
    off_xyz_gs_points = torch.from_numpy(off_xyz_gs_points).to(gs_points.device)  # [N_gs, 3]
    off_xyz_mesh = torch.from_numpy(off_xyz_mesh).to(gs_points.device)  # [N_mesh, 3]

    return {'off_xyz_gs_points':off_xyz_gs_points, 
            'off_xyz_mesh': off_xyz_mesh,
            'mesh_vertices': prev_verts,
            'mesh_faces': prev_faces}
    
    
    

def edit_mesh(
    gaussians: gaussian_model,
    d_xyz: torch.Tensor,
    d_normal: torch.Tensor,
    fid: torch.Tensor,
    deform: deform,
    deform_back: deform,
    appearance: appearance_net,
    offset_dict: dict,
    enable_smooth_vert_color=False,
    disable_cano_w_color=False
):
    """use canonical mesh to edit the mesh

    Args:
        gaussians (gaussian_model): Gaussians model
        d_xyz (torch.Tensor): Predicted xyz offset [N, 3]
        d_normal (torch.Tensor): Predicted normal offset [N, 3]
        fid (torch.Tensor): Time label [N, 1]
        deform_back (deform_back): Backward deformation network
        appearance (appearance_net): Appearance network
        disable_cano_w_color: If turn on, the canonical mesh is white model, without color information
    """
    off_xyz_gs_points = offset_dict['off_xyz_gs_points']
    off_xyz_mesh = offset_dict['off_xyz_mesh']
    mesh_vertices = offset_dict['mesh_vertices']
    mesh_faces = offset_dict['mesh_faces']

    # -----------
    # Use the points in `off_xyz_gs_points` that are marked as `inf` to identify and remove the corresponding
    #  points in `gaussians.get_xyz`, `d_xyz`, and `d_normal`.
    d_xyz = d_xyz[~torch.isinf(off_xyz_gs_points).any(dim=1)]
    d_normal = d_normal[~torch.isinf(off_xyz_gs_points).any(dim=1)]
    gaussian_canonical_xyz = gaussians.get_xyz[~torch.isinf(off_xyz_gs_points).any(dim=1)]
    gaussian_canonical_normal = gaussians.get_normal[~torch.isinf(off_xyz_gs_points).any(dim=1)]

    dpsr_points = gaussian_canonical_xyz + d_xyz + off_xyz_gs_points[~torch.isinf(off_xyz_gs_points).any(dim=1)]
    normals = gaussian_canonical_normal + d_normal
    # -----------
    
    # deformed mesh from gs_points
    
    verts, faces = build_mesh_from_gs(gaussians, dpsr_points, normals)
    verts_d = verts.clone()
    faces_d = faces.clone()

    # =============canoncial mesh==============
    
    if disable_cano_w_color:
        # means that the canonical mesh has not color information
        if type(mesh_vertices) == torch.Tensor:
            canonical_verts = mesh_vertices    
        else:
            canonical_verts = torch.from_numpy(mesh_vertices).to(dpsr_points.device) 
    else:
        canonical_verts, faces = build_mesh_from_gs(gaussians, gaussians.get_xyz, gaussians.get_normal)
    
    N_canonical = canonical_verts.shape[0]
    time_input_canonical = fid.unsqueeze(0).expand(N_canonical, -1)
    # ----deformed mesh from canonical mesh----
    verts2 = obtain_deformed_verts(canonical_verts, deform, time_input_canonical)


    if type(off_xyz_mesh) == torch.Tensor:
        off_xyz_mesh = off_xyz_mesh.detach().cpu().numpy()
    
    off_xyz_mesh[np.isinf(off_xyz_mesh)] = 0 # represent the deleted vertices, which should be set to 0
    

    if disable_cano_w_color:
        indices = build_indices((verts2+torch.from_numpy(off_xyz_mesh).to(verts.device)).detach().cpu().numpy(),
                                verts.detach().cpu().numpy())
        mesh_deform_back_dxyz, _, _, _ = deform_back.step(verts2.detach(), time_input_canonical)    
        mesh_canonical_xyz = verts2 + mesh_deform_back_dxyz
        vtx_color = appearance.step(mesh_canonical_xyz, time_input_canonical)
        vtx_color = vtx_color[indices]
    else:
        verts = verts2 + torch.from_numpy(off_xyz_mesh).to(verts.device)
        vtx_color = offset_dict['cano_edited_colors']

    
    if disable_cano_w_color:
        return verts, faces, vtx_color, verts_d, faces_d
    else:
        if enable_smooth_vert_color:
            print("smooth the vertex color")
            verts = smooth_vertices(vertices=verts, faces=faces)
        return verts, faces, vtx_color, verts_d, faces_d
