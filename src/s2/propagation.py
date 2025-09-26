
import torch
import numpy as np
from s2.utils import *

from s1.scene import DeformModelNormal as deform_back
from s1.scene import AppearanceModel as appearance_net
from s1.scene import GaussianModelDPSRDynamicAnchor as gaussian_model



SMALL_NUMBER = 1e-6

def extract_new_obj(edited_verts, edited_faces, edited_indexs, edited_colors, new_indices):
    # map edited indices to vertices
    edited_id_to_vert = {idx: pos for idx, pos in zip(edited_indexs, edited_verts)}
    edited_id_to_color = {idx: c for idx, c in zip(edited_indexs, edited_colors)}

    # extract new object
    new_object = {idx: edited_id_to_vert[idx] for idx in new_indices}
    new_object_keys = sorted(new_object.keys())
    new_obj_verts = np.array([new_object[k] for k in new_object_keys])
    new_obj_colors = np.array([edited_id_to_color[k] for k in new_object_keys])
    new_obj_indexs = np.array(new_object_keys)

    old_to_new = {old_id: i for i, old_id in enumerate(new_object_keys)}

    # find faces that are completely made up of new indices
    new_obj_faces_global = [
        face for face in edited_faces
        if all(edited_indexs[v_id] in new_indices for v_id in face)
    ]

    # remap face indices
    new_obj_faces = [
        [old_to_new[edited_indexs[v_id]] for v_id in face]
        for face in new_obj_faces_global
    ]
    new_obj_faces = np.array(new_obj_faces)

    # one can use this func: save_mesh_by_trimesh to visualize the colored mesh
    return {'verts': new_obj_verts, 'faces': new_obj_faces, 'indexs': new_obj_indexs, 'colors': new_obj_colors}


def obtain_offsets(gs_points, prev_mesh_path, edited_mesh_path, extra_param=None, edited_mesh=None,  mesh_rescaling_ratio=1.0):
    """
    The indices in `edited_indexs` start from 1 when writing; newly added objects do not have an index value by default.
    When certain vertices from the original object are deleted, there may be gaps in the indices within `edited_indexs`.
    Therefore, when assigning indices to new objects, the first index with a value of 0 is found, and numbering starts from `max(cano_indexs) + 1`.
    """
    cano_verts, cano_faces, cano_indexs, cano_colors = plydata_read(prev_mesh_path)
    if edited_mesh is not None:
        edited_verts = edited_mesh['verts']
        edited_faces = edited_mesh['faces']
        edited_indexs = edited_mesh['indexs']
        edited_colors = edited_mesh['colors']
    else:
        edited_verts, edited_faces, edited_indexs, edited_colors = plydata_read(edited_mesh_path)
        edited_verts = np.array(edited_verts) / mesh_rescaling_ratio     
        # Note: for the object that has been motioned by rigging in blender, it needs to be re-scaled back now.


    if edited_colors is None:
        # If the edited_colors does not exist, create a colors that is all white
        edited_colors = np.ones_like(edited_verts, dtype=np.uint8) * 255
    
    zero_positions = np.where(edited_indexs == 0)[0]  # zero_positions denotes the index set of the newly added points

    num_new_points = len(zero_positions)
    if num_new_points > 0:
        """
        This part of the process addresses the scenario where newly added objects may not have indices initially, such as objects generated directly from img-2-3d.
        On the other hand, for objects trained using Gaussian Splatting (GS) and saved manually, indices are assigned during the process.
        Additionally, it is possible to manually assign indices to objects downloaded from the internet or generated via img-2-3d before editing.
        """
        max_cano = max(cano_indexs) if len(cano_indexs) > 0 else 0

        # allocate new ids for the new points starting from max_cano + 1
        new_ids = np.arange(max_cano + 1, max_cano + 1 + num_new_points)

        # fill the new ids into the zero positions of edited_indexs
        edited_indexs[zero_positions] = new_ids


    cano_indices_set = set(cano_indexs)
    edited_indices_set = set(edited_indexs)

    common_indices = cano_indices_set & edited_indices_set
    deleted_indices = cano_indices_set - edited_indices_set
    new_indices = edited_indices_set - cano_indices_set


    cano_index_to_vertex = {idx: pos for idx, pos in zip(cano_indexs, cano_verts)}
    edited_index_to_vertex = {idx: pos for idx, pos in zip(edited_indexs, edited_verts)}
    edited_index_to_color = {idx: c for idx, c in zip(edited_indexs, edited_colors)}
    vertex_index_to_cano_idx = {idx: i for i, idx in enumerate(cano_indexs)}

    # initialize the vertex differences, and fill the common indices
    off_xyz_mesh = np.full((len(cano_verts), 3), np.inf, dtype=np.float32)
    cano_edited_colors = np.full((len(cano_verts), 3), np.inf, dtype=np.float32)
    
    for idx in common_indices:
        i = vertex_index_to_cano_idx[idx]
        cano_pos = cano_index_to_vertex[idx]
        edited_pos = edited_index_to_vertex[idx]
        off_xyz_mesh[i] = edited_pos - cano_pos
        cano_edited_colors[i] = edited_index_to_color[idx]
        
        
    # build indices from gs_points to mesh_vertices
    indices = build_indices(cano_verts, gs_points)
    
    off_xyz_gs_points = off_xyz_mesh[indices]
    off_xyz_gs_points = torch.from_numpy(off_xyz_gs_points).to(gs_points.device)  # [N_gs, 3]
    off_xyz_mesh = torch.from_numpy(off_xyz_mesh).to(gs_points.device)  # [N_mesh, 3]
    cano_edited_colors = torch.from_numpy(np.array(cano_edited_colors).astype(np.float32)[:,:3]/255).to(gs_points.device)  # [N_mesh, 3]
    
    output = {}
    if extra_param is not None:
        # first, extract new object
        new_obj = extract_new_obj(edited_verts, edited_faces, edited_indexs, edited_colors, new_indices)
        
        # then, calculate the relative displacement between the bind point and all points of the new object
        mesh_bind_point_index = extra_param['binding_index'] 
        
        # Note: If the index of an object does not start from 1, the following line will raise an error.
        mesh_bind_point_index = vertex_index_to_cano_idx[mesh_bind_point_index]
        
        mesh_bind_point_pos = cano_index_to_vertex[mesh_bind_point_index]
        gs_bind_point_idx = build_indices(gs_points, mesh_bind_point_pos)

        output['mesh_bind_point_pos'] = mesh_bind_point_pos
        output['gs_bind_point_idx'] = gs_bind_point_idx
        output['new_obj_attrs'] = new_obj
        output['binding_index'] = mesh_bind_point_index

    
    output['off_xyz_gs_points'] = off_xyz_gs_points
    output['off_xyz_mesh'] = off_xyz_mesh
    output['mesh_vertices'] = cano_verts
    output['mesh_faces'] = cano_faces
    output['cano_edited_colors'] = cano_edited_colors
    output['edited_mesh'] = {'verts': edited_verts, 'faces': edited_faces, 'indexs': edited_indexs, 'colors': edited_colors}
    
    return output
    

def editing_composition(
    gaussians: gaussian_model,
    d_xyz: torch.Tensor,
    d_normal: torch.Tensor,
    fid: torch.Tensor,
    deform: deform_back,
    deform_back: deform_back,
    appearance: appearance_net,
    offset_dict: dict,
    disable_cano_w_color: bool = False,
    enable_zero_motion_for_newobj: bool = False
):

    verts, faces, vtx_color, verts_d, faces_d = edit_mesh(gaussians, d_xyz, d_normal, fid, deform, deform_back, appearance, offset_dict, disable_cano_w_color=disable_cano_w_color)
    verts_ori_copy = verts.clone()
    faces_ori_copy = faces.clone()


    gs_bind_point_idx = offset_dict['gs_bind_point_idx']
    new_obj_attrs = offset_dict['new_obj_attrs']
    
    
    deformed_bind_point = obtain_deformed_verts(gaussians.get_xyz[gs_bind_point_idx].unsqueeze(0), 
                                                deform, fid.unsqueeze(0))
    deformed_bind_point -= torch.from_numpy(offset_dict['mesh_bind_point_pos']).to(verts.device)

    if new_obj_attrs['verts'].shape[0] != 0:
        new_obj_verts = torch.from_numpy(new_obj_attrs['verts']).to(verts.device)
        if not enable_zero_motion_for_newobj:
            new_obj_verts += deformed_bind_point
        new_obj_verts_copy = new_obj_verts.clone()
        new_obj_faces = torch.from_numpy(new_obj_attrs['faces']).to(verts.device).int()
        new_obj_faces_copy = new_obj_faces.clone()
        new_obj_faces += len(verts)
        new_obj_colors = torch.from_numpy(new_obj_attrs['colors']).to(verts.device).float()
        if new_obj_colors.max() > 1:
            new_obj_colors /= 255.0
        
        verts = torch.cat([verts, new_obj_verts], dim=0)
        faces = torch.cat([faces, new_obj_faces], dim=0)
        vtx_color = torch.cat([vtx_color, new_obj_colors], dim=0)
        
        # If you want to render only the new object and the motion is the binding point's motion, uncomment the following code
        # verts = new_obj_verts_copy
        # faces = new_obj_faces_copy
        # vtx_color = new_obj_colors
        

        new_obj_faces_d = torch.from_numpy(new_obj_attrs['faces']).to(verts.device).int() + len(verts_d)
        verts_d = torch.cat([verts_d, new_obj_verts], dim=0)
        faces_d = torch.cat([faces_d, new_obj_faces_d], dim=0)
    else:
        new_obj_verts_copy = None
        new_obj_faces_copy = None

    

    return verts, faces, vtx_color, verts_d, faces_d, new_obj_verts_copy, new_obj_faces_copy, verts_ori_copy, faces_ori_copy 
    


"""

General issues:
1. when the depth of two objects are different, there might be some blending in some areas, in this case, we only need to adjust the depth of the newly added object slightly (z-axis)
"""