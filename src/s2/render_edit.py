import sys
import os
import datetime
import os.path as osp
import torch
import uuid
import datetime
from tqdm import tqdm
import random
from argparse import ArgumentParser, Namespace
import numpy as np
import imageio
import cv2
from PIL import Image
import nvdiffrast.torch as dr

from s1.scene import Scene
from s1.scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from s1.scene import DeformModelNormal as deform_model
from s1.scene import DeformModelNormalSep as deform_model_sep
from s1.scene import AppearanceModel as appearance_model

from s1.gaussian_renderer import render
from s1.utils.general_utils import safe_state
from s1.utils.visualize_utils import get_normal
from s1.utils.renderer import mesh_shape_renderer, updated_mesh_renderer
from s1.utils.system_utils import load_config_from_file, merge_config
from s1.arguments import ModelParams, PipelineParams, OptimizationParams

from s2.propagation import obtain_offsets, save_mesh_from_gs, plydata_read, editing_composition



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(dataset, opt, checkpoint):
    # Load models
    ## Gaussian model
    gaussians = gaussian_model(
        dataset.sh_degree,
        grid_res=dataset.grid_res,
        density_thres=opt.init_density_threshold,
        dpsr_sig=opt.dpsr_sig,
    )
    
    scene = Scene(dataset, gaussians, shuffle=False)
    ## Deform forward model
    deform = deform_model(
        is_blender=dataset.is_blender, is_6dof=dataset.is_6dof, model_name="deform"
    )
    deform_normal = deform_model_sep(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform_normal",
    )
    ## Deform backward model
    deform_back = deform_model(
        is_blender=dataset.is_blender, is_6dof=dataset.is_6dof, model_name="deform_back"
    )
    deform_back_normal = deform_model_sep(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform_back_normal",
    )
    ## Appearance model
    appearance = appearance_model(is_blender=dataset.is_blender)
    
    ## Load checkpoint
    if checkpoint:
        gaussians.load_ply(checkpoint, iteration=args.load_iteration)
        deform.load_weights(checkpoint, iteration=args.load_iteration)
        deform_normal.load_weights(checkpoint, iteration=args.load_iteration)
        deform_back.load_weights(checkpoint, iteration=args.load_iteration)
        deform_back_normal.load_weights(checkpoint, iteration=args.load_iteration)
        appearance.load_weights(checkpoint, iteration=args.load_iteration)
    return gaussians, scene, deform, deform_normal, deform_back, deform_back_normal, appearance

@torch.no_grad()
def rendering_trajectory(dataset, opt, pipe, checkpoint, fps=24):
    glctx = dr.RasterizeCudaContext()
    gaussians, scene, deform, deform_normal, deform_back, _, appearance = load_models(dataset, opt, checkpoint)

    # Compose camera trajectory
    viewpoint_cam_stack = scene.getCameras().copy()
    
    final_images = []
    mask_images = []
    mask_of_original_obj_images = []
    mask_of_new_obj_images = []
    
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    normal_list = []
    mesh_image_np_list = []
    
    viewpoint_cam_stack = sorted(viewpoint_cam_stack, key=lambda x: x.tid)
    

    if args.canonical_mesh_path:
        if not osp.exists(args.canonical_mesh_path):
            os.makedirs(osp.dirname(args.canonical_mesh_path), exist_ok=True)
    
    
    # Only change the viewpoint_cam.vid to other values, can adjust the render view
    # If you want to globally rotate the object a large angle, e.g., >40 degrees, we recommend to change the viewpoint_cam.vid to other values. 
    # Also, you are encouraged to revise the camera pose, instead of changing the viewpoint_cam.vid. 
    sssselected_render_view = 0 # default: 0, choices: [0, 3, 6, 9, 12, 15, 18]
    print(f"selected_render_view: {sssselected_render_view}")
    viewpoint_cam_stack = [viewpoint_cam for viewpoint_cam in viewpoint_cam_stack if viewpoint_cam.vid == sssselected_render_view]
    

    for idx, viewpoint_cam in tqdm(enumerate(viewpoint_cam_stack)):
        
        fid = viewpoint_cam.fid
        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)

        # Query the gaussians
        d_xyz, d_rotation, d_scaling, _ = deform.step(gaussians.get_xyz.detach(), time_input)
        d_normal = deform_normal.step(gaussians.get_xyz.detach(), time_input)
        
        if args.do_save_canonical_mesh:
            # ===================================First step: save a mesh===================================
            xyz_for_saving = gaussians.get_xyz
            normal_for_saving = gaussians.get_normal
            # --------------------------------
            # If you want to save the mesh with color after (gs+deform), uncomment the following lines
            # if args.save_vert_color:
            #     xyz_for_saving += d_xyz
            #     normal_for_saving += d_normal
            # --------------------------------
        
            save_mesh_from_gs(gaussians=gaussians,
                            xyz=xyz_for_saving,
                            normals=normal_for_saving,
                            save_path=args.canonical_mesh_path,
                            start_index=args.start_index,
                            save_vert_color=args.save_vert_color,
                            appearance=appearance, fid=fid, deform=deform, deform_back=deform_back,
                            enable_smooth_vert_color=args.enable_smooth_vert_color,
                            is_canonical=args.is_canonical)
            exit()
            # =============================================================================================
      
        args.extra_param = {'binding_index': args.binding_index}
        
        if idx == 0: 
            offset_dict = obtain_offsets(gs_points=gaussians.get_xyz, 
                                    prev_mesh_path=args.canonical_mesh_path, 
                                    edited_mesh_path=args.edited_mesh_path,
                                    extra_param=args.extra_param,
                                    mesh_rescaling_ratio=args.mesh_rescaling_ratio)
                
        
        verts, faces, vtx_color, verts_d, faces_d, new_obj_verts, new_obj_faces, verts_ori, faces_ori  = editing_composition(
            gaussians,
            d_xyz,
            d_normal,
            fid,
            deform,
            deform_back,
            appearance,
            offset_dict,
            disable_cano_w_color=args.disable_cano_w_color,
            enable_zero_motion_for_newobj=args.enable_zero_motion_for_newobj
        )



        # Note: When the rotation angle is large, this should also be disabled; When the mask looks like an extra piece, this should also be disabled
        verts_d, faces_d = None, None  
        # Render the mesh
        mask, mesh_image, depth, mask_of_new_obj, mask_ori_obj = updated_mesh_renderer(
            glctx,
            verts,
            faces,
            vtx_color,
            viewpoint_cam,
            True,
            verts_d=verts_d,
            faces_d=faces_d,
            new_obj_verts=new_obj_verts, new_obj_faces=new_obj_faces,
            verts_ori=verts_ori, faces_ori=faces_ori,
        )
        
        if verts_d is not None:
            # alpha blending
            # [h,w,1]-->[1,h,w]
            mask_  = torch.permute(mask, (2, 0, 1))
            depth_  = torch.permute(depth, (2, 0, 1))
            
            img_bg = torch.ones_like(mask_)
            depth_bg = torch.zeros_like(depth_)
            depth = mask_ * depth_ + (1 - mask_) * depth_bg
            mesh_image = mask_ * mesh_image + (1 - mask_) * img_bg
            # [1,h,w]-->[h,w,1]
            depth = torch.permute(depth, (1, 2, 0))
        

        
        gt_depth_path = viewpoint_cam.image_path.replace("rgbs", "depths")
        # # # read gt-depth and transform to tensor
        gt_depth = Image.open(gt_depth_path).convert("L")
        gt_depth = torch.tensor(np.array(gt_depth), dtype=torch.float32, device="cuda")
        gt_depth_np = gt_depth.detach().cpu().numpy()
        gt_depth_np = np.repeat(gt_depth_np[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        
        mask_ = mask.clone()
        mask_[mask_ > 0] = 1
        # tensor to bool
        mask_ = mask_.bool()
        depth[mask_] = (depth[mask_] - depth[mask_].min()) / (depth[mask_].max() - depth[mask_].min()) * 255
        depth[~mask_] = 0
        depth = np.squeeze(depth.cpu().numpy())
        pred_depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        
        # # # -----------------------------------------------------
        pred_normal = get_normal(depth)
        # mask the background of the normal, set the background to 255
        mask_np = mask.repeat(1,1,3).detach().cpu().numpy() * 255
        
        pred_normal = pred_normal* (mask_np/255) + (255-mask_np)
        pred_normal = pred_normal.astype(np.uint8)
        normal_list.append(pred_normal)
        # # # -----------------------------------------------------

        mask_of_new_obj_np = mask_of_new_obj.repeat(1,1,3).detach().cpu().numpy() * 255
        mask_ori_obj_np = mask_ori_obj.repeat(1,1,3).detach().cpu().numpy() * 255

        mask_inter = np.logical_and(mask_of_new_obj_np, mask_ori_obj_np).astype(np.uint8) * 255
        mask_ori_obj_np = mask_ori_obj_np - mask_inter
        mask_ori_obj_np[mask_ori_obj_np < 10] = 0
        mesh_image_np = mesh_image.permute(1, 2, 0).detach().cpu().numpy() * 255

        # Render the mesh itself
        mesh_image_shape = mesh_shape_renderer(verts, faces, viewpoint_cam)
        mesh_image_shape_np = mesh_image_shape.detach().cpu().numpy() * 255
        # Compose the final image
        gt_img_np = np.array(viewpoint_cam.original_image.permute(1, 2, 0).detach().cpu().numpy()* 255)
        
        d_xyz = d_xyz + offset_dict['off_xyz_gs_points'] 
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
            dataset.is_6dof,
        )
        gs_image = render_pkg["render"]
        gs_image = torch.clamp(gs_image, 0.0, 1.0)
        gs_image = gs_image.permute(1, 2, 0).detach().cpu().numpy() * 255
        
        final_img = np.hstack([cv2.resize(gt_img_np, gs_image.shape[:2][::-1]), 
                               gs_image, mask_np, mesh_image_np, mesh_image_shape_np, pred_depth, 
                               cv2.resize(gt_depth_np, gs_image.shape[:2][::-1])])
        
        # downsample
        final_img = cv2.resize(final_img, (final_img.shape[1] // 2, final_img.shape[0] // 2))
        final_images.append(final_img)
        mask_images.append(mask_np)
        mask_of_new_obj_images.append(mask_of_new_obj_np)
        mask_of_original_obj_images.append(mask_ori_obj_np)
        mesh_image_np_list.append(mesh_image_np)

    
    # Save the final video
    final_images = np.stack(final_images).astype(np.uint8)
    # No transposition needed

    # Save the mp4
    imageio.mimwrite(
        osp.join(args.save_path, "video.mp4"),
        final_images,
        fps=fps,
        codec="libx264"
    )

    # Save the normal video
    normal_list = np.stack(normal_list).astype(np.uint8)
    imageio.mimwrite(
        osp.join(args.save_path, "normal.mp4"),
        normal_list,
        fps=fps,
        codec="libx264"
    )

    # Save the mask video
    mask_images = np.stack(mask_images).astype(np.uint8)
    imageio.mimwrite(
        osp.join(args.save_path, "mask.mp4"),
        mask_images,
        fps=fps,
        codec="libx264"
    )
    
    # Save the mask of new obj video
    mask_of_new_obj_images = np.stack(mask_of_new_obj_images).astype(np.uint8)
    imageio.mimwrite(
        osp.join(args.save_path, "mask_of_new_obj.mp4"),
        mask_of_new_obj_images,
        fps=fps,
        codec="libx264"
    )
    
    # Save the mask of original obj video
    mask_of_original_obj_images = np.stack(mask_of_original_obj_images).astype(np.uint8)
    imageio.mimwrite(
        osp.join(args.save_path, "mask_of_original_obj.mp4"),
        mask_of_original_obj_images,
        fps=fps,
        codec="libx264"
    )

    # save the mesh_image_np video
    mesh_image_np_list = np.stack(mesh_image_np_list).astype(np.uint8)
    imageio.mimwrite(
        osp.join(args.save_path, "mesh_image.mp4"),
        mesh_image_np_list,
        fps=fps,
        codec="libx264"
    )
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--do_save_canonical_mesh", action="store_true", default=False)
    parser.add_argument("--save_vert_color", action="store_true", default=False, help='if true, save the vertex color')
    parser.add_argument("--enable_smooth_vert_color", action="store_true", default=False, help='if true, the vertex mesh will be smoothed')
    parser.add_argument("--disable_cano_w_color", action="store_true", default=False, help='if true, the canonical mesh will be rendered with color')
    parser.add_argument("--is_canonical", action="store_true", default=False, help='if true, save the canonical mesh with color, rather than the 1st mesh')
    parser.add_argument("--enable_zero_motion_for_newobj", action="store_true", default=False, help='if true, the new object will not have motion, for example, chair on the ground')
    
    parser.add_argument("--fps", type=int, default=10)
    # parser.add_argument("--total_frames", type=int, default=200)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--binding_index", type=int, default=10)
    parser.add_argument("--start_index", type=int, default=1, 
                        help="This parameter is used to save the canonical mesh of the second object when adding two objects. The index of the second object starts from the last one of the first object +1")
    
    # parser.add_argument("--camera_radius", type=float, default=4.0)
    # parser.add_argument("--camera_lookat", type=float, nargs="+", default=[0, 0, 0])
    # parser.add_argument("--camera_elevation", type=float, default=1.0)
    parser.add_argument("--mesh_rescaling_ratio", type=float, default=1, help="If you enlarge the mesh N times for rigging when editing the model, you need to set this parameter to N")
    
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--edited_mesh_path", type=str, default=None)
    parser.add_argument("--canonical_mesh_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None, help="the path to save the images and videos", required=True)
    parser.add_argument("--save_name", type=str, default=None, help='name of the output folder')
    

    # Fix random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = parser.parse_args(sys.argv[1:])

    if 'wo_color' in args.canonical_mesh_path:
        args.disable_cano_w_color = True
        print('-----------------------------disable_cano_w_color-----------------------------')
    
    if 'w_color' in args.canonical_mesh_path:
        args.disable_cano_w_color = False
        args.save_vert_color = True
        print('-----------------------------enable_cano_w_color, save_vert_color-----------------------------')
    
    
    # Load config file
    if args.config:
        config_data = load_config_from_file(args.config)
        combined_args = merge_config(config_data, args)
        args = Namespace(**combined_args)

    lp = lp.extract(args)
    op = op.extract(args)
    pp = pp.extract(args)

    # Updating save path
    unique_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_name = osp.basename(lp.source_path)
    folder_name = f"rendering-traj-{data_name}-{unique_str}" if not args.save_name else args.save_name
    
    if not lp.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        lp.model_path = os.path.join("./output/", unique_str[0:10])
    args.save_path = osp.join(args.save_path, folder_name)
    # Set up output folder
    print("Output folder: {}".format(lp.model_path))
    if not args.do_save_canonical_mesh:
        os.makedirs(args.save_path, exist_ok=True)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    rendering_trajectory(lp, op, pp, args.start_checkpoint, args.fps)

    # All done
    print("\nRendering complete.")
