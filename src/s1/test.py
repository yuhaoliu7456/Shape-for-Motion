import os, sys
import datetime
import torch
import uuid
import cv2
import imageio
import random

import os.path as osp
from tqdm import tqdm
import numpy as np
from PIL import Image

import trimesh
import nvdiffrast.torch as dr
from argparse import ArgumentParser, Namespace

from s1.scene import Scene
from s1.scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from s1.scene import DeformModelNormal as deform_model
from s1.scene import DeformModelNormalSep as deform_model_sep
from s1.scene import AppearanceModel as appearance_model

from s1.utils.renderer import mesh_renderer, mesh_shape_renderer
from s1.utils.general_utils import safe_state
from s1.utils.system_utils import load_config_from_file, merge_config
from s1.arguments import ModelParams, PipelineParams, OptimizationParams
from s1.gaussian_renderer import render



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def rendering_trajectory(dataset, opt, pipe, checkpoint, fps=24):
    args.model_path = dataset.model_path
    
    ## Gaussian model
    gaussians = gaussian_model(
        dataset.sh_degree,
        grid_res=dataset.grid_res,
        density_thres=opt.init_density_threshold,
        dpsr_sig=opt.dpsr_sig,
    )
    glctx = dr.RasterizeCudaContext()
    scene = Scene(dataset, gaussians, shuffle=False)
    
    ## Deform forward model
    deform = deform_model(is_blender=dataset.is_blender, is_6dof=dataset.is_6dof, model_name="deform")
    deform_normal = deform_model_sep(is_blender=dataset.is_blender,is_6dof=dataset.is_6dof,model_name="deform_normal",)

    ## Deform backward model
    deform_back = deform_model(is_blender=dataset.is_blender, is_6dof=dataset.is_6dof, model_name="deform_back")
    deform_back_normal = deform_model_sep(is_blender=dataset.is_blender,is_6dof=dataset.is_6dof,model_name="deform_back_normal",)

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
        print("Loading checkpoint from: {}".format(args.load_iteration))

    # Compose camera trajectory
    viewpoint_cam_stack = scene.getCameras().copy()
    
    # Create folders
    os.makedirs(osp.join(dataset.model_path,'mesh'), exist_ok=True)
    
    # misc settings
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    final_images = []
    for idx, viewpoint_cam in tqdm(enumerate(viewpoint_cam_stack)):
        
        fid = viewpoint_cam.fid
        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)

        # Query the gaussians
        d_xyz, d_rotation, d_scaling, _ = deform.step(gaussians.get_xyz.detach(), time_input)
        d_normal = deform_normal.step(gaussians.get_xyz.detach(), time_input)

        # Query the GT image
        gt_img_np = np.array(viewpoint_cam.original_image.permute(1, 2, 0).detach().cpu().numpy()* 255)
        
        # add the gaussian render part
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

        # Query the mesh rendering rgb
        mask, mesh_image, depth, verts, faces, vtx_color = mesh_renderer(
            glctx,
            gaussians,
            d_xyz,
            d_normal,
            fid,
            deform_back,
            appearance,
            False,
            True,
            viewpoint_cam,
        )
        
        # save mesh
        # verts, faces = gaussians.export_mesh(deform, deform_normal, t=fid)
        # mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy())
        # mesh.export(osp.join(dataset.model_path,'mesh', f"mesh_tid_{str(viewpoint_cam.tid)}.ply"))

        gt_depth_path = viewpoint_cam.image_path.replace("rgbs", "depths")
        gt_depth = Image.open(gt_depth_path).convert("L")
        gt_depth = torch.tensor(np.array(gt_depth), dtype=torch.float32, device="cuda")
        gt_depth_np = gt_depth.detach().cpu().numpy()
        gt_depth_np = np.repeat(gt_depth_np[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        
        mask_ = mask.clone()
        mask_[mask_ > 0] = 1
        mask_ = mask_.bool()
        depth[mask_] = (depth[mask_] - depth[mask_].min()) / (depth[mask_].max() - depth[mask_].min()) * 255
        depth[~mask_] = 0
        depth = np.squeeze(depth.cpu().numpy())
        pred_depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        
        mask_np = mask.repeat(1,1,3).detach().cpu().numpy() * 255
        mesh_image_np = mesh_image.permute(1, 2, 0).detach().cpu().numpy() * 255

        # Render the mesh itself
        mesh_image_shape = mesh_shape_renderer(verts, faces, viewpoint_cam)
        mesh_image_shape_np = mesh_image_shape.detach().cpu().numpy() * 255
        
        # Compose the final image
        final_img = np.hstack([gt_img_np, gs_image, mask_np, mesh_image_np, mesh_image_shape_np, pred_depth, gt_depth_np])
        final_img = cv2.resize(final_img, (final_img.shape[1] // 2, final_img.shape[0] // 2))
        final_images.append(final_img)
        

    # Save the final video
    final_images = np.stack(final_images).astype(np.uint8)

    imageio.mimwrite(
        osp.join(dataset.model_path, "video.mp4"),
        final_images,
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
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--load_iteration", type=int, default=-1)
    
    # Fix random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = parser.parse_args(sys.argv[1:])

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

    # Set up output folder
    lp.model_path = osp.join(lp.model_path, "test_"+args.start_checkpoint.split("/")[-1])
    print("Output folder: {}".format(lp.model_path))
    os.makedirs(lp.model_path, exist_ok=True)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    rendering_trajectory(lp, op, pp, args.start_checkpoint, args.fps)

    # All done
    print("\nRendering complete.")
