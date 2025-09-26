import os, sys
import json
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
import trimesh
from PIL import Image
import nvdiffrast.torch as dr

from s1.scene import Scene
from s1.scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from s1.scene import DeformModelNormal as deform_model
from s1.scene import DeformModelNormalSep as deform_model_sep
from s1.scene import AppearanceModel as appearance_model
from s1.nvdiffrast_utils import regularizer
from s1.gaussian_renderer import render
from s1.utils.renderer import mesh_renderer
from s1.utils.loss_utils import l1_loss, ssim, ScaleAndShiftInvariantLoss
from s1.utils.general_utils import safe_state, get_linear_noise_func
from s1.utils.image_utils import get_psnr

from s1.utils.system_utils import load_config_from_file, merge_config, code_backup
from s1.arguments import ModelParams, PipelineParams, OptimizationParams



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training(
    dataset,
    opt,
    pipe,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    log_every=1000,
):
    if checkpoint and opt.resume_iter > 0:
        first_iter = opt.resume_iter
    else:
        first_iter = opt.first_iter
    args.model_path = dataset.model_path
    
    ## Gaussian model
    gaussians = gaussian_model(
        dataset.sh_degree,
        grid_res=dataset.grid_res,
        density_thres=opt.init_density_threshold,
        dpsr_sig=opt.dpsr_sig,
    )

    glctx = dr.RasterizeCudaContext()
    
    scene = Scene(dataset, gaussians, shuffle=True)
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
    
    # Load checkpoint
    if checkpoint:
        gaussians.load_ply(checkpoint, iteration=opt.resume_iter)
        deform.load_weights(checkpoint, iteration=opt.resume_iter)
        deform_normal.load_weights(checkpoint, iteration=opt.resume_iter)
        deform_back.load_weights(checkpoint, iteration=opt.resume_iter)
        deform_back_normal.load_weights(checkpoint, iteration=opt.resume_iter)
        appearance.load_weights(checkpoint, iteration=opt.resume_iter)
    
    # Training setup
    gaussians.training_setup(opt)
    deform.train_setting(opt)
    deform_normal.train_setting(opt)
    deform_back.train_setting(opt)
    deform_back_normal.train_setting(opt)
    appearance.train_setting(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(
        lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000
    )
    first_iter += 1

    DPSR_ITER = opt.dpsr_iter
    ANCHOR_ITER = opt.anchor_iter
    ANCHOR_EVERY = opt.anchor_interval
    NORMAL_WARMUP_ITER = 2000

    
    ssi_loss = ScaleAndShiftInvariantLoss(alpha=0, reduction='a').cuda()

    # inp frames
    inp_viewpoint_stack = [] # first row in the matrix
    multi_viewpoint_stack = [] # first column(novel views of the 1st frame) in the matrix
    novel_viewpoint_stack = [] # the rest of the matrix

    inp_viewpoint_stack_for_use = None
    novel_viewpoint_stack_for_use = None
    first_multi_views_for_use = None
    start_id = int(dataset.time_index.split('-')[0]) if dataset.time_index is not None else 0
    for each_camera in scene.getCameras().copy():
        if each_camera.vid == 0:
            inp_viewpoint_stack.append(each_camera)
        else:
            novel_viewpoint_stack.append(each_camera)
        if each_camera.tid == start_id:
            multi_viewpoint_stack.append(each_camera)

    time_interval=0 # useless for blender data
    
    for iteration in range(first_iter, opt.iterations + 1):
        torch.cuda.empty_cache()
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        deform.update_learning_rate(iteration)
        deform_normal.update_learning_rate(iteration)
        deform_back.update_learning_rate(iteration)
        deform_back_normal.update_learning_rate(iteration)
        appearance.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not inp_viewpoint_stack_for_use:
            inp_viewpoint_stack_for_use = inp_viewpoint_stack.copy()
        if not novel_viewpoint_stack_for_use:
            novel_viewpoint_stack_for_use = novel_viewpoint_stack.copy()
        if not first_multi_views_for_use:
            first_multi_views_for_use = multi_viewpoint_stack.copy()

        if iteration > opt.sample_interval_novel_views and iteration % opt.sample_interval_novel_views == 0:
            viewpoint_cam = novel_viewpoint_stack_for_use.pop(random.randint(0, len(novel_viewpoint_stack_for_use) - 1))
        else:
            # sample from inp_viewpoint_stack
            viewpoint_cam = inp_viewpoint_stack_for_use.pop(random.randint(0, len(inp_viewpoint_stack_for_use) - 1))
            
        if iteration < opt.warm_up and len(first_multi_views_for_use) > 0:
            viewpoint_cam = first_multi_views_for_use.pop((random.randint(0, len(first_multi_views_for_use) - 1)))
        
        
        fid = viewpoint_cam.fid
        # Deform the gaussians to time step t
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling, d_normal = 0.0, 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = (
                0
                if dataset.is_blender
                else torch.randn(1, 1, device="cuda").expand(N, -1)
                * time_interval
                * smooth_term(iteration)
            )
            d_xyz, d_rotation, d_scaling, _ = deform.step(
                gaussians.get_xyz.detach(), time_input + ast_noise
            )
            
            if iteration >= DPSR_ITER + NORMAL_WARMUP_ITER:
                d_normal = deform_normal.step(
                    gaussians.get_xyz.detach(), time_input + ast_noise
                )
            else:
                d_normal = 0.0
        # Gaussian splatting
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
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        losses = {}
        psnr = {}

        # Deform the time step t gaussian back to canonical space
        if iteration < opt.warm_up:
            d_xyz_back, d_rotation_back, d_scaling_back, d_normal_back = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = (
                0
                if dataset.is_blender
                else torch.randn(1, 1, device="cuda").expand(N, -1)
                * time_interval
                * smooth_term(iteration)
            )
            deformed_xyz = gaussians.get_xyz + d_xyz
            d_xyz_back, d_rotation_back, d_scaling_back, _ = deform_back.step(
                deformed_xyz.detach(), time_input + ast_noise
            )
            ## Calculate the cycle consistency loss
            cycle_loss_xyz = l1_loss(-d_xyz_back, d_xyz)
            cycle_loss_rotation = l1_loss(-d_rotation_back, d_rotation)
            cycle_loss_scaling = l1_loss(-d_scaling_back, d_scaling)
            if iteration >= DPSR_ITER + NORMAL_WARMUP_ITER:
                d_normal_back = deform_back_normal.step(
                    gaussians.get_xyz.detach(), time_input + ast_noise
                )
                cycle_loss_normal = l1_loss(-d_normal_back, d_normal)
                cycle_loss = (
                    cycle_loss_xyz
                    + cycle_loss_rotation
                    + cycle_loss_scaling
                    + cycle_loss_normal
                ) / 4.0
            else:
                cycle_loss = (
                    cycle_loss_xyz + cycle_loss_rotation + cycle_loss_scaling
                ) / 3.0
            losses["cycle_loss"] = cycle_loss

        # DPSR normal initialization, resample
        
        if iteration == DPSR_ITER:
            gaussians.normal_initialization(args, opt, dataset, deform, d_xyz, d_rotation, d_scaling)
        
        if iteration >= DPSR_ITER:
            # DPSR
            freeze_pos = iteration < DPSR_ITER + opt.normal_warm_up
            mask, mesh_image, pred_depth, verts, faces, _ = mesh_renderer(
                glctx,
                gaussians,
                d_xyz,
                d_normal,
                fid,
                deform_back,
                appearance,
                freeze_pos,
                dataset.white_background,
                viewpoint_cam,
            )
        
            gt_image = viewpoint_cam.original_image.cuda() # [3, h, w] range: [0,1]
            gt_mask = viewpoint_cam.gt_alpha_mask.cuda()  # [h,w,1] range: [0,1]
            gt_depth = viewpoint_cam.gt_depth.cuda() # [h,w,1] range: [0,1]

            ###  Depth loss
            depth_loss_value =  ssi_loss(pred_depth.permute(2,0,1), gt_depth.permute(2,0,1), gt_mask.permute(2,0,1))
            if viewpoint_cam.vid == 0:
                losses["depth_loss"] = depth_loss_value * opt.depth_loss_weight 
            else:
                losses["depth_loss"] = depth_loss_value * opt.novel_views_depth_loss_weight
                
            ### Mask loss
            mask_loss = l1_loss(mask, gt_mask)
            if viewpoint_cam.vid == 0:
                losses["mask_loss"] = mask_loss * 100 * opt.mask_loss_weight
            else:
                losses["mask_loss"] = mask_loss * 100 * opt.novel_views_mask_loss_weight

            ### mesh image loss
            Ll1 = l1_loss(mesh_image, gt_image)
            mesh_img_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(mesh_image, gt_image))
            if viewpoint_cam.vid == 0:
                losses["mesh_img_loss"] = mesh_img_loss * opt.mesh_img_loss_weight
            else:
                losses["mesh_img_loss"] = mesh_img_loss * opt.novel_views_mesh_img_loss_weight
            

            psnr["mesh_img_psnr"] = get_psnr(mesh_image.detach(), gt_image.detach())
            ## Laplacian loss
            laplacian_scale = 1000 * lp.laplacian_loss_weight
            t_iter = iteration / opt.iterations
            laplacian_loss = (
                regularizer.laplace_regularizer_const(verts, faces.long())
                * laplacian_scale
                * (1 - t_iter)
            )
            losses["laplacian_loss"] = laplacian_loss
            ## Anchoring loss
            if (
                iteration > ANCHOR_ITER
                and iteration % ANCHOR_EVERY == 0
                and lp.use_anchor > 0
            ):
                print(f"Anchoring at iteration {iteration} under fid {fid}")
                anchor_loss = gaussians.anchor_mesh(
                    verts,
                    faces,
                    deform,
                    deform_back,
                    fid,
                    search_radius=opt.anchor_search_radius,
                    topn=opt.anchor_topn,
                    bs=opt.anchor_n_1_bs,
                    increase_bs=opt.anchor_0_1_bs,
                )
                losses["anchor_loss"] = anchor_loss * opt.anchor_loss_weight
            
        # Gaussian image loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        
        img_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        if viewpoint_cam.tid!=0 and viewpoint_cam.vid!=0:
            img_loss *= args.novel_views_gs_loss_weight
            

        losses["img_loss"] = img_loss * opt.img_loss_weight

        # Log psnr
        psnr["img_psnr"] = get_psnr(image.detach(), gt_image.detach())

        ## Total loss
        loss = 0
        for k, v in losses.items():
            loss += v
        loss.backward()

        # log mesh and images
        if iteration % log_every == 0:
            with torch.no_grad():
                if iteration > DPSR_ITER:
                    mask_np = mask.repeat(1, 1, 3).detach().cpu().numpy() * 255
                    mask_np = np.clip(mask_np, 0, 255)
                    mesh_img_np = (
                        mesh_image.permute(1, 2, 0).detach().cpu().numpy() * 255
                    )
                    mesh_img_np = np.clip(mesh_img_np, 0, 255)
                    gt_mask_np = gt_mask.repeat(1, 1, 3).detach().cpu().numpy() * 255
                    imageio.imwrite(
                        osp.join(
                            args.model_path, "logs", f"mesh_image_{iteration }.jpg"
                        ),
                        mesh_img_np.astype(np.uint8),
                    )
                    imageio.imwrite(
                        osp.join(args.model_path, "logs", f"mask_{iteration}.jpg"),
                        mask_np.astype(np.uint8),
                    )
                    imageio.imwrite(
                        osp.join(args.model_path, "logs", f"gt_mask_{iteration}.jpg"),
                        gt_mask_np.astype(np.uint8),
                    )
            img_np = image.permute(1, 2, 0).detach().cpu().numpy() * 255
            img_np = np.clip(img_np, 0, 255)
            gt_image_np = gt_image.permute(1, 2, 0).detach().cpu().numpy() * 255
            imageio.imwrite(
                osp.join(args.model_path, "logs", f"image_{iteration}.jpg"),
                img_np.astype(np.uint8),
            )
            imageio.imwrite(
                osp.join(args.model_path, "logs", f"gt_image_{iteration}.jpg"),
                gt_image_np.astype(np.uint8),
            )
            
        if iteration % log_every == 0:
            if iteration >= DPSR_ITER:
                
                verts, faces = gaussians.export_mesh(deform, deform_normal, t=fid)
                mesh = trimesh.Trimesh(
                    verts.detach().cpu().numpy(), faces.detach().cpu().numpy()
                )
                mesh.export(
                    osp.join(args.model_path, "logs_geo", f"mesh_{iteration}_tid_{str(viewpoint_cam.tid)}.ply")
                )


        iter_end.record()


        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                log_str = {}
                for k in losses.keys():
                    log_str[k] = f"{losses[k].item():.04f}"
                for k in psnr.keys():
                    log_str[k] = f"{psnr[k]:.04f}"
                progress_bar.set_postfix(log_str)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()


            # Log and save
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                deform_normal.save_weights(args.model_path, iteration)
                deform_back.save_weights(args.model_path, iteration)
                deform_back_normal.save_weights(args.model_path, iteration)
                appearance.save_weights(args.model_path, iteration)

            # Densification and pruning
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        args.prune_threshold,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                deform.optimizer.step()
                deform_normal.optimizer.step()
                deform_back.optimizer.step()
                deform_back_normal.optimizer.step()
                appearance.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad(set_to_none=True)
                deform_normal.optimizer.zero_grad(set_to_none=True)
                deform_back.optimizer.zero_grad(set_to_none=True)
                deform_back_normal.optimizer.zero_grad(set_to_none=True)
                appearance.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_path = osp.join(args.model_path, "checkpoint")
                os.makedirs(save_path, exist_ok=True)
                gaussians.save_ply(
                    osp.join(save_path, "pointcloud_{}.ply".format(iteration))
                )

        torch.cuda.empty_cache()






if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[25000,  50000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=[25000, 50000],
    )
    parser.add_argument("--start_checkpoint", type=str, default=None, help="Path to checkpoint to start from")
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--config", type=str, default=None)
    
    # Fix random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Load config file
    if args.config:
        config_data = load_config_from_file(args.config)
        combined_args = merge_config(config_data, args)
        args = Namespace(**combined_args)

    print("Optimizing " + args.model_path)
    lp = lp.extract(args)
    op = op.extract(args)
    pp = pp.extract(args)

    # Updating save path
    # unique_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_str = '' ## NOTE: we use gpu number as the unique string 
    if args.expname != 'debug':
        folder_name = f"{unique_str}-{args.expname}"
    else:
        folder_name = "debug"
        
    if not lp.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        lp.model_path = os.path.join("./output/", unique_str[0:10])
    lp.model_path = osp.join(lp.model_path, folder_name)
    # Set up output folder
    print("Output folder: {}".format(lp.model_path))
    os.makedirs(lp.model_path, exist_ok=True)
    os.makedirs(osp.join(lp.model_path, "logs"), exist_ok=True)
    os.makedirs(osp.join(lp.model_path, "logs_geo"), exist_ok=True)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Save all parameters into file
    combined_args = vars(Namespace(**vars(lp), **vars(op), **vars(pp)))
    # Convert namespace to JSON string
    args_json = json.dumps(combined_args, indent=4)
    # Write JSON string to a text file
    with open(osp.join(lp.model_path, "cfg_args.txt"), "w") as output_file:
        output_file.write(args_json)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp,
        op,
        pp,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.log_every,
    )


    # All done
    print("\nTraining complete.")
