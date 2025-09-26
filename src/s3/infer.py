import os
import torch
import logging
import argparse

from copy import deepcopy

from diffusers.models.model_loading_utils import load_state_dict
from diffusers.utils import export_to_video

from s3.models.controlnet_sdv import ControlNetSDVModel
from s3.models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from s3.pipeline import StableVideoDiffusionPipelineControlNet
from s3.dataset import data_preparation



# Main script
if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run inference for Stable Video Diffusion ControlNet")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-video-diffusion-img2vid", help="Path to the pretrained model")
    parser.add_argument("--ori_input_folder", type=str, required=True, help="Folder containing validation images")
    parser.add_argument("--edited_mask_folder", type=str, required=True, help="Folder containing validation mask images")
    parser.add_argument("--edited_geometry_folder", type=str, required=True, help="Folder containing validation control images")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save output images")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing the model weights")
    parser.add_argument("--height", type=int, default=512, help="Height of the output images")
    parser.add_argument("--width", type=int, default=768, help="Width of the output images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--in_channels", type=int, default=14, help="Number of input channels for the UNet model")
    parser.add_argument("--revise_first_layer", default=True, help="Whether to revise the first layer of the UNet.")
    parser.add_argument("--in_channels_controlnet", type=int, default=14, help="Number of input channels for the ControlNet model")
    parser.add_argument("--mask_bg_type", type=str, default="random", help="The type of mask ref.")
    parser.add_argument("--mask_ref_type", type=str, default="gray", help="The type of mask ref.")
    parser.add_argument("--mask_guide_type", type=str, default="black", help="The type of mask guide.")
    parser.add_argument("--max_guidance_scale", type=float, default=3.0, help="Maximum guidance scale for the controlnet.")
    parser.add_argument("--min_guidance_scale", type=float, default=1.0, help="Minimum guidance scale for the controlnet.")
    parser.add_argument("--conditioning_channels", type=int, default=4, help="The number of conditioning channels. (used in controlnet)")
    parser.add_argument("--mask_inpaint_type", type=str, default="gray", help="The type of mask inpaint.")
    parser.add_argument("--min_dilation", type=int, default=8, help="The minimum dilation of the mask.")
    parser.add_argument("--max_dilation", type=int, default=15, help="The maximum dilation of the mask.")
    parser.add_argument("--edited_texture_folder",type=str,default=None,help=("the path of the original ref image"),)
    parser.add_argument("--ori_mask_folder",type=str,default=None,help=("the path of the original mask image"),)
    parser.add_argument("--excluded_mask_folder", type=str, default=None, help="The excluded mask folder for inpainting, particularly when conduct the adding operation.")
    parser.add_argument("--use_excluded_mask_for_local_edit", action="store_true", help="Whether to use the excluded mask for local edit.")
    parser.add_argument("--cache_dir", type=str, default="/apdcephfs_jn/share_302245012/pretrained_models", help="The cache directory.")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="The number of inference steps.")
    


    # Parse arguments
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir , exist_ok=True)

    # Prepare validation data
    val_data = data_preparation(args=args)
    
    # Load the models
    print(f"Loading weights from {args.weights_dir}")
    args.controlnet_weights = os.path.join(args.weights_dir, "model.safetensors")
    args.unet_weights = os.path.join(args.weights_dir, "model_1.safetensors")
    
    assert os.path.exists(args.controlnet_weights), f"ControlNet weights not found at {args.controlnet_weights}"
    assert os.path.exists(args.unet_weights), f"UNet weights not found at {args.unet_weights}" 

    # Load the ControlNet model
    backup_unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                                             cache_dir=args.cache_dir)

    controlnet = ControlNetSDVModel.from_unet(backup_unet, 
                                              revise_first_layer=args.revise_first_layer, in_channels_controlnet=args.in_channels_controlnet,
                                              conditioning_channels=args.conditioning_channels,)
    controlnet.load_state_dict(load_state_dict(args.controlnet_weights))

    # Load the UNet model
    new_config = deepcopy(backup_unet.config)
    new_config["in_channels"] = args.in_channels
    unet = UNetSpatioTemporalConditionControlNetModel.from_config(new_config)
    state_dict = load_state_dict(args.unet_weights)
    unet.load_state_dict(state_dict)
    logger.info("ControlNet and UNet loaded successfully")

    # Create the pipeline
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(args.pretrained_model_name_or_path, 
                                                                                controlnet=controlnet,unet=unet,     
                                                                                cache_dir=args.cache_dir)
    pipeline.enable_model_cpu_offload()
    # Additional pipeline configurations can be added here
    #pipeline.enable_xformers_memory_efficient_attention()
    
    # Set random seed
    generator = torch.Generator().manual_seed(args.seed)

    
    edited_output = pipeline(image=val_data['ori_frame'], 
                                ori_mask=val_data['ori_mask'],
                                edited_geometry=val_data['edited_geometry'], 
                                edited_mask=val_data['edited_mask'],
                                conditional_bg=val_data['bg'],
                                inpaint_mask=val_data['inpaint_mask'],
                                decode_chunk_size=8,
                                motion_bucket_id=10,
                                width=args.width,
                                height=args.height,
                                generator=generator, 
                                num_inference_steps=args.num_inference_steps,
                                min_guidance_scale=args.min_guidance_scale,
                                max_guidance_scale=args.max_guidance_scale,
                                ).frames

    save_path = os.path.join(args.output_dir, os.path.basename(args.edited_mask_folder).split('.')[0]+ ".mp4")
    export_to_video(val_data['ori_rgb'], save_path.replace(".mp4", "_ori.mp4"), fps=7)

    # Export the output to video
    count = 0
    while os.path.exists(save_path.replace(".mp4", f"_{count}.mp4")):
        count += 1
    save_path = save_path.replace(".mp4", f"_{count}.mp4")
    
    export_to_video(edited_output[0], save_path, fps=7)
    print('Saving to: ', save_path)