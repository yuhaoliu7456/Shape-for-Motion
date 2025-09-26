import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from PIL import Image

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--ckpt-root', type=str, default=None, required=True)
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    
    depth_anything.load_state_dict(torch.load(f'{args.ckpt_root}/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename)

        depth = depth_anything.infer_image(raw_image, args.input_size)
        depth_np = depth.copy()
        # depth = (depth - depth.min()) / (depth.max() - depth.min())

        mask_img = cv2.imread(filename.replace('rgbs', 'masks').replace('jpg', 'png'))
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)/255
        mask = mask_img[:, :] > 0.1
        # import pdb; pdb.set_trace()
        depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
        depth[~mask] = 0

        depth_pt = depth.copy()

        depth = (depth * 255).astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # # add the transformation from depth to normal
        a=np.pi * 2.0
        bg_th = 0.1
        x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
        y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
        z = np.ones_like(x) * a
        x[depth_pt < bg_th] = 0
        y[depth_pt < bg_th] = 0
        normal = np.stack([x, y, z], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
        normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        # save the normal image using PIL
        normal_image = Image.fromarray(normal_image)
        # normal_image.save('normal.png')

        # normal_image.resize((768, 512)).save('/mnt/petrelfs/liuyuhao/Code/SVD-ControlNet/samples/camel/normal/'+os.path.basename(filename))
        # Image.fromarray(depth).resize((768, 512)).save('/mnt/petrelfs/liuyuhao/Code/SVD-ControlNet/samples/camel/depth/'+os.path.basename(filename))

        # import pdb; pdb.set_trace()

        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
            output_normal_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png').replace('depth', 'normal')
            os.makedirs(os.path.dirname(output_normal_path), exist_ok=True)
            normal_image.save(output_normal_path)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)