# Shape-for-Motion: Precise and Consistent Video Editing with 3D Proxy [Siggraph Asia 2025]

### [Project Page](https://shapeformotion.github.io//) | [Video](https://shapeformotion.github.io/video/demo.mp4) | [Paper](https://arxiv.org/abs/2506.22432) | [Data](https://1drv.ms/f/c/411e3f963c5f74e5/EpJ0BN4VwbBGsUr5-cleSvcBPPd3Nj0BMwXBt1vhu7qONw?e=hJZRGv)

Authors: [Yuhao LIU](https://yuhaoliu7456.github.io/)<sup>1</sup>, [Tengfei Wang](https://tengfei-wang.github.io/)<sup>2</sup>, [Fang Liu](https://fawnliu.github.io/)<sup>1</sup>, [Zhenwei Wang](https://zhenwwang.github.io/)<sup>1</sup>, [Rynson W.H. Lau](https://www.cs.cityu.edu.hk/~rynson/)<sup>1</sup>  
Affiliations: <sup>1</sup>City University of Hong Kong, <sup>2</sup>Tencent Hunyuan  

## [Project Page](https://shapeformotion.github.io/) | [Paper](https://arxiv.org/abs/2506.22432)

<p align="center">
  <img src="./imgs/teaser.jpg" alt="Shape-for-Motion Teaser"  style="width:80%">
</p>


We propose a 3D-aware video editing framework, Shape-for-Motion, to support precise and consistent video object manipulation by reconstructing an editable 3D mesh to serve as control signals for video generation.

## Installation

```
conda create -n shape-for-motion python=3.9
conda activate shape-for-motion

# Install PyTorch
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Other packages
bash install.sh
```


## Testing

1. Refer to [Weights](./pretrained_weights/ckpt.md) to download pre-trained models to `Shape-for-Motion/pretrained_weights`.



2. Run the following command to extract frames from the video (refer to [Data](./data/DATA.md) for more details):
```bash
bash scripts/s1_prepare_data.sh NAME_OF_SAMPLE_VIDEO.mp4 X1 Y1 X2 Y2
# Note: 
# (1) NAME_OF_SAMPLE_VIDEO.mp4 must exist under data/raw_videos
# (2) X/Y[1/2] denotes the coordinates of two sampled points used in SAM segmentation.

```


3. Start reconstructing the object:

```bash
bash scripts/s1_train.sh NAME_OF_SAMPLE_VIDEO EXP_NAME_1
```

4. Test the optimized model:
```bash
bash scripts/s1_test.sh NAME_OF_SAMPLE_VIDEO EXP_NAME_1
# Make sure that EXP_NAME_1 is the same as the one used during optimization

# After testing, save the canonical mesh
bash s2_save_cano.sh  NAME_OF_SAMPLE_VIDEO EXP_NAME_1
```




5. Manual editing

```
First, save the canonical mesh. Import it into Blender, make the desired manual edits, and save it as a new canonical mesh. Then propagate the edits from canonical space to all frames to generate the edited images.

Note that when saving the canonical mesh, we insert indices for each vertex. Remember not to modify these indices during editing; they are automatically saved after you manually edit the mesh.
```

6. Propagate the editing

```
bash s2_propagate_edit.sh
```

7. Generative rendering

```bash
# Before generative rendering, prepare the used data first:
bash scripts/s3_prepare_data.sh SAVE_NAME

bash s3_test.sh SAVE_NAME
```

## Toy Examples

We have provided several toy examples so that you do not need to perform reconstruction or manual editing. You only need to follow the steps below. For more details, please refer to [Example.md](./Example.md).


## Citation
If you find our code or paper helps, please consider citing:
```
@article{liu2025shape,
  title={Shape-for-Motion: Precise and Consistent Video Editing with 3D Proxy},
  author={Liu, Yuhao and Wang, Tengfei and Liu, Fang and Wang, Zhenwei and Lau, Rynson WH},
  journal={arXiv preprint arXiv:2506.22432},
  year={2025}
}
```


Thanks to the authors of [DG-Mesh](https://github.com/Isabella98Liu/DG-Mesh/), [Deformable 3DGS](https://ingra14m.github.io/Deformable-Gaussians/), and [Diffusers](https://github.com/huggingface/diffusers/)  for their excellent code!