
All models are stored in `Shape-for-Motion/pretrained_weights` by default, and the file structure is as follows
```shell
Shape-for-Motion
  ├──pretrained_weights
  │  ├──ckpt.md
  │  ├──shape-for-motion
  │  │  ├──README.md
  │  │  ├──model_1.safetensors # denoising UNet
  │  │  ├──model.safetensors   # ControlNet
  │  │  ├──random_states_0.pkl
  │  │  ├──scheduler.bin
  │  ├──depth_anything_v2_vitl.pth
  │  ├──sv3d_p.safetensors
  │  ├──sam2_hiera_large.pt
  │  ├──models--stabilityai--stable-video-diffusion-img2vid
  ├──...
```

## Download Shape-for-Motion model
To download the Shape-for-Motion model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
# Switch to the directory named 'Shape-for-Motion'
cd Shape-for-Motion/pretrained_weights
# Use the huggingface-cli tool to download the model
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download LeoLau/Shape-for-Motion --local-dir ./shape-for-motion
```

<details>
<summary>💡Tips for using huggingface-cli (network problem)</summary>

##### 1. Using HF-Mirror

If you encounter slow download speeds in China, you can try a mirror to speed up the download process. For example,

```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download xxx
```

##### 2. Resume Download

`huggingface-cli` supports resuming downloads. If the download is interrupted, you can just rerun the download 
command to resume the download process.

Note: If an `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` like error occurs during the download 
process, you can ignore the error and rerun the download command.

</details>

 
## Download other pre-trained ckpts
```shell
cd Shape-for-Motion/pretrained_weights
bash download.sh
```