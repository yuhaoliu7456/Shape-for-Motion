# cd $DEPTH_ANYTHING_V2_ROOT
# source /opt/conda/etc/profile.d/conda.sh
# conda activate dgmesh
input_path="/apdcephfs/private_yuhaliu/Code/Shpe-for-Motion/data/s1_processed/otter_long_5s/raw-rgbs/00000.png"
depth_path="/apdcephfs/private_yuhaliu/Code/v4-decouple/utils/normal.png"
ckpt_root="/apdcephfs_jn/share_302245012/yuhaliu/Weights"

python run_w_normal.py \
    --img-path $input_path \
    --grayscale \
    --input-size 1080 \
    --outdir $depth_path \
    --pred-only \
    --ckpt-root $ckpt_root
rm -rf $depth_path/*.mp4