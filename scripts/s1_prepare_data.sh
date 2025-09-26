#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || { echo "CAN NOT switch to the folder"; exit 1; }
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0"
Project_ROOT=$(pwd)
CONDA_ENV="shape-for-motion"

# ---------------The following parameters need to be manually set case-by-case----------------
DATA_ROOT="${Project_ROOT}/data/s1_processed" # the root folder to save the data
ckpt_root="${Project_ROOT}/pretrained_weights"


SAMPLE_VIDEO=$1 # name of the video in the data/raw_videos folder, e.g., "example.mp4" or "pexel-dogs311.mp4"
input_video="${Project_ROOT}/data/raw_videos/$SAMPLE_VIDEO"


# two points of the object in the first frame, used for SAM mask extraction
i_x1=$2 #431
i_y1=$3 #169
i_x2=$4 #430
i_y2=$5 #249
 
frame_rate=30 # by default is 30
num_frames=200 # by default 21
image_frame_ratio=0.8 # 0.8 by default (when crop the object, the ratio of the object to the image)
# -----------------------------------------------------------------------

filename=$(basename ${input_video})
FOLDER="${filename%.*}" # for mp4 input
if [[ ${input_video} == *.mp4 ]]; then
    FOLDER="${filename%.*}" # for mp4 input
else
    FOLDER=$(basename $(dirname ${input_video})) # for directory path input
fi

echo ${FOLDER}-------------------------------
output_path="${Project_ROOT}/outputs/s1/${FOLDER}"


SCRIPT_ROOT="${Project_ROOT}/utils"
SV3D_ROOT="${Project_ROOT}/tools/SV3D"
SAM_ROOT="${Project_ROOT}/tools/SAM2"
DEPTH_ANYTHING_V2_ROOT="${Project_ROOT}/tools/Depth-Anything-V2"


# create target folders
novel_views_folder=$DATA_ROOT/$FOLDER/novelviews
mkdir -p $novel_views_folder
mkdir -p $DATA_ROOT/$FOLDER/raw-rgbs
mkdir -p $DATA_ROOT/$FOLDER/raw-masks
mkdir -p $DATA_ROOT/$FOLDER/raw-depths
mkdir -p $DATA_ROOT/$FOLDER/processed_rgbs
mkdir -p $DATA_ROOT/$FOLDER/processed_masks
mkdir -p $DATA_ROOT/$FOLDER/processed_depths

input_path=$DATA_ROOT/$FOLDER/raw-rgbs
mask_path=$DATA_ROOT/$FOLDER/raw-masks
depth_path=$DATA_ROOT/$FOLDER/raw-depths



# 1.  extracted frames
echo "---------------extract frames--------------"
cd $SCRIPT_ROOT
python raw_frames_extraction_from_video.py $input_video $DATA_ROOT/$FOLDER/raw-rgbs $frame_rate $num_frames


# 2. extract masks
echo "---------------Start SAM--------------"
cd $SAM_ROOT
source /opt/conda/etc/profile.d/conda.sh
conda activate $CONDA_ENV
python video_pred.py $input_path $mask_path $i_x1 $i_y1 $i_x2 $i_y2 $ckpt_root/sam2_hiera_large.pt


# # 3. extract depths
echo "---------------Start Depth-Anything-V2--------------"
cd $DEPTH_ANYTHING_V2_ROOT
source /opt/conda/etc/profile.d/conda.sh
conda activate $CONDA_ENV
python run.py \
    --img-path $input_path \
    --grayscale \
    --input-size 1080 \
    --outdir $depth_path \
    --pred-only \
    --ckpt-root $ckpt_root
rm -rf $depth_path/*.mp4



# 3. padding and masking
cd $SCRIPT_ROOT
echo "---------------Start padding and masking--------------"
python prepare_data_for_novelviews_gen.py $input_path $mask_path $depth_path $DATA_ROOT/$FOLDER $image_frame_ratio



echo "---------------Start SV3D--------------"
cd $SV3D_ROOT
source /opt/conda/etc/profile.d/conda.sh
conda activate $CONDA_ENV
python scripts/sampling/sv3d.py \
    --sv3d_version sv3d_p \
    --elevations_deg 0 \
    --input_path $DATA_ROOT/$FOLDER/processed_rgbs \
    --output_folder $DATA_ROOT/$FOLDER \
    --num_steps 20 \
    --ckpt_path $ckpt_root/sv3d_p.safetensors


    
echo "---------------Start Segmentation--------------"
cd $SCRIPT_ROOT
# Iterate through each folder in novel_views_folder
for folder in "$novel_views_folder"/*/; do
    if [ -d $folder ]; then
        # Check if rgb/p_0 subfolder exists
        rgb_path=$folder/rgbs/p_0
        if [ -d $rgb_path ]; then
            mask_path=$folder/masks/p_0
            echo "$mask_path"
            mkdir -p $mask_path

            python mask_gen_by_threshold.py $rgb_path $mask_path

        else
            echo "No rgb/p_0 subfolder found in $folder"
        fi
    fi
    echo "Done with $folder"
done



# # generate depth for views 
echo "---------------Start Depth-Anything-V2--------------"
cd $DEPTH_ANYTHING_V2_ROOT
source /opt/conda/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Iterate through each folder in novel_views_folder
for folder in "$novel_views_folder"/*/; do
    if [ -d "$folder" ]; then
        # Check if rgb/p_0 subfolder exists
        rgb_path="$folder/rgbs/p_0"
        if [ -d $rgb_path ]; then
            depth_path=$folder/depths/p_0
            echo $depth_path
            mkdir -p $depth_path

            python run.py \
                --img-path $rgb_path \
                --grayscale \
                --input-size 576 \
                --outdir $depth_path \
                --pred-only \
                --ckpt-root $ckpt_root
        else
            echo "No rgb/p_0 subfolder found in $folder"
        fi
    fi
    echo "Done with $folder"
done


# Generate camera parameters
echo "---------------Start Camera Parameters Generation--------------"
cd $SCRIPT_ROOT
python camerapose_gen_for_sv3d.py $FOLDER $DATA_ROOT


# write the config file
echo "---------------Start Config File Generation--------------"
cd $SCRIPT_ROOT
python write_config.py ${Project_ROOT}/src/s1/configs/base.yaml "${Project_ROOT}/data/s1_processed/${FOLDER}/${FOLDER}.yaml" $DATA_ROOT/$FOLDER ${output_path} ${FOLDER}.json ${FOLDER}.json


