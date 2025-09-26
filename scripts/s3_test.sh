#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || { echo "CAN NOT switch to the folder"; exit 1; }
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
gpuid=0
export CUDA_VISIBLE_DEVICES=$gpuid
Project_ROOT=$(pwd)



SAVE_NAME=$1 # This SAVE_NAME corresponds to the 'SAVE_NAME' in the s3_prepare_data.sh
DATA_ROOT=${Project_ROOT}/data/s2_edited/edited_videos/${SAVE_NAME}

echo $DATA_ROOT
echo $SAVE_NAME
    

python -m s3.infer \
    --cache_dir ${Project_ROOT}/pretrained_weights \
    --weights_dir ${Project_ROOT}/pretrained_weights/shape-for-motion \
    --output_dir ${Project_ROOT}/outputs/s3 \
    --ori_input_folder $DATA_ROOT/ori-rgb/$SAVE_NAME.mp4 \
    --ori_mask_folder $DATA_ROOT/ori-mask/$SAVE_NAME.mp4 \
    --edited_geometry_folder  $DATA_ROOT/edited-normal/$SAVE_NAME.mp4 \
    --edited_mask_folder $DATA_ROOT/edited-mask/$SAVE_NAME.mp4 \
    --edited_texture_folder $DATA_ROOT/edited-rgb/$SAVE_NAME.mp4 \
    --excluded_mask_folder $DATA_ROOT/excluded-mask/$SAVE_NAME.mp4 \
    --height 512 \
    --width 768 \

    # If you found that the inpainted background is not very good, you can try to use larger dilation range by enlarging the min_dilation and max_dilation to 8 and 15, respectively.


echo 'The results are saved under the ' ${Project_ROOT}/outputs/s3