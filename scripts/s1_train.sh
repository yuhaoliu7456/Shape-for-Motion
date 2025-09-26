#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || { echo "CAN NOT switch to the folder"; exit 1; }
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES='0'
Project_ROOT=$(pwd)


FOLDER=$1 # folder name
EXP=$2 # experiment name


rate=2  # novel view sample rate
# loss weights
depth_loss_weight=1  # depth loss weight for inp video
m=0.1  # novel_views_mask_loss_weight
s=0.1  # novel_views_mesh_img_loss_weight
d=0.1  # novel_views_depth_loss_weight for novel views
g=0.1  # novel_views_gs_loss_weight

# Basicly, the above loss weights are the most proper for most cases. 
# There are also other two sets of loss weights you can adjust if you want to. 
# depth_loss_weight: m : s : d : g
# [1, 1, 1, 0.1, 1], [10, 0.1, 0.1, 1, 0.1]



python -m s1.train \
        --config "${Project_ROOT}/data/s1_processed/${FOLDER}/${FOLDER}.yaml" \
        --expname $EXP \
        --depth_loss_weight ${depth_loss_weight} \
        --sample_interval_novel_views ${rate} \
        --novel_views_mask_loss_weight ${m} \
        --novel_views_mesh_img_loss_weight ${s} \
        --novel_views_depth_loss_weight ${d} \
        --novel_views_gs_loss_weight ${g}

echo 'The experiment is saved under the' ${Project_ROOT}/outputs/s1/${FOLDER}/${EXP}