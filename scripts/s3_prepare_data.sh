#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || { echo "CAN NOT switch to the folder"; exit 1; }
export PYTHONPATH="$(pwd):$PYTHONPATH"
gpuid=0
export CUDA_VISIBLE_DEVICES=$gpuid
Project_ROOT=$(pwd)


FOLDER=$1 # folder name
EDIT_MESH_NAME=$2 # manually edited mesh name by blender(e.g., rotate_l20, scale_10, etc.)
SAVE_NAME="$1_$2" # save name for the output folder for this editing

echo 'prepare data for video: ' ${SAVE_NAME}

data_root="${Project_ROOT}/data/s1_processed/${FOLDER}"
edited_root="${Project_ROOT}/outputs/s2/${SAVE_NAME}"
save_root="${Project_ROOT}/data/s2_edited/edited_videos/${SAVE_NAME}"
config_path="${Project_ROOT}/data/s1_processed/${FOLDER}/${FOLDER}.yaml"

python utils/prepare_data_for_rendering.py ${data_root} ${SAVE_NAME} ${edited_root} ${save_root} ${config_path}

