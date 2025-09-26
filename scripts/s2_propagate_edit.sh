#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || { echo "CAN NOT switch to the folder"; exit 1; }
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
gpuid=0
export CUDA_VISIBLE_DEVICES=$gpuid
Project_ROOT=$(pwd)
# Project_ROOT=/apdcephfs/private_yuhaliu/Code/Shpe-for-Motion




FOLDER=$1 # folder name
EXP=$2 # experiment name

EDIT_MESH_NAME=$3 # manually edited mesh name by blender(e.g., rotate_l20, scale_10, etc.)
SAVE_NAME="$1_$3" # save name for the output folder for this editing
IF_ADD_OBJ=$4 # int type: 0 or 1

echo "FOLDER: ${FOLDER}, EDIT_MESH_NAME: ${EDIT_MESH_NAME}, Exp: ${EXP}, save_name: ${SAVE_NAME} -------"


if [ $IF_ADD_OBJ -eq 1 ]; then
    
    # # # # ===========script: add a static MESH to a GS model===========
    BINDING_INDEX=98485
    # binding_index denote the index of the vertex in the canonical mesh.
    # It is used to associate the new object with the canonical mesh at the binding point.

    python -m s2.render_edit \
        --config "${Project_ROOT}/data/s1_processed/${FOLDER}/${FOLDER}.yaml" \
        --start_checkpoint ${Project_ROOT}/outputs/s1/${FOLDER}/${EXP} \
        --canonical_mesh_path ${Project_ROOT}/data/s2_edited/editing_files/${FOLDER}/canonical_wo_color.ply \
        --edited_mesh_path ${Project_ROOT}/data/s2_edited/editing_files/${FOLDER}/${EDIT_MESH_NAME}.ply \
        --binding_index ${BINDING_INDEX} \
        --save_path ${Project_ROOT}/outputs/s2/ \
        --save_name ${SAVE_NAME}
        # --mesh_rescaling_ratio 10 \
        # --enable_zero_motion_for_newobj # If the new object is completely static, set this parameter to ensure that the new object does not move
else
    # # # # ===========script:  used for editing, like rotate, delete, scale, texture editing
    # # IF you want to do texture editing, you need to set the canonical mesh with color: i.e., canonical_wo_color --> canonical_w_color
    python -m s2.render_edit \
        --config "${Project_ROOT}/data/s1_processed/${FOLDER}/${FOLDER}.yaml" \
        --start_checkpoint ${Project_ROOT}/outputs/s1/${FOLDER}/${EXP} \
        --canonical_mesh_path ${Project_ROOT}/data/s2_edited/editing_files/${FOLDER}/canonical_wo_color.ply \
        --edited_mesh_path ${Project_ROOT}/data/s2_edited/editing_files/${FOLDER}/${EDIT_MESH_NAME}.ply  \
        --save_path ${Project_ROOT}/outputs/s2/ \
        --save_name ${SAVE_NAME} \
        --mesh_rescaling_ratio 20 \
fi










echo "The results are saved under the" ${Project_ROOT}/outputs/s2/${SAVE_NAME}




# example
# bash scripts/s2_propagate_edit.sh 0 fp-sora-bear -gpu1-depth_loss_weight1_rate2_m1_s1_d0.1_g1 rotate_l20
# bash scripts/s2_propagate_edit.sh 0 framepack-robot1 -gpu1-depth_loss_weight1_rate2_m1_s1_d0.1_g1 wocolor_scale20_sitdown_rotatel20.ply
# TODO: 把otter，cat（加texture）和car（加tree）的当作demo case整理出来




# bash scripts/s2_propagate_edit.sh car-turn demo car_w_tree_98485
# bash scripts/s2_propagate_edit.sh cat-pikachu-0 demo rotate_texture