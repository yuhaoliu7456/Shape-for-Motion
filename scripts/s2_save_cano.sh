#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || { echo "CAN NOT switch to the folder"; exit 1; }
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
gpuid=0
export CUDA_VISIBLE_DEVICES=$gpuid
Project_ROOT=$(pwd)

FOLDER=$1 # folder name
EXP=$2 # experiment name



echo "FOLDER: ${FOLDER}, EXP: ${EXP}"

# ===========script: save canonical mesh======================
# With the canonical mesh without color, you can only do motion editing
python -m s2.render_edit \
    --config "${Project_ROOT}/data/s1_processed/${FOLDER}/${FOLDER}.yaml" \
    --start_checkpoint ${Project_ROOT}/outputs/s1/${FOLDER}/${EXP} \
    --canonical_mesh_path ${Project_ROOT}/data/s2_edited/editing_files/${FOLDER}/canonical_wo_color.ply \
    --do_save_canonical_mesh \
    --save_path ${Project_ROOT}/data/s2_edited/editing_files/${FOLDER} \

# With the canonical mesh with color, you can do texture editing, optional texture+motion editing
 python -m s2.render_edit \
    --config "${Project_ROOT}/data/s1_processed/${FOLDER}/${FOLDER}.yaml" \
    --start_checkpoint ${Project_ROOT}/outputs/s1/${FOLDER}/${EXP} \
    --canonical_mesh_path ${Project_ROOT}/data/s2_edited/editing_files/${FOLDER}/canonical_w_color.ply \
    --do_save_canonical_mesh \
    --save_path ${Project_ROOT}/data/s2_edited/editing_files/${FOLDER} \
    # --enable_smooth_vert_color    # by default: false. If true, the mesh will be smoothed
# ============================================================

