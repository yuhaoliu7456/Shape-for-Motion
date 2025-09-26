#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || { echo "CAN NOT switch to the folder"; exit 1; }
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES='0'
Project_ROOT=$(pwd)


FOLDER=$1 # folder name
EXP=$2 # experiment name

python -m s1.test_ori \
        --config "${Project_ROOT}/data/s1_processed/${FOLDER}/${FOLDER}.yaml" \
        --start_checkpoint ${Project_ROOT}/outputs/s1/${FOLDER}/${EXP}


echo 'The results are saved under the' ${Project_ROOT}/outputs/s1/${FOLDER}/test_${EXP}