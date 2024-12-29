#!/bin/bash

# save a copy of source files before slurm schedule, and run the code using submitted version. 

export EXPNAME=videopanda_7b
export PYTHONPATH_PROJECT=$(pwd)
echo "Project root: ${PYTHONPATH_PROJECT}"

if [[ ${EXPNAME} != *"debug"* ]]; then
    mkdir -p checkpoints/${EXPNAME}/src_files
    # echo "Saving source files..."
    rsync -rlp ${PYTHONPATH_PROJECT}/* ${PYTHONPATH_PROJECT}/checkpoints/${EXPNAME}/src_files/ --exclude wandb --exclude checkpoints --exclude docs --exclude examples --exclude images --exclude BAAI --exclude openai --exclude OpenGVLab --exclude playground --exclude lmsys --exclude cache_dir || exit 1
    echo "Saving source files...Finished! EXPNAME: ${EXPNAME}"
    
    cd ${PYTHONPATH_PROJECT}/checkpoints/${EXPNAME}/src_files
    touch "$(date "+%Y-%m-%d|%H:%M:%S").txt"
    cd -
else
    echo "Debuging...Do not save source files..."
fi

sbatch scripts/videopanda/slurm_videopanda_7b_all.sh