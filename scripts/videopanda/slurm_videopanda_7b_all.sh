#!/bin/bash
#SBATCH --job-name=videopanda_7b
#SBATCH --output=slurm-%x-%j.log
#SBATCH --account=your_account_name
#SBATCH --partition=your_account_name
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=32

#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=480GB

#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email

# ################## Please Edit Here #########################

## keyword in EXPNAME: 
# debug: do no save source files; minidata: mini-dataset with 10 samples
export EXPNAME=$SLURM_JOB_NAME # the same as job-name above
echo $EXPNAME

export RANDSEED=42
echo $RANDSEED

# module load cuda/12.1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export DATA_ROOT=/path/to/data_root     # e.g. /home/username/datasets/Video-LLaVA

# ###########################################

echo "Running on host" $(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR="$MASTER_ADDR

export CUDA_LAUNCH_BLOCKING=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=20
# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
# export NCCL_P2P_DISABLE=1

# ###########################################

echo "Run started at: "
STARTTIME=$(date "+%Y-%m-%d|%H:%M:%S")
echo ${STARTTIME}

set -x

# run previous saved copy of source files (submitted version) instead of current version. 
if [[ ${EXPNAME} != *"debug"* ]]; then
    cd ${PYTHONPATH_PROJECT}/checkpoints/${EXPNAME}/src_files
    ln -s ${PYTHONPATH_PROJECT}/cache_dir .
    # ln -s ${PYTHONPATH_PROJECT}/lmsys .
else
    echo "Debuging...Do not save source files..."
fi

echo "New PYTHONPATH: $(pwd)"
export PYTHONPATH=$(pwd)

# ####################### pre-align ####################################################
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Prealign started at: "
STARTTIME=$(date "+%Y-%m-%d|%H:%M:%S")
echo ${STARTTIME}

export IMAGE_FOLDER=${DATA_ROOT}/train
export VIDEO_FOLDER=${DATA_ROOT}/train
export JSON_FOLDER=${DATA_ROOT}/train/train_json
export VIT_PATH=videopanda/config/videopanda-vision-LB-patch14-anypixel-448
export VIDEO_PATH=videopanda/config/videopanda-video-LB-patch14-anypixel-448
export VIT_PATH_TEACHER=LanguageBind/LanguageBind_Image
export VIDEO_PATH_TEACHER=LanguageBind/LanguageBind_Video_merge
export BASE_LR=4e-5
export LEARNIG_RATE=4e-4

export CKPT_PATH=${PYTHONPATH_PROJECT}/checkpoints/EVE-7B-Pretrain-v1.0
export SAVE_PATH=${EXPNAME}/videopanda_prtr0
echo "ckpt: ${CKPT_PATH}, save: ${SAVE_PATH}"

# --data_path ${JSON_FOLDER}/llava_image_.json ${JSON_FOLDER}/valley_.json \
srun --cpus-per-task=${SLURM_CPUS_PER_TASK} torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    videopanda/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed ./scripts/zero2.json \
    --version v1 \
    --data_stage prealign \
    --seed $RANDSEED \
    --group_by_modality_length True \
    --data_path ${JSON_FOLDER}/valley_.json \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VIT_PATH} \
    --vision_tower_teacher ${VIT_PATH_TEACHER} \
    --requires_image_distill False \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower ${VIDEO_PATH} \
    --video_tower_teacher ${VIDEO_PATH_TEACHER} \
    --requires_video_distill True \
    --tune_vision_tower True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${PYTHONPATH_PROJECT}/checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --mm_projector_lr ${LEARNIG_RATE} \
    --vision_tower_lr ${LEARNIG_RATE} \
    --video_projector_lr ${LEARNIG_RATE} \
    --video_tower_lr ${LEARNIG_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} \
    || exit 1

# # # ####################### pre-train ####################################################
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Pretrain started at: "
STARTTIME=$(date "+%Y-%m-%d|%H:%M:%S")
echo ${STARTTIME}
unset LEARNIG_RATE

export CKPT_PATH=${PYTHONPATH_PROJECT}/checkpoints/${SAVE_PATH}
export SAVE_PATH=${EXPNAME}/videopanda_prtr1
echo "ckpt: ${CKPT_PATH}, save: ${SAVE_PATH}"

# --data_path ${JSON_FOLDER}/llava_image_.json ${JSON_FOLDER}/valley_.json \
srun --cpus-per-task=${SLURM_CPUS_PER_TASK} torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    videopanda/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed ./scripts/zero2.json \
    --version v1 \
    --data_stage pretrain \
    --seed $RANDSEED \
    --group_by_modality_length True \
    --data_path ${JSON_FOLDER}/valley_.json \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VIT_PATH} \
    --vision_tower_teacher ${VIT_PATH_TEACHER} \
    --requires_image_distill False \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower ${VIDEO_PATH} \
    --video_tower_teacher ${VIDEO_PATH_TEACHER} \
    --requires_video_distill True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${PYTHONPATH_PROJECT}/checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} \
    || exit 1

# ####################### fine-tune ####################################################
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Finetune started at: "
STARTTIME=$(date "+%Y-%m-%d|%H:%M:%S")
echo ${STARTTIME}

export BASE_LR=2e-5
export LEARNIG_RATE=2e-5
export CKPT_PATH=${PYTHONPATH_PROJECT}/checkpoints/${SAVE_PATH}
export SAVE_PATH=${EXPNAME}/videopanda_fitu
echo "ckpt: ${CKPT_PATH}, save: ${SAVE_PATH}"

# --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/videochatgpt_tune_.json ${JSON_FOLDER}/nlp_tune.json \
srun --cpus-per-task=${SLURM_CPUS_PER_TASK} torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    videopanda/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed ./scripts/zero2.json \
    --version v1 \
    --data_stage finetune \
    --seed $RANDSEED \
    --data_path ${JSON_FOLDER}/videochatgpt_tune_.json \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VIT_PATH} \
    --vision_tower_teacher ${VIT_PATH_TEACHER} \
    --requires_image_distill False \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower ${VIDEO_PATH} \
    --video_tower_teacher ${VIDEO_PATH_TEACHER} \
    --requires_video_distill True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${PYTHONPATH_PROJECT}/checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --mm_projector_lr ${LEARNIG_RATE} \
    --vision_tower_lr ${LEARNIG_RATE} \
    --video_projector_lr ${LEARNIG_RATE} \
    --video_tower_lr ${LEARNIG_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} \
    || exit 1

echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Train completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date

mv ${PYTHONPATH_PROJECT}/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log ${PYTHONPATH_PROJECT}/checkpoints/${EXPNAME}/${STARTTIME}.log

unlink cache_dir
# unlink lmsys