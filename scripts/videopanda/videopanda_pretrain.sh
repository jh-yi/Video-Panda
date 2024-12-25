#!/bin/bash
# export NCCL_SOCKET_IFNAME=eno1,eno2 # eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
# export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_HCA=mlx5_2,mlx5_5
export MASTER_PORT=23456
export QUOTA=reserved
export CUDA_DEVICE_MAX_CONNECTIONS=1

set -x

# ################## Please Edit Here #########################

# module load cuda/12.1
export EXPNAME=videopanda_7b_f8_LB448_new   #  8 frames, LanguageBind, 448x448
export RANDSEED=42
export DATA_ROOT=/path/to/data_root         # e.g. /home/username/datasets/Video-LLaVA

export NNODES=16    # 1
export GPUS_PER_NODE=4
export CPUS_PER_TASK=32

# ###########################################

export PYTHONPATH_PROJECT=$(pwd)
export IMAGE_FOLDER=${DATA_ROOT}/train
export VIDEO_FOLDER=${DATA_ROOT}/train
export JSON_FOLDER=${DATA_ROOT}/train/train_json
export VIT_PATH=videopanda/config/videopanda-vision-LB-patch14-anypixel-448
export VIDEO_PATH=videopanda/config/videopanda-video-LB-patch14-anypixel-448
export VIT_PATH_TEACHER=LanguageBind/LanguageBind_Image
export VIDEO_PATH_TEACHER=LanguageBind/LanguageBind_Video_merge

export BASE_LR=4e-5

export CKPT_PATH=${PYTHONPATH_PROJECT}/checkpoints/${EXPNAME}/videopanda_prtr0
export SAVE_PATH=${EXPNAME}/videopanda_prtr1
echo "ckpt: ${CKPT_PATH}, save: ${SAVE_PATH}"

# --data_path ${JSON_FOLDER}/llava_image_.json ${JSON_FOLDER}/valley_.json \
torchrun --nproc_per_node=$GPUS_PER_NODE --nnode=$NNODES --node_rank=$1 --master_addr=$2 --master_port=$MASTER_PORT \
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
    --requires_image_distill True \
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
    # 2>&1 | tee logs/${SAVE_PATH}-rank$1-$(date "+%Y-%m-%d|%H:%M:%S").log