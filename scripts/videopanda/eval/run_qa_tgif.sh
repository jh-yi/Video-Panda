# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/videopanda/eval/run_qa_tgif.sh

# module load cuda/12.1
CKPT_NAME=Video-Panda-7B/videopanda_fitu
GPT_Zero_Shot_QA="/path/to/data_root/eval/GPT_Zero_Shot_QA" # e.g. /home/username/datasets/Video-LLaVA/eval/GPT_Zero_Shot_QA


model_path="checkpoints/${CKPT_NAME}"
cache_dir="./cache_dir"
video_dir="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/mp4"
gt_file_question="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/test_a.json"
output_dir="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${CKPT_NAME}"
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videopanda/eval/video/run_inference_video_qa.py \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done