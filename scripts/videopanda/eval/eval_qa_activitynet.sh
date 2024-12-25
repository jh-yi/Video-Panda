# bash scripts/videopanda/eval/eval_qa_activitynet.sh

# module load cuda/12.1
output_name=videopanda_7b_f8_LB448/videopanda_fitu
GPT_Zero_Shot_QA="/path/to/data_root/eval/GPT_Zero_Shot_QA" # e.g. /home/username/datasets/Video-LLaVA/eval/GPT_Zero_Shot_QA


pred_path="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${output_name}/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${output_name}/gpt3-0.25"
output_json="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${output_name}/results.json"
api_key="YOUR_API_KEY"
api_base="https://api.openai.com/v1"
num_tasks=8

python3 videopanda/eval/video/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}