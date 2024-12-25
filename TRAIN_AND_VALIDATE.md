

## Training
Specify your `DATA_ROOT` according to the data preparation.
- Stage 1 prealignment script: [videopanda_prealign.sh](scripts/videopanda/videopanda_prealign.sh). 
- Stage 2 pretraining script: [videopanda_pretrain.sh](scripts/videopanda/videopanda_pretrain.sh). 
- Stage 3 tuning script: [videopanda_finetune.sh](scripts/videopanda/videopanda_finetune.sh).

For slurm user (with multiple nodes), you can also use [slurm_sbatch_launcher.sh](scripts/videopanda/slurm_sbatch_launcher.sh) and [slurm_videopanda_7b_all.sh](scripts/videopanda/slurm_videopanda_7b_all.sh) to run three stages at a time. Please specify your `EXPNAME` and `job-name`, `account`, `partition`, `DATA_ROOT` in both scripts respectively. 

## Validating
Our video validation code comes from Video-ChatGPT, thanks for their contribution! 

You can refer to the official repository for validation, but we also provide [off-the-shelf](scripts/videopanda/eval) scripts. Please specify your `CKPT_NAME` and `GPT_Zero_Shot_QA` in each script. 

### MSRVTT-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/videopanda/eval/run_qa_msrvtt.sh
```

2. GPT-Assistant evaluation.
```Shell
 bash scripts/videopanda/eval/eval_qa_msrvtt.sh
```

### MSVD-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/videopanda/eval/run_qa_msvd.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/videopanda/eval/eval_qa_msvd.sh
```

### TGIF-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/videopanda/eval/run_qa_tgif.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/videopanda/eval/eval_qa_tgif.sh
```

### ActivityNet-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/videopanda/eval/run_qa_activitynet.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/videopanda/eval/eval_qa_activitynet.sh
```