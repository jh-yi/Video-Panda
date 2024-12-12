## Data preparation

Please refer to Video-LLaVA for more details for data preparation. 

### Data for training

- The videos pretraining dataset is from [Valley](https://github.com/RupertLuo/Valley).
- The videos tuning dataset is from [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT).
- Download the training annotations. You can download from [Baidu Disk](https://pan.baidu.com/s/1vPZswad5auXlDrmV7JJpdg?pwd=lj8b), [Google Disk](https://drive.google.com/file/d/1zGRyVSUMoczGq6cjQFmT0prH67bu2wXD/view?usp=sharing) or [Peking University Disk](https://disk.pku.edu.cn:443/link/E8BFEFF8EB55E92DEEA232EB094FDB4C)

You can download the processed data on [Hugging Face](https://huggingface.co/datasets/LanguageBind/Video-LLaVA/tree/main), or from Baidu Disk as follows. 
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Baidu Disk</th>
    </tr>
    </tr>
    <tr align="center">
        <td>Video pretraining</td><td><a href="https://pan.baidu.com/s/1jluOimE7mmihEBfnpwwCew?pwd=jyjz">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Video tuning</td><td><a href="https://pan.baidu.com/s/10hJ_U7wVmYTUo75YHc_n8g?pwd=g1hf">Link</a></td>
    </tr>
</table>
</div>

After downloading all of them, organize the data as follows in ```DATA_ROOT```. 

```Shell
DATA_ROOT
├── valley
└── videochatgpt_tune
```

### Data for validating
- Videos and annotations can be downloaded from Video-ChatGPT. Video-LLaVA also provide the processed data as follows.
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Baidu Disk</th><th>Google Disk</th><th>Peking University Disk</th>
    </tr>
    <tr align="center">
        <td>Activitynet_Zero_Shot_QA</td><td><a href="https://pan.baidu.com/s/1d_AVx9Mz_57nA3exhQZGyA?pwd=9amr ">Link</a></td><td>-</td><td>-</td>
    </tr>
    </tr>
    <tr align="center">
        <td>MSRVTT_Zero_Shot_QA</td><td><a href="https://pan.baidu.com/s/1QHUtwHXm4Vc-Wc12XFCFsA?pwd=1rj8">Link</a></td><td><a href="https://drive.google.com/file/d/1yXh9lz7flQ5Ui2IRSd6Qi6RqSEeUJwl3/view?usp=drive_link">Link</a></td><td>-</td>
    </tr>
    </tr>
    <tr align="center">
        <td>MSVD_Zero_Shot_QA</td><td><a href="https://pan.baidu.com/s/1PJSHkjHG2BPl_ddUnBj9AA?pwd=jj34">Link</a></td><td><a href="https://drive.google.com/file/d/1_q4eiSdb7i8P3Hmh4lCfgY1uBGyzU_7X/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/8B0D01747D8AA65534820B7E60CBFEFC">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>TGIF_Zero_Shot_QA</td><td><a href="https://pan.baidu.com/s/11ubtWbTtubyBmN9UPvAyow?pwd=98yr">Link</a></td><td><a href="https://drive.google.com/file/d/1so6L9rg_gdC8Segur7rKML-ffd4Ix_I6/view?usp=drive_link">Link</a></td><td><a href="https://disk.pku.edu.cn:443/link/B9AB387EFE8817158F181FF3D7A97163">Link</a></td>
    </tr>
</table>
</div>

After downloading all of them, organize the data as follows in `eval`.

```Shell
eval
└── GPT_Zero_Shot_QA
    ├── Activitynet_Zero_Shot_QA
    ├── MSRVTT_Zero_Shot_QA
    ├── MSVD_Zero_Shot_QA
    └── TGIF_Zero_Shot_QA
```

## Training
Specify your `DATA_ROOT` according to the data preparation.
- Stage 1 prealignment script: [prealign.sh](scripts/v1_5/prealign.sh). 
- Stage 2 pretraining script: [pretrain.sh](scripts/v1_5/pretrain.sh). 
- Stage 3 tuning script: [finetune.sh](scripts/v1_5/finetune.sh).

## Validating
Our video validation code comes from Video-ChatGPT, thanks for their contribution! 

You can refer to the official repository for validation, but we also provide [off-the-shelf](scripts/v1_5/eval) scripts.

### MSRVTT-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_qa_msrvtt.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/v1_5/eval/eval_qa_msrvtt.sh
```

### MSVD-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_qa_msvd.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/v1_5/eval/eval_qa_msvd.sh
```

### TGIF-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_qa_tgif.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/v1_5/eval/eval_qa_tgif.sh
```

### ActivityNet-QA
1. Inference to get the result.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/run_qa_activitynet.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/v1_5/eval/eval_qa_activitynet.sh
```