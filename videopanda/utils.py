import torch
import shutil
import os
import pathlib

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def save_source(save_path):
    
    src_path = pathlib.Path(__file__).resolve().parents[1]
    dst_path = os.path.join(save_path, "src_files")
    i = 0
    while os.path.exists(dst_path + str(i)): 
        i += 1   
    dst_path = dst_path + str(i)
    shutil.copytree(src=src_path, dst=dst_path, ignore=shutil.ignore_patterns('*.pyc', 'tmp*', '.*', 'data', 'logs', 'wandb', '*.lprof', 'checkpoints', 'docs', 'examples', 'images', 'BAAI', 'openai', 'OpenGVLab', 'playground', '*.out', '*.log', 'LICENSE', 'cache_dir', 'lmsys')) 
    print("Saving source files to ", dst_path)

