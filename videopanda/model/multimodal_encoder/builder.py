import os
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower

# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_vision_tower_teacher', getattr(image_tower_cfg, 'image_tower', None))
    if image_tower.endswith('LanguageBind_Image'):
        return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'video_tower_teacher', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower.endswith('LanguageBind_Video_merge'):
        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================
