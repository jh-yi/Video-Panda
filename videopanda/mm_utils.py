import base64
from io import BytesIO
import torch
from PIL import Image
import math
from transformers import StoppingCriteria

from videopanda.constants import IMAGE_TOKEN_INDEX
from torchvision.transforms.functional import pil_to_tensor
from videopanda.model.dataloader import _get_rawvideo_dec
from videopanda.constants import NUM_FRAMES
import numpy as np

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, processor, model_cfg=None):

    def compute_basehw(raw_image, processor):
        width, height = raw_image.size
        max_size = processor.image_size_teacher

        if width > height:
            new_width = max_size
            val_patch = int(math.ceil(max_size * height / width / processor.patch_stride_teacher))
            new_height = val_size = val_patch * processor.patch_stride_teacher
        else:
            new_height = max_size
            val_patch = int(math.ceil(max_size * width / height / processor.patch_stride_teacher))
            new_width = val_size = val_patch * processor.patch_stride_teacher
        
        max_patch = max_size // processor.patch_stride_teacher
        assert max_size % processor.patch_stride_teacher == 0
        pre_size = (max_patch - val_patch) // 2 * processor.patch_stride_teacher
        post_size = max_size - pre_size - val_size
        assert post_size % processor.patch_stride_teacher == 0

        return [new_width, new_height, pre_size, val_size, post_size]

    def resize2require(raw_image, all_sizes, background_color):
        width, height = raw_image.size
        max_size = max(width, height)
        assert max_size == sum(all_sizes)

        image = Image.new(raw_image.mode, (max_size, max_size), background_color)
        if width > height:
            image.paste(raw_image, (0, all_sizes[0]))
            image_mask = torch.cat([torch.zeros(1, all_sizes[0], max_size),
                                    torch.ones(1, all_sizes[1], max_size),
                                    torch.zeros(1, all_sizes[2], max_size)], dim=1)
        else:
            image.paste(raw_image, (all_sizes[0], 0))
            image_mask = torch.cat([torch.zeros(1, max_size, all_sizes[0]),
                                    torch.ones(1, max_size, all_sizes[1]),
                                    torch.zeros(1, max_size, all_sizes[2])], dim=2)
        return image, image_mask

    ratio = processor.image_size // processor.image_size_teacher
    assert ratio >= 1
    assert processor.image_size % processor.image_size_teacher == 0
    try:
        background_color = tuple(int(x*255) for x in processor.image_mean)
    except:
        background_color = tuple(int(x*255) for x in processor.image_mean[0][0])

    new_images = []
    for raw_image in images:
        basehw = compute_basehw(raw_image, processor)
        needhw = [value * ratio for value in basehw]
        image = raw_image.resize((needhw[0], needhw[1]))
        if needhw[0] == needhw[1]:
            image_mask = torch.ones(1, needhw[1], needhw[0])
        else:
            image, image_mask = resize2require(image, needhw[2:], background_color)

        if 'LanguageBindVideo' in repr(processor):
            tmp_image = pil_to_tensor(image).unsqueeze(1)
            image = processor.preprocess(tmp_image, return_tensors='pt')['pixel_values'][0].permute(1,0,2,3)[0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # raise NotImplementedError

        image = torch.cat([image, image_mask], dim=0)
        new_images.append(image)

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def process_video_images(video_path, image_processor, video_processor, use_context_image=False):

    raw_frames, raw_context_images = _get_rawvideo_dec(video_path, image_processor, video_processor,
                                                frame_resolution=video_processor.image_size, max_frames=NUM_FRAMES,
                                                num_video_frames=NUM_FRAMES,
                                                num_context_images=NUM_FRAMES)

    def compute_basehw(raw_image, processor):
        width, height = raw_image.size
        max_size = processor.image_size_teacher

        if width > height:
            new_width = max_size
            val_patch = int(math.ceil(max_size * height / width / processor.patch_stride_teacher))
            new_height = val_size = val_patch * processor.patch_stride_teacher
        else:
            new_height = max_size
            val_patch = int(math.ceil(max_size * width / height / processor.patch_stride_teacher))
            new_width = val_size = val_patch * processor.patch_stride_teacher
        
        max_patch = max_size // processor.patch_stride_teacher
        assert max_size % processor.patch_stride_teacher == 0
        pre_size = (max_patch - val_patch) // 2 * processor.patch_stride_teacher
        post_size = max_size - pre_size - val_size
        assert post_size % processor.patch_stride_teacher == 0

        return [new_width, new_height, pre_size, val_size, post_size]
    
    def compute_basehw_video(raw_image, processor):
        width, height = raw_image.size
        max_size = processor.image_size_teacher

        if width > height:
            new_width = max_size
            val_patch = int(math.ceil(max_size * height / width / processor.patch_stride_teacher))
            new_height = val_size = val_patch * processor.patch_stride_teacher
        else:
            new_height = max_size
            val_patch = int(math.ceil(max_size * width / height / processor.patch_stride_teacher))
            new_width = val_size = val_patch * processor.patch_stride_teacher
        
        max_patch = max_size // processor.patch_stride_teacher
        assert max_size % processor.patch_stride_teacher == 0
        pre_size = (max_patch - val_patch) // 2 * processor.patch_stride_teacher
        post_size = max_size - pre_size - val_size
        assert post_size % processor.patch_stride_teacher == 0

        return [new_width, new_height, pre_size, val_size, post_size]
    
    def resize2require_video(raw_images, all_sizes, background_color):
        width, height = raw_images[0].size
        max_size = max(width, height)
        assert max_size == sum(all_sizes)
        
        images = [Image.new(raw_image.mode, (max_size, max_size), background_color) for raw_image in raw_images]

        if width > height:
            for image, raw_image in zip(images, raw_images):
                image.paste(raw_image, (0, all_sizes[0]))
            image_mask = torch.cat([torch.zeros(1, all_sizes[0], max_size),
                                    torch.ones(1, all_sizes[1], max_size),
                                    torch.zeros(1, all_sizes[2], max_size)], dim=1)
        else:
            for image, raw_image in zip(images, raw_images):
                image.paste(raw_image, (all_sizes[0], 0))
            image_mask = torch.cat([torch.zeros(1, max_size, all_sizes[0]),
                                    torch.ones(1, max_size, all_sizes[1]),
                                    torch.zeros(1, max_size, all_sizes[2])], dim=2)
        return images, image_mask
    
    # frames
    raw_frames = [Image.fromarray(i.astype('uint8'), 'RGB') for i in raw_frames]
    ratio = video_processor.image_size // video_processor.image_size_teacher
    assert ratio >= 1
    assert video_processor.image_size % video_processor.image_size_teacher == 0
    background_color = tuple(int(x*255) for x in video_processor.image_mean[0][0])
    
    basehw = compute_basehw_video(raw_frames[0], video_processor)
    needhw = [value * ratio for value in basehw]

    frames = [i.resize((needhw[0], needhw[1])) for i in raw_frames]    
    del raw_frames

    if needhw[0] == needhw[1]:
        video_mask = torch.ones(1, needhw[1], needhw[0])
    else:
        frames, video_mask = resize2require_video(frames, needhw[2:], background_color)
    
    if 'LanguageBind' in repr(video_processor):
        tmp_frames = torch.stack([pil_to_tensor(frame) for frame in frames]).permute(1,0,2,3)
        frames = video_processor.preprocess(tmp_frames, return_tensors='pt')['pixel_values'][0].permute(1,0,2,3)
    else: # e.g. internvideo
        frames = video_processor.preprocess([np.asarray(frame, np.float32) for frame in frames])['pixel_values']
        raise NotImplementedError
    frames = [torch.cat([i, video_mask], dim=0) for i in frames]

    # Padding frames
    slice_len = len(frames)
    if slice_len < 1:
        pass
    else:
        while len(frames) < NUM_FRAMES:
            frames.append(torch.zeros((4, video_processor.image_size, video_processor.image_size)))

    return None, frames


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [
        tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)): # e.g. [1,319, ....], [-299, -200], [1, 29871, ...]
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1].split('_')[0]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # TODO
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"
        offset = min(output_ids.shape[1] -
                     self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(
            output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
