from abc import ABC, abstractmethod

import torch

from videopanda.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
                               IMAGE_TOKEN_INDEX, CHUNK_SIZE)

from .multimodal_projector.builder import build_vision_projector, build_video_projector
from .multimodal_encoder.vision_tokenizer import VisionCompressor, VideoTokenizer, VideoCompressor
import json
from einops import rearrange
from .multimodal_encoder.builder import build_image_tower
from .multimodal_encoder.builder import build_video_tower

class VideoPandaMetaModel:
    def __init__(self, config):
        super(VideoPandaMetaModel, self).__init__(config)

        if hasattr(config, "mm_video_tower"):
            with open(config.mm_video_tower+'/preprocessor_config.json', 'r') as f:
                video_config = json.load(f)
            self.video_tower = VideoTokenizer(config.video_hidden_size, config, video_config)
            self.video_projector = build_video_projector(config)
            
            if config.requires_video_distill:
                if 'LanguageBind' in config.video_tower_teacher:
                    for k in video_config.keys():   # merge two configs
                        config.__setattr__(k, video_config[k])
                    self.video_teacher = build_video_tower(config, delay_load=True)
                else:
                    raise NotImplementedError

                self.video_teacher.requires_grad_(False)
                self.video_tower_compressor = VideoCompressor(self.video_teacher,
                                                              self.video_tower.image_processor,
                                                              llm_size=config.hidden_size, 
                                                              num_layer=config.num_hidden_layers,
                                                              chunk_size=video_config['num_frames'])

        if hasattr(config, "mm_vision_tower"):
            
            self.vision_tower = self.video_tower # shared
            self.mm_projector = self.video_projector # shared

            if config.requires_image_distill:
                if 'LanguageBind' in config.mm_vision_tower_teacher:
                    self.image_teacher = build_image_tower(config, delay_load=True)
                else:
                    raise NotImplementedError
                self.image_teacher.requires_grad_(False)
                self.vision_tower_compressor = VisionCompressor(self.image_teacher, 
                                                                self.vision_tower.image_processor,
                                                                llm_size=config.hidden_size, 
                                                                num_layer=config.num_hidden_layers)
        
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_image_teacher(self):
        return self.image_teacher
    
    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower
    
    def get_video_teacher(self):
        return self.video_teacher
    
    def get_vision_tower_compressor(self):
        return self.vision_tower_compressor

    def get_video_tower_compressor(self):
        return self.video_tower_compressor

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        video_tower = model_args.video_tower
        vision_tower_teacher = model_args.vision_tower_teacher
        video_tower_teacher = model_args.video_tower_teacher
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_video_mm_mlp_adapter = model_args.pretrain_video_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.mm_video_tower = video_tower
        self.config.mm_vision_tower_teacher = vision_tower_teacher
        self.config.mm_video_tower_teacher = video_tower_teacher
        self.config.requires_image_distill = model_args.requires_image_distill
        self.config.requires_video_distill = model_args.requires_video_distill
        # ==========================================================================

        vision_tower = self.vision_tower
        video_tower = self.video_tower

        # In case it is frozen by LoRA
        for p in self.vision_tower.parameters():
            p.requires_grad = True
        for p in self.video_tower.parameters():
            p.requires_grad = True

        # ==========================================================================

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.video_hidden_size = video_tower.hidden_size

        # ==========================================================================

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'video_projector', None) is None:
            self.video_projector = build_video_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.video_projector.parameters():
                p.requires_grad = True

        # ==========================================================================

        if pretrain_mm_mlp_adapter is not None:
            print(f'Load mm_mlp_adapter from {pretrain_mm_mlp_adapter}')
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_aw(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_aw(mm_projector_weights, 'mm_projector'))
            
        if pretrain_video_mm_mlp_adapter is not None:
            print(f'Load video mm_mlp_adapter from {pretrain_video_mm_mlp_adapter}')
            video_mm_projector_weights = torch.load(pretrain_video_mm_mlp_adapter, map_location='cpu')

            def get_aw(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.video_projector.load_state_dict(get_aw(video_mm_projector_weights, 'video_projector'))

        # ==========================================================================

        if model_args.requires_image_distill:
            if 'LanguageBind' in model_args.vision_tower_teacher:
                self.image_teacher.load_model()

        if model_args.requires_video_distill:    
            if 'LanguageBind' in model_args.video_tower_teacher:
                self.video_teacher.load_model()

class VideoPandaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_image_teacher(self):
        return self.get_model().get_image_teacher()
    
    def get_image_loss(self):
        return self.get_model().get_vision_tower_compressor()

    def encode_images(self, images):
        if self.config.unified_pel:
            images = images.unsqueeze(1)    # B,C,H,W->B,1,C,H,WW
        vision_tower = self.get_model().get_vision_tower()
        return vision_tower(images, self.get_model().mm_projector)

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def get_video_teacher(self):
        return self.get_model().get_video_teacher()
    
    def get_video_loss(self):
        return self.get_model().get_video_tower_compressor()
    
    def encode_videos(self, frames, batch_size):

        frames = rearrange(frames, '(b t) c h w -> b t c h w', b=batch_size)
        num_chunks = frames.shape[1] // CHUNK_SIZE

        video_features = []
        video_patch_hws = []
        for i in range(batch_size):
            cur_video = frames[i]                                   # current video of shape (t, c, h, w)
            chunks = cur_video.chunk(num_chunks, dim=0)             # segment-wise temporal modeling, compatible with long videos
            chunk_batch = torch.stack(chunks, dim=0)                # (num_chunks, 4, c, h, w), new batch dimension for processing all chunks at once
            # print(chunk_batch.shape)
            patch_embeds, patch_hw = self.get_model().get_video_tower()(chunk_batch, self.get_model().video_projector)
            # video_features[i] = chunk_features[:, 1:] # remove cls token
            video_features.extend(patch_embeds)
            video_patch_hws.extend(patch_hw)
    
        return video_features, video_patch_hws

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, frames):
        # B = input_ids.shape[0]
        vision_tower = self.get_vision_tower()
        video_tower = self.get_video_tower()
        if (vision_tower is None and video_tower is None) or (images is None and frames is None) or input_ids.shape[1] == 1:
            if past_key_values is not None and (vision_tower is not None or video_tower is not None) and (images is not None or frames is not None) and input_ids.shape[1] == 1:
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None, None

        ############## prepare ids
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        ################# visual features
        video_idx = [idx for idx, vid in enumerate(frames) if vid is not None and len(vid)>1]
        videos_minibatch = torch.stack([f for idx in video_idx for f in frames[idx]]) if len(video_idx) > 0 else None    # mini_b*t, c=4, h, w or None
        # v_input_ids = [input_id for i, input_id in enumerate(input_ids) if i in video_idx]
        image_idx = [idx for idx, img in enumerate(images) if img is not None and len(img)==1]
        images_minibatch = torch.stack([i for idx in image_idx for i in images[idx]]) if len(image_idx) > 0 else None    # mini_b c=4 h w or None
        # i_input_ids = [input_id for i, input_id in enumerate(input_ids) if i in image_idx]
        assert len(video_idx) + len(image_idx) == len(input_ids)        
        
        tmp_image_features = [None] * (len(image_idx) + len(video_idx))
        tmp_feature_lens = [None] * (len(image_idx) + len(video_idx))
        tmp_patch_hws = [None] * (len(image_idx) + len(video_idx))
    
        video_features_minibatch, video_patch_hws, image_features_minibatch, patch_hws = None, None, None, None
        if videos_minibatch is not None:    # b1*t, 4, h, w     &   b1*t, 4, h_, w_
            video_features_minibatch, video_patch_hws = self.encode_videos(videos_minibatch, batch_size=len(video_idx))
            b = len(video_idx)
            t = videos_minibatch.shape[0] // len(video_idx)
            
            for i, pos in enumerate(video_idx): 
                tmp_image_features[pos] = [video_features_minibatch[i*t+j] for j in range(t)]
                tmp_feature_lens[pos] = [(video_features_minibatch[i*t+j]).shape[0] for j in range(t)]
                tmp_patch_hws[pos] = [video_patch_hws[i*t+j] for j in range(t)]
        
        if images_minibatch is not None:
            image_features_minibatch, patch_hws = self.encode_images(images_minibatch)
            for i, pos in enumerate(image_idx): 
                tmp_image_features[pos] = image_features_minibatch[i]
                tmp_feature_lens[pos] = [image_features_minibatch[i].shape[0]]
                tmp_patch_hws[pos] = patch_hws[i]
        
        patch_hws = tmp_patch_hws
        new_tmp = []
        for image in tmp_image_features: # tmp_image_features: [(L_i, 4096), [(L_v, 4096), (L_v, 4096), ...]]
            # print(len(new_tmp), len(image))
            if isinstance(image, list): # video
                t = len(image)
                for i in range(t):
                    new_tmp.append(image[i])
                # print('add video')
            else:
                new_tmp.append(image)
        image_features = new_tmp
        # print(len(image_features), *[i.shape for i in image_features])
        # print(len(image_features), image_features[0].shape)
        
        # ====================================================================================================
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False): # TODO
                    cur_new_input_embeds.append(self.get_model().embed_tokens(
                        cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(
                        cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(
                        cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full(
                            (cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_labels.append(
                            cur_labels[image_token_start+1:image_token_start+2])
                        cur_labels = cur_labels[image_token_start+2:]
                else: # cut cur_input_ids into clips (image token as delimiter)
                    cur_text_embeds = self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    if cur_text_embeds.shape[0] > 0:
                        cur_new_input_embeds.append(cur_text_embeds)
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_text_labels = cur_labels[:image_token_start]
                        if cur_text_labels.shape[0] > 0:
                            cur_new_labels.append(cur_text_labels)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        ##### deprecated due to information loss, instead try to increase context length
        ## Truncate sequences to max length as image embeddings can make the sequence longer
        # tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        # tokenizer_model_max_length = getattr(self.config, 'max_position_embeddings', None)
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        #     new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        #     print("#tokens exceed limit: ", getattr(self.config, 'max_position_embeddings', None))
        
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":           # left padding
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
            else:                                                                           # right padding
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, patch_hws, tmp_feature_lens

    # deprecated
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False