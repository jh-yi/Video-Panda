from typing import List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..videopanda_arch import VideoPandaMetaForCausalLM, VideoPandaMetaModel
import os
import math

class VideoPandaConfig(LlamaConfig):
    model_type = "videopanda"


class VideoPandaLlamaModel(VideoPandaMetaModel, LlamaModel):
    config_class = VideoPandaConfig

    def __init__(self, config: LlamaConfig):
        super(VideoPandaLlamaModel, self).__init__(config)

class VideoPandaLlamaForCausalLM(LlamaForCausalLM, VideoPandaMetaForCausalLM):
    config_class = VideoPandaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VideoPandaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_teacher: Optional[torch.FloatTensor] = None,
        frames: Optional[torch.FloatTensor] = None,
        frames_teacher: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        _input_ids = input_ids
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, patch_hw, feature_lens = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images, frames)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )
        hidden_states = outputs[0]  ## bsz, toklen, 4096
        logits = self.lm_head(hidden_states)

        loss = None
        loss1 = None
        image_loss = None
        video_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss1 = loss_fct(shift_logits, shift_labels)
            loss = loss1
        
        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output
        
        video_idx_frames = [idx for idx, vid in enumerate(frames) if vid is not None and len(vid)>1]      
        if frames_teacher is not None and (self.config.requires_image_distill or self.config.requires_video_distill):
            video_idx = [idx for idx, vid in enumerate(frames_teacher) if vid is not None and len(vid)>1]
            image_idx = [idx for idx, img in enumerate(images_teacher) if img is not None and len(img)==1]
            images_teacher_minibatch = torch.stack([i for idx in image_idx for i in images_teacher[idx]]) if len(image_idx) > 0 else None    # mini_b c=4 h w or None
            frames_teacher_minibatch = torch.stack([f for idx in video_idx for f in frames_teacher[idx]]) if len(video_idx) > 0 else None    # mini_b*t, c=4, h, w or None
            stacked_hidden_states = torch.stack(outputs.hidden_states, dim=2)

            if self.config.requires_image_distill and self.training and images_teacher_minibatch is not None:
                image_patch_hw = [item for idx,item in enumerate(patch_hw) if idx in image_idx] 
                image_loss = self.get_image_loss()(
                    torch.stack([item for idx, item in enumerate(_input_ids) if idx in image_idx]),
                    images_teacher_minibatch,
                    torch.stack([item for idx, item in enumerate(stacked_hidden_states) if idx in image_idx]),
                    image_patch_hw,
                    [item for idx, item in enumerate(feature_lens) if idx in image_idx],
                    )                                                                             
                loss = loss + image_loss
                
            if self.config.requires_video_distill and self.training and frames_teacher_minibatch is not None:
                video_patch_hws = [item for idx,item in enumerate(patch_hw) if idx in video_idx]
                video_loss = self.get_video_loss()(
                        torch.stack([item for idx, item in enumerate(_input_ids) if idx in video_idx]),
                        frames_teacher_minibatch, 
                        torch.stack([item for idx, item in enumerate(stacked_hidden_states) if idx in video_idx]),
                        video_patch_hws, 
                        [item for idx, item in enumerate(feature_lens) if idx in video_idx],
                        image_patch_hw=None)
                loss = loss + video_loss

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print("llm loss", loss1.item(), "image loss", image_loss.item() if image_loss is not None else None, "video loss", video_loss.item() if video_loss is not None else None)
        
        if os.environ.get('EXPNAME') and 'debug' in os.environ['EXPNAME'] and inputs_embeds is not None:
            print("#tokens: {}. Token padding: {}%. Video minibach: b={}".format(outputs[0].shape, round(((inputs_embeds==0).sum() / math.prod(inputs_embeds.shape)).item()*100), len(video_idx_frames)))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "frames": kwargs.get("frames", None),
            }
        )
        return model_inputs


AutoConfig.register("videopanda", VideoPandaConfig)
AutoModelForCausalLM.register(VideoPandaConfig, VideoPandaLlamaForCausalLM)