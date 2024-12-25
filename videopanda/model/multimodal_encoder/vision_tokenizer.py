import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from videopanda.constants import IMAGE_TOKEN_INDEX, NUM_FRAMES
from einops import rearrange
from timm.models.layers import DropPath
from collections import OrderedDict
from videopanda.model.multimodal_encoder.languagebind.image.processing_image import LanguageBindImageProcessor
from videopanda.model.multimodal_encoder.languagebind.video.processing_video import LanguageBindVideoProcessor
import math

class LocalAttention(nn.Module):
    def __init__(self, input_size, conv_stride, num_heads=8):
        super().__init__()
        self.conv_stride = conv_stride
        self.num_heads = num_heads
        self.scale = input_size ** -0.5

        self.q = nn.Sequential(nn.LayerNorm(input_size), nn.Linear(input_size, input_size, bias=False))
        self.kv = nn.Sequential(nn.LayerNorm(input_size), nn.Linear(input_size, input_size * 2, bias=False))
        self.proj = nn.Linear(input_size, input_size)

    def forward(self, features):
        reduce_features = F.avg_pool2d(features, kernel_size=self.conv_stride, stride=self.conv_stride)
        B, C, H, W = features.shape
        _, _, h, w = reduce_features.shape
        N = self.conv_stride ** 2

        reduce_features = reduce_features.flatten(2).transpose(-2, -1)
        patch_q = self.q(reduce_features).reshape(B, h * w, self.num_heads, -1).permute(0, 2, 1, 3).unsqueeze(-2)
        features = features.unfold(2, self.conv_stride, self.conv_stride).unfold(3, self.conv_stride, self.conv_stride)
        features = features.contiguous().view(B, C, h * w, self.conv_stride, self.conv_stride)
        patch_kv = self.kv(features.flatten(3).permute(0, 2, 3, 1))
        patch_kv = patch_kv.reshape(B, h * w, N, 2, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5)
        patch_attn = (patch_q * self.scale * patch_kv[0]).sum(-1)
        patch_attn = patch_attn.softmax(dim=-1)

        aggre_features = (patch_attn.unsqueeze(-1) * patch_kv[1]).sum(-2)
        aggre_features = aggre_features.transpose(1, 2).reshape(B, h * w, -1)

        return reduce_features + self.proj(aggre_features)

class GlobalAttention(nn.Module):
    def __init__(self, input_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = input_size ** -0.5

        self.q = nn.Sequential(nn.LayerNorm(input_size), nn.Linear(input_size, input_size, bias=False))
        self.kv = nn.Sequential(nn.LayerNorm(input_size), nn.Linear(input_size, input_size * 2, bias=False))
        self.proj = nn.Linear(input_size, input_size)
    
    def forward(self, class_feature, features):

        B, N, C = features.shape
        class_feature = class_feature.repeat(B, 1, 1)

        patch_q, patch_kv = self.q(class_feature), self.kv(features)
        patch_q = patch_q.reshape(B, 1, self.num_heads, -1).transpose(1, 2)
        patch_kv = patch_kv.reshape(B, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        patch_attn = (patch_q * self.scale * patch_kv[0]).sum(-1)
        patch_attn = patch_attn.softmax(dim=-1)

        aggre_features = (patch_attn.unsqueeze(-1) * patch_kv[1]).sum(-2)
        aggre_features = aggre_features.reshape(B, 1, -1)
        
        return class_feature + self.proj(aggre_features)

# Adapted from https://github.com/OpenGVLab/UniFormerV2
class Local_MHRA(nn.Module):
    def __init__(self, d_model, dw_reduction=1.5, pos_kernel_size=3):
        super().__init__() 

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(re_d_model, re_d_model, kernel_size=(pos_kernel_size, 1, 1), stride=(1, 1, 1), padding=(padding, 0, 0), groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        # logger.info('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x):
        return self.pos_embed(x)
    
# Adapted from https://github.com/OpenGVLab/UniFormerV2
class Extractor(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None, mlp_factor=4.0, dropout=0.0, drop_path=0.0,):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_mlp)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_mlp, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        # zero init
        # nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.out_proj.weight, 0.)
        nn.init.constant_(self.attn.out_proj.bias, 0.)
        # nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.)

    def attention(self, x, y): # x=query:1,N,C; y:tL,N,C
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3) # T, N, h, D -> N, h, T, D
        k = k.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim ** 0.5))

        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x, y):
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class VisionCompressor(nn.Module):
    def __init__(self, teacher_layer, processor, llm_size, num_layer, layer_rate=4, num_heads=8):
        super().__init__()

        self.layer_idx = num_layer - torch.arange(0, num_layer + 1, layer_rate)
        self.num_layer = len(self.layer_idx)

        self.teacher_layer = teacher_layer
        if teacher_layer.__dict__.get('image_tower_name') and teacher_layer.__dict__['image_tower_name'] == 'LanguageBind/LanguageBind_Image':
            teacher_hid_size = 1024
        else:
            raise NotImplementedError

        self.norm_layer = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(llm_size), nn.Linear(llm_size, teacher_hid_size, bias=False))  # 4096 -> teacher_hid_size
            for _ in range(self.num_layer)])

        self.input_layer = nn.Linear(llm_size, teacher_hid_size)
        self.output_layer = nn.Linear(teacher_hid_size, teacher_hid_size)
        self.num_heads = num_heads
        self.scale = teacher_hid_size ** -0.5
        
        max_patch = processor.image_size // processor.patch_stride // processor.conv_stride
        max_patch_teacher = processor.image_size_teacher // processor.patch_stride_teacher
        assert (max_patch % max_patch_teacher == 0) or (max_patch_teacher % max_patch == 0)

        self.mask_stride_teacher = processor.patch_stride_teacher
        self.patch_ratio = max_patch // max_patch_teacher

    def cross_attention(self, vision_input, vision_outputs):
        N, L, C = vision_outputs.shape

        patch_q = vision_outputs[:, :1].reshape(N, 1, self.num_heads, -1).transpose(1, 2)
        patch_kv = vision_outputs[:, 1:].reshape(N, L - 1, self.num_heads, -1).transpose(1, 2)

        cross_attn = (patch_q * self.scale * patch_kv).sum(-1).softmax(dim=-1)
        fuse_ouptuts = (cross_attn.unsqueeze(-1) * patch_kv).sum(-2)

        return vision_input + self.output_layer(fuse_ouptuts.reshape(N, -1))
    
    def compute_cosineloss(self, pred_feature, teacher_feature):
        loss_func = nn.CosineSimilarity(dim=-1)
        return 1 - loss_func(pred_feature, teacher_feature).mean()

    def forward(self, input_ids, images, all_features, patch_hws, feature_lens):   

        B, N, L, D = all_features.shape
        BT, _, _, _ = images.shape
        T = BT // B
        feature_lens = [i for item in feature_lens for i in item]

        teacher_2d_masks = F.avg_pool2d(images[:, -1:, :, :], kernel_size=self.mask_stride_teacher, stride=self.mask_stride_teacher)
        # assert len(torch.where(teacher_2d_masks % 1)[0]) == 0
        teacher_features = self.teacher_layer(images[:, :-1, :, :])   # remove mask

        distill_loss, count_image = 0, 0
        for i in range(BT):
            H, W = patch_hws[i]
            if_exist_image = IMAGE_TOKEN_INDEX in input_ids[i//T]

            if i % T == 0:
                idx_str = torch.where(input_ids[i//T] == IMAGE_TOKEN_INDEX)[0][0] if if_exist_image else 0
            idx_end = idx_str + min(N, feature_lens[i])
            i_features = all_features[i//T, idx_str:idx_end]
            idx_str = idx_end
            
            if if_exist_image:
                i_features = i_features.permute(1, 2, 0)
                
                if feature_lens[i] % H != 0:
                    i_features = i_features[:, :, 1:] # remove CLS token
                    feature_lens[i] -= 1
                
                i_features = i_features.reshape(L, D, H, feature_lens[i]//H)
                if feature_lens[i] == H*(W+1):
                    i_features = i_features[:, :, :, :-1] # remove SPLIT token
                    feature_lens[i] -= H

                if self.patch_ratio > 1:
                    i_features = F.avg_pool2d(i_features, kernel_size=self.patch_ratio, stride=self.patch_ratio)
                i_features = i_features.flatten(2).permute(2, 0, 1)

            # optional: This is a hack. merging features from selective layers
            vision_feature = []
            for j in range(self.num_layer):
                vision_feature.append(self.norm_layer[j](i_features[:, self.layer_idx[j]]))
            vision_feature = self.cross_attention(self.input_layer(i_features[:, -1]), torch.stack(vision_feature, dim=1))
            # vision_feature = self.input_layer(i_features[:, -1])

            if if_exist_image:
                if H != W:
                    teacher_1d_mask = teacher_2d_masks[i, 0].view(-1).bool() # remove padded image tokens
                    teacher_feature = teacher_features[i][teacher_1d_mask]
                else:
                    teacher_feature = teacher_features[i]

                # TODO: double check
                patch_ratio_t = int(math.sqrt(teacher_feature.shape[0] // vision_feature.shape[0]))
                if patch_ratio_t > 1:
                    H_t = W_t = int(math.sqrt(teacher_feature.shape[0]))
                    teacher_feature = rearrange(teacher_feature, '(h w) d -> d h w', h=H_t)
                    teacher_feature = F.avg_pool2d(teacher_feature, kernel_size=patch_ratio_t, stride=patch_ratio_t)
                    teacher_feature = rearrange(teacher_feature, 'd h w -> (h w) d')

                if teacher_feature.shape[0] > 0:
                    count_image += 1
                    distill_loss += self.compute_cosineloss(vision_feature, teacher_feature)
                else:
                    assert vision_feature.shape[0] == 8*8 # padded frames TODO: adaptive
                    distill_loss += self.compute_cosineloss(vision_feature, vision_feature) # this is a hacky fix, for deepspeed zero3 to work
                    
            else:
                distill_loss += self.compute_cosineloss(vision_feature, vision_feature.detach())

        return distill_loss / max(count_image, 1.0)

class VideoTokenizer(nn.Module):
    def __init__(self, input_size, config, video_config):
        super().__init__()

        self.is_loaded = True
        self.hidden_size = input_size
        
        if 'LB' in config.mm_video_tower:
            self.image_processor = LanguageBindVideoProcessor(video_config)
        else:
            raise NotImplementedError
        
        patch_stride, conv_stride = self.image_processor.patch_stride, self.image_processor.conv_stride
        self.patch_stride, self.conv_stride = patch_stride, conv_stride

        self.patch_embedding = nn.Conv2d(3, input_size, kernel_size=patch_stride, stride=patch_stride, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(input_size))
        self.split_embedding = nn.Parameter(torch.randn(input_size))

        dw_reduction = 1.5
        mlp_factor = 4
        mlp_dropout = 0.5
        cls_dropout = 0.5
        n_dim = self.hidden_size
        num_classes = n_dim
        self.frozen = False
        if input_size == 1408: n_head = 11      # InterVideoV2
        elif input_size == 1024: n_head = 8     # LanguageBind or CLIP
        elif input_size == 1152: n_head = 8     # Siglip, 8 9 12
        else: raise NotImplementedError
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, n_dim))

        # LSTE
        self.lmhra1 = Local_MHRA(self.hidden_size, dw_reduction=dw_reduction)
        self.lmhra2 = Local_MHRA(self.hidden_size, dw_reduction=dw_reduction)   # deprecated
        self.dpe = nn.Conv3d(n_dim, n_dim, kernel_size=3, stride=1, padding=1, bias=True, groups=n_dim)
        nn.init.constant_(self.dpe.bias, 0.)

        # GSTRA
        self.dec = Extractor(n_dim, n_head, mlp_factor=mlp_factor, dropout=mlp_dropout, drop_path=0,)
        self.proj = nn.Sequential(nn.LayerNorm(n_dim), nn.Dropout(cls_dropout), nn.Linear(n_dim, num_classes),)
        if not self.frozen:
            self.balance = nn.Parameter(torch.zeros((n_dim)))
            self.sigmoid = nn.Sigmoid()
        
        # LSD
        self.local_attention = LocalAttention(input_size, conv_stride)

        # FSRA
        self.global_attention = GlobalAttention(input_size)
    
    def forward(self, pixel_values, modules):
        B, T, _, _, _ = pixel_values.shape
        pixel_values = rearrange(pixel_values, 'b t c h w -> (b t) c h w')
        pixel_values, pixel_masks = pixel_values[:, :-1, :, :], pixel_values[:, -1:, :, :]  # remove mask

        if pixel_values.dtype != self.dtype:
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=self.dtype))
        else:
            patch_embeds = self.patch_embedding(pixel_values)
        patch_masks = F.avg_pool2d(pixel_masks, kernel_size=self.patch_stride, stride=self.patch_stride) # BT, 1, h, w
        # assert len(torch.where(patch_masks % 1)[0]) == 0
        _, C, h_, w_ = patch_embeds.shape   # bt, c, h, w

        if T > 1:
            tmp_patch_embeds = patch_embeds.clone()
            # =================LSTE
            patch_embeds = rearrange(patch_embeds, '(b t) c h w -> b c t h w', b=B).contiguous()
            patch_embeds = patch_embeds + self.lmhra1(patch_embeds)
            tmp_feats = self.dpe(patch_embeds)      # b c t h w
            patch_embeds = tmp_feats + patch_embeds # b c t h w

            # =================GSTRA
            patch_embeds = patch_embeds.permute(2,3,4,0,1).contiguous().view(-1, B, C) # thw, B, C
            cls_token = self.temporal_cls_token.repeat(1, B, 1)
            cls_token = self.dec(cls_token, patch_embeds)[0, :, :] # -> 1, B, C -> B, C

            patch_embeds = rearrange(patch_embeds, '(t h w) b c -> (b t) c h w', t=T, h=h_, w=w_)
            patch_embeds = patch_embeds + tmp_patch_embeds  # optional: dense connection

        patch_embeds_, patch_hw_ = [], []
        for i in range(patch_embeds.shape[0]): # remove padded patches
            if patch_masks[i, 0].sum() == 0:
                patch_embed = patch_embeds[i, :, :16, :16]
            else:
                nonzero_indices = torch.nonzero(patch_masks[i, 0], as_tuple=False)
                h1, w1 = nonzero_indices[0]
                h2, w2 = nonzero_indices[-1]
                patch_embed = patch_embeds[i, :, h1:h2+1, w1:w2+1]

            H, W = patch_embed.shape[1:]
            h, w = H // self.conv_stride, W // self.conv_stride

            # ===============LSD
            patch_embed = self.local_attention(patch_embed.unsqueeze(0))            # chw->1chw-> 1,hw,C

            # ===============FSRA
            class_embed = self.class_embedding[None, None, :].to(dtype=self.dtype)  # CLS token C-> 1，1，C
            class_embed = self.global_attention(class_embed, patch_embed)[0]        # B=1, 1, C -> 1, C

            patch_embed = patch_embed.transpose(-2, -1).reshape(-1, h, w)
            split_embed = self.split_embedding[:, None, None].repeat(1, h, 1)
            patch_embed = torch.cat([patch_embed, split_embed.to(dtype=self.dtype)], dim=-1)
            patch_embed = patch_embed.flatten(1).transpose(0, 1)                    # h(w+1), C

            if T > 1: # fuse CLS tokens
                weight = self.sigmoid(self.balance)
                class_embed = self.proj((1 - weight) * cls_token[i//T].unsqueeze(0) + weight * class_embed)
                
            patch_embeds_.append(modules(torch.cat([class_embed, patch_embed], dim=0)))
            patch_hw_.append(torch.LongTensor([h, w]).to(self.device))

        return patch_embeds_, patch_hw_
    
    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    @property
    def device(self):
        return self.patch_embedding.weight.device

class VideoCompressor(nn.Module):
    def __init__(self, teacher_layer, processor, llm_size, num_layer, layer_rate=4, num_heads=8, chunk_size=4):
        super().__init__()

        self.layer_idx = num_layer - torch.arange(0, num_layer + 1, layer_rate)
        self.num_layer = len(self.layer_idx)    
        self.chunk_size = chunk_size

        self.teacher_layer = teacher_layer
        if teacher_layer.__dict__.get('video_tower_name') and teacher_layer.__dict__['video_tower_name'] == 'LanguageBind/LanguageBind_Video_merge':
            teacher_hid_size = 1024
        else:
            raise NotImplementedError
    
        self.norm_layer = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(llm_size), nn.Linear(llm_size, teacher_hid_size, bias=False))  # 4096 -> teacher_hid_size
            for _ in range(self.num_layer)])

        self.input_layer = nn.Linear(llm_size, teacher_hid_size)
        self.output_layer = nn.Linear(teacher_hid_size, teacher_hid_size)
        self.num_heads = num_heads
        self.scale = teacher_hid_size ** -0.5
        
        max_patch = processor.image_size // processor.patch_stride // processor.conv_stride
        max_patch_teacher = processor.image_size_teacher // processor.patch_stride_teacher
        assert (max_patch % max_patch_teacher == 0) or (max_patch_teacher % max_patch == 0)

        self.mask_stride_teacher = processor.patch_stride_teacher
        self.patch_ratio = max_patch // max_patch_teacher

    def cross_attention(self, vision_input, vision_outputs):
        N, L, C = vision_outputs.shape

        patch_q = vision_outputs[:, :1].reshape(N, 1, self.num_heads, -1).transpose(1, 2)
        patch_kv = vision_outputs[:, 1:].reshape(N, L - 1, self.num_heads, -1).transpose(1, 2)

        cross_attn = (patch_q * self.scale * patch_kv).sum(-1).softmax(dim=-1)
        fuse_ouptuts = (cross_attn.unsqueeze(-1) * patch_kv).sum(-2)

        return vision_input + self.output_layer(fuse_ouptuts.reshape(N, -1))
    
    def compute_cosineloss(self, pred_feature, teacher_feature):
        loss_func = nn.CosineSimilarity(dim=-1)
        return 1 - loss_func(pred_feature, teacher_feature).mean()
    
    def compute_mseloss(self, pred_feature, teacher_feature):
        loss_func = nn.MSELoss()
        return loss_func(pred_feature, teacher_feature)

    def forward(self, input_ids, images, all_features, patch_hw, feature_lens, image_patch_hw=None):

        B, N, L, D = all_features.shape
        BT, _, H2, W2 = images.shape
        T = BT // B
        num_chunks = T // self.chunk_size
        patch_hw = [i for item in patch_hw for i in item]
        feature_lens = [i for item in feature_lens for i in item]

        teacher_2d_masks = F.avg_pool2d(images[:, -1:, :, :], kernel_size=self.mask_stride_teacher, stride=self.mask_stride_teacher)
        assert len(torch.where(teacher_2d_masks % 1)[0]) == 0

        # images exist
        if image_patch_hw is not None:
            tmp_list = [(i[0]*(i[1]+1)+1).item() for i in image_patch_hw]
            n_image_tokens = [sum(tmp_list[i:i+T]) for i in range(0, len(tmp_list), T)]
        
        if self.teacher_layer.__dict__.get('video_tower_name') and self.teacher_layer.__dict__['video_tower_name'] == 'LanguageBind/LanguageBind_Video_merge':
            images = rearrange(images, '(b t) c h w -> b t c h w', b=B)
            teacher_features = []
            for i in range(B):
                cur_images = images[i]
                chunks = cur_images.chunk(num_chunks, dim=0)
                chunk_batch = torch.stack(chunks, dim=0)
                chunk_batch = rearrange(chunk_batch, 'b t c h w -> b c t h w')
                tmp_teacher_features = self.teacher_layer((chunk_batch[:, :-1, :, :, :]))
                tmp_teacher_features = rearrange(tmp_teacher_features, 'b t l d -> (b t) l d')[:, 1:, :]
                
                teacher_features.extend(tmp_teacher_features)
        else:
            raise NotImplementedError

        distill_loss, count_image = 0, 0
        for i in range(BT):
            H, W = patch_hw[i]
            if_exist_image = IMAGE_TOKEN_INDEX in input_ids[i//T] and teacher_2d_masks[i, 0].sum() > 0

            if i % T == 0:
                idx_str = torch.where(input_ids[i//T] == IMAGE_TOKEN_INDEX)[0][0] if if_exist_image else 0

                if image_patch_hw is not None:
                    idx_str += n_image_tokens[i // T]
                
            idx_end = idx_str + min(N, feature_lens[i])
            i_features = all_features[i//T, idx_str:idx_end]
            idx_str = idx_end

            if if_exist_image:
                i_features = i_features.permute(1, 2, 0)

                # # video-level tokens in the first frame
                # if i % T == 0 and feature_lens[i] != feature_lens[i+1]:
                #     i_features = i_features[:, :, 128:] # video ST-CLS tokens   TODO adaptive 
                #     feature_lens[i] -= 128

                if feature_lens[i] % H != 0: # 1 + H*(W+1)
                    i_features = i_features[:, :, 1:] # remove CLS token
                    feature_lens[i] -= 1

                i_features = i_features.reshape(L, D, H, feature_lens[i]//H)
                if feature_lens[i] == H*(W+1):
                    i_features = i_features[:, :, :, :-1] # remove SPLIT token
                    feature_lens[i] -= H

                if self.patch_ratio > 1:
                    i_features = F.avg_pool2d(i_features, kernel_size=self.patch_ratio, stride=self.patch_ratio)
                i_features = i_features.flatten(2).permute(2, 0, 1)

            # optional: This is a hack. merging features from selective layers
            vision_feature = []
            for j in range(self.num_layer):
                vision_feature.append(self.norm_layer[j](i_features[:, self.layer_idx[j]]))
            vision_feature = self.cross_attention(self.input_layer(i_features[:, -1]),
                                                  torch.stack(vision_feature, dim=1))
            # vision_feature = self.input_layer(i_features[:, -1])

            if if_exist_image:
                teacher_nonzero_indices = torch.nonzero(teacher_2d_masks[i, 0], as_tuple=False)
                h1, w1 = teacher_nonzero_indices[0]
                h2, w2 = teacher_nonzero_indices[-1]       
                
                if H != W:
                    teacher_1d_mask = teacher_2d_masks[i, 0].view(-1).bool() # remove padded image tokens
                    teacher_feature = teacher_features[i][teacher_1d_mask]
                    assert teacher_feature.shape[0] == (h2-h1+1) * (w2-w1+1)
                else:
                    teacher_feature = teacher_features[i]

                patch_ratio_t = int(math.sqrt(teacher_feature.shape[0] // vision_feature.shape[0]))
                if patch_ratio_t > 1:
                    teacher_feature = rearrange(teacher_feature, '(h w) d -> d h w', h=h2-h1+1)
                    teacher_feature = F.avg_pool2d(teacher_feature, kernel_size=patch_ratio_t, stride=patch_ratio_t)
                    teacher_feature = rearrange(teacher_feature, 'd h w -> (h w) d')

                if teacher_feature.shape[0] > 0:
                    count_image += 1
                    distill_loss += self.compute_cosineloss(vision_feature, teacher_feature)
                else:
                    assert vision_feature.shape[0] == 8*8 # padded frames TODO: adaptive
                    distill_loss += self.compute_cosineloss(vision_feature, vision_feature) # this is a hacky fix, for deepspeed zero3 to work
            else:
                distill_loss += self.compute_cosineloss(vision_feature, vision_feature.detach())

        return distill_loss / max(count_image, 1.0)
