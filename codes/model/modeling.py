import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from transformers import BertModel, ResNetModel
from transformers.models.clip import CLIPModel


@dataclass
class EncoderOutputs:
    image_embed: Optional[torch.FloatTensor] = None
    text_embed: Optional[torch.FloatTensor] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    text_hidden_states: Optional[torch.FloatTensor] = None

# 输入实体和提及的特征张量：entity_text_cls: [B, N, D]; entity_image_cls: [B, N, D]; entity_text_tokens: [B, N, Lt, D]; entity_image_tokens: [B, N, Lv, D]
# mention_text_cls: [B, D]; mention_image_cls: [B, D]; mention_text_tokens: [B, Lt, D]; mention_image_tokens: [B, Lv, D]

class CLIPEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained(self.args.clip_model, local_files_only=getattr(args, 'local_files', True))
        self.clip.requires_grad_(False)
        self.image_cls_fc = nn.Linear(self.clip.config.projection_dim, args.model.dim)
        self.text_cls_fc = nn.Linear(self.clip.config.projection_dim, args.model.dim)
        self.image_tokens_fc = nn.Linear(self.clip.config.vision_config.hidden_size, args.model.dim)
        self.text_tokens_fc = nn.Linear(self.clip.config.text_config.hidden_size, args.model.dim)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None):
        clip_output = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        text_embeds = self.text_cls_fc(clip_output.text_embeds)
        image_embeds = self.image_cls_fc(clip_output.image_embeds)
        text_tokens = self.text_tokens_fc(clip_output.text_model_output[0])
        image_tokens = self.image_tokens_fc(clip_output.vision_model_output[0])
        return text_embeds, image_embeds, text_tokens, image_tokens


class BertResnetEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_model, local_files_only=getattr(args, 'local_files', True))
        self.resnet = ResNetModel.from_pretrained(args.resnet_model, local_files_only=getattr(args, 'local_files', True))
        self.bert.requires_grad_(False)
        self.resnet.requires_grad_(False)
        self.text_fc = nn.Linear(768, args.model.dim)
        self.image_fc = nn.Linear(2048, args.model.dim)

    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        resnet_output = self.resnet(pixel_values=pixel_values)
        text_hidden_states = self.text_fc(bert_output.last_hidden_state)
        text_embeds = torch.mean(text_hidden_states, dim=1)
        shape = resnet_output.last_hidden_state.shape
        image_hidden_states = self.image_fc(resnet_output.last_hidden_state.reshape(shape[0], -1, shape[1]))
        image_embeds = torch.mean(image_hidden_states, dim=1)
        return text_embeds, image_embeds, text_hidden_states, image_hidden_states


class CandidateAwareAGSFRefiner(nn.Module):
    def __init__(self, dim, cond_dim=None, hidden_dim=None, d_state=16, d_conv=4, expand=2, alpha_init=0.0, dropout=0.05):
        super().__init__()
        cond_dim = cond_dim or dim
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.feat_norm = nn.LayerNorm(dim)
        self.cond_proj = nn.Linear(cond_dim, dim)
        self.in_proj = nn.Linear(dim, dim)
        self.cond_in_proj = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.constant_(self.gate[-2].bias, -2.0)

    def forward(self, feats, cond, mask=None):
        squeeze_candidate = False
        if feats.dim() == 3:
            feats = feats.unsqueeze(1)
            squeeze_candidate = True
        if feats.dim() != 4:
            raise ValueError(f'feats must be [B,N,L,D] or [B,L,D], got {tuple(feats.shape)}')
        batch_size, num_candidates, seq_len, dim = feats.shape
        if dim != self.dim:
            raise ValueError(f'feats last dim must be {self.dim}, got {dim}')
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, num_candidates, -1)
        elif cond.dim() == 3 and cond.shape[1] == 1 and num_candidates != 1:
            cond = cond.expand(-1, num_candidates, -1)
        elif cond.dim() != 3:
            raise ValueError(f'cond must be [B,N,D] or [B,D], got {tuple(cond.shape)}')
        if cond.shape[:2] != feats.shape[:2]:
            raise ValueError(f'cond shape {tuple(cond.shape)} is not aligned with feats shape {tuple(feats.shape)}')

        mask_float = None
        if mask is not None:
            if mask.dim() == 2:
                if squeeze_candidate:
                    mask = mask.unsqueeze(1)
                else:
                    mask = mask.unsqueeze(1).expand(-1, num_candidates, -1)
            if mask.shape != (batch_size, num_candidates, seq_len):
                raise ValueError(f'mask shape {tuple(mask.shape)} is not aligned with feats shape {tuple(feats.shape)}')
            mask_float = mask.to(dtype=feats.dtype, device=feats.device).unsqueeze(-1)

        cond_proj = self.cond_proj(cond)
        cond_tokens = cond_proj.unsqueeze(2).expand(-1, -1, seq_len, -1)
        norm_feats = self.feat_norm(feats)
        h = self.act(self.in_proj(norm_feats) + self.cond_in_proj(cond_tokens))
        if mask_float is not None:
            h = h * mask_float
        h_flat = h.reshape(batch_size * num_candidates, seq_len, dim)
        mixed_fwd = self.mamba(h_flat)
        mixed_bwd = torch.flip(self.mamba(torch.flip(h_flat, dims=[1])), dims=[1])
        mixed = self.out_proj((mixed_fwd + mixed_bwd) * 0.5)
        mixed = self.dropout(mixed).reshape(batch_size, num_candidates, seq_len, dim)
        if mask_float is not None:
            mixed = mixed * mask_float
        gate_input = torch.cat([feats, cond_tokens, feats * cond_tokens, mixed], dim=-1)
        gate = self.gate(gate_input)
        delta = self.alpha * gate * mixed
        if mask_float is not None:
            delta = delta * mask_float
        refined = self.out_norm(feats + delta)
        if squeeze_candidate:
            refined = refined.squeeze(1)
        return refined


class PairAwareEvidenceCandidateSelector(nn.Module):  # 该 refiner 模块结合 TECS 与 MVCS
    def __init__(self, dim, hidden_dim=None, alpha_init=0.05, dropout=0.05):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.feat_norm = nn.LayerNorm(dim)
        self.cond_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(              # 判断每个 token 应该被增强的程度，输入：norm_feats + cond_tokens + norm_feats * cond_tokens + abs(norm_feats - cond_tokens)
            nn.Linear(dim * 4, hidden_dim),     # [B, N, L, 4D] -> [B, N, L, hidden_dim]
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),           # [B, N, L, hidden_dim] -> [B, N, L, 1]
            nn.Sigmoid(),                       # 输出门控权重 [B, N, L, 1]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.constant_(self.gate[-2].bias, -2.0)

    def forward(self, feats, cond, mask=None):
        squeeze_candidate = False
        if feats.dim() == 3:
            feats = feats.unsqueeze(1)
            squeeze_candidate = True
        if feats.dim() != 4:
            raise ValueError(f'feats must be [B,N,L,D] or [B,L,D], got {tuple(feats.shape)}')
        batch_size, num_candidates, seq_len, dim = feats.shape
        if dim != self.dim:
            raise ValueError(f'feats last dim must be {self.dim}, got {dim}')
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, num_candidates, -1)
        elif cond.dim() == 3 and cond.shape[1] == 1 and num_candidates != 1:
            cond = cond.expand(-1, num_candidates, -1)
        elif cond.dim() != 3:
            raise ValueError(f'cond must be [B,N,D] or [B,D], got {tuple(cond.shape)}')
        if cond.shape != (batch_size, num_candidates, dim):
            raise ValueError(f'cond shape {tuple(cond.shape)} is not aligned with feats shape {tuple(feats.shape)}')

        mask_float = None
        if mask is not None:
            if mask.dim() == 2:
                if squeeze_candidate:
                    mask = mask.unsqueeze(1)
                else:
                    mask = mask.unsqueeze(1).expand(-1, num_candidates, -1)
            if mask.shape != (batch_size, num_candidates, seq_len):
                raise ValueError(f'mask shape {tuple(mask.shape)} is not aligned with feats shape {tuple(feats.shape)}')
            mask_float = mask.to(dtype=feats.dtype, device=feats.device).unsqueeze(-1)

        norm_feats = self.feat_norm(feats)    # 维度归一化后的特征 [B, N, L, D]
        cond_tokens = self.cond_proj(cond).unsqueeze(2).expand(-1, -1, seq_len, -1)  # [B, N, D] -> 增加 token 维度 [B, N, 1, D] -> 扩展为 [B, N, L, D]
        gate_input = torch.cat([              # 构造 pair-aware 输入，将每个候选的特征拼接为 [B, N, L, D*4]
            norm_feats,
            cond_tokens,
            norm_feats * cond_tokens,
            torch.abs(norm_feats - cond_tokens),
        ], dim=-1)
        gate = self.gate(gate_input)          # 每个 token 一个增强选择权重 [B, N, L, 4D] -> [B, N, L, 1]
        if mask_float is not None:
            gate = gate * mask_float
        delta = self.alpha * gate * self.value_proj(norm_feats)   # 将门控控制的增强量加入原始 feats，重要 token 的 gate 更大，不重要 token 的 gate 更小
        if mask_float is not None:
            delta = delta * mask_float
        selected = self.out_norm(feats + delta)                   # 残差更新：原始 feats + gate 控制的 delta，[B, N, L, D] + [B, N, L, D] -> [B, N, L, D]
        if squeeze_candidate:                                     # 恢复原始输入维度，如果输入为 [B,L,D]，需要去除 candidate 维度
            selected = selected.squeeze(1)                        # [B, N, L, D] -> [B, L, D]
            gate = gate.squeeze(1)                                # [B, N, L, 1] -> [B, L, 1]，每个 token 的增强权重
        return selected, gate

# PASM 使用当前 mention 的语义作为条件，通过双向 Mamba 建模候选实体文本内部的上下文关系，并以门控残差方式增强与当前 mention 更相关的实体语义 token。
# 语义挖掘
class PairAwareAdaptiveSemanticMining(nn.Module):
    def __init__(self, dim, hidden_dim=None, d_state=16, d_conv=4, expand=2, alpha_init=0.0, dropout=0.05):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.token_norm = nn.LayerNorm(dim)    # [D] → [D]
        self.cond_proj = nn.Linear(dim, dim)   # [D] → [D]
        self.token_proj = nn.Linear(dim, dim)  # [D] → [D]
        self.cond_token_proj = nn.Linear(dim, dim)  # [D] → [D]
        self.act = nn.SiLU()
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)  # [Lt, D] → [Lt, D]
        self.state_proj = nn.Linear(dim, dim)       # 投影双向 Mamba 输出 	[D] → [D]
        self.dropout = nn.Dropout(dropout)
        self.semantic_gate = nn.Sequential(         # 产生细粒度语义门控 [4D] → [D]
            nn.Linear(dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)            # [D] → [D]
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))  # 	控制 PASM 残差增量整体强度
        nn.init.constant_(self.semantic_gate[-2].bias, -2.0)

    def forward(self, tokens, cond, mask):  # Selected Entity Text Tokens [B, N, Lt, D] mention_semantic_cond_expand [B, N, D]   entity_text_mask [B, N, Lt]
        if tokens.dim() != 4:
            raise ValueError(f'tokens must be [B,N,L,D], got {tuple(tokens.shape)}')
        batch_size, num_candidates, seq_len, dim = tokens.shape
        if dim != self.dim:
            raise ValueError(f'tokens last dim must be {self.dim}, got {dim}')
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, num_candidates, -1)
        elif cond.dim() == 3 and cond.shape[1] == 1 and num_candidates != 1:
            cond = cond.expand(-1, num_candidates, -1)
        elif cond.dim() != 3:
            raise ValueError(f'cond must be [B,D] or [B,N,D], got {tuple(cond.shape)}')
        if cond.shape != (batch_size, num_candidates, dim):
            raise ValueError(f'cond shape {tuple(cond.shape)} is not aligned with tokens shape {tuple(tokens.shape)}')
        if mask.dim() != 3 or mask.shape != tokens.shape[:3]:   # 确保区分是padding token 还是有效token
            raise ValueError(f'mask must be [B,N,L], got {tuple(mask.shape)}')

        mask_float = mask.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(-1)   # [B, N, Lt] -> [B, N, Lt, 1]， 使得 mask 可以与 token 相乘
        cond_tokens = self.cond_proj(cond).unsqueeze(2).expand(-1, -1, seq_len, -1)    # [B, N, D] -> con_proj [B, N, D] -> unsqueeze [B, N, 1, D] -> expand [B, N, Lt, D]
        norm_tokens = self.token_norm(tokens)  # 归一化 entity token

        # 投影后的逐元素 加法融合： C 不被附加到 X 的末尾；C 被映射为一个与 X 同维度的 条件偏移量，逐维加入到 token 表示中。 也就是 类似于给每个 token 添加条件偏置
        hidden = self.act(self.token_proj(norm_tokens) + self.cond_token_proj(cond_tokens))  # [B, N, Lt, D] + [B, N, Lt, D] -> [B, N, Lt, D] ，二者融合：每个 entity_token 面对不同 mention，得到的 hidden_state 可以不同
        # 有效token：保持其 hidden 值； padding token：所有D 个维度都被置为0
        hidden = hidden * mask_float           # [B, N, Lt, D] * [B, N, Lt, 1] -> [B, N, Lt, D]， 使得 padding token 的 hidden_state 为 0 （mamba 本身不像 transformer attention 那样在调用中直接接收 padding mask，mamba 前将 padding state 清零， mamba 后还会再次清零输出）
        # mamba 是沿着 Lt 维度建模，不会在不同的 candidate/mention 之间传播状态
        hidden_flat = hidden.reshape(batch_size * num_candidates, seq_len, dim)   # [B, N, Lt, D] reshape -> [B*N, Lt, D],因为 mamba 需要处理一批序列，这里把 mention-candidate pair [Lt, D] 看作一条独立的entity text sequence,共有 B*N 条 mention-candidate pair
        state_fwd = self.mamba(hidden_flat)    # [B × N, Lt, D] -> [B × N, Lt, D] 按照entity 文本从前到后的顺序建模
        state_bwd = torch.flip(self.mamba(torch.flip(hidden_flat, dims=[1])), dims=[1])  # 沿着 token 序列维度翻转建模之后再翻转回来
        semantic_state = self.state_proj((state_fwd + state_bwd) * 0.5)                  # [B × N, Lt, D]， 不是 concat 而是 average ，所以输出维度D 不变
        semantic_state = self.dropout(semantic_state).reshape(batch_size, num_candidates, seq_len, dim)  # [B × N, Lt, D] -> [B, N, Lt, D]
        semantic_state = semantic_state * mask_float   # [B, N, Lt, D] * [B, N, Lt, 1] -> [B, N, Lt, D]， 使得 padding token 的 semantic_state 为 0, 对 Mamba 输出的二次屏蔽，padding 位置最终不会保留语义状态
        gate_input = torch.cat([tokens, cond_tokens, tokens * cond_tokens, semantic_state], dim=-1)  # [B, N, Lt, D] + [B, N, Lt, D] + [B, N, Lt, D] + [B, N, Lt, D]上下文感知的双向语义状态 -> [B, N, Lt, 4D]
        semantic_gate = self.semantic_gate(gate_input) * mask_float                                  # [B, N, Lt, 4D] -> [B, N, Lt, D]， 每个 token 的每个特征维度都有单独的门控值（0，1）， 控制应该从双向 semantic state 中吸收多少更新信息
        delta = self.alpha * semantic_gate * semantic_state                                          # [B, N, Lt, D] * [B, N, Lt, D] -> [B, N, Lt, D]， 控制 PASM 残差增量整体强度
        mined_tokens = self.out_norm(tokens + delta)                                                 # [B, N, Lt, D] + [B, N, Lt, D] -> [B, N, Lt, D]， 残差更新：原始 tokens + 门控控制的 delta
        return mined_tokens, semantic_gate   # mined_tokens [B, N, Lt, D] semantic_gate [B, N, Lt, D]:每个 token 的每个特征维度都有单独的门控值


class AnchorEvidencePool(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.token_proj = nn.Linear(dim, dim)
        self.anchor_proj = nn.Linear(dim, dim)
        self.score_proj = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, anchor, mask=None):
        if tokens.dim() != 4:
            raise ValueError(f'tokens must be [B,N,L,D], got {tuple(tokens.shape)}')
        if anchor.dim() != 3:
            raise ValueError(f'anchor must be [B,N,D], got {tuple(anchor.shape)}')
        if tokens.shape[:2] != anchor.shape[:2]:
            raise ValueError(f'anchor shape {tuple(anchor.shape)} is not aligned with tokens shape {tuple(tokens.shape)}')
        hidden = torch.tanh(self.token_proj(tokens) + self.anchor_proj(anchor).unsqueeze(-2))
        scores = self.score_proj(self.dropout(hidden)).squeeze(-1)
        if mask is not None:
            if mask.shape != tokens.shape[:3]:
                raise ValueError(f'mask shape {tuple(mask.shape)} is not aligned with tokens shape {tuple(tokens.shape)}')
            mask_bool = mask.to(dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~mask_bool, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=-2)
        return self.norm(pooled)

class MentionImageEvidencePool(nn.Module):  # 接收的是 mention_image_tokens: ([B,L_v,D])  +  mention_image_cls: ([B,D])
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.token_proj = nn.Linear(dim, dim)  # 视觉 token 投影：[B, L_v, D] → [B, L_v, D]，图像token 中哪些语义维度适合用于决定视觉证据的重要性
        self.anchor_proj = nn.Linear(dim, dim) # anchor 投影：[B, D] → [B, D]，在这张图片的整体语义下，哪些局部视觉 token 最值得作为证据保留
        self.score_proj = nn.Linear(dim, 1)    # 将每个 token 的融合表示 映射为一个 标量 score：[B, L_v, D] → [B, L_v, 1]
        self.dropout = nn.Dropout(dropout)     # 防止 attention 打分器过度依赖少量激活维度，从而降低过拟合
        self.norm = nn.LayerNorm(dim)          # 输出归一化：[B, D] → [B, D]

    def forward(self, tokens, anchor, mask=None):
        if tokens.dim() != 3 or anchor.dim() != 2:
            raise ValueError('strict mention visual pool expects [B,L,D] tokens and [B,D] anchor')
        if tokens.shape[0] != anchor.shape[0] or tokens.shape[-1] != anchor.shape[-1]:
            raise ValueError('strict mention visual pool inputs are incompatible')
        hidden = torch.tanh(self.token_proj(tokens) + self.anchor_proj(anchor).unsqueeze(1)) # 每个局部视觉 token 在当前 mention 整图语义条件下的匹配表示：hidden: [B, L_v, D]
        scores = self.score_proj(self.dropout(hidden)).squeeze(-1)  # 计算每个视觉 token 的注意力分数
        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError('strict mention visual mask has incompatible shape')
            scores = scores.masked_fill(~mask.to(dtype=torch.bool), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)  # 归一化为注意力权重：[B, L_v] → [B, L_v]，表示第 (b) 个 mention 中第 (l) 个视觉 token 对最终视觉证据的重要程度。
        return self.norm(torch.sum(tokens * weights.unsqueeze(-1), dim=1))   # 加权汇聚并归一化， [B, D]

# 去掉 MVCS 后使用 mention 自身的 CLS 表示，从其视觉 token 中自适应地挑选重要视觉区域；但是这份视觉选择仅针对“这张 mention 图片”，不再针对“这张图片与某一个候选实体的匹配关系”。
class MentionTextQueryPool(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.token_proj = nn.Linear(dim, dim)    # [B, L, D] -> [B, L, D]
        self.anchor_proj = nn.Linear(dim, dim)   # [B, D] -> [B, D]
        self.score_proj = nn.Linear(dim, 1)      # [B, L, D] -> [B, L, 1]
        self.dropout = nn.Dropout(dropout)       # 训练时对 hidden 值进行 dropout，防止过拟合，维度不变
        self.norm = nn.LayerNorm(dim)            # 对 pooled 结果进行 LayerNorm，[B,D] -> [B,D]

    def forward(self, tokens, anchor, mask):
        if tokens.dim() != 3:
            raise ValueError(f'tokens must be [B,L,D], got {tuple(tokens.shape)}')
        if anchor.dim() != 2:
            raise ValueError(f'anchor must be [B,D], got {tuple(anchor.shape)}')
        if mask.dim() != 2:
            raise ValueError(f'mask must be [B,L], got {tuple(mask.shape)}')
        if tokens.shape[0] != anchor.shape[0] or tokens.shape[-1] != anchor.shape[-1]:
            raise ValueError(f'anchor shape {tuple(anchor.shape)} is not aligned with tokens shape {tuple(tokens.shape)}')
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f'mask shape {tuple(mask.shape)} is not aligned with tokens shape {tuple(tokens.shape)}')
        hidden = torch.tanh(self.token_proj(tokens) + self.anchor_proj(anchor).unsqueeze(1))   # 每个 token 与同一个 anchor 交互，[B,L,D] + [B,1,D] -> [B,L,D]，得到 token 与 anchor 的匹配表示
        scores = self.score_proj(self.dropout(hidden)).squeeze(-1)  # [B,L,D] -> dropout 后 [B,L,D] -> 投影 [B,L,1] -> 去掉最后一维 [B,L]：表示每个 token 与 anchor 的相关程度
        mask_bool = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~mask_bool, torch.finfo(scores.dtype).min)   # 将无效 token 的 score 设置为最小值，使其经过 softmax 后权重接近 0，输出维度为 [B,L]
        weights = torch.softmax(scores, dim=-1)                             # [B,L] -> [B,L]：每个样本内部所有 token 的权重和为 1
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)           # tokens [B,L,D] 与 weights.unsqueeze [B,L,1] 相乘得到 [B,L,D]，对 token 维度加权求和得到 [B,D] 的 summary 表示
        return self.norm(pooled)  # 将 mention text token 信息压缩为一个 text-only semantic summary


class CandidateAttentionRead(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = math.sqrt(dim)

    def forward(self, query, tokens, mask):
        if tokens.dim() != 4:
            raise ValueError(f'tokens must be [B,N,L,D], got {tuple(tokens.shape)}')
        batch_size, num_candidates, seq_len, dim = tokens.shape
        if query.dim() == 2:
            query = query.unsqueeze(1).expand(-1, num_candidates, -1)
        elif query.dim() == 3 and query.shape[1] == 1 and num_candidates != 1:
            query = query.expand(-1, num_candidates, -1)
        elif query.dim() != 3:
            raise ValueError(f'query must be [B,D] or [B,N,D], got {tuple(query.shape)}')
        if query.shape != (batch_size, num_candidates, dim):
            raise ValueError(f'query shape {tuple(query.shape)} is not aligned with tokens shape {tuple(tokens.shape)}')
        if mask.dim() == 2:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(f'mask shape {tuple(mask.shape)} is not aligned with tokens shape {tuple(tokens.shape)}')
            mask = mask.unsqueeze(1).expand(-1, num_candidates, -1)
        elif mask.dim() == 3:
            if mask.shape != (batch_size, num_candidates, seq_len):
                raise ValueError(f'mask shape {tuple(mask.shape)} is not aligned with tokens shape {tuple(tokens.shape)}')
        else:
            raise ValueError(f'mask must be [B,L] or [B,N,L], got {tuple(mask.shape)}')

        q = self.query_proj(query).unsqueeze(-2)
        k = self.key_proj(tokens)
        v = self.value_proj(tokens)
        scores = torch.sum(q * k, dim=-1) / self.scale
        mask = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        read = torch.sum(self.dropout(attn).unsqueeze(-1) * v, dim=-2)
        r = self.norm(query + self.out_proj(read))
        return r, attn


class ScoreAwareGatedResidualReliability(nn.Module):  # 输入的是 [B, N, 9D] 特征 + raw_score[B, N, 3]
    def __init__(self, dim, hidden_dim=None, dropout=0.05, gate_bias_init=-2.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.score_proj = nn.Sequential(   # 把人工构造的21维分数关系特征 编码为 D 维的 score_features
            nn.LayerNorm(21),
            nn.Linear(21, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.input_norm = nn.LayerNorm(dim * 10)
        self.delta_net = nn.Sequential(            # [B, N, 10D] -> [B, N, 3]： delta 是施加到融合权重 logits上的残差修正量
            nn.Linear(dim * 10, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
        )
        self.gate_net = nn.Sequential(             # [B, N, 10D] -> [B, N, 1]: gate值代表 当前这个 mention-candidate对 是否有足够证据让模型偏离默认融合策略；也就是分数接近0 则几乎严重预设的融合先验 越接近1 允许delta改写融合权重
            nn.Linear(dim * 10, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.delta_net[-1].weight, mean=0.0, std=1e-3)   # 小方差初始化 delta_net 最后层权重
        nn.init.zeros_(self.delta_net[-1].bias)                          # delta_net 最后一层的偏置初始化为0，因为 delta 是残差修正量，不需要偏置
        nn.init.constant_(self.gate_net[-2].bias, gate_bias_init)        # 初始化门控偏置，减少早期随机动态权重导致的分支坍塌

    def forward(self, evidence_features, raw_scores):
        branch_mean = raw_scores.mean(dim=1, keepdim=True)                # raw_scores: [B, N, 3] -> branch_mean: [B, 1, 3], 沿着candidate维度N 求平均：某个mention下，对每个评分分支  计算全部候选的平均得分，作为该候选列表的相对基准
        branch_std = raw_scores.std(dim=1, keepdim=True, unbiased=False)  # raw_scores: [B, N, 3] -> branch_std: [B, 1, 3], 沿着candidate维度N 求标准差：某个mention下，对每个评分分支 计算全部候选的标准差，作为该候选列表的相对波动性
        branch_max = raw_scores.max(dim=1, keepdim=True).values           # raw_scores: [B, N, 3] -> branch_max: [B, 1, 3], 沿着candidate维度N 求最大值：某个mention下，对每个评分分支 计算全部候选的最大得分，作为该候选列表的相对最高分
        branch_min = raw_scores.min(dim=1, keepdim=True).values           # raw_scores: [B, N, 3] -> branch_min: [B, 1, 3], 沿着candidate维度N 求最小值：某个mention下，对每个评分分支 计算全部候选的最小得分，作为该候选列表的相对最低分
        score_centered = raw_scores - branch_mean                         # [B， N， 3] 相对于候选均值的中心化分数
        score_z = score_centered / (branch_std + 1e-6)                    # [B, N, 3], 沿着candidate维度N 求标准化得分：某个mention下，对每个评分分支计算全部候选的得分与平均分的差值，再除以标准差，得到标准化得分
        score_to_max = raw_scores - branch_max                            # [B, N, 3], 沿着candidate维度N 求得分与最大值的差值：某个mention下，对每个评分分支计算距当前最优候选的差值
        score_range_pos = (raw_scores - branch_min) / (branch_max - branch_min + 1e-6)  # [B, N, 3], 沿着candidate维度N 求得分与最小值的差值，再除以最大值与最小值的差值，得到标准化得分 min-max 归一化的候选相对位置
        semantic_score = raw_scores[..., 0:1]
        joint_score = raw_scores[..., 1:2]
        global_score = raw_scores[..., 2:3]                                # 取出以上三个 [B, N, 1] 作为三个评分分支的得分
        branch_agreement = torch.cat([                                     # 计算三个评分分支之间的差异，形成 6 个新的分数关系特征：[B, N, 6]
            semantic_score - joint_score,
            semantic_score - global_score,
            joint_score - global_score,
            torch.abs(semantic_score - joint_score),
            torch.abs(semantic_score - global_score),
            torch.abs(joint_score - global_score),
        ], dim=-1)
        score_context = torch.cat([                            # [B, N, 21] 拼接为 21 维分数上下文
            raw_scores,                                        # [B, N, 3]
            score_centered,                                    # [B, N, 3]
            score_z,                                           # [B, N, 3]
            score_to_max,                                      # [B, N, 3]
            score_range_pos,                                   # [B, N, 3]
            branch_agreement,                                  # [B, N, 6]
        ], dim=-1)
        score_features = self.score_proj(score_context)         # 将 21 维分数上下文变为 D 维表征[B, N, D]
        reliability_input = torch.cat([evidence_features, score_features], dim=-1)  # [B, N, 9D] + [B, N, D] -> [B, N, 10D] 合并证据与分数上下文，不仅看某个分支分数高不高，还看支持这个分数的跨模态证据是否可信
        reliability_input = self.input_norm(reliability_input)
        delta = torch.tanh(self.delta_net(reliability_input))   #  [B, N, 3] 预测权重调整量，并且限制每个分支的权重 logit 修正范围
        gate = self.gate_net(reliability_input)                 #  [B, N, 1] 预测动态调整门，表示当前样本应该多大程度使用动态权重修正
        return delta, gate, score_features


class VERA(nn.Module):    # 整体模型，包含多个子模块串联
    """VERA multimodal entity-linking model.

    Three reliable-matching scores are aggregated as the final score:
      1. semantic_interaction: pair-aware reader inner product plus a learnable
         text-CLS residual (semantic_text_residual_scale).
      2. joint: cross-modal fusion of the semantic reader outputs with the
         visual evidence anchors (z_m_vis, z_e_vis). This is where the visual
         signal enters the decision now that the standalone visual_support
         score has been removed.
      3. global_residual: fixed convex combination of text-CLS global alignment
         and multimodal global alignment (global_text_ratio).

    Textual evidence selectors build lightweight candidate evidence tokens, PASM
    performs the only state-space semantic mining step, and semantic evidence then
    calibrates visual grounding before score-aware reliability aggregation.
    """

    NUM_SCORES = 3

    def __init__(self, args):
        super().__init__()
        dim = args.model.dim
        dropout = float(getattr(args.model, 'caler_dropout', 0.05))
        cond_dropout = float(getattr(args.model, 'condition_dropout', dropout))
        pool_dropout = float(getattr(args.model, 'evidence_pool_dropout', dropout))
        reader_dropout = float(getattr(args.model, 'semantic_reader_dropout', dropout))
        d_state = int(getattr(args.model, 'agsf_d_state', 16))
        d_conv = int(getattr(args.model, 'agsf_d_conv', 4))
        expand = int(getattr(args.model, 'agsf_expand', 2))
        alpha_init = float(getattr(args.model, 'agsf_alpha_init', 0.0))

        self.mention_text_condition_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Dropout(cond_dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        # 构造 text-only 的 entity semantic condition
        self.entity_text_condition_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Dropout(cond_dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        selector_hidden = int(getattr(args.model, 'selector_hidden_dim', dim))
        selector_dropout = float(getattr(args.model, 'selector_dropout', dropout))
        selector_alpha_init = float(getattr(args.model, 'selector_alpha_init', 0.05))
        self.entity_text_selector = PairAwareEvidenceCandidateSelector(
            dim=dim,
            hidden_dim=selector_hidden,
            alpha_init=selector_alpha_init,
            dropout=selector_dropout,
        )
        pasm_hidden = int(getattr(args.model, 'pasm_hidden_dim', dim))
        pasm_dropout = float(getattr(args.model, 'pasm_dropout', dropout))
        pasm_alpha_init = float(getattr(args.model, 'pasm_alpha_init', 0.0))
        self.pasm_semantic_miner = PairAwareAdaptiveSemanticMining(
            dim=dim,
            hidden_dim=pasm_hidden,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            alpha_init=pasm_alpha_init,
            dropout=pasm_dropout,
        )
        self.mention_text_query_pool = MentionTextQueryPool(dim, dropout=pool_dropout)
        self.entity_text_query_pool = AnchorEvidencePool(dim, dropout=pool_dropout)
        self.pasm_evidence_pool = AnchorEvidencePool(dim, dropout=pool_dropout)
        self.mention_semantic_query_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Dropout(reader_dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.entity_semantic_query_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Dropout(reader_dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.mention_to_entity_text_reader = CandidateAttentionRead(dim, dropout=reader_dropout)
        self.entity_to_mention_text_reader = CandidateAttentionRead(dim, dropout=reader_dropout)
        self.mention_image_pool = MentionImageEvidencePool(dim, dropout=pool_dropout)
        self.entity_vis_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        visual_gate_hidden = int(getattr(args.model, 'visual_gate_hidden_dim', dim))
        visual_gate_dropout = float(getattr(args.model, 'visual_gate_dropout', dropout))
        visual_gate_bias_init = float(getattr(args.model, 'visual_gate_bias_init', 0.0))

        self.mention_visual_gate = nn.Sequential(
            nn.Linear(dim * 3, visual_gate_hidden),
            nn.SiLU(),
            nn.Dropout(visual_gate_dropout),
            nn.Linear(visual_gate_hidden, dim),
            nn.Sigmoid(),
        )
        self.entity_visual_gate = nn.Sequential(
            nn.Linear(dim * 3, visual_gate_hidden),
            nn.SiLU(),
            nn.Dropout(visual_gate_dropout),
            nn.Linear(visual_gate_hidden, dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.mention_visual_gate[-2].bias, visual_gate_bias_init)
        nn.init.constant_(self.entity_visual_gate[-2].bias, visual_gate_bias_init)

        self.mention_joint_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.entity_joint_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.global_mention_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.global_entity_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))

        semantic_text_residual_init = float(getattr(args.model, 'semantic_text_residual_init', 0.2))
        global_text_ratio = float(getattr(args.model, 'global_text_ratio', 0.7))
        if global_text_ratio < 0.0 or global_text_ratio > 1.0:
            raise ValueError('global_text_ratio must be in [0, 1]')
        self.semantic_text_residual_scale = nn.Parameter(torch.tensor(semantic_text_residual_init, dtype=torch.float32))
        self.register_buffer('global_text_ratio', torch.tensor(global_text_ratio, dtype=torch.float32))

        score_weights = getattr(args.model, 'reliable_score_weights', [0.43, 0.17, 0.40])
        score_weights = torch.tensor([float(w) for w in score_weights], dtype=torch.float32)
        if score_weights.numel() != self.NUM_SCORES:
            raise ValueError(f'reliable_score_weights must contain {self.NUM_SCORES} values: semantic_interaction, joint, global_residual')
        if torch.any(score_weights < 0):
            raise ValueError('reliable_score_weights must be non-negative')
        if float(score_weights.sum()) <= 0.0:
            raise ValueError('reliable_score_weights sum must be positive')
        score_weights = score_weights / score_weights.sum()
        self.register_buffer('reliable_score_weights', score_weights)
        self.score_aggregation = getattr(args.model, 'score_aggregation', 'score_aware_gated_residual')
        if self.score_aggregation not in ('fixed', 'score_aware_gated_residual'):
            raise ValueError("score_aggregation must be 'fixed' or 'score_aware_gated_residual'")
        reliability_hidden = int(getattr(args.model, 'reliability_hidden_dim', dim * 2))
        reliability_dropout = float(getattr(args.model, 'reliability_dropout', dropout))
        self.reliability_delta_scale = float(getattr(args.model, 'reliability_delta_scale', 0.3))
        reliability_gate_bias_init = float(getattr(args.model, 'reliability_gate_bias_init', -2.0))
        reliability_prior = getattr(args.model, 'reliability_prior', score_weights.tolist())
        reliability_prior = torch.tensor([float(w) for w in reliability_prior], dtype=torch.float32)
        if reliability_prior.numel() != self.NUM_SCORES:
            raise ValueError(f'reliability_prior must contain {self.NUM_SCORES} values')
        if torch.any(reliability_prior <= 0):
            raise ValueError('reliability_prior values must be positive')
        reliability_prior = reliability_prior / reliability_prior.sum()
        self.register_buffer('reliability_prior_logits', torch.log(reliability_prior))
        self.reliability_estimator = ScoreAwareGatedResidualReliability(
            dim=dim,
            hidden_dim=reliability_hidden,
            dropout=reliability_dropout,
            gate_bias_init=reliability_gate_bias_init,
        )

    def forward(self, entity_text_cls, entity_text_tokens, mention_text_cls, mention_text_tokens, entity_image_cls, entity_image_tokens, mention_image_cls, mention_image_tokens, mention_text_mask, entity_text_mask):
        batch_size, num_candidates, dim = entity_text_cls.shape
        if entity_text_tokens.shape[:2] != (batch_size, num_candidates):
            raise ValueError('entity_text_tokens must be [B,N,Lt,D]')
        if entity_image_cls.shape != entity_text_cls.shape:
            raise ValueError('entity_image_cls must match entity_text_cls shape')
        if mention_text_cls.shape != (batch_size, dim):
            raise ValueError('mention_text_cls must be [B,D]')
        if mention_text_tokens.dim() != 3 or mention_text_tokens.shape[0] != batch_size or mention_text_tokens.shape[-1] != dim:
            raise ValueError('mention_text_tokens must be [B,Lt,D]')
        if mention_image_cls.shape != (batch_size, dim):
            raise ValueError('mention_image_cls must be [B,D]')
        if mention_text_mask.dim() != 2 or mention_text_mask.shape != mention_text_tokens.shape[:2]:
            raise ValueError(f'mention_text_mask must be [B,Lt], got {tuple(mention_text_mask.shape)}')
        if entity_text_mask.dim() != 3 or entity_text_mask.shape != entity_text_tokens.shape[:3]:
            raise ValueError(f'entity_text_mask must be [B,N,Lt], got {tuple(entity_text_mask.shape)}')

        # pooled_mention_text：把 mention text 的 token 信息压缩成一个 text-only semantic summary
        pooled_mention_text = self.mention_text_query_pool(mention_text_tokens, mention_text_cls, mention_text_mask)

        # mention_semantic_cond 保证 semantic branch 中 mention query 来自文本特征，不使用 mention image
        mention_semantic_cond = self.mention_text_condition_proj(torch.cat([mention_text_cls, pooled_mention_text], dim=-1))
        mention_semantic_cond_expand = mention_semantic_cond.unsqueeze(1).expand(-1, num_candidates, -1)

        pooled_entity_text_self = self.entity_text_query_pool(entity_text_tokens, entity_text_cls, mask=entity_text_mask)
        # entity_text_cond 保证 semantic branch 中 entity query 来自文本特征，不使用 entity image
        entity_text_cond = self.entity_text_condition_proj(torch.cat([entity_text_cls, pooled_entity_text_self], dim=-1))

        # 经过 TECS 挑选增强后的特征：[B, N, Lt, D]
        selected_entity_text_tokens, entity_text_candidate_gate = self.entity_text_selector(
            entity_text_tokens, mention_semantic_cond_expand, mask=entity_text_mask
        )
        mined_entity_text_tokens, pasm_semantic_gate = self.pasm_semantic_miner(
            selected_entity_text_tokens, mention_semantic_cond_expand, entity_text_mask
        )  # 	([B,N,L_t,D])	经 PECS 和 PASM 挖掘后的候选实体文本 token

        pasm_entity_evidence = self.pasm_evidence_pool(
            mined_entity_text_tokens, mention_semantic_cond_expand, mask=entity_text_mask
        )  # ([B,N,D])	形成可直接与 mention 对齐的 PASM 证据，用于 PASM evidence loss

        # PASM 挖掘出的候选实体文本证据与 mention 语义条件之间的余弦匹配分数
        # 显示监督 PASM 挖掘出的实体语义证据的判别能力，来查看 PASM 单独做的好不好
        pasm_evidence_logits = torch.sum(
            nn.functional.normalize(mention_semantic_cond, dim=-1).unsqueeze(1) * nn.functional.normalize(pasm_entity_evidence, dim=-1),
            dim=-1,
        )
        #  broadcasting 让一个 mention 分别与它的 (N) 个候选进行比较 ： [B, D]→ [B, 1, D]
        # 第 (b) 个 mention 的语义条件，与第 (n) 个候选实体经 PASM 提取出的语义证据之间的余弦相似度，逐维相乘。
        # 相乘后的结果是[B, N, D]，沿特征维度D 求和，得到[B, N]，即每个候选实体与 mention 的语义匹配分数：batch 中每个 mention 对其 N 个候选实体的 PASM evidence 匹配分数。

        # Candidate-independent mention visual evidence is pooled before candidate broadcasting.
        z_m_vis_base = self.mention_image_pool(mention_image_tokens, mention_image_cls)
        z_m_vis = z_m_vis_base.unsqueeze(1).expand(-1, num_candidates, -1)             # 一份 candidate-independent mention visual summary 被复制给所有候选，不再是针对每个候选不同的 mention 视觉特征表示了
        # entity-side visual evidence [B, N, D]
        mention_text_tokens_expand = mention_text_tokens.unsqueeze(1).expand(-1, num_candidates, -1, -1)
        mention_semantic_query = self.mention_semantic_query_proj(torch.cat([mention_text_cls, pooled_mention_text], dim=-1))
        # PASM 后的mined_entity_text_tokens [B, N, Le, D]  --->  [B, N, D] 自身语义摘要
        pooled_entity_text = self.entity_text_query_pool(mined_entity_text_tokens, entity_text_cond, mask=entity_text_mask)
        # [B, N, D] + [B, N, D] = [B, N, 2D] ---> entity_semantic_query_proj ---> [B, N, D]:entity 用来查询 mention 文本的语义 query
        entity_semantic_query = self.entity_semantic_query_proj(torch.cat([entity_text_cls, pooled_entity_text], dim=-1))

        r_mention_to_entity, alpha_m2e = self.mention_to_entity_text_reader(
            mention_semantic_query, mined_entity_text_tokens, entity_text_mask
        ) # query[B, D], key[B, N, Lt, D], q 先扩展为 [B, N, 1, D] 之后与 k 逐token计算分数[B, N, Le]，再 softmax 得到权重 [B, N, Le]，最后与 value[B, N, Le, D] 相乘得到加权和 [B, N, D]，最后带residual 得到 [B, N, D]
          # alpha_m2e[b, n, l]：第 b 个 mention 在评价第 n 个 candidate 时，对该 candidate 第 l 个 entity text token 的关注程度。

        r_entity_to_mention, alpha_e2m = self.entity_to_mention_text_reader(
            entity_semantic_query, mention_text_tokens_expand, mention_text_mask
        ) # q [B, N, D], k [B, N, Lm, D], mention_text_mask[B, Lm] --->  [B, N, D]; alpha_e2m [B, N, Lm]
          # r_entity_to_mention[b, n]： 第 n 个 entity 用自己的语义 query，从当前 mention 的文本 token 中读取到的证据

        # Entity-side visual evidence [B, N, D].
        z_e_vis = self.entity_vis_proj(torch.cat([entity_text_cls, entity_image_cls], dim=-1))
        # Mention-side visual gate 用 semantic m2e 的 reader outputs 去校准mention视觉证据： [B, N, D] + [B, N, D] + [B, N, D]*[B, N, D] = [B, N, 3D] ---> mention_visual_gate 层 ---> [B, N, D]，每个值都在 [0, 1] 之间。g_m_vis[b,n]表示 第D 个视觉特征维度保留多少
        g_m_vis = self.mention_visual_gate(torch.cat([
            r_mention_to_entity,                    # mention semantic query 从第 n 个 candidate entity 的 PASM-refined 文本描述中读取到的语义证据。
            z_m_vis,
            r_mention_to_entity * z_m_vis,          # 两者同向且都较强：该维度相乘结果较大且为正，表示文本和视觉可能一致
        ], dim=-1))
        # 同上，最终维度为：[B, N, D]  用 semantic e2m 的 reader outputs 去校准 entity 视觉证据
        g_e_vis = self.entity_visual_gate(torch.cat([
            r_entity_to_mention,
            z_e_vis,
            r_entity_to_mention * z_e_vis,
        ], dim=-1))
        z_m_vis_grounded = g_m_vis * z_m_vis       # 逐维gate 得到语义校准后的 mention 侧视觉证据
        z_e_vis_grounded = g_e_vis * z_e_vis       # 逐维gate 得到语义校准后的 entity 侧视觉证据
        z_m_joint = self.mention_joint_proj(torch.cat([r_mention_to_entity, z_m_vis_grounded], dim=-1))   # [B, N, D] + [B, N, D] = [B, N, 2D] ---> mention_joint_proj ---> [B, N, D]： 对于当前men-ent pair， mention侧 文本-视觉 联合表示
        z_e_joint = self.entity_joint_proj(torch.cat([r_entity_to_mention, z_e_vis_grounded], dim=-1))    # 同上，对于当前 mention-candidate pair，entity 侧的联合匹配表示：entity 从 mention 文本中读到的语义证据 与 经过语义校准的 entity 多模态视觉表示 的融合结果。
        g_m = self.global_mention_proj(torch.cat([mention_text_cls, mention_image_cls], dim=-1))          # [B, D] + [B, D] = [B, 2D] ---> global_mention_proj ---> [B, D]:mention 的全局多模态融合表示
        g_e = self.global_entity_proj(torch.cat([entity_text_cls, entity_image_cls], dim=-1))             # [B, N, D] + [B, N, D] = [B, N, 2D] ---> global_entity_proj ---> [B, N, D]: 每个候选实体的全局多模态融合表示

        # 双向局部文本证据一致性 [B, N, D] * [B, N, D] = [B, N]  :若 mention 从 entity 读到的证据与 entity 从 mention 读到的证据方向一致，则 reader_score 更高
        reader_score = torch.sum(r_mention_to_entity * r_entity_to_mention, dim=-1)
        #  文本全局匹配[B, N]
        text_global_score = torch.sum(mention_text_cls.unsqueeze(1) * entity_text_cls, dim=-1)
        #  全局多模态匹配 [B, N]
        multimodal_global_score = torch.sum(g_m.unsqueeze(1) * g_e, dim=-1)

        semantic_interaction_score = reader_score + self.semantic_text_residual_scale * text_global_score  # 主文本语义分支 [B, N]
        # [B, N, D] * [B, N, D] =  [B, N, D],沿着 D 维 求和 -> [B, N]: 对第 b 个 mention 和第 n 个 candidate 而言，双方经过文本引导、视觉 grounding 后的联合表示是否一致
        joint_score = torch.sum(z_m_joint * z_e_joint, dim=-1)
        # [B, N] * global_text_ratio + [B, N] * (1.0 - global_text_ratio) = [B, N]:名称中的 residual 表示它作为一个额外的全局匹配路径，补充前面更复杂的细粒度分支
        global_residual_score = self.global_text_ratio * text_global_score + (1.0 - self.global_text_ratio) * multimodal_global_score
        scores = (semantic_interaction_score, joint_score, global_residual_score)
        raw_scores = torch.stack(scores, dim=-1) # 将三个分支堆叠成为一个张量，三个 [B, N]沿着最后一维 stack [B, N, 3]
        if self.score_aggregation == 'fixed':    # 固定权重，所有样本共享同一权重 [0.43, 0.17, 0.40]
            reliability_alpha = self.reliable_score_weights.view(1, 1, -1).expand(batch_size, num_candidates, -1)  # [3] -> [1, 1, 3] -> [B, N, 3] 即 每个 ment-ent pair 都使用完全相同的权重
            reliability_delta = torch.zeros_like(reliability_alpha)                                                # [B, N, 3] 每个元素都初始化为 0
            reliability_gate = torch.zeros(batch_size, num_candidates, 1, dtype=raw_scores.dtype, device=raw_scores.device)   # [B, N, 1] 每个元素都初始化为 0，固定模式不允许动态模块根据当前样本调整权重。
            reliability_logits = self.reliability_prior_logits.view(1, 1, -1).expand(batch_size, num_candidates, -1)  # [3] -> [1, 1, 3] -> [B, N, 3] 即 每个 ment-ent pair 都使用完全相同的权重
        else:                                    # 动态 score-aware reliability 模式
            g_m_pair = g_m.unsqueeze(1).expand(-1, num_candidates, -1)
            evidence_features = torch.cat([
                r_mention_to_entity,
                r_entity_to_mention,
                z_m_vis_grounded,
                z_e_vis_grounded,
                g_m_pair,
                g_e,
                r_mention_to_entity * r_entity_to_mention,
                z_m_vis_grounded * z_e_vis_grounded,
                g_m_pair * g_e,
            ], dim=-1)
            reliability_delta, reliability_gate, score_features = self.reliability_estimator(evidence_features, raw_scores)
            reliability_logits = self.reliability_prior_logits.view(1, 1, -1) + self.reliability_delta_scale * reliability_gate * reliability_delta   #[B, N, 3] 动态可靠性权重修正的全局幅度系数reliability_delta_scale
            reliability_alpha = torch.softmax(reliability_logits, dim=-1)  # [B, N, 3]
        final_score = torch.sum(reliability_alpha * raw_scores, dim=-1)    # [B, N]
        evidence = {
            'alpha_m2e': alpha_m2e,       # [B,N,L_e]:mention 读取第 (n) 个 candidate 的 entity text token 时的注意力分布
            'alpha_e2m': alpha_e2m,       # [B,N,L_m]:entity 读取第 (n) 个 candidate 的 mention text token 时的注意力分布
            'visual_gate_m': g_m_vis,     # [B, N, D]:mention 侧视觉增距的逐维保留比例
            'visual_gate_e': g_e_vis,     # [B, N, D]:entity  侧视觉证据的逐维保留比例
            'entity_text_candidate_gate': entity_text_candidate_gate,    # [B,N,L_e,1]：第 (b) 个 mention 对第 (n) 个 candidate 的第 (l) 个 entity text token 的候选选择/残差增强强度。
            'pasm_semantic_gate': pasm_semantic_gate,                    # [B,N,L_e,D]每个 token 的每个特征维度一个门控
            'pasm_evidence_logits': pasm_evidence_logits,                # [B, N] 计算是 mention 的语义条件与每个 candidate 的 PASM evidence 的余弦相似度
            'reliability_alpha': reliability_alpha,
            'reliability_logits': reliability_logits,
            'reliability_delta': reliability_delta,
            'reliability_gate': reliability_gate,
            'branch_scores_raw': raw_scores.detach(),
            'final_score_raw': final_score.detach(),
            'semantic_interaction_score': semantic_interaction_score.detach(),
            'joint_score': joint_score.detach(),
            'global_residual_score': global_residual_score.detach(),
            'reader_score': reader_score.detach(),
            'text_global_score': text_global_score.detach(),
            'multimodal_global_score': multimodal_global_score.detach(),
            'semantic_text_residual_scale': self.semantic_text_residual_scale.detach(),
        }
        return final_score, scores, evidence
# final_score：用于 主排序交叉熵损失、最终候选排序；
# score: 用于辅助分支交叉熵损失、分支性能分析；
# evidence： 辅助损失、训练监控、可解释性分析
