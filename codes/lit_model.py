import os
from io import BytesIO
import h5py
import PIL
import io
import math
import copy
import datetime
import torch
import ujson
import random
import time
import numpy as np
import transformers
import lightning as L
import lightning.pytorch as pl
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm as tqdm
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, DataCollatorForSeq2Seq, T5TokenizerFast, \
    AutoModelForSeq2SeqLM, AutoProcessor, CLIPProcessor, tokenization_utils_fast, AutoImageProcessor, CLIPImageProcessor
from codes.model.modeling import EncoderOutputs, CLIPEncoder, BertResnetEncoder, VERA
from codes.utils.functions import load_json_file, load_jsonl_file
from transformers import get_linear_schedule_with_warmup, get_constant_schedule


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class LightModule(L.LightningModule):
    BRANCH_NAMES = ('semantic_interaction', 'joint', 'global_residual')

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        self.save_hyperparameters()

        self.eval_info = []
        backbone = getattr(args, 'backbone', 'clip')
        self.encoder = CLIPEncoder(args) if backbone == 'clip' else BertResnetEncoder(args)
        self.matcher: VERA = VERA(args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.clip_model if backbone == 'clip' else 'bert-base-uncased',
            local_files_only=getattr(args, 'local_files', True)
        )

        if backbone == 'clip':
            self.image_processor = CLIPProcessor.from_pretrained(
                args.clip_model,
                local_files_only=getattr(args, 'local_files', True)
            ).image_processor
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(
                args.clip_model if backbone == 'clip' else args.resnet_model,
                local_files_only=getattr(args, 'local_files', True)
            )
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.branch_names = self.BRANCH_NAMES
        loss_args = getattr(args, 'loss', None)
        aux_loss_weights = getattr(loss_args, 'aux_loss_weights', None) if loss_args is not None else None
        if aux_loss_weights is None:
            aux_loss_weights = [1.0] * len(self.branch_names)
        aux_loss_weights = torch.tensor([float(w) for w in aux_loss_weights], dtype=torch.float32)
        if aux_loss_weights.numel() != len(self.branch_names):
            raise ValueError(f'aux_loss_weights must contain {len(self.branch_names)} values')
        if torch.any(aux_loss_weights < 0):
            raise ValueError('aux_loss_weights must be non-negative')
        aux_weight_sum = aux_loss_weights.sum()
        if float(aux_weight_sum) <= 0.0:
            raise ValueError('aux_loss_weights sum must be positive')
        self.register_buffer('aux_loss_weights', aux_loss_weights / aux_weight_sum)
        self.sparse_weight = float(getattr(loss_args, 'sparse_weight', 0.0)) if loss_args is not None else 0.0
        self.sparse_eps = float(getattr(loss_args, 'sparse_eps', 1e-8)) if loss_args is not None else 1e-8
        self.pasm_evidence_weight = float(getattr(loss_args, 'pasm_evidence_weight', 0.0)) if loss_args is not None else 0.0
        self.pasm_temperature = float(getattr(loss_args, 'pasm_temperature', 0.07)) if loss_args is not None else 0.07
        if self.pasm_temperature <= 0.0:
            raise ValueError('pasm_temperature must be positive')
        self.clip_processor = _transform(224)

    @staticmethod
    def _attention_entropy(alpha: torch.Tensor, eps: float, mask: torch.Tensor = None, normalize: bool = False) -> torch.Tensor:
        alpha_safe = alpha.clamp(min=eps)
        entropy = -(alpha * torch.log(alpha_safe)).sum(dim=-1)
        if normalize:
            if mask is None:
                valid_len = torch.full_like(entropy, alpha.shape[-1], dtype=alpha.dtype)
            else:
                if mask.dim() == 2 and alpha.dim() == 3:
                    mask = mask.unsqueeze(1).expand_as(alpha)
                if mask.shape != alpha.shape:
                    raise ValueError(f'mask shape {tuple(mask.shape)} is not aligned with alpha shape {tuple(alpha.shape)}')
                valid_len = mask.to(dtype=alpha.dtype, device=alpha.device).sum(dim=-1)
            valid_len = valid_len.clamp(min=2.0)
            entropy = entropy / torch.log(valid_len)
        return entropy

    @staticmethod
    def _score_rank(score: torch.Tensor) -> torch.Tensor:
        return torch.argsort(torch.argsort(score, dim=-1, descending=True), dim=-1, descending=False) + 1

    @staticmethod
    def _score_margin(score: torch.Tensor):
        pos_score = score[:, 0]
        negmax_score = score[:, 1:].max(dim=1).values
        return pos_score, negmax_score, pos_score - negmax_score

    @staticmethod
    def _masked_gate_stats(gate: torch.Tensor, mask: torch.Tensor = None):
        gate = gate.squeeze(-1)
        if mask is not None:
            if mask.dim() == 2 and gate.dim() == 3:
                mask = mask.unsqueeze(1).expand_as(gate)
            if mask.shape != gate.shape:
                raise ValueError(f'mask shape {tuple(mask.shape)} is not aligned with gate shape {tuple(gate.shape)}')
            mask = mask.to(dtype=torch.bool, device=gate.device)
            values = gate.masked_select(mask)
            masked_gate = gate.masked_fill(~mask, torch.finfo(gate.dtype).min)
            max_mean = masked_gate.max(dim=-1).values.mean()
        else:
            values = gate.reshape(-1)
            max_mean = gate.max(dim=-1).values.mean()
        return values.mean(), values.std(), max_mean


    def _log_score_diagnostics(self, prefix: str, name: str, scores: torch.Tensor, labels: torch.Tensor, batch_size: int):
        with torch.no_grad():
            pos = scores[:, 0]
            negmax = scores[:, 1:].max(dim=1).values
            margin = pos - negmax
            acc = (scores.argmax(dim=1) == labels).float().mean()
        self.log(f'{prefix}/{name}_pos_mean', pos.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_negmax_mean', negmax.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_margin_mean', margin.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_acc', acc, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def _log_tensor_distribution(self, prefix: str, name: str, value: torch.Tensor, batch_size: int):
        value = value.detach().float().reshape(-1)
        self.log(f'{prefix}/{name}_mean', value.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_std', value.std(unbiased=False), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_min', value.min(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_max', value.max(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        qs = torch.quantile(value, torch.tensor([0.25, 0.5, 0.75], device=value.device, dtype=value.dtype))
        self.log(f'{prefix}/{name}_q25', qs[0], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_q50', qs[1], on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_q75', qs[2], on_epoch=True, sync_dist=True, batch_size=batch_size)
        k = min(10, value.numel())
        self.log(f'{prefix}/{name}_top10_mean', value.topk(k).values.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)

    def _log_attention_diagnostics(self, prefix: str, name: str, alpha: torch.Tensor, batch_size: int):
        entropy = self._attention_entropy(alpha, self.sparse_eps).detach().float()
        alpha_max = alpha.detach().float().max(dim=-1).values
        self.log(f'{prefix}/{name}_entropy_mean', entropy.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_entropy_std', entropy.std(unbiased=False), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_max_mean', alpha_max.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/{name}_max_std', alpha_max.std(unbiased=False), on_epoch=True, sync_dist=True, batch_size=batch_size)

    def training_step(self, batch):
        mention_batch = batch['mention_input_dict']
        entity_batch = batch['entity_input_dict']

        mention_text_embeds, mention_image_embed, mention_text_tokens, mention_image_tokens = self.encoder(
            **mention_batch)
        entity_text_cls, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = self.encoder(
            **entity_batch)

        batch_size = mention_text_embeds.shape[0]
        total_entity = entity_text_cls.shape[0]
        entity_text_cls = entity_text_cls.reshape(batch_size, total_entity // batch_size, -1)
        entity_image_embeds = entity_image_embeds.reshape(batch_size, total_entity // batch_size, -1)
        length, dim = entity_text_seq_tokens.shape[-2:]
        entity_text_seq_tokens = entity_text_seq_tokens.reshape(batch_size, total_entity // batch_size, length, dim)
        entity_text_mask = entity_batch['attention_mask'].reshape(batch_size, total_entity // batch_size, -1)
        length, dim = mention_image_tokens.shape[-2:]
        entity_image_patch_tokens = entity_image_patch_tokens.reshape(batch_size, total_entity // batch_size, length,
                                                                      dim)

        labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        logits, other_logits, evidence = self.matcher(entity_text_cls=entity_text_cls,
                                            entity_text_tokens=entity_text_seq_tokens,
                                            mention_text_cls=mention_text_embeds,
                                            mention_text_tokens=mention_text_tokens,
                                            entity_image_cls=entity_image_embeds,
                                            entity_image_tokens=entity_image_patch_tokens,
                                            mention_image_cls=mention_image_embed,
                                            mention_image_tokens=mention_image_tokens,
                                            mention_text_mask=mention_batch['attention_mask'],
                                            entity_text_mask=entity_text_mask)
        loss_final = self.ce_loss(logits, labels)  # 最终融合分数的主交叉熵损失
        loss = loss_final
        aux_losses = []
        if other_logits is not None and isinstance(other_logits, (list, tuple)):
            if len(other_logits) != len(self.branch_names):
                raise ValueError(
                    f'matcher returned {len(other_logits)} auxiliary scores but '
                    f'{len(self.branch_names)} branches are configured'
                )
            aux_losses = [self.ce_loss(_, labels) for _ in other_logits]     #  三条辅助分支交叉熵损失
            loss_args = getattr(self.args, 'loss', None)
            final_weight = getattr(loss_args, 'final_weight', 1.0) if loss_args is not None else 1.0    # 最终融合分数的主交叉熵损失权重
            aux_weight = getattr(loss_args, 'aux_weight', 1.0) if loss_args is not None else 1.0        # 三条辅助分支交叉熵损失权重
            weighted_aux_loss = sum(branch_loss * weight for branch_loss, weight in zip(aux_losses, self.aux_loss_weights))  # 三条辅助分支交叉熵损失加权求和：aux_loss_weights: [0.48, 0.12, 0.40]
            loss = final_weight * loss_final + aux_weight * weighted_aux_loss
        # 该稀疏损失通过最小化 mention→entity 和 entity→mention 两个文本注意力分布的熵，迫使模型将匹配证据集中到少数最关键的 token 上，从而减少无关文本的干扰并增强可解释性。
        sparse_loss = None
        if self.sparse_weight > 0.0 and evidence is not None:
            alpha_m2e = evidence.get('alpha_m2e')   #  [B, N, L_e]：mention 在阅读某个 candidate entity 的文本时，应该关注 entity 的哪些 token。
            alpha_e2m = evidence.get('alpha_e2m')   #  [B, N, L_m]：entity 在阅读某个 mention 的文本时，应该关注 mention 的哪些 token。
            if alpha_m2e is not None and alpha_e2m is not None:
                h_m2e = self._attention_entropy(alpha_m2e, self.sparse_eps) # [B, N]：第 b 个 mention 在读取第 n 个 entity 文本时，注意力分布有多分散。
                h_e2m = self._attention_entropy(alpha_e2m, self.sparse_eps) # [B, N]：第 b 个 entity 在读取第 n 个 mention 文本时，注意力分布有多分散。
                sparse_loss = h_m2e.mean() + h_e2m.mean()                   # 组合双向注意力熵，得到稀疏损失；鼓励两种方向的 attention 集中
                loss = loss + self.sparse_weight * sparse_loss              # 将稀疏损失加入总损失
                self._log_attention_diagnostics('train', 'alpha_m2e', alpha_m2e, batch_size)
                self._log_attention_diagnostics('train', 'alpha_e2m', alpha_e2m, batch_size)
                self.log('train/alpha_m2e_entropy_mean', h_m2e.detach().mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/alpha_e2m_entropy_mean', h_e2m.detach().mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                h_m2e_norm = self._attention_entropy(alpha_m2e, self.sparse_eps, mask=entity_text_mask, normalize=True)
                mention_text_mask_expand = mention_batch['attention_mask'].unsqueeze(1).expand_as(alpha_e2m)
                h_e2m_norm = self._attention_entropy(alpha_e2m, self.sparse_eps, mask=mention_text_mask_expand, normalize=True)
                self.log('train/alpha_m2e_entropy_norm_mean', h_m2e_norm.detach().mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/alpha_e2m_entropy_norm_mean', h_e2m_norm.detach().mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
    # PASM 不能只作为一个隐式中间层；它输出的 entity semantic evidence 本身也必须具有辨别正确候选的能力。
        pasm_evidence_loss = None
        if self.pasm_evidence_weight > 0.0:
            if evidence is None or evidence.get('pasm_evidence_logits') is None:
                raise ValueError('pasm_evidence_logits is required when pasm_evidence_weight > 0')
            # pasm_evidence_logits[B, N] 即：mention 语义条件与 该 实体 的 PASM 挖掘后实体语义证据之间的相似度
            pasm_evidence_logits = evidence['pasm_evidence_logits'] / self.pasm_temperature    # 除以 temperature 明显拉大 candidate 之间的 logit 差异
            pasm_evidence_loss = self.ce_loss(pasm_evidence_logits, labels)
            loss = loss + self.pasm_evidence_weight * pasm_evidence_loss

        self.log('loss', loss.detach(), on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('train/loss_final', loss_final.detach(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        for branch_name, branch_loss in zip(self.branch_names, aux_losses):
            self.log(f'train/loss_{branch_name}', branch_loss.detach(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        self._log_score_diagnostics('train', 'final', logits.detach(), labels, batch_size)
        for branch_name, branch_logits in zip(self.branch_names, other_logits or []):
            self._log_score_diagnostics('train', branch_name, branch_logits.detach(), labels, batch_size)
        if sparse_loss is not None:
            self.log('train/loss_sparse', sparse_loss.detach(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        if pasm_evidence_loss is not None:
            self.log('train/pasm_evidence_loss', pasm_evidence_loss.detach(), on_epoch=True, sync_dist=True, batch_size=batch_size)
        if evidence is not None:
            visual_gate_m = evidence.get('visual_gate_m')
            visual_gate_e = evidence.get('visual_gate_e')
            if visual_gate_m is not None and visual_gate_e is not None:
                self.log('train/visual_gate_m_mean', visual_gate_m.detach().mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/visual_gate_e_mean', visual_gate_e.detach().mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/visual_gate_m_std', visual_gate_m.detach().std(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/visual_gate_e_std', visual_gate_e.detach().std(), on_epoch=True, sync_dist=True, batch_size=batch_size)
            entity_text_candidate_gate = evidence.get('entity_text_candidate_gate')
            if entity_text_candidate_gate is not None:
                entity_text_candidate_gate = entity_text_candidate_gate.detach()
                gate_mean, gate_std, gate_max_mean = self._masked_gate_stats(entity_text_candidate_gate, entity_text_mask)
                self.log('train/entity_text_candidate_gate_mean', gate_mean, on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/entity_text_candidate_gate_std', gate_std, on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/entity_text_candidate_gate_max_mean', gate_max_mean, on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/entity_text_selector_alpha', self.matcher.entity_text_selector.alpha.detach(), on_epoch=True, sync_dist=True, batch_size=batch_size)
            pasm_semantic_gate = evidence.get('pasm_semantic_gate')
            if pasm_semantic_gate is not None:
                pasm_semantic_gate = pasm_semantic_gate.detach()
                self.log('train/pasm_gate_mean', pasm_semantic_gate.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/pasm_gate_std', pasm_semantic_gate.std(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/pasm_alpha', self.matcher.pasm_semantic_miner.alpha.detach(), on_epoch=True, sync_dist=True, batch_size=batch_size)
            pasm_evidence_logits = evidence.get('pasm_evidence_logits')
            if pasm_evidence_logits is not None:
                with torch.no_grad():
                    pasm_evidence_logits = pasm_evidence_logits.detach()
                    pasm_pos = pasm_evidence_logits[:, 0]
                    pasm_negmax = pasm_evidence_logits[:, 1:].max(dim=1).values
                    pasm_margin = pasm_pos - pasm_negmax
                self.log('train/pasm_evidence_pos_mean', pasm_pos.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/pasm_evidence_negmax_mean', pasm_negmax.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/pasm_evidence_margin_mean', pasm_margin.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/pasm_evidence_acc', (pasm_evidence_logits.argmax(dim=1) == labels).float().mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
            reliability_alpha = evidence.get('reliability_alpha')
            if reliability_alpha is not None:
                reliability_alpha = reliability_alpha.detach()
                self.log('train/reliability_sem_mean', reliability_alpha[..., 0].mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_joint_mean', reliability_alpha[..., 1].mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_global_mean', reliability_alpha[..., 2].mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_sem_std', reliability_alpha[..., 0].std(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_joint_std', reliability_alpha[..., 1].std(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_global_std', reliability_alpha[..., 2].std(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                reliability_entropy = -(reliability_alpha * torch.log(reliability_alpha.clamp(min=self.sparse_eps))).sum(dim=-1)
                self.log('train/reliability_entropy_mean', reliability_entropy.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_sem_min', reliability_alpha[..., 0].min(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_sem_max', reliability_alpha[..., 0].max(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_joint_min', reliability_alpha[..., 1].min(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_joint_max', reliability_alpha[..., 1].max(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_global_min', reliability_alpha[..., 2].min(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_global_max', reliability_alpha[..., 2].max(), on_epoch=True, sync_dist=True, batch_size=batch_size)
            reliability_gate = evidence.get('reliability_gate')
            if reliability_gate is not None:
                reliability_gate = reliability_gate.detach()
                self.log('train/reliability_gate_mean', reliability_gate.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_gate_std', reliability_gate.std(), on_epoch=True, sync_dist=True, batch_size=batch_size)
            reliability_delta = evidence.get('reliability_delta')
            if reliability_delta is not None:
                reliability_delta = reliability_delta.detach()
                self.log('train/reliability_delta_abs_mean', reliability_delta.abs().mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log('train/reliability_delta_std', reliability_delta.std(), on_epoch=True, sync_dist=True, batch_size=batch_size)

        if other_logits is not None:
            for branch_name, branch_logit in zip(self.branch_names, other_logits):
                with torch.no_grad():
                    branch_pos, branch_negmax, branch_margin = self._score_margin(branch_logit)
                self.log(f'train/{branch_name}_pos_mean', branch_pos.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log(f'train/{branch_name}_negmax_mean', branch_negmax.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log(f'train/{branch_name}_margin_mean', branch_margin.mean(), on_epoch=True, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        mention_idx = batch['mention_idx']
        mention_input_dict = batch['mention_input_dict']
        entity_input_dict = batch['entity_input_dict']

        mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
            self.encoder(**mention_input_dict)
        entity_text_embeds, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
            self.encoder(**entity_input_dict)

        batch_size = mention_text_embeds.shape[0]
        total_entity = entity_text_embeds.shape[0]
        entity_text_embeds = entity_text_embeds.reshape(batch_size, total_entity // batch_size, -1)
        entity_image_embeds = entity_image_embeds.reshape(batch_size, total_entity // batch_size, -1)
        length, dim = entity_text_seq_tokens.shape[-2:]
        entity_text_seq_tokens = entity_text_seq_tokens.reshape(batch_size, total_entity // batch_size, length, dim)
        entity_text_mask = entity_input_dict['attention_mask'].reshape(batch_size, total_entity // batch_size, -1)
        length, dim = mention_image_patch_tokens.shape[-2:]
        entity_image_patch_tokens = entity_image_patch_tokens.reshape(batch_size, total_entity // batch_size, length,
                                                                      dim)

        score, branch_scores, _ = self.matcher(entity_text_embeds, entity_text_seq_tokens,
                                               mention_text_embeds, mention_text_seq_tokens,
                                               entity_image_embeds, entity_image_patch_tokens,
                                               mention_image_embeds, mention_image_patch_tokens,
                                               mention_input_dict['attention_mask'], entity_text_mask)

        rank = self._score_rank(score)
        tgt_rank = rank[torch.arange(score.shape[0]), 0]
        eval_record = dict(rank=tgt_rank.cpu(), all_rank=rank.cpu(), idx=mention_idx.cpu())
        if branch_scores is not None:
            for branch_name, branch_score in zip(self.branch_names, branch_scores):
                branch_rank = self._score_rank(branch_score)
                eval_record[f'{branch_name}_rank'] = branch_rank[torch.arange(branch_score.shape[0]), 0].cpu()
                eval_record[f'{branch_name}_all_rank'] = branch_rank.cpu()
        self.eval_info.append(eval_record)

    def on_validation_epoch_end(self):
        all_eval_info = self.all_gather(self.eval_info)
        save_folder = os.path.join('./rank_save', self.args.task, self.args.run_name + f'-{self.time}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        all_rank = []
        mention_idx = []
        for _ in all_eval_info:
            batch_all_rank = _['all_rank'].cpu()
            batch_mention_idx = _['idx'].cpu()
            all_rank.append(batch_all_rank.reshape(-1, batch_all_rank.shape[-1]))
            mention_idx.append(batch_mention_idx.reshape(-1))
        all_rank = torch.concat(all_rank, dim=0).numpy()
        mention_idx = torch.concat(mention_idx, dim=0).numpy()

        ranks = torch.concat([_['rank'].cpu().flatten() for _ in all_eval_info]).numpy()
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits2 = (ranks <= 2).mean()
        hits1 = (ranks <= 1).mean()

        branch_all_ranks = {}
        branch_rank_metrics = {}
        for branch_name in self.branch_names:
            rank_key = f'{branch_name}_rank'
            all_rank_key = f'{branch_name}_all_rank'
            if all(rank_key in _ and all_rank_key in _ for _ in all_eval_info):
                branch_ranks = torch.concat([_[rank_key].cpu().flatten() for _ in all_eval_info]).numpy()
                branch_all_rank = torch.concat([_[all_rank_key].cpu().reshape(-1, _[all_rank_key].shape[-1]) for _ in all_eval_info], dim=0).numpy()
                branch_all_ranks[branch_name] = branch_all_rank
                branch_rank_metrics[branch_name] = branch_ranks
                self.log(f"Val/{branch_name}_hits5", (branch_ranks <= 5).mean(), sync_dist=True, rank_zero_only=True)
                self.log(f"Val/{branch_name}_hits3", (branch_ranks <= 3).mean(), sync_dist=True, rank_zero_only=True)
                self.log(f"Val/{branch_name}_hits1", (branch_ranks <= 1).mean(), sync_dist=True, rank_zero_only=True)
                self.log(f"Val/{branch_name}_mr", branch_ranks.mean(), sync_dist=True, rank_zero_only=True)
                self.log(f"Val/{branch_name}_mrr", (1. / branch_ranks).mean(), sync_dist=True, rank_zero_only=True)

        self.log("Val/hits20", hits20, sync_dist=True, rank_zero_only=True)
        self.log("Val/hits10", hits10, sync_dist=True, rank_zero_only=True)
        self.log("Val/hits5", hits5, sync_dist=True, rank_zero_only=True)
        self.log("Val/hits3", hits3, sync_dist=True, rank_zero_only=True)
        self.log("Val/hits2", hits2, sync_dist=True, rank_zero_only=True)
        self.log("Val/hits1", hits1, sync_dist=True, rank_zero_only=True)
        self.log("Val/mr", ranks.mean(), sync_dist=True, rank_zero_only=True)
        self.log("Val/mrr", (1. / ranks).mean(), sync_dist=True, rank_zero_only=True)
        self.eval_info.clear()

        if self.trainer.is_global_zero:
            print('Saving {}'.format(os.path.join(save_folder, 'valid_all_rank.npy')))
            print('Saving {}'.format(os.path.join(save_folder, 'valid_mention_idx.npy')))
            np.save(os.path.join(save_folder, 'valid_all_rank.npy'), all_rank)
            np.save(os.path.join(save_folder, 'valid_mention_idx.npy'), mention_idx)
            for branch_name, branch_all_rank in branch_all_ranks.items():
                branch_path = os.path.join(save_folder, f'valid_{branch_name}_all_rank.npy')
                print('Saving {}'.format(branch_path))
                np.save(branch_path, branch_all_rank)
                np.save(os.path.join(save_folder, f'valid_{branch_name}_rank.npy'), branch_rank_metrics[branch_name])

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        mention_idx = batch['mention_idx']
        mention_input_dict = batch['mention_input_dict']
        entity_input_dict = batch['entity_input_dict']

        mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
            self.encoder(**mention_input_dict)
        entity_text_embeds, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
            self.encoder(**entity_input_dict)

        batch_size = mention_text_embeds.shape[0]
        total_entity = entity_text_embeds.shape[0]
        entity_text_embeds = entity_text_embeds.reshape(batch_size, total_entity // batch_size, -1)
        entity_image_embeds = entity_image_embeds.reshape(batch_size, total_entity // batch_size, -1)
        length, dim = entity_text_seq_tokens.shape[-2:]
        entity_text_seq_tokens = entity_text_seq_tokens.reshape(batch_size, total_entity // batch_size, length, dim)
        entity_text_mask = entity_input_dict['attention_mask'].reshape(batch_size, total_entity // batch_size, -1)
        length, dim = mention_image_patch_tokens.shape[-2:]
        entity_image_patch_tokens = entity_image_patch_tokens.reshape(batch_size, total_entity // batch_size, length,
                                                                      dim)

        score, branch_scores, _ = self.matcher(entity_text_embeds, entity_text_seq_tokens,
                                               mention_text_embeds, mention_text_seq_tokens,
                                               entity_image_embeds, entity_image_patch_tokens,
                                               mention_image_embeds, mention_image_patch_tokens,
                                               mention_input_dict['attention_mask'], entity_text_mask)

        rank = self._score_rank(score)
        tgt_rank = rank[torch.arange(score.shape[0]), 0]
        eval_record = dict(rank=tgt_rank.cpu(), all_rank=rank.cpu(), idx=mention_idx.cpu())
        if branch_scores is not None:
            for branch_name, branch_score in zip(self.branch_names, branch_scores):
                branch_rank = self._score_rank(branch_score)
                eval_record[f'{branch_name}_rank'] = branch_rank[torch.arange(branch_score.shape[0]), 0].cpu()
                eval_record[f'{branch_name}_all_rank'] = branch_rank.cpu()
        self.eval_info.append(eval_record)

    def on_test_epoch_end(self):
        all_eval_info = self.all_gather(self.eval_info)
        save_folder = os.path.join('./rank_save', self.args.task, self.args.run_name + f'-{self.time}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        all_rank = []
        mention_idx = []
        for _ in all_eval_info:
            batch_all_rank = _['all_rank'].cpu()
            batch_mention_idx = _['idx'].cpu()
            all_rank.append(batch_all_rank.reshape(-1, batch_all_rank.shape[-1]))
            mention_idx.append(batch_mention_idx.reshape(-1))
        all_rank = torch.concat(all_rank, dim=0).numpy()
        mention_idx = torch.concat(mention_idx, dim=0).numpy()

        ranks = torch.concat([_['rank'].cpu().flatten() for _ in all_eval_info]).numpy()
        branch_all_ranks = {}
        branch_rank_metrics = {}
        for branch_name in self.branch_names:
            rank_key = f'{branch_name}_rank'
            all_rank_key = f'{branch_name}_all_rank'
            if all(rank_key in _ and all_rank_key in _ for _ in all_eval_info):
                branch_ranks = torch.concat([_[rank_key].cpu().flatten() for _ in all_eval_info]).numpy()
                branch_all_rank = torch.concat([_[all_rank_key].cpu().reshape(-1, _[all_rank_key].shape[-1]) for _ in all_eval_info], dim=0).numpy()
                branch_all_ranks[branch_name] = branch_all_rank
                branch_rank_metrics[branch_name] = branch_ranks
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits2 = (ranks <= 2).mean()
        hits1 = (ranks <= 1).mean()

        self.log("Test/hits20", hits20, sync_dist=True, rank_zero_only=True)
        self.log("Test/hits10", hits10, sync_dist=True, rank_zero_only=True)
        self.log("Test/hits5", hits5, sync_dist=True, rank_zero_only=True)
        self.log("Test/hits3", hits3, sync_dist=True, rank_zero_only=True)
        self.log("Test/hits2", hits2, sync_dist=True, rank_zero_only=True)
        self.log("Test/hits1", hits1, sync_dist=True, rank_zero_only=True)
        self.log("Test/mr", ranks.mean(), sync_dist=True, rank_zero_only=True)
        self.log("Test/mrr", (1. / ranks).mean(), sync_dist=True, rank_zero_only=True)
        for branch_name, branch_ranks in branch_rank_metrics.items():
            self.log(f"Test/{branch_name}_hits20", (branch_ranks <= 20).mean(), sync_dist=True, rank_zero_only=True)
            self.log(f"Test/{branch_name}_hits10", (branch_ranks <= 10).mean(), sync_dist=True, rank_zero_only=True)
            self.log(f"Test/{branch_name}_hits5", (branch_ranks <= 5).mean(), sync_dist=True, rank_zero_only=True)
            self.log(f"Test/{branch_name}_hits3", (branch_ranks <= 3).mean(), sync_dist=True, rank_zero_only=True)
            self.log(f"Test/{branch_name}_hits2", (branch_ranks <= 2).mean(), sync_dist=True, rank_zero_only=True)
            self.log(f"Test/{branch_name}_hits1", (branch_ranks <= 1).mean(), sync_dist=True, rank_zero_only=True)
            self.log(f"Test/{branch_name}_mr", branch_ranks.mean(), sync_dist=True, rank_zero_only=True)
            self.log(f"Test/{branch_name}_mrr", (1. / branch_ranks).mean(), sync_dist=True, rank_zero_only=True)
        self.eval_info.clear()

        if self.trainer.is_global_zero:
            print('Saving {}'.format(os.path.join(save_folder, 'test_all_rank.npy')))
            print('Saving {}'.format(os.path.join(save_folder, 'test_mention_idx.npy')))
            np.save(os.path.join(save_folder, 'test_all_rank.npy'), all_rank)
            np.save(os.path.join(save_folder, 'test_mention_idx.npy'), mention_idx)
            for branch_name, branch_all_rank in branch_all_ranks.items():
                branch_path = os.path.join(save_folder, f'test_{branch_name}_all_rank.npy')
                print('Saving {}'.format(branch_path))
                np.save(branch_path, branch_all_rank)
                np.save(os.path.join(save_folder, f'test_{branch_name}_rank.npy'), branch_rank_metrics[branch_name])

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches

        # Linear LR Scaling: scale learning rate with batch size
        # Reference: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (Goyal et al., 2017)
        reference_batch_size = 16  # baseline batch size
        lr_scale = self.args.data.batch_size / reference_batch_size

        # Support both old format (learning_rate) and new format (optimization.learning_rate)
        base_lr = getattr(self.args, 'learning_rate', None)
        if base_lr is None:
            optimization_args = getattr(self.args, 'optimization', None)
            if optimization_args is not None:
                base_lr = getattr(optimization_args, 'learning_rate', 3e-5)
            else:
                base_lr = 3e-5

        scaled_lr = base_lr * lr_scale * torch.cuda.device_count()

        optimizer = torch.optim.AdamW(self.parameters(), lr=scaled_lr, betas=(0.9, 0.999))

        # Scale warmup steps proportionally (5% of total steps)
        warmup_steps = int(total_steps * 0.05)

        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def setup(self, stage: str):
        data_args = self.args.data.get(self.args.task)
        self.data_args = data_args

        self.data_args.entity = os.path.join(self.data_args.folder, self.data_args.entity)
        self.data_args.train_data = os.path.join(self.data_args.folder, self.data_args.train_data)
        self.data_args.valid_data = os.path.join(self.data_args.folder, self.data_args.valid_data)
        self.data_args.test_data = os.path.join(self.data_args.folder, self.data_args.test_data)
        self.data_args.image_h5 = os.path.join(self.data_args.folder, self.data_args.image_h5)

        if not hasattr(self, 'entity'):
            self.entity = load_jsonl_file(self.data_args.entity, desc='Entity', key='qid')
            self.qid2id = {d['qid']: d['id'] for _, d in self.entity.items()}
            image_h5py_file = h5py.File(data_args.image_h5, 'r')
            self.entity_image_ds = image_h5py_file['entity_image']
            self.mention_image_ds = image_h5py_file['mention_image']

        if stage == 'fit':
            if not hasattr(self, 'train_data') and not hasattr(self, 'valid_data'):
                self.train_data = load_jsonl_file(self.data_args.train_data, desc='Train')
                if getattr(self.args.data, 'percentage', 1) < 1:
                    self.train_data = self.train_data[:int(len(self.train_data) * getattr(self.args.data, 'percentage', 1))]

                for idx in range(len(self.train_data)):
                    self.train_data[idx].update({'idx': idx})
                self.valid_data = load_jsonl_file(self.data_args.valid_data, desc='Valid')
                for idx in range(len(self.valid_data)):
                    self.valid_data[idx].update({'idx': idx})

        elif stage == 'test':
            if not hasattr(self, 'test_data'):
                self.test_data = load_jsonl_file(self.data_args.test_data, desc='Test')
                for idx in range(len(self.test_data)):
                    self.test_data[idx].update({'idx': idx})

    def load_image_from_h5py(self, key_list, is_mention):
        if is_mention:
            image_bytes = [self.mention_image_ds.get(key, None) for key in key_list]
        else:
            image_bytes = [self.entity_image_ds.get(key, None) for key in key_list]
        image_objs = [Image.open(io.BytesIO(np.array(_))) if _ is not None else Image.new('RGB', (224, 224), 'white') for _ in image_bytes]
        pixel_values = torch.stack([self.clip_processor(_) for _ in image_objs])
        return pixel_values

    def select_candidates(self, candidate_list):
        gt_qid = candidate_list[0]
        random_candidates = random.sample(candidate_list[1:], self.args.data.num_train_candidate)
        return [gt_qid] + random_candidates

    def train_collator(self, batch, is_eval):
        mention_text = [_['mentions'] + '. ' + _['sentence'] for _ in batch]
        mention_idx = torch.tensor([_['idx'] for _ in batch], dtype=torch.int)
        mention_image_file = [_['imgPath'].split('/')[-1].split('.')[0] for _ in batch]
        entity_candidates_qid = sum([_['candidates'] if is_eval else self.select_candidates(_['candidates']) for _ in batch], [])
        entity_candidates_dict = [self.entity[qid] for qid in entity_candidates_qid]
        entity_candidates_text = [d['name'] + '. ' + d.get('desc', '') for d in entity_candidates_dict]

        mention_input_dict = self.tokenizer(mention_text, truncation=True, padding=True,
                                            max_length=self.args.data.text_max_length, return_tensors='pt')

        mention_pixel_values = self.load_image_from_h5py(mention_image_file, is_mention=True)
        mention_input_dict['pixel_values'] = mention_pixel_values

        entity_input_dict = self.tokenizer(entity_candidates_text, truncation=True, padding=True,
                                           max_length=self.args.data.text_max_length, return_tensors='pt')
        entity_pixel_values = self.load_image_from_h5py(entity_candidates_qid, is_mention=False)
        entity_input_dict['pixel_values'] = entity_pixel_values

        return {
            'mention_idx': mention_idx,
            'mention_input_dict': mention_input_dict,
            'entity_input_dict': entity_input_dict,
        }

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.args.data.batch_size,
                          num_workers=self.args.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          collate_fn=lambda x: self.train_collator(x, is_eval=False))

    def val_dataloader(self):
        return DataLoader(self.valid_data,
                          batch_size=self.args.data.eval_batch_size,
                          num_workers=self.args.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=lambda x: self.train_collator(x, is_eval=True))

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.args.data.eval_batch_size,
                          num_workers=self.args.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          collate_fn=lambda x: self.train_collator(x, is_eval=True))
