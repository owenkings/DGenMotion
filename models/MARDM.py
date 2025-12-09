import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CLIP'))
import clip
import math
from functools import partial
from typing import List, Optional, Tuple
from timm.models.vision_transformer import Mlp
from models.DiffMLPs import DiffMLPs_models
from utils.eval_utils import eval_decorator
from utils.train_utils import lengths_to_mask, uniform, get_mask_subset_prob, cosine_schedule


#################################################################################
#                           FSQ Grid Snapping (推理时量化)                        #
#################################################################################
def snap_to_fsq_grid(z: torch.Tensor, half_levels: torch.Tensor) -> torch.Tensor:
    """
    将 Diffusion 的连续输出吸附到最近的 FSQ 网格点。
    
    这是 FSQ-MARDM 的关键后处理步骤！
    
    原理：
    - Diffusion 模型预测的是连续值（如 0.51, -0.82）
    - FSQ 网格点是离散的（如 [-1, -0.5, 0, 0.5, 1]）
    - 必须将连续输出"吸附"到最近的网格点，才能让 Decoder 看到"熟悉"的信号
    
    Args:
        z: Diffusion 的预测输出 [B, T, D]，理论上在 [-1, 1] 范围
        half_levels: FSQ 的半级参数 (L-1)/2, shape [D]
                     例如 level=5 时，half_level=2
                     
    Returns:
        z_snapped: 吸附到网格后的值 [B, T, D]，保证在 FSQ 的离散网格上
        
    Example:
        >>> half_levels = torch.tensor([3.5, 3.5, 3.5, 2.0, 2.0, 2.0])  # FSQ_AE_High
        >>> z = torch.tensor([0.51, -0.82, 0.123])
        >>> z_snapped = snap_to_fsq_grid(z, half_levels[:3])
        >>> # z_snapped ≈ [0.571, -0.857, 0.143] (最近网格点)
    """
    # 确保 half_levels 在正确的设备上
    if z.device != half_levels.device:
        half_levels = half_levels.to(z.device)
    
    # 1. 缩放到网格索引空间 (Scale to Grid Index Space)
    #    例如：0.51 * 2.0 = 1.02
    z_scaled = z * half_levels
    
    # 2. 量化到最近整数 (Round to Nearest Integer)
    #    例如：round(1.02) = 1
    z_quantized = torch.round(z_scaled)
    
    # 3. 反缩放回 [-1, 1] (Scale back to normalized range)
    #    例如：1 / 2.0 = 0.5
    z_snapped = z_quantized / half_levels
    
    # 4. 截断 (Clamp) - 防止 CFG 导致的数值溢出
    #    Classifier-Free Guidance 可能会预测出超出范围的值（如 1.2）
    #    必须拉回合法范围
    return z_snapped.clamp(-1, 1)


#################################################################################
#                                      MARDM                                    #
#################################################################################
class MARDM(nn.Module):
    def __init__(self, ae_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.2, clip_dim=512,
                 diffmlps_batch_mul=4, diffmlps_model='SiT-XL', cond_drop_prob=0.1,
                 clip_version='ViT-B/32', **kargs):
        super(MARDM, self).__init__()

        self.ae_dim = ae_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
            self.num_actions = kargs.get('num_actions', 1)
            self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
        # --------------------------------------------------------------------------
        # MAR Tranformer
        print('Loading MARTransformer...')
        self.input_process = InputProcess(self.ae_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        self.MARTransformer = nn.ModuleList([
            MARTransBlock(self.latent_dim, num_heads, mlp_size=ff_size, drop_out=self.dropout) for _ in range(num_layers)
        ])

        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.mask_latent = nn.Parameter(torch.zeros(1, 1, self.ae_dim))

        self.apply(self.__init_weights)
        for block in self.MARTransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        # --------------------------------------------------------------------------
        # DiffMLPs
        print('Loading DiffMLPs...')
        self.DiffMLPs = DiffMLPs_models[diffmlps_model](target_channels=self.ae_dim, z_channels=self.latent_dim)
        self.diffmlps_batch_mul = diffmlps_batch_mul

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def load_and_freeze_clip(self, clip_version):
        import importlib
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'CLIP'))
        clip_local = importlib.import_module('clip')
        clip_model, clip_preprocess = clip_local.load(clip_version, device='cpu', jit=False)
        assert torch.cuda.is_available()
        clip_local.model.convert_weights(clip_model)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model
    
    def _get_clip_tokenize(self):
        """动态获取 clip tokenize 函数，避免 pickle 问题"""
        import importlib
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'CLIP'))
        clip_local = importlib.import_module('clip')
        return clip_local.tokenize

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        tokenize = self._get_clip_tokenize()
        text = tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, latents, cond, padding_mask, force_mask=False, mask=None):
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.input_process(latents)
        cond = self.cond_emb(cond)
        x = self.position_enc(x)
        x = x.permute(1, 0, 2)
        if mask is not None: # hard pseudo reorder, in practice, after pe, for transformer encoder architecture should be near same without hard because of bidirectional attention.
            sort_indices = torch.argsort(mask.to(torch.float), dim=1)
            x = torch.gather(x, dim=1, index=sort_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            inverse_indices = torch.argsort(sort_indices, dim=1)
            padding_mask = torch.gather(padding_mask, dim=1, index=sort_indices)

        for block in self.MARTransformer:
            x = block(x, cond, padding_mask)
        if mask is not None:
            x = torch.gather(x, dim=1, index=inverse_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        return x

    def forward_loss(self, latents, y, m_lens):
        latents = latents.permute(0, 2, 1)
        b, l, d = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1), latents, torch.zeros_like(latents))

        target = latents.clone().detach()
        input = latents.clone()

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        rand_time = uniform((b,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_masked = (l * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((b, l), device=device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)
        mask &= non_pad_mask
        mask_rlatents = get_mask_subset_prob(mask, 0.1)
        rand_latents = torch.randn_like(input)
        input = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input)
        mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
        input = torch.where(mask_mlatents.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), input)

        z = self.forward(input, cond_vector, ~non_pad_mask, force_mask)
        target = target.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        z = z.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)
        mask = mask.reshape(b * l).repeat(self.diffmlps_batch_mul)
        target = target[mask]
        z = z[mask]
        loss = self.DiffMLPs(z=z, target=target)

        return loss

    def forward_with_CFG(self, latents, cond_vector, padding_mask, cfg=3, mask=None, force_mask=False, hard_pseudo_reorder=False):
        if hard_pseudo_reorder:
            reorder_mask = mask.clone()
        else:
            reorder_mask = None
        if force_mask:
            return self.forward(latents, cond_vector, padding_mask, force_mask=True, mask=reorder_mask)

        logits = self.forward(latents, cond_vector, padding_mask, mask=reorder_mask)
        if cfg != 1:
            aux_logits = self.forward(latents, cond_vector, padding_mask, force_mask=True, mask=reorder_mask)
            mixed_logits = torch.cat([logits, aux_logits], dim=0)
        else:
            mixed_logits = logits
        b, l, d = mixed_logits.size()
        if mask is not None:
            mask2 = torch.cat([mask, mask], dim=0).reshape(b * l)
            mixed_logits = (mixed_logits.reshape(b * l, d))[mask2]
        else:
            mixed_logits = mixed_logits.reshape(b * l, d)
        output = self.DiffMLPs.sample(mixed_logits, 1, cfg)
        if cfg != 1:
            scaled_logits, _ = output.chunk(2, dim=0)
        else:
            scaled_logits = output
        if mask is not None:
            latents = latents.reshape(b//2 * l, self.ae_dim)
            latents[mask.reshape(b//2 * l)] = scaled_logits
            scaled_logits = latents.reshape(b//2, l, self.ae_dim)

        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 force_mask=False,
                 hard_pseudo_reorder=False
                 ):
        device = next(self.parameters()).device
        l = max(m_lens)
        b = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, l)

        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(b, l, self.ae_dim).to(device),
                          self.mask_latent.repeat(b, l, 1))
        masked_rand_schedule = torch.where(padding_mask, 1e5, torch.rand_like(padding_mask, dtype=torch.float))

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), latents)
            logits = self.forward_with_CFG(latents, cond_vector=cond_vector, padding_mask=padding_mask,
                                                  cfg=cond_scale, mask=is_mask, force_mask=force_mask, hard_pseudo_reorder=hard_pseudo_reorder)
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)

            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        return latents.permute(0,2,1)

    @torch.no_grad()
    @eval_decorator
    def edit(self,
             conds,
             latents,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             force_mask=False,
             edit_mask=None,
             padding_mask=None,
             hard_pseudo_reorder=False,
             ):

        device = next(self.parameters()).device
        l = latents.shape[-1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask == None:
            padding_mask = ~lengths_to_mask(m_lens, l)

        if edit_mask == None:
            mask_free = True
            latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(latents.shape[0], l, self.ae_dim).to(device),
                                  latents.permute(0, 2, 1))
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(latents.shape[0], l, self.ae_dim).to(device),
                              latents.permute(0, 2, 1))
            latents = torch.where(edit_mask.unsqueeze(-1),
                              self.mask_latent.repeat(latents.shape[0], l, 1), latents)
            masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5)

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = 0.16 if mask_free else cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(
                dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(latents.shape[0], latents.shape[1], 1), latents)
            logits = self.forward_with_CFG(latents, cond_vector=cond_vector, padding_mask=padding_mask,
                                                  cfg=cond_scale, mask=is_mask, force_mask=force_mask, hard_pseudo_reorder=hard_pseudo_reorder)
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)

            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        latents = torch.where(edit_mask.unsqueeze(-1), latents, latents)
        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        return latents.permute(0,2,1)

#################################################################################
#                                   FSQ-MARDM                                   #
#################################################################################
class FSQ_MARDM(nn.Module):
    """
    FSQ-MARDM: 结合 FSQ 的 Masked Autoregressive Diffusion Model
    
    核心改动：
    - Diffusion 的预测目标从 512-dim 连续向量变为 d-dim FSQ 量化坐标
    - 这是 "Denoising to Grid" 的核心思想
    - FSQ 坐标保持拓扑结构，便于 Diffusion 学习
    
    Args:
        ae_dim: AE 输出维度 (512)，用于 Transformer 输入
        fsq_dim: FSQ 维度 (d=5 或 6)，用于 DiffMLP 预测目标
        其他参数与 MARDM 相同
    """
    
    def __init__(self, ae_dim, fsq_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.2, clip_dim=512,
                 diffmlps_batch_mul=4, diffmlps_model='SiT-XL', cond_drop_prob=0.1,
                 clip_version='ViT-B/32', **kargs):
        super(FSQ_MARDM, self).__init__()

        self.ae_dim = ae_dim
        self.fsq_dim = fsq_dim  # FSQ 量化维度
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
            self.num_actions = kargs.get('num_actions', 1)
            self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
        
        # --------------------------------------------------------------------------
        # MAR Transformer (输入仍然是 ae_dim=512)
        print('Loading FSQ-MARTransformer...')
        self.input_process = InputProcess(self.ae_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        self.MARTransformer = nn.ModuleList([
            MARTransBlock(self.latent_dim, num_heads, mlp_size=ff_size, drop_out=self.dropout) for _ in range(num_layers)
        ])

        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.mask_latent = nn.Parameter(torch.zeros(1, 1, self.ae_dim))

        self.apply(self.__init_weights)
        for block in self.MARTransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        # --------------------------------------------------------------------------
        # DiffMLPs (预测目标改为 fsq_dim)
        print(f'Loading DiffMLPs for FSQ (target_dim={fsq_dim})...')
        self.DiffMLPs = DiffMLPs_models[diffmlps_model](target_channels=self.fsq_dim, z_channels=self.latent_dim)
        self.diffmlps_batch_mul = diffmlps_batch_mul

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def load_and_freeze_clip(self, clip_version):
        import importlib
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'CLIP'))
        clip_local = importlib.import_module('clip')
        clip_model, clip_preprocess = clip_local.load(clip_version, device='cpu', jit=False)
        assert torch.cuda.is_available()
        clip_local.model.convert_weights(clip_model)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model
    
    def _get_clip_tokenize(self):
        """动态获取 clip tokenize 函数，避免 pickle 问题"""
        import importlib
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'CLIP'))
        clip_local = importlib.import_module('clip')
        return clip_local.tokenize

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        tokenize = self._get_clip_tokenize()
        text = tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, latents, cond, padding_mask, force_mask=False, mask=None):
        """
        Transformer forward pass
        
        Args:
            latents: AE 编码的 512-dim 表示 [B, T, 512]
            cond: 条件向量
            padding_mask: 填充掩码
            
        Returns:
            z: Transformer 输出 [B, T, latent_dim]
        """
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.input_process(latents)
        cond = self.cond_emb(cond)
        x = self.position_enc(x)
        x = x.permute(1, 0, 2)
        if mask is not None:
            sort_indices = torch.argsort(mask.to(torch.float), dim=1)
            x = torch.gather(x, dim=1, index=sort_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            inverse_indices = torch.argsort(sort_indices, dim=1)
            padding_mask = torch.gather(padding_mask, dim=1, index=sort_indices)

        for block in self.MARTransformer:
            x = block(x, cond, padding_mask)
        if mask is not None:
            x = torch.gather(x, dim=1, index=inverse_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        return x

    def forward_loss(self, latents, fsq_targets, y, m_lens):
        """
        计算训练损失
        
        Args:
            latents: AE 编码的 512-dim 表示 [B, 512, T]
            fsq_targets: FSQ 量化坐标 [B, T, d]，Diffusion 的预测目标
            y: 文本条件
            m_lens: 序列长度
            
        Returns:
            loss: Diffusion 损失
        """
        latents = latents.permute(0, 2, 1)  # [B, T, 512]
        b, l, _ = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1), latents, torch.zeros_like(latents))
        fsq_targets = torch.where(non_pad_mask.unsqueeze(-1), fsq_targets, torch.zeros_like(fsq_targets))

        target = fsq_targets.clone().detach()  # Diffusion 目标是 FSQ 坐标
        input = latents.clone()

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        rand_time = uniform((b,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_masked = (l * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((b, l), device=device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)
        mask &= non_pad_mask
        mask_rlatents = get_mask_subset_prob(mask, 0.1)
        rand_latents = torch.randn_like(input)
        input = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input)
        mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
        input = torch.where(mask_mlatents.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), input)

        z = self.forward(input, cond_vector, ~non_pad_mask, force_mask)
        
        # DiffMLPs 预测 FSQ 坐标
        target = target.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)  # [B*L*mul, d]
        z = z.reshape(b * l, -1).repeat(self.diffmlps_batch_mul, 1)  # [B*L*mul, latent_dim]
        mask = mask.reshape(b * l).repeat(self.diffmlps_batch_mul)
        target = target[mask]
        z = z[mask]
        loss = self.DiffMLPs(z=z, target=target)

        return loss

    def forward_with_CFG(self, latents, cond_vector, padding_mask, cfg=3, mask=None, force_mask=False, hard_pseudo_reorder=False):
        """带 Classifier-Free Guidance 的前向"""
        if hard_pseudo_reorder:
            reorder_mask = mask.clone()
        else:
            reorder_mask = None
        if force_mask:
            return self.forward(latents, cond_vector, padding_mask, force_mask=True, mask=reorder_mask)

        logits = self.forward(latents, cond_vector, padding_mask, mask=reorder_mask)
        if cfg != 1:
            aux_logits = self.forward(latents, cond_vector, padding_mask, force_mask=True, mask=reorder_mask)
            mixed_logits = torch.cat([logits, aux_logits], dim=0)
        else:
            mixed_logits = logits
        b, l, d = mixed_logits.size()
        if mask is not None:
            mask2 = torch.cat([mask, mask], dim=0).reshape(b * l)
            mixed_logits = (mixed_logits.reshape(b * l, d))[mask2]
        else:
            mixed_logits = mixed_logits.reshape(b * l, d)
        
        # DiffMLP 输出 FSQ 坐标
        output = self.DiffMLPs.sample(mixed_logits, 1, cfg)
        
        if cfg != 1:
            scaled_logits, _ = output.chunk(2, dim=0)
        else:
            scaled_logits = output
        if mask is not None:
            # 创建 FSQ 维度的 latents 用于填充
            fsq_latents = torch.zeros(b//2 * l, self.fsq_dim, device=latents.device)
            fsq_latents[mask.reshape(b//2 * l)] = scaled_logits
            scaled_logits = fsq_latents.reshape(b//2, l, self.fsq_dim)

        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 force_mask=False,
                 hard_pseudo_reorder=False,
                 ae=None  # 需要传入 FSQ_AE 用于解码
                 ):
        """
        生成 FSQ 坐标
        
        Returns:
            fsq_coords: FSQ 量化坐标 [B, T, d]
            如果传入 ae，则返回解码后的动作 [B, T, 67]
        """
        device = next(self.parameters()).device
        l = max(m_lens)
        b = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, l)

        # 初始化为 mask token (ae_dim=512)
        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(b, l, self.ae_dim).to(device),
                          self.mask_latent.repeat(b, l, 1))
        
        # FSQ 坐标初始化
        fsq_coords = torch.zeros(b, l, self.fsq_dim).to(device)
        
        masked_rand_schedule = torch.where(padding_mask, 1e5, torch.rand_like(padding_mask, dtype=torch.float))

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), latents)
            
            # 生成 FSQ 坐标
            pred_fsq = self.forward_with_CFG(latents, cond_vector=cond_vector, padding_mask=padding_mask,
                                             cfg=cond_scale, mask=is_mask, force_mask=force_mask, 
                                             hard_pseudo_reorder=hard_pseudo_reorder)
            
            fsq_coords = torch.where(is_mask.unsqueeze(-1), pred_fsq, fsq_coords)
            
            # 如果有 AE，将 FSQ 坐标转换回 ae_dim 用于下一步
            if ae is not None:
                # 通过 post_fsq 将 FSQ 坐标转回 512-dim
                latents_update = ae.post_fsq(pred_fsq)  # [B, T, 512]
                latents = torch.where(is_mask.unsqueeze(-1), latents_update, latents)

            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        fsq_coords = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(fsq_coords), fsq_coords)
        
        # ========== ✅ [CRITICAL FIX] FSQ Grid Snapping ==========
        # Diffusion 输出的是连续值，必须吸附到 FSQ 网格点
        # 这一步消除了生成过程中的"网格间隙噪音"，是 FSQ-MARDM 的画龙点睛之笔
        if ae is not None and hasattr(ae, 'fsq'):
            half_levels = ae.fsq._half_levels
            fsq_coords = snap_to_fsq_grid(fsq_coords, half_levels)
        # =========================================================
        
        return fsq_coords  # [B, T, d]

    @torch.no_grad()
    @eval_decorator  
    def edit(self,
             conds,
             latents,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             force_mask=False,
             edit_mask=None,
             padding_mask=None,
             hard_pseudo_reorder=False,
             ae=None
             ):
        """编辑模式（与 generate 类似）"""
        device = next(self.parameters()).device
        l = latents.shape[-1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask is None:
            padding_mask = ~lengths_to_mask(m_lens, l)

        if edit_mask is None:
            mask_free = True
            latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(latents.shape[0], l, self.ae_dim).to(device),
                                  latents.permute(0, 2, 1))
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros(latents.shape[0], l, self.ae_dim).to(device),
                              latents.permute(0, 2, 1))
            latents = torch.where(edit_mask.unsqueeze(-1),
                              self.mask_latent.repeat(latents.shape[0], l, 1), latents)
            masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5)

        fsq_coords = torch.zeros(latents.shape[0], l, self.fsq_dim).to(device)

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            rand_mask_prob = 0.16 if mask_free else cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))

            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(latents.shape[0], latents.shape[1], 1), latents)
            pred_fsq = self.forward_with_CFG(latents, cond_vector=cond_vector, padding_mask=padding_mask,
                                             cfg=cond_scale, mask=is_mask, force_mask=force_mask, 
                                             hard_pseudo_reorder=hard_pseudo_reorder)
            
            fsq_coords = torch.where(is_mask.unsqueeze(-1), pred_fsq, fsq_coords)
            
            if ae is not None:
                latents_update = ae.post_fsq(pred_fsq)
                latents = torch.where(is_mask.unsqueeze(-1), latents_update, latents)

            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)

        fsq_coords = torch.where(edit_mask.unsqueeze(-1), fsq_coords, fsq_coords)
        fsq_coords = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(fsq_coords), fsq_coords)
        
        # ========== ✅ [CRITICAL FIX] FSQ Grid Snapping ==========
        # 与 generate 函数一致，确保编辑模式也输出精确的网格点
        if ae is not None and hasattr(ae, 'fsq'):
            half_levels = ae.fsq._half_levels
            fsq_coords = snap_to_fsq_grid(fsq_coords, half_levels)
        # =========================================================
        
        return fsq_coords


#################################################################################
#                                     MARDM Zoos                                #
#################################################################################
def mardm_ddpm_xl(**kwargs):
    return MARDM(latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16, dropout=0.2, clip_dim=512,
                 diffmlps_model="DDPM-XL", diffmlps_batch_mul=4, cond_drop_prob=0.1, **kwargs)
def mardm_sit_xl(**kwargs):
    return MARDM(latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16, dropout=0.2, clip_dim=512,
                 diffmlps_model="SiT-XL", diffmlps_batch_mul=4, cond_drop_prob=0.1, **kwargs)

# FSQ-MARDM 模型
def fsq_mardm_sit_xl(fsq_dim=5, **kwargs):
    """FSQ-MARDM with SiT-XL DiffMLP"""
    return FSQ_MARDM(latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16, dropout=0.2, clip_dim=512,
                     diffmlps_model="SiT-XL", diffmlps_batch_mul=4, cond_drop_prob=0.1, 
                     fsq_dim=fsq_dim, **kwargs)

def fsq_mardm_ddpm_xl(fsq_dim=5, **kwargs):
    """FSQ-MARDM with DDPM-XL DiffMLP"""
    return FSQ_MARDM(latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16, dropout=0.2, clip_dim=512,
                     diffmlps_model="DDPM-XL", diffmlps_batch_mul=4, cond_drop_prob=0.1,
                     fsq_dim=fsq_dim, **kwargs)

MARDM_models = {
    'MARDM-DDPM-XL': mardm_ddpm_xl, 
    'MARDM-SiT-XL': mardm_sit_xl,
    'FSQ-MARDM-SiT-XL': fsq_mardm_sit_xl,
    'FSQ-MARDM-DDPM-XL': fsq_mardm_ddpm_xl,
}

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def modulate_here(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute((1, 0, 2))
        x = self.poseEmbedding(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.2):
        super().__init__()
        assert embed_dim % 8 == 0
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, mask):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            mask = mask[:, None, None, :]
            att = att.masked_fill(mask != 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y


class MARTransBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_size=1024, drop_out=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, drop_out_rate=drop_out)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = mlp_size
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate_here(self.norm1(x), shift_msa, scale_msa), mask=padding_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate_here(self.norm2(x), shift_mlp, scale_mlp))
        return x
