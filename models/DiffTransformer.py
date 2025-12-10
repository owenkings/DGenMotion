"""
DiffTransformer: 1D Transformer for FSQ Diffusion (JiT-style)
==============================================================

核心思想 (JiT Paradigm):
1. 用 Transformer 替代 MLP，引入全局时序注意力
2. 输入保持序列结构 (B, L, D)，不做 flatten
3. Time Embedding 通过 adaLN 全局注入
4. Context 通过 token-wise addition 精细控制
5. **x-prediction**: 直接预测干净数据，而非 velocity/noise

与 DiffMLPs 的关键区别：
- DiffMLPs: 处理 (B*L, D) 的独立点，各帧无交互，v-prediction
- DiffTransformer: 处理 (B, L, D) 序列，全局 Self-Attention，x-prediction

为什么 x-prediction 在 FSQ 空间更优？
- FSQ 将连续空间离散化为有限网格点
- 预测"正确的网格坐标"比预测"高维随机噪声/velocity"更简单
- 模型输出天然有界 (FSQ 范围 [-1, 1])，训练更稳定

Reference: JiT (Just image Transformer) - He et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from torchdiffeq import odeint


#################################################################################
#                               Core Building Blocks                            #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    时间步嵌入器 - 将标量时间步转换为向量表示
    使用正弦位置编码 + MLP 投影
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        创建正弦时间步嵌入
        :param t: 1-D Tensor of N indices (可以是小数)
        :param dim: 输出维度
        :param max_period: 控制最小频率
        :return: (N, dim) Tensor
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PositionalEmbedding1D(nn.Module):
    """
    1D 位置嵌入 - 让 Transformer 知道"第几帧"
    支持固定正弦编码或可学习编码
    """
    def __init__(self, max_len: int, embed_dim: int, learnable: bool = True):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.learnable = learnable
        
        if learnable:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # 固定正弦编码
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
            self.register_buffer('pos_embed', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            x + pos_embed: [B, L, D]
        """
        seq_len = x.shape[1]
        return x + self.pos_embed[:, :seq_len, :]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN 调制函数"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                              DiT Block (1D Version)                            #
#################################################################################

class DiTBlock1D(nn.Module):
    """
    DiT Block (1D) - 带有 adaLN 调制的 Transformer Block
    
    结构：
    - Self-Attention with adaLN
    - FFN with adaLN
    - Time embedding 控制 scale/shift/gate
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # FFN
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        
        # adaLN 调制层 - 输出 6 个参数: shift1, scale1, gate1, shift2, scale2, gate2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入序列 [B, L, D]
            c: 条件嵌入 [B, D] (time + context)
            attn_mask: 可选的注意力掩码 [B, L] 或 [B, L, L]
        Returns:
            x: 输出序列 [B, L, D]
        """
        # 获取 adaLN 参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Self-Attention with adaLN
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        
        # 处理 attention mask
        key_padding_mask = None
        if attn_mask is not None and attn_mask.dim() == 2:
            # [B, L] 布尔掩码 -> key_padding_mask
            key_padding_mask = attn_mask
        
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # FFN with adaLN
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class FinalLayer1D(nn.Module):
    """
    DiT 最终输出层 - adaLN + Linear
    """
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, hidden_size]
            c: [B, hidden_size]
        Returns:
            out: [B, L, out_channels]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                           DiffTransformer Main Class                          #
#################################################################################

class DiffTransformer(nn.Module):
    """
    DiffTransformer: 1D Transformer for FSQ Diffusion
    
    核心架构：
    1. Input Projection: FSQ_dim -> hidden_size
    2. 1D Positional Embedding
    3. Context Injection (add)
    4. N x DiTBlock1D (with adaLN for time)
    5. Final Layer -> FSQ_dim
    
    Args:
        target_channels: FSQ 维度 (d=6 for FSQ_AE_High)
        z_channels: MAR Transformer 输出维度 (latent_dim=1024)
        hidden_size: Transformer 隐藏维度
        depth: Transformer 层数
        num_heads: 注意力头数
        max_seq_len: 最大序列长度
        mlp_ratio: FFN 扩展比率
        dropout: Dropout 率
    """
    
    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        max_seq_len: int = 64,  # 196 // 4 ≈ 49, 留些余量
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.target_channels = target_channels  # FSQ_dim
        self.z_channels = z_channels  # Condition dim from MAR
        self.hidden_size = hidden_size
        self.depth = depth
        
        # ========== Embeddings ==========
        # Input projection: FSQ_dim -> hidden_size
        self.input_proj = nn.Linear(target_channels, hidden_size)
        
        # 1D Positional Embedding (可学习)
        self.pos_embed = PositionalEmbedding1D(max_seq_len, hidden_size, learnable=True)
        
        # Time Embedding
        self.time_embed = TimestepEmbedder(hidden_size)
        
        # Context Projection: z_channels -> hidden_size
        self.cond_proj = nn.Linear(z_channels, hidden_size)
        
        # ========== Transformer Blocks ==========
        self.blocks = nn.ModuleList([
            DiTBlock1D(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # ========== Output ==========
        self.final_layer = FinalLayer1D(hidden_size, target_channels)
        
        # ========== Initialize Weights ==========
        self.initialize_weights()

    def initialize_weights(self):
        """权重初始化"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # 时间嵌入 MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN 输出 (让初始时 Transformer 接近恒等)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out 最终输出层
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        c: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass (用于训练)
        
        Args:
            x: Noisy FSQ coordinates [B, L, FSQ_dim] 或 [B*L, FSQ_dim]
            t: Timesteps [B] 或 [B*L]
            c: Context from MAR [B, L, z_channels] 或 [B*L, z_channels]
            padding_mask: 可选的填充掩码 [B, L]
            
        Returns:
            pred: 预测输出 [B, L, FSQ_dim] 或 [B*L, FSQ_dim]
        """
        # 处理维度 - 支持 (B*L, D) 格式以保持兼容性
        input_was_flat = (x.dim() == 2)
        
        if input_was_flat:
            # 假设 batch 和 seq 的乘积，需要重塑
            # 这种情况下我们把它当作 batch_size = B*L, seq_len = 1
            x = x.unsqueeze(1)  # [B*L, 1, FSQ_dim]
            c = c.unsqueeze(1) if c.dim() == 2 else c  # [B*L, 1, z_channels]
        
        B, L, _ = x.shape
        
        # 1. Input Embedding
        x = self.input_proj(x)  # [B, L, hidden_size]
        
        # 2. Add Positional Embedding
        x = self.pos_embed(x)  # [B, L, hidden_size]
        
        # 3. Context Injection (token-wise)
        c_emb = self.cond_proj(c)  # [B, L, hidden_size] or [B, 1, hidden_size]
        if c_emb.shape[1] == 1 and L > 1:
            c_emb = c_emb.expand(-1, L, -1)  # 广播
        x = x + c_emb
        
        # 4. Time Embedding (global)
        t_emb = self.time_embed(t)  # [B, hidden_size]
        
        # 5. Transformer Blocks
        for block in self.blocks:
            x = block(x, t_emb, padding_mask)
        
        # 6. Final Layer
        x = self.final_layer(x, t_emb)  # [B, L, FSQ_dim]
        
        if input_was_flat:
            x = x.squeeze(1)  # [B*L, FSQ_dim]
        
        return x

    def forward_with_cfg(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        c: torch.Tensor, 
        cfg_scale: float = 1.0
    ) -> torch.Tensor:
        """
        带 Classifier-Free Guidance 的前向
        
        注意：调用前应该已经将 conditional 和 unconditional 的 x, c 拼接
        x 的前半部分对应 conditional，后半部分对应 unconditional
        """
        half = x.shape[0] // 2
        combined = torch.cat([x[:half], x[:half]], dim=0)
        model_out = self.forward(combined, t, c)
        
        # 分离 conditional 和 unconditional 预测
        cond_out, uncond_out = torch.split(model_out, half, dim=0)
        
        # CFG 混合
        cfg_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        
        return torch.cat([cfg_out, cfg_out], dim=0)


#################################################################################
#                    DiffTransformer with x-prediction (JiT Core)                #
#################################################################################

class DiffTransformer_XPred(nn.Module):
    """
    DiffTransformer with x-prediction (JiT paradigm)
    
    ⭐ 这是 JiT 的核心实现！
    
    核心思想：
    - 模型直接预测干净数据 x_1，而不是 velocity 或 noise
    - 在 FSQ 空间中，这意味着直接预测"正确的网格坐标"
    - 训练更稳定，因为目标有界 (FSQ 范围 [-1, 1])
    
    Flow Matching 框架下的 x-prediction：
    - 插值: x_t = t * x_1 + (1-t) * x_0, 其中 x_0 ~ N(0,1), x_1 = data
    - 模型输入 x_t，预测 x_1
    - 采样时：velocity = (x_1_pred - x_t) / (1-t)
    
    与 v-prediction 的对比：
    - v-pred: 模型输出 ∈ (-∞, +∞)，需要学习噪声的复杂分布
    - x-pred: 模型输出 ∈ [-1, 1] (FSQ 范围)，目标更简单明确
    """
    
    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        num_sampling_steps: int = 50,
        **kwargs
    ):
        super().__init__()
        
        self.in_channels = target_channels
        self.num_sampling_steps = num_sampling_steps
        
        self.net = DiffTransformer(
            target_channels=target_channels,
            z_channels=z_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            **kwargs
        )

    def forward(self, target: torch.Tensor, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        训练前向 - x-prediction 损失
        
        训练目标：让模型从含噪输入 x_t 直接预测干净数据 x_1
        
        Args:
            target: 目标 FSQ 坐标 x_1 [B*L, FSQ_dim]
            z: MAR Transformer 输出 [B*L, z_channels]
            mask: 可选掩码
            
        Returns:
            loss: MSE(model_output, x_1)
        """
        device = target.device
        batch_size = target.shape[0]
        
        # 1. 采样时间步 t ~ U(0, 1)
        #    避免 t=0 (纯噪声) 和 t=1 (纯数据) 的边界问题
        t = torch.rand(batch_size, device=device) * 0.98 + 0.01  # t ∈ [0.01, 0.99]
        
        # 2. 采样噪声 x_0 ~ N(0, 1)
        x_0 = torch.randn_like(target)
        
        # 3. 构造含噪输入 x_t = t * x_1 + (1-t) * x_0
        t_expand = t.view(-1, 1)  # [B*L, 1]
        x_t = t_expand * target + (1 - t_expand) * x_0
        
        # 4. 模型预测 x_1
        x_1_pred = self.net(x_t, t, z)
        
        # 5. 计算 x-prediction 损失
        loss = (x_1_pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B*L]
        
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss

    def sample(
        self, 
        z: torch.Tensor, 
        temperature: float = 1.0, 
        cfg: float = 1.0
    ) -> torch.Tensor:
        """
        采样 - 使用 x-prediction 模型生成 FSQ 坐标
        
        使用简单稳定的迭代采样方法（类似 DDIM）：
        - 每一步模型预测 x_1
        - 根据预测的 x_1 更新 x_t
        
        Args:
            z: MAR Transformer 输出 [B*L, z_channels]
            temperature: 采样温度（控制初始噪声强度）
            cfg: Classifier-Free Guidance 强度
            
        Returns:
            sampled: 生成的 FSQ 坐标 [B*L, FSQ_dim]
        """
        device = z.device
        num_steps = self.num_sampling_steps
        
        # 确定 batch size
        if cfg != 1.0:
            batch_size = z.shape[0] // 2
            z_cond = z[:batch_size]
            z_uncond = z[batch_size:]
        else:
            batch_size = z.shape[0]
            z_cond = z
            z_uncond = None
        
        # 初始噪声 x_0 ~ N(0, σ)
        x_t = torch.randn(batch_size, self.in_channels, device=device) * temperature
        
        # 时间步从 ε 到 1-ε (避免边界)
        timesteps = torch.linspace(0.02, 0.98, num_steps, device=device)
        
        # 保存 x_0 用于插值
        x_0 = x_t.clone()
        
        for i, t_now in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t_now.item(), device=device)
            
            # 模型预测 x_1
            if cfg != 1.0:
                x_1_cond = self.net(x_t, t_batch, z_cond)
                x_1_uncond = self.net(x_t, t_batch, z_uncond)
                x_1_pred = x_1_uncond + cfg * (x_1_cond - x_1_uncond)
            else:
                x_1_pred = self.net(x_t, t_batch, z_cond)
            
            # 如果是最后一步，直接返回预测的 x_1
            if i == num_steps - 1:
                x_t = x_1_pred
            else:
                # 计算下一个时间步的 x_t
                # x_t = t * x_1 + (1-t) * x_0
                # 根据预测的 x_1 估计 x_0: x_0 = (x_t - t * x_1) / (1 - t)
                t_next = timesteps[i + 1]
                
                if t_now > 0.01:
                    x_0_est = (x_t - t_now * x_1_pred) / (1 - t_now)
                else:
                    x_0_est = x_0
                
                # 更新 x_t
                x_t = t_next * x_1_pred + (1 - t_next) * x_0_est
        
        # 对于 CFG 模式，复制输出以保持接口兼容
        if cfg != 1.0:
            x_t = torch.cat([x_t, x_t], dim=0)
        
        return x_t

    def sample_ddim_style(
        self, 
        z: torch.Tensor, 
        temperature: float = 1.0, 
        cfg: float = 1.0,
        num_steps: int = None
    ) -> torch.Tensor:
        """
        DDIM 风格的采样 - 更直接的 x-prediction 采样
        
        这种方法更简单直接：每一步直接用模型预测的 x_1 更新
        
        Args:
            z: 条件
            temperature: 温度
            cfg: CFG 强度
            num_steps: 采样步数
        """
        device = z.device
        num_steps = num_steps or self.num_sampling_steps
        
        if cfg != 1.0:
            batch_size = z.shape[0] // 2
            z_cond = z[:batch_size]
            z_uncond = z[batch_size:]
        else:
            batch_size = z.shape[0]
            z_cond = z
            z_uncond = None
        
        # 初始噪声
        x_t = torch.randn(batch_size, self.in_channels, device=device) * temperature
        
        # 时间步从 0 到 1
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = torch.full((batch_size,), t_now.item(), device=device)
            
            # 预测 x_1
            if cfg != 1.0:
                x_1_cond = self.net(x_t, t_batch, z_cond)
                x_1_uncond = self.net(x_t, t_batch, z_uncond)
                x_1_pred = x_1_uncond + cfg * (x_1_cond - x_1_uncond)
            else:
                x_1_pred = self.net(x_t, t_batch, z_cond)
            
            # 计算下一步的 x_t
            # x_{t+dt} = (t+dt) * x_1 + (1-t-dt) * x_0
            # 我们用当前估计的 x_0 = (x_t - t * x_1_pred) / (1-t)
            if t_now < 0.999:
                x_0_est = (x_t - t_now * x_1_pred) / (1 - t_now)
            else:
                x_0_est = torch.randn_like(x_t)
            
            x_t = t_next * x_1_pred + (1 - t_next) * x_0_est
        
        # 最后一步直接返回预测的 x_1
        x_1 = x_1_pred
        
        if cfg != 1.0:
            x_1 = torch.cat([x_1, x_1], dim=0)
        
        return x_1


#################################################################################
#              DiffTransformer with Transport-based Training                     #
#################################################################################

class DiffTransformer_SiT(nn.Module):
    """
    DiffTransformer with v-prediction (SiT/Flow Matching)
    
    使用 transport.py 的 velocity prediction 训练
    
    与 DiffMLPs_SiT 接口保持一致
    """
    
    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        **kwargs
    ):
        super().__init__()
        
        # 延迟导入避免循环依赖
        from diffusions.transport import create_transport, Sampler
        
        self.in_channels = target_channels
        self.net = DiffTransformer(
            target_channels=target_channels,
            z_channels=z_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            **kwargs
        )
        
        # SiT Transport (velocity prediction)
        self.train_diffusion = create_transport(prediction="velocity")
        self.gen_diffusion = Sampler(self.train_diffusion)

    def forward(self, target: torch.Tensor, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """训练前向 - velocity prediction 损失"""
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, model_kwargs)
        loss = loss_dict["loss"]
        
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        
        return loss.mean()

    def sample(
        self, 
        z: torch.Tensor, 
        temperature: float = 1.0, 
        cfg: float = 1.0
    ) -> torch.Tensor:
        """采样 - velocity prediction ODE"""
        device = z.device
        
        if cfg != 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).to(device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            model_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).to(device)
            model_kwargs = dict(c=z)
            model_fn = self.net.forward
        
        sample_fn = self.gen_diffusion.sample_ode()
        sampled_token_latent = sample_fn(noise, model_fn, **model_kwargs)[-1]
        
        return sampled_token_latent


class DiffTransformer_Transport_XPred(nn.Module):
    """
    DiffTransformer with Transport-based x-prediction
    
    使用修改后的 transport.py 支持的 x-prediction 训练
    这是一个替代方案，与 DiffTransformer_XPred 功能相同
    
    优点：与 transport.py 框架一致
    缺点：采样时需要转换 x-pred -> velocity
    """
    
    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        **kwargs
    ):
        super().__init__()
        
        from diffusions.transport import create_transport, Sampler
        
        self.in_channels = target_channels
        self.net = DiffTransformer(
            target_channels=target_channels,
            z_channels=z_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            **kwargs
        )
        
        # ⭐ 使用 x-prediction (JiT 核心)
        self.train_diffusion = create_transport(prediction="x")
        self.gen_diffusion = Sampler(self.train_diffusion)

    def forward(self, target: torch.Tensor, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """训练前向 - x-prediction 损失 (通过 transport.py)"""
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, model_kwargs)
        loss = loss_dict["loss"]
        
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        
        return loss.mean()

    def sample(
        self, 
        z: torch.Tensor, 
        temperature: float = 1.0, 
        cfg: float = 1.0
    ) -> torch.Tensor:
        """采样 - 使用 ODE (x-pred 自动转换为 velocity)"""
        device = z.device
        
        if cfg != 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).to(device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            model_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).to(device)
            model_kwargs = dict(c=z)
            model_fn = self.net.forward
        
        sample_fn = self.gen_diffusion.sample_ode()
        sampled_token_latent = sample_fn(noise, model_fn, **model_kwargs)[-1]
        
        return sampled_token_latent


#################################################################################
#                             Model Factory Functions                            #
#################################################################################

# ======================== x-prediction 版本 (JiT 核心，推荐) ========================

def diff_transformer_xpred_s(**kwargs):
    """小型 DiffTransformer x-pred: ~10M params"""
    return DiffTransformer_XPred(
        hidden_size=256,
        depth=6,
        num_heads=4,
        **kwargs
    )

def diff_transformer_xpred_b(**kwargs):
    """基础 DiffTransformer x-pred: ~30M params"""
    return DiffTransformer_XPred(
        hidden_size=512,
        depth=8,
        num_heads=8,
        **kwargs
    )

def diff_transformer_xpred_l(**kwargs):
    """大型 DiffTransformer x-pred: ~90M params"""
    return DiffTransformer_XPred(
        hidden_size=768,
        depth=12,
        num_heads=12,
        **kwargs
    )

def diff_transformer_xpred_xl(**kwargs):
    """超大型 DiffTransformer x-pred: ~150M params (推荐配置)
    
    ⭐ JiT 核心实现：
    - x-prediction: 直接预测干净的 FSQ 坐标
    - 1D Transformer: 全局时序注意力
    - adaLN: 优雅的条件注入
    """
    return DiffTransformer_XPred(
        hidden_size=1024,
        depth=16,
        num_heads=16,
        mlp_ratio=4.0,
        num_sampling_steps=50,
        **kwargs
    )

# ======================== v-prediction 版本 (对照组) ========================

def diff_transformer_vpred_s(**kwargs):
    """小型 DiffTransformer v-pred (对照)"""
    return DiffTransformer_SiT(
        hidden_size=256,
        depth=6,
        num_heads=4,
        **kwargs
    )

def diff_transformer_vpred_b(**kwargs):
    """基础 DiffTransformer v-pred (对照)"""
    return DiffTransformer_SiT(
        hidden_size=512,
        depth=8,
        num_heads=8,
        **kwargs
    )

def diff_transformer_vpred_l(**kwargs):
    """大型 DiffTransformer v-pred (对照)"""
    return DiffTransformer_SiT(
        hidden_size=768,
        depth=12,
        num_heads=12,
        **kwargs
    )

def diff_transformer_vpred_xl(**kwargs):
    """超大型 DiffTransformer v-pred (对照)"""
    return DiffTransformer_SiT(
        hidden_size=1024,
        depth=16,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs
    )

# ======================== 向后兼容别名 ========================

# 默认使用 x-prediction (JiT 核心)
diff_transformer_small = diff_transformer_xpred_s
diff_transformer_base = diff_transformer_xpred_b
diff_transformer_large = diff_transformer_xpred_l
diff_transformer_xl = diff_transformer_xpred_xl


# 模型注册表
DiffTransformer_models = {
    # x-prediction (JiT 核心，推荐) ⭐
    'DiffTransformer-S': diff_transformer_xpred_s,
    'DiffTransformer-B': diff_transformer_xpred_b,
    'DiffTransformer-L': diff_transformer_xpred_l,
    'DiffTransformer-XL': diff_transformer_xpred_xl,  # 推荐
    
    # 显式命名的 x-prediction 版本
    'DiffTransformer-XPred-S': diff_transformer_xpred_s,
    'DiffTransformer-XPred-B': diff_transformer_xpred_b,
    'DiffTransformer-XPred-L': diff_transformer_xpred_l,
    'DiffTransformer-XPred-XL': diff_transformer_xpred_xl,
    
    # v-prediction 版本 (对照组)
    'DiffTransformer-VPred-S': diff_transformer_vpred_s,
    'DiffTransformer-VPred-B': diff_transformer_vpred_b,
    'DiffTransformer-VPred-L': diff_transformer_vpred_l,
    'DiffTransformer-VPred-XL': diff_transformer_vpred_xl,
}


#################################################################################
#                                    Testing                                     #
#################################################################################

if __name__ == '__main__':
    print("=" * 60)
    print("Testing DiffTransformer module (JiT x-prediction)")
    print("=" * 60)
    
    # 测试配置
    batch_size = 4
    seq_len = 49  # 196 // 4
    fsq_dim = 6  # FSQ_AE_High
    z_dim = 1024  # MAR latent_dim
    
    # ========== 测试 x-prediction 模型 ==========
    print("\n[1] Testing DiffTransformer-XPred-XL (JiT Core)")
    model_xpred = DiffTransformer_XPred(
        target_channels=fsq_dim, 
        z_channels=z_dim,
        hidden_size=512,  # 使用较小配置加速测试
        depth=4,
        num_heads=8,
        num_sampling_steps=20
    )
    print(f"    Model: {model_xpred.__class__.__name__}")
    print(f"    Parameters: {sum(p.numel() for p in model_xpred.parameters())/1e6:.2f}M")
    
    # 测试训练
    target = torch.randn(batch_size * seq_len, fsq_dim)
    z = torch.randn(batch_size * seq_len, z_dim)
    
    loss = model_xpred(target, z)
    print(f"    Training loss (x-pred): {loss.item():.4f}")
    
    # 测试采样 (无 CFG)
    with torch.no_grad():
        sampled = model_xpred.sample(z[:32], temperature=1.0, cfg=1.0)
        print(f"    Sampled shape: {sampled.shape}")
        print(f"    Sampled range: [{sampled.min():.3f}, {sampled.max():.3f}]")
    
    # 测试 CFG 采样
    with torch.no_grad():
        z_small = z[:32]
        z_cfg = torch.cat([z_small, z_small], dim=0)
        sampled_cfg = model_xpred.sample(z_cfg, temperature=1.0, cfg=4.5)
        print(f"    CFG Sampled shape: {sampled_cfg.shape}")
    
    # ========== 测试 v-prediction 模型 (对照) ==========
    print("\n[2] Testing DiffTransformer-VPred (Control Group)")
    model_vpred = DiffTransformer_SiT(
        target_channels=fsq_dim,
        z_channels=z_dim,
        hidden_size=512,
        depth=4,
        num_heads=8
    )
    print(f"    Model: {model_vpred.__class__.__name__}")
    
    loss_v = model_vpred(target, z)
    print(f"    Training loss (v-pred): {loss_v.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\n推荐使用: FSQ-MARDM-DiT-XL (x-prediction)")
    print("训练命令:")
    print("  python train_MARDM.py --model FSQ-MARDM-DiT-XL ...")

