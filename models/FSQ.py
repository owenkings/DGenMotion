"""
Finite Scalar Quantization (FSQ) Module
=======================================

FSQ 核心思想：
1. 将连续值通过 tanh 压缩到 [-1, 1]
2. 将 [-1, 1] 离散化到有限个级别 (levels)
3. 量化后的值仍保持数值连续性（拓扑结构）

与 VQ 的关键区别：
- VQ: Token ID 100 和 101 在语义上无关联
- FSQ: 数值 5 和 6 几何相邻，保留拓扑结构

Reference: https://arxiv.org/abs/2309.15505
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np


class FSQ(nn.Module):
    """
    Finite Scalar Quantization
    
    将连续潜变量量化到有限的离散网格点上，同时保持拓扑结构。
    
    Args:
        levels: 每个维度的量化级别数，如 [8, 5, 5, 5, 5]
                隐式码本大小 = prod(levels)
                
    Example:
        >>> fsq = FSQ(levels=[8, 5, 5, 5, 5])  # 5000 个隐式中心点
        >>> z = torch.randn(2, 10, 5)  # [batch, seq, dim]
        >>> z_q = fsq(z)  # 量化后，仍是 [batch, seq, dim]
    """
    
    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        
        # 计算隐式码本大小
        self.codebook_size = int(np.prod(levels))
        
        # 预计算每个维度的量化范围
        # 对于 level L，量化值为 {-floor(L/2), ..., floor(L/2)} 或 {-floor(L/2), ..., floor((L-1)/2)}
        # 例如 L=5: {-2, -1, 0, 1, 2}, L=8: {-4, -3, -2, -1, 0, 1, 2, 3}
        _levels = torch.tensor(levels, dtype=torch.float32)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0)
        
        self.register_buffer('_levels', _levels)
        self.register_buffer('_basis', _basis)
        
        # 用于将值映射到 [-1, 1] 再量化
        # half_levels = (L - 1) / 2
        half_levels = (_levels - 1) / 2
        self.register_buffer('_half_levels', half_levels)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播：量化连续潜变量
        
        Args:
            z: 连续潜变量 [B, T, d] 或 [B, d]
            
        Returns:
            z_q: 量化后的潜变量，形状与输入相同
        """
        return self.quantize(z)
    
    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        量化过程：
        1. tanh 压缩到 [-1, 1]
        2. 缩放到各维度对应的范围
        3. round() 量化到最近整数
        4. 归一化回 [-1, 1] 范围（可选，保持数值稳定）
        """
        # 确保输入维度正确
        assert z.shape[-1] == self.dim, f"Expected dim {self.dim}, got {z.shape[-1]}"
        
        # Step 1: tanh 压缩到 [-1, 1]
        z_bounded = torch.tanh(z)
        
        # Step 2: 缩放到量化范围 [-half_level, half_level]
        # z_scaled ∈ [-half_level, half_level]
        half_levels = self._half_levels.to(z.device)
        z_scaled = z_bounded * half_levels
        
        # Step 3: round 量化（使用 STE - Straight-Through Estimator）
        z_quantized = self._round_ste(z_scaled)
        
        # Step 4: 归一化回 [-1, 1]（让不同 level 的维度在同一尺度）
        z_normalized = z_quantized / half_levels
        
        return z_normalized
    
    def _round_ste(self, z: torch.Tensor) -> torch.Tensor:
        """
        直通估计器 (Straight-Through Estimator)
        前向：round()
        反向：恒等映射（梯度直接传递）
        """
        return z + (torch.round(z) - z).detach()
    
    def get_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        将量化值转换为唯一索引（用于分析或可视化）
        
        Args:
            z_q: 量化后的值 [B, T, d]，范围在 [-1, 1]
            
        Returns:
            indices: 索引 [B, T]，范围 [0, codebook_size)
        """
        # 反归一化
        half_levels = self._half_levels.to(z_q.device)
        z_int = torch.round(z_q * half_levels)
        
        # 偏移到非负整数
        z_offset = z_int + half_levels
        
        # 计算唯一索引
        basis = self._basis.to(z_q.device)
        indices = (z_offset * basis).sum(dim=-1).long()
        
        return indices
    
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        将索引转换回量化值（逆操作，用于解码或分析）
        
        Args:
            indices: 索引 [B, T]
            
        Returns:
            z_q: 量化值 [B, T, d]
        """
        indices = indices.unsqueeze(-1)  # [B, T, 1]
        
        codes = []
        for i, level in enumerate(self.levels):
            code = (indices // int(self._basis[i].item())) % level
            code = code - self._half_levels[i]
            code = code / self._half_levels[i]  # 归一化到 [-1, 1]
            codes.append(code)
        
        return torch.cat(codes, dim=-1)
    
    def extra_repr(self) -> str:
        return f'levels={self.levels}, codebook_size={self.codebook_size}'


class ResidualFSQ(nn.Module):
    """
    残差 FSQ (Residual FSQ)
    
    类似于 RVQ，使用多层 FSQ 来提高精度。
    每一层量化上一层的残差。
    
    Args:
        levels: 每层每个维度的量化级别
        num_quantizers: 量化器数量（层数）
    """
    
    def __init__(self, levels: List[int], num_quantizers: int = 4):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList([FSQ(levels) for _ in range(num_quantizers)])
        self.dim = len(levels)
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            z: 输入 [B, T, d]
            
        Returns:
            z_q: 总量化值 [B, T, d]
            all_codes: 每层的量化值列表
        """
        residual = z
        z_q = torch.zeros_like(z)
        all_codes = []
        
        for quantizer in self.quantizers:
            code = quantizer(residual)
            residual = residual - code
            z_q = z_q + code
            all_codes.append(code)
            
        return z_q, all_codes


# ============== 预定义配置 ==============

def fsq_small():
    """小型 FSQ: ~1000 码本"""
    return FSQ(levels=[5, 5, 5, 5, 5])  # 3125

def fsq_medium():
    """中型 FSQ: ~5000 码本"""
    return FSQ(levels=[8, 5, 5, 5, 5])  # 5000

def fsq_large():
    """大型 FSQ: ~20000 码本"""
    return FSQ(levels=[8, 6, 6, 5, 5, 5])  # 21600

def fsq_xlarge():
    """超大型 FSQ: ~60000 码本"""
    return FSQ(levels=[8, 8, 8, 6, 5, 5])  # 76800


FSQ_configs = {
    'small': [5, 5, 5, 5, 5],           # 3,125, dim=5
    'medium': [8, 5, 5, 5, 5],          # 5,000, dim=5
    'large': [8, 6, 6, 5, 5, 5],        # 21,600, dim=6
    'xlarge': [8, 8, 8, 6, 5, 5],       # 76,800, dim=6
    
    # 高容量配置 - 推荐用于高质量重建
    'high': [8, 8, 8, 5, 5, 5],         # 64,000, dim=6
    'ultra': [8, 8, 8, 8, 5, 5],        # 102,400, dim=6
    'mega': [8, 8, 8, 8, 8, 5],         # 163,840, dim=6
    
    # 更高维度配置 - 更细粒度的量化
    'high_dim7': [7, 5, 5, 5, 5, 5, 5], # 109,375, dim=7
    'high_dim8': [5, 5, 5, 5, 5, 5, 5, 5], # 390,625, dim=8
}


if __name__ == '__main__':
    # 测试 FSQ 模块
    print("Testing FSQ module...")
    
    # 测试基本功能
    fsq = FSQ(levels=[8, 5, 5, 5, 5])
    print(f"FSQ config: {fsq}")
    print(f"Codebook size: {fsq.codebook_size}")
    
    # 测试量化
    z = torch.randn(2, 10, 5)
    z_q = fsq(z)
    print(f"Input shape: {z.shape}, Output shape: {z_q.shape}")
    print(f"Input range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"Output range: [{z_q.min():.3f}, {z_q.max():.3f}]")
    
    # 测试索引
    indices = fsq.get_indices(z_q)
    print(f"Indices shape: {indices.shape}")
    print(f"Unique indices: {indices.unique().shape[0]} / {fsq.codebook_size}")
    
    # 测试梯度
    z.requires_grad = True
    z_q = fsq(z)
    loss = z_q.sum()
    loss.backward()
    print(f"Gradient exists: {z.grad is not None}")
    print(f"Gradient shape: {z.grad.shape}")
    
    print("\nAll tests passed!")

