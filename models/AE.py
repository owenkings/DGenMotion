import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from models.FSQ import FSQ, FSQ_configs

#################################################################################
#                                         AE                                    #
#################################################################################
class AE(nn.Module):
    def __init__(self, input_width=67, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        self.output_emb_width = output_emb_width
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

    def preprocess(self, x):
        x = x.permute(0, 2, 1).float()
        return x

    def encode(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        return x_encoder

    def forward(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_out = self.decoder(x_encoder)
        return x_out

    def decode(self, x):
        x_out = self.decoder(x)
        return x_out

#################################################################################
#                                      FSQ-AE                                   #
#################################################################################
class FSQ_AE(nn.Module):
    """
    FSQ-AE: 集成 Finite Scalar Quantization 的自编码器
    
    结构：
    Input (67-dim) → Encoder → 512-dim → Linear(512→d) → FSQ → Linear(d→512) → Decoder → Output (67-dim)
    
    与普通 AE 的区别：
    - Bottleneck 处使用 FSQ 量化
    - 潜空间落在规整的"隐式网格"上
    - 保持拓扑结构（数值相邻的量化值在语义上也相近）
    
    Args:
        input_width: 输入动作维度 (67)
        output_emb_width: 编码器输出维度 (512)
        fsq_levels: FSQ 量化级别列表，如 [8, 5, 5, 5, 5]
        其他参数与 AE 相同
    """
    
    def __init__(self, input_width=67, output_emb_width=512, down_t=2, stride_t=2, 
                 width=512, depth=3, dilation_growth_rate=3, activation='relu', norm=None,
                 fsq_levels: List[int] = [8, 5, 5, 5, 5]):
        super().__init__()
        
        self.fsq_dim = len(fsq_levels)  # FSQ 维度 (d)
        self.output_emb_width = output_emb_width
        self.fsq_levels = fsq_levels
        
        # Encoder (与 AE 相同)
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        
        # FSQ Bottleneck
        self.pre_fsq = nn.Linear(output_emb_width, self.fsq_dim)
        self.fsq = FSQ(levels=fsq_levels)
        self.post_fsq = nn.Linear(self.fsq_dim, output_emb_width)
        
        # Decoder (与 AE 相同)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
    
    def preprocess(self, x):
        """将输入转换为 (B, C, T) 格式"""
        x = x.permute(0, 2, 1).float()
        return x
    
    def encode(self, x) -> torch.Tensor:
        """
        编码并通过 FSQ 量化
        
        Args:
            x: 输入动作 [B, T, 67]
            
        Returns:
            z_out: 量化后的潜变量 [B, 512, T/4]，用于 MARDM Transformer
        """
        x_in = self.preprocess(x)  # [B, 67, T]
        x_encoder = self.encoder(x_in)  # [B, 512, T/4]
        
        # FSQ 量化
        x_encoder = x_encoder.permute(0, 2, 1)  # [B, T/4, 512]
        z = self.pre_fsq(x_encoder)  # [B, T/4, d]
        z_q = self.fsq(z)  # [B, T/4, d] - 量化后
        z_out = self.post_fsq(z_q)  # [B, T/4, 512]
        
        return z_out.permute(0, 2, 1)  # [B, 512, T/4]
    
    def encode_with_fsq(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码并返回完整信息（包括 FSQ 量化坐标）
        
        Args:
            x: 输入动作 [B, T, 67]
            
        Returns:
            z_out: 量化后的潜变量 [B, 512, T/4]
            z_fsq: FSQ 量化坐标 [B, T/4, d]，这是 Diffusion 的预测目标
        """
        x_in = self.preprocess(x)  # [B, 67, T]
        x_encoder = self.encoder(x_in)  # [B, 512, T/4]
        
        # FSQ 量化
        x_encoder = x_encoder.permute(0, 2, 1)  # [B, T/4, 512]
        z = self.pre_fsq(x_encoder)  # [B, T/4, d]
        z_fsq = self.fsq(z)  # [B, T/4, d] - FSQ 量化坐标
        z_out = self.post_fsq(z_fsq)  # [B, T/4, 512]
        
        return z_out.permute(0, 2, 1), z_fsq  # [B, 512, T/4], [B, T/4, d]
    
    def get_fsq_target(self, x) -> torch.Tensor:
        """
        仅获取 FSQ 量化坐标（Diffusion 的预测目标）
        
        Args:
            x: 输入动作 [B, T, 67]
            
        Returns:
            z_fsq: FSQ 量化坐标 [B, T/4, d]
        """
        x_in = self.preprocess(x)  # [B, 67, T]
        x_encoder = self.encoder(x_in)  # [B, 512, T/4]
        
        x_encoder = x_encoder.permute(0, 2, 1)  # [B, T/4, 512]
        z = self.pre_fsq(x_encoder)  # [B, T/4, d]
        z_fsq = self.fsq(z)  # [B, T/4, d]
        
        return z_fsq
    
    def decode(self, x) -> torch.Tensor:
        """
        从 512-dim 潜变量解码
        
        Args:
            x: 潜变量 [B, 512, T/4]
            
        Returns:
            motion: 重建动作 [B, T, 67]
        """
        x_out = self.decoder(x)
        return x_out
    
    def decode_from_fsq(self, z_fsq: torch.Tensor) -> torch.Tensor:
        """
        从 FSQ 量化坐标直接解码
        
        Args:
            z_fsq: FSQ 量化坐标 [B, T/4, d]
            
        Returns:
            motion: 重建动作 [B, T, 67]
        """
        z_out = self.post_fsq(z_fsq)  # [B, T/4, 512]
        z_out = z_out.permute(0, 2, 1)  # [B, 512, T/4]
        return self.decoder(z_out)
    
    def forward(self, x) -> torch.Tensor:
        """
        完整的前向传播（编码 → FSQ 量化 → 解码）
        
        Args:
            x: 输入动作 [B, T, 67]
            
        Returns:
            x_out: 重建动作 [B, T, 67]
        """
        x_in = self.preprocess(x)  # [B, 67, T]
        x_encoder = self.encoder(x_in)  # [B, 512, T/4]
        
        # FSQ 量化
        x_encoder = x_encoder.permute(0, 2, 1)  # [B, T/4, 512]
        z = self.pre_fsq(x_encoder)  # [B, T/4, d]
        z_q = self.fsq(z)  # [B, T/4, d]
        z_out = self.post_fsq(z_q)  # [B, T/4, 512]
        z_out = z_out.permute(0, 2, 1)  # [B, 512, T/4]
        
        # 解码
        x_out = self.decoder(z_out)  # [B, T, 67]
        return x_out
    
    def get_codebook_usage(self, x) -> dict:
        """
        分析码本使用情况（用于监控训练）
        
        Returns:
            dict: 包含 unique_codes, usage_rate 等统计信息
        """
        z_fsq = self.get_fsq_target(x)  # [B, T/4, d]
        indices = self.fsq.get_indices(z_fsq)  # [B, T/4]
        
        unique_indices = indices.unique()
        usage_rate = len(unique_indices) / self.fsq.codebook_size
        
        return {
            'unique_codes': len(unique_indices),
            'total_codes': self.fsq.codebook_size,
            'usage_rate': usage_rate,
            'indices': indices
        }


#################################################################################
#                                      AE Zoos                                  #
#################################################################################
def ae(**kwargs):
    return AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None, **kwargs)

def fsq_ae_small(**kwargs):
    """小型 FSQ-AE: ~3,125 隐式码本"""
    return FSQ_AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                  dilation_growth_rate=3, activation='relu', norm=None,
                  fsq_levels=FSQ_configs['small'], **kwargs)

def fsq_ae_medium(**kwargs):
    """中型 FSQ-AE: ~5,000 隐式码本"""
    return FSQ_AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                  dilation_growth_rate=3, activation='relu', norm=None,
                  fsq_levels=FSQ_configs['medium'], **kwargs)

def fsq_ae_large(**kwargs):
    """大型 FSQ-AE: ~21,600 隐式码本"""
    return FSQ_AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                  dilation_growth_rate=3, activation='relu', norm=None,
                  fsq_levels=FSQ_configs['large'], **kwargs)

def fsq_ae_xlarge(**kwargs):
    """超大型 FSQ-AE: ~76,800 隐式码本"""
    return FSQ_AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                  dilation_growth_rate=3, activation='relu', norm=None,
                  fsq_levels=FSQ_configs['xlarge'], **kwargs)

def fsq_ae_high(**kwargs):
    """高容量 FSQ-AE: ~64,000 隐式码本 (推荐用于高质量重建)"""
    return FSQ_AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                  dilation_growth_rate=3, activation='relu', norm=None,
                  fsq_levels=FSQ_configs['high'], **kwargs)

def fsq_ae_ultra(**kwargs):
    """超高容量 FSQ-AE: ~102,400 隐式码本"""
    return FSQ_AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                  dilation_growth_rate=3, activation='relu', norm=None,
                  fsq_levels=FSQ_configs['ultra'], **kwargs)

def fsq_ae_mega(**kwargs):
    """极高容量 FSQ-AE: ~163,840 隐式码本"""
    return FSQ_AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                  dilation_growth_rate=3, activation='relu', norm=None,
                  fsq_levels=FSQ_configs['mega'], **kwargs)

def fsq_ae_high_dim7(**kwargs):
    """高维度 FSQ-AE: ~109,375 隐式码本, dim=7"""
    return FSQ_AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                  dilation_growth_rate=3, activation='relu', norm=None,
                  fsq_levels=FSQ_configs['high_dim7'], **kwargs)

def fsq_ae_high_dim8(**kwargs):
    """高维度 FSQ-AE: ~390,625 隐式码本, dim=8"""
    return FSQ_AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                  dilation_growth_rate=3, activation='relu', norm=None,
                  fsq_levels=FSQ_configs['high_dim8'], **kwargs)

def fsq_ae_custom(fsq_levels: List[int], **kwargs):
    """自定义 FSQ-AE"""
    return FSQ_AE(output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                  dilation_growth_rate=3, activation='relu', norm=None,
                  fsq_levels=fsq_levels, **kwargs)

AE_models = {
    'AE_Model': ae,
    'FSQ_AE_Small': fsq_ae_small,       # 3,125
    'FSQ_AE_Medium': fsq_ae_medium,     # 5,000
    'FSQ_AE_Large': fsq_ae_large,       # 21,600
    'FSQ_AE_XLarge': fsq_ae_xlarge,     # 76,800
    'FSQ_AE_High': fsq_ae_high,         # 64,000 (推荐)
    'FSQ_AE_Ultra': fsq_ae_ultra,       # 102,400
    'FSQ_AE_Mega': fsq_ae_mega,         # 163,840
    'FSQ_AE_HighDim7': fsq_ae_high_dim7, # 109,375, dim=7
    'FSQ_AE_HighDim8': fsq_ae_high_dim8, # 390,625, dim=8
}

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
class Encoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, input_emb_width=3, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class nonlinearity(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=0.2):
        super(ResConv1DBlock, self).__init__()
        padding = dilation
        self.norm = norm

        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig
        return x
