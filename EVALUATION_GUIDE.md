# MARDM项目评估指南

本指南提供了对MARDM项目进行全面评估的详细说明。

## 📋 目录

1. [评估概述](#评估概述)
2. [环境准备](#环境准备)
3. [评估脚本说明](#评估脚本说明)
4. [运行评估](#运行评估)
5. [评估指标说明](#评估指标说明)
6. [结果解读](#结果解读)
7. [常见问题](#常见问题)

---

## 评估概述

本评估系统包含以下内容：

### 评估的模型
- **AutoEncoder (AE)**: 运动数据压缩与重建模型
- **MARDM-SiT-XL**: 基于Scalable Interpolant Transformers的扩散模型（最佳性能）
- **MARDM-DDPM-XL**: 基于DDPM的扩散模型

### 评估的指标

#### 质量指标
- **FID (Fréchet Inception Distance)**: 生成质量，越低越好
- **Diversity**: 生成多样性
- **R-Precision (TOP1/TOP2/TOP3)**: 文本-动作检索准确率，越高越好
- **Matching Score**: 文本-动作匹配度，越低越好
- **Multimodality**: 多模态生成能力，越高越好
- **CLIP-Score**: 文本-动作对齐分数，越高越好
- **MAE**: 重建误差（仅AE），越低越好

#### 性能指标
- **推理时间**: 单次前向传播时间
- **生成时间**: 完整生成一个动作序列的时间
- **吞吐量**: 每秒处理样本数
- **内存占用**: GPU/CPU内存使用

### 数据集
- **HumanML3D (t2m)**: 主要数据集，包含约15,000个动作序列
- **KIT-ML (kit)**: 辅助数据集

---

## 环境准备

### 1. 检查环境

运行环境检查脚本：

```bash
python check_environment.py --dataset_name t2m
```

这将检查：
- Python包依赖
- 目录结构
- 模型文件
- 数据集文件
- 评估脚本

### 2. 确保环境完整

如果检查通过，你会看到：
```
✓ 系统检查通过，可以运行评估
```

如果有缺失项，请根据提示补充。

---

## 评估脚本说明

### 核心评估脚本

#### 1. `check_environment.py`
**用途**: 检查环境和文件完整性  
**运行时间**: < 1分钟  
**输出**: 环境检查报告

```bash
python check_environment.py --dataset_name t2m
```

#### 2. `evaluation_AE.py`
**用途**: 评估AutoEncoder模型  
**运行时间**: 约30-60分钟  
**输出**: `checkpoints/t2m/AE/eval/eval.log`

```bash
python evaluation_AE.py --name AE --dataset_name t2m
```

**参数说明**:
- `--name`: 模型名称（默认: AE）
- `--dataset_name`: 数据集名称（t2m或kit）
- `--num_workers`: 数据加载线程数（默认: 4）

#### 3. `evaluation_MARDM.py`
**用途**: 评估MARDM模型  
**运行时间**: 约2-4小时  
**输出**: `checkpoints/t2m/MARDM_*/eval/eval.log`

```bash
# 评估SiT-XL
python evaluation_MARDM.py \
    --name MARDM_SiT_XL \
    --model "MARDM-SiT-XL" \
    --dataset_name t2m \
    --cfg 4.5

# 评估DDPM-XL
python evaluation_MARDM.py \
    --name MARDM_DDPM_XL \
    --model "MARDM-DDPM-XL" \
    --dataset_name t2m \
    --cfg 4.5
```

**参数说明**:
- `--name`: 模型名称
- `--model`: 模型类型
- `--dataset_name`: 数据集名称
- `--cfg`: Classifier-free guidance scale（t2m用4.5，kit用2.5）
- `--time_steps`: 采样步数（默认: 18）
- `--cal_mm`: 是否计算多模态指标（默认: True）

#### 4. `performance_profiling.py`
**用途**: 性能分析（推理时间、内存等）  
**运行时间**: 约30-60分钟  
**输出**: `evaluation_results/performance_profile_*.json`

```bash
python performance_profiling.py --dataset_name t2m
```

#### 5. `generate_evaluation_report.py`
**用途**: 生成综合评估报告  
**运行时间**: < 1分钟  
**输出**: `evaluation_results/evaluation_report_*.md`

```bash
python generate_evaluation_report.py --dataset_name t2m
```

### 辅助脚本

#### `comprehensive_evaluation.py`
自动运行所有评估（不推荐，建议分步运行）

#### `quick_evaluation.py`
快速测试（需要CLIP等依赖）

#### `run_full_evaluation.sh`
完整评估的Shell脚本

---

## 运行评估

### 方案A: 分步运行（推荐）

这种方式可以更好地监控进度和处理错误。

#### 步骤1: 环境检查
```bash
python check_environment.py --dataset_name t2m
```

#### 步骤2: 评估AE模型
```bash
python evaluation_AE.py --name AE --dataset_name t2m --num_workers 4
```

预计时间: 30-60分钟  
输出位置: `checkpoints/t2m/AE/eval/eval.log`

#### 步骤3: 评估MARDM-SiT-XL
```bash
python evaluation_MARDM.py \
    --name MARDM_SiT_XL \
    --model "MARDM-SiT-XL" \
    --dataset_name t2m \
    --cfg 4.5 \
    --num_workers 4 \
    --time_steps 18
```

预计时间: 2-4小时  
输出位置: `checkpoints/t2m/MARDM_SiT_XL/eval/eval.log`

#### 步骤4: 评估MARDM-DDPM-XL
```bash
python evaluation_MARDM.py \
    --name MARDM_DDPM_XL \
    --model "MARDM-DDPM-XL" \
    --dataset_name t2m \
    --cfg 4.5 \
    --num_workers 4 \
    --time_steps 18
```

预计时间: 2-4小时  
输出位置: `checkpoints/t2m/MARDM_DDPM_XL/eval/eval.log`

#### 步骤5: 性能分析
```bash
python performance_profiling.py --dataset_name t2m
```

预计时间: 30-60分钟  
输出位置: `evaluation_results/performance_profile_*.json`

#### 步骤6: 生成报告
```bash
python generate_evaluation_report.py --dataset_name t2m
```

预计时间: < 1分钟  
输出位置: `evaluation_results/evaluation_report_*.md`

### 方案B: 使用Shell脚本

```bash
bash run_full_evaluation.sh t2m
```

这会自动运行所有评估步骤。预计总时间: 5-9小时

### 方案C: 后台运行

对于长时间运行的评估，建议使用nohup或screen：

```bash
# 使用nohup
nohup bash run_full_evaluation.sh t2m > evaluation.log 2>&1 &

# 或使用screen
screen -S mardm_eval
bash run_full_evaluation.sh t2m
# 按 Ctrl+A, D 分离会话
```

---

## 评估指标说明

### 质量指标详解

#### FID (Fréchet Inception Distance)
- **含义**: 衡量生成动作分布与真实动作分布的距离
- **范围**: [0, +∞)
- **越低越好**: FID < 1.0 为优秀，< 2.0 为良好
- **计算方式**: 基于特征空间的均值和协方差

#### Diversity
- **含义**: 生成动作的多样性
- **范围**: 取决于数据集
- **理想值**: 接近真实数据的多样性
- **计算方式**: 随机采样动作对之间的平均距离

#### R-Precision
- **含义**: 给定文本描述，能否从候选集中检索到对应动作
- **TOP1/TOP2/TOP3**: 前1/2/3名中包含正确动作的比例
- **范围**: [0, 1]
- **越高越好**: > 0.5 为良好，> 0.6 为优秀

#### Matching Score
- **含义**: 文本embedding与动作embedding之间的距离
- **范围**: [0, +∞)
- **越低越好**: 表示文本和动作更匹配

#### Multimodality
- **含义**: 同一文本描述生成多个不同动作的能力
- **范围**: 取决于数据集
- **越高越好**: 表示模型能生成更多样的动作

#### CLIP-Score
- **含义**: 基于CLIP模型的文本-动作对齐分数
- **范围**: 通常 [0, 1]
- **越高越好**: > 0.5 为良好

#### MAE (仅AE模型)
- **含义**: 重建动作与原始动作的平均绝对误差
- **范围**: [0, +∞)
- **越低越好**: < 0.1 为优秀

### 性能指标详解

#### 推理时间
- **含义**: 模型单次前向传播所需时间
- **单位**: 毫秒(ms)
- **影响因素**: 模型大小、序列长度、硬件

#### 生成时间
- **含义**: 完整生成一个动作序列的时间（包括采样步骤）
- **单位**: 毫秒(ms)
- **影响因素**: 采样步数、序列长度、CFG scale

#### 吞吐量
- **含义**: 每秒可处理的样本数
- **单位**: samples/sec
- **越高越好**: 表示处理速度更快

#### 内存占用
- **含义**: 推理过程中的峰值GPU内存
- **单位**: MB或GB
- **影响因素**: 模型大小、批大小、序列长度

---

## 结果解读

### 查看评估结果

#### 1. 查看原始日志
```bash
# AE模型
cat checkpoints/t2m/AE/eval/eval.log

# MARDM-SiT-XL
cat checkpoints/t2m/MARDM_SiT_XL/eval/eval.log

# MARDM-DDPM-XL
cat checkpoints/t2m/MARDM_DDPM_XL/eval/eval.log
```

#### 2. 查看综合报告
```bash
# Markdown格式（推荐）
cat evaluation_results/evaluation_report_t2m_latest.md

# JSON格式
cat evaluation_results/evaluation_report_t2m_*.json
```

#### 3. 查看性能分析
```bash
cat evaluation_results/performance_profile_t2m_*.json
```

### 结果示例

典型的评估结果（HumanML3D数据集）：

#### AE模型
```
FID: 0.XXX ± 0.XXX
Diversity: X.XXX ± 0.XXX
R-Precision TOP1: 0.XXX ± 0.XXX
Matching Score: X.XXX ± 0.XXX
MAE: 0.XXX ± 0.XXX
```

#### MARDM-SiT-XL（预期最佳）
```
FID: 0.XXX ± 0.XXX (应该是最低的)
Diversity: X.XXX ± 0.XXX
R-Precision TOP1: 0.XXX ± 0.XXX (应该是最高的)
Matching Score: X.XXX ± 0.XXX
Multimodality: X.XXX ± 0.XXX
CLIP-Score: 0.XXX ± 0.XXX
```

#### MARDM-DDPM-XL
```
FID: 0.XXX ± 0.XXX
Diversity: X.XXX ± 0.XXX
R-Precision TOP1: 0.XXX ± 0.XXX
Matching Score: X.XXX ± 0.XXX
Multimodality: X.XXX ± 0.XXX
CLIP-Score: 0.XXX ± 0.XXX
```

### 对比分析

评估完成后，报告会自动生成对比分析，包括：

1. **模型性能对比表**: 所有指标的横向对比
2. **改进百分比**: SiT vs DDPM的性能差异
3. **性能分析**: 推理速度、内存占用对比
4. **改进建议**: 基于评估结果的优化建议

---

## 常见问题

### Q1: 评估需要多长时间？
**A**: 
- 环境检查: < 1分钟
- AE评估: 30-60分钟
- MARDM-SiT-XL评估: 2-4小时
- MARDM-DDPM-XL评估: 2-4小时
- 性能分析: 30-60分钟
- **总计**: 约5-9小时

### Q2: 可以只评估部分模型吗？
**A**: 可以。每个评估脚本都是独立的，你可以选择只运行需要的部分。

### Q3: 评估时GPU内存不足怎么办？
**A**: 
- 减小batch_size（在evaluation脚本中修改）
- 使用更少的num_workers
- 关闭其他占用GPU的程序

### Q4: 如何在CPU上运行评估？
**A**: 评估脚本会自动检测并使用CPU，但速度会非常慢（可能需要数天）。

### Q5: 评估结果的置信区间是什么意思？
**A**: 每个指标运行20次取平均，置信区间表示结果的可靠性（95%置信度）。

### Q6: 为什么我的结果和论文不一致？
**A**: 可能的原因：
- 数据集版本不同
- 随机种子不同
- 评估器版本不同
- 硬件差异（CPU vs GPU）

### Q7: 如何保存评估结果？
**A**: 所有结果自动保存在：
- `checkpoints/[dataset]/[model]/eval/eval.log`: 原始日志
- `evaluation_results/`: 综合报告和性能分析

### Q8: 评估中断了怎么办？
**A**: 可以从中断的地方继续：
- 每个模型的评估是独立的
- 重新运行对应的evaluation脚本即可
- 之前的结果会被覆盖

### Q9: 如何评估KIT-ML数据集？
**A**: 将所有命令中的`--dataset_name t2m`改为`--dataset_name kit`，CFG scale改为2.5。

### Q10: 如何添加自定义评估指标？
**A**: 
1. 修改`utils/eval_utils.py`中的评估函数
2. 在`evaluation_*.py`中添加指标收集
3. 更新`generate_evaluation_report.py`以显示新指标

---

## 附录

### 文件结构
```
MARDM/
├── check_environment.py          # 环境检查
├── evaluation_AE.py              # AE评估
├── evaluation_MARDM.py           # MARDM评估
├── performance_profiling.py      # 性能分析
├── generate_evaluation_report.py # 报告生成
├── run_full_evaluation.sh        # 完整评估脚本
├── EVALUATION_GUIDE.md           # 本指南
├── checkpoints/                  # 模型文件
│   └── t2m/
│       ├── AE/
│       │   └── eval/eval.log
│       ├── MARDM_SiT_XL/
│       │   └── eval/eval.log
│       └── MARDM_DDPM_XL/
│           └── eval/eval.log
└── evaluation_results/           # 评估结果
    ├── environment_check_*.json
    ├── performance_profile_*.json
    └── evaluation_report_*.md
```

### 相关资源

- **论文**: [MARDM论文链接]
- **代码仓库**: [GitHub链接]
- **数据集**: HumanML3D, KIT-ML
- **预训练模型**: Google Drive链接

### 联系方式

如有问题，请：
1. 查看本指南的常见问题部分
2. 检查GitHub Issues
3. 联系项目维护者

---

**最后更新**: 2025-12-03

