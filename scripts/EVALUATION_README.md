# MARDM项目评估系统

本文档介绍为MARDM项目新增的完整评估系统。

## 🎯 评估系统概述

本评估系统提供了对MARDM项目进行全面、系统化评估的工具集，包括：

### ✅ 已完成的工作

1. **环境检查工具** (`check_environment.py`)
   - 检查Python包依赖
   - 验证目录结构
   - 确认模型文件存在
   - 验证数据集完整性
   - 检查评估脚本

2. **综合评估脚本** (`comprehensive_evaluation.py`)
   - 自动化运行所有评估
   - 解析评估日志
   - 收集所有指标
   - 生成JSON格式结果

3. **性能分析工具** (`performance_profiling.py`)
   - 测量推理时间
   - 监控内存使用
   - 计算吞吐量
   - 分析不同配置下的性能
   - 批处理性能测试
   - 采样步数影响分析

4. **报告生成器** (`generate_evaluation_report.py`)
   - 整合所有评估结果
   - 生成Markdown格式报告
   - 创建对比表格
   - 提供改进建议
   - 自动计算性能差异

5. **快速测试工具** (`quick_evaluation.py`)
   - 快速验证系统状态
   - 测试模型加载
   - 验证推理功能
   - 生成测试报告

6. **完整评估脚本** (`run_full_evaluation.sh`)
   - 一键运行所有评估
   - 自动保存日志
   - 分步骤执行
   - 错误处理

7. **详细文档** (`EVALUATION_GUIDE.md`)
   - 完整的使用指南
   - 指标说明
   - 结果解读
   - 常见问题解答

## 📊 评估指标

### 质量指标
| 指标 | 说明 | 目标 |
|------|------|------|
| FID | Fréchet Inception Distance | 越低越好 |
| Diversity | 生成多样性 | 接近真实数据 |
| R-Precision | 检索准确率 (TOP1/2/3) | 越高越好 |
| Matching Score | 文本-动作匹配度 | 越低越好 |
| Multimodality | 多模态生成能力 | 越高越好 |
| CLIP-Score | 文本-动作对齐 | 越高越好 |
| MAE | 重建误差 (仅AE) | 越低越好 |

### 性能指标
| 指标 | 说明 | 单位 |
|------|------|------|
| 推理时间 | 单次前向传播时间 | ms |
| 生成时间 | 完整生成时间 | ms |
| 吞吐量 | 每秒处理样本数 | samples/sec |
| 内存占用 | GPU/CPU内存使用 | MB/GB |

## 🚀 快速开始

**⚠️ 重要说明**：
- 评估辅助脚本位于 `scripts/` 目录
- 所有命令应从**项目根目录**运行
- Shell脚本会自动切换到正确的目录

### 1. 环境检查（可选）
```bash
cd /data/tiany/MARDM
python scripts/check_environment.py --dataset_name t2m
```

### 2. 运行评估

#### 方案A: 分步运行（推荐）
```bash
cd /data/tiany/MARDM

# 步骤1: 评估AE
python evaluation_AE.py --name AE --dataset_name t2m

# 步骤2: 评估MARDM-SiT-XL
python evaluation_MARDM.py --name MARDM_SiT_XL --model "MARDM-SiT-XL" --dataset_name t2m --cfg 4.5

# 步骤3: 评估MARDM-DDPM-XL
python evaluation_MARDM.py --name MARDM_DDPM_XL --model "MARDM-DDPM-XL" --dataset_name t2m --cfg 4.5

# 步骤4: 性能分析
python scripts/performance_profiling.py --dataset_name t2m

# 步骤5: 生成报告
python scripts/generate_evaluation_report.py --dataset_name t2m
```

#### 方案B: 一键运行（最简单）
```bash
cd /data/tiany/MARDM
bash scripts/run_full_evaluation.sh t2m
```

### 3. 查看结果
```bash
# 查看综合报告
cat evaluation_results/evaluation_report_t2m_latest.md

# 查看原始日志
cat checkpoints/t2m/*/eval/eval.log
```

## 📁 文件说明

### 新增的评估工具

| 文件 | 功能 | 运行时间 |
|------|------|----------|
| `check_environment.py` | 环境和文件检查 | < 1分钟 |
| `comprehensive_evaluation.py` | 综合评估（自动化） | 5-9小时 |
| `performance_profiling.py` | 性能分析 | 30-60分钟 |
| `generate_evaluation_report.py` | 报告生成 | < 1分钟 |
| `quick_evaluation.py` | 快速测试 | 5-10分钟 |
| `run_full_evaluation.sh` | 完整评估脚本 | 5-9小时 |
| `EVALUATION_GUIDE.md` | 详细使用指南 | - |
| `EVALUATION_README.md` | 本文档 | - |

### 原有的评估脚本（已存在）

| 文件 | 功能 | 运行时间 |
|------|------|----------|
| `evaluation_AE.py` | AE模型评估 | 30-60分钟 |
| `evaluation_MARDM.py` | MARDM模型评估 | 2-4小时 |

## 📈 评估流程

```
┌─────────────────────┐
│  环境检查            │
│  check_environment  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  评估AE模型         │
│  evaluation_AE      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  评估MARDM-SiT-XL   │
│  evaluation_MARDM   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  评估MARDM-DDPM-XL  │
│  evaluation_MARDM   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  性能分析           │
│  performance_prof   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  生成报告           │
│  generate_report    │
└─────────────────────┘
```

## 📊 输出结果

### 目录结构
```
evaluation_results/
├── environment_check_t2m_20251203_140019.json
├── performance_profile_t2m_20251203_150000.json
├── evaluation_report_t2m_20251203_160000.json
└── evaluation_report_t2m_latest.md

checkpoints/t2m/
├── AE/eval/eval.log
├── MARDM_SiT_XL/eval/eval.log
└── MARDM_DDPM_XL/eval/eval.log

logs/
├── eval_ae_t2m_20251203_140000.log
├── eval_mardm_sit_t2m_20251203_150000.log
├── eval_mardm_ddpm_t2m_20251203_160000.log
└── performance_profile_t2m_20251203_170000.log
```

### 报告内容

生成的Markdown报告包含：

1. **评估概述**
   - 评估时间和数据集
   - 系统信息

2. **模型性能指标**
   - 所有模型的对比表格
   - 每个模型的详细指标

3. **性能分析**
   - 测试环境信息
   - 推理时间分析
   - 内存使用情况
   - 批处理性能
   - 采样步数影响

4. **指标说明**
   - 每个指标的详细解释
   - 评判标准

5. **结论与建议**
   - 模型对比分析
   - 改进建议

## 🔧 高级用法

### 自定义评估参数

#### 修改采样步数
```bash
python evaluation_MARDM.py \
    --name MARDM_SiT_XL \
    --model "MARDM-SiT-XL" \
    --dataset_name t2m \
    --cfg 4.5 \
    --time_steps 25  # 默认18
```

#### 修改CFG scale
```bash
python evaluation_MARDM.py \
    --name MARDM_SiT_XL \
    --model "MARDM-SiT-XL" \
    --dataset_name t2m \
    --cfg 5.0  # 默认4.5
```

#### 禁用多模态评估（加速）
```bash
python evaluation_MARDM.py \
    --name MARDM_SiT_XL \
    --model "MARDM-SiT-XL" \
    --dataset_name t2m \
    --cfg 4.5 \
    --cal_mm  # 这个flag会禁用多模态评估
```

### 后台运行

对于长时间评估，建议使用后台运行：

```bash
# 使用nohup
nohup bash run_full_evaluation.sh t2m > evaluation.log 2>&1 &

# 查看进度
tail -f evaluation.log

# 使用screen
screen -S mardm_eval
bash run_full_evaluation.sh t2m
# 按 Ctrl+A, D 分离会话
# 重新连接: screen -r mardm_eval
```

## 📝 评估清单

在运行评估前，请确认：

- [ ] 环境检查通过 (`check_environment.py`)
- [ ] 所有模型文件存在
- [ ] 数据集完整
- [ ] GPU可用（或接受CPU的慢速度）
- [ ] 有足够的磁盘空间（至少10GB）
- [ ] 有足够的时间（5-9小时）

## 🐛 故障排除

### 常见问题

1. **CUDA out of memory**
   - 减小batch_size
   - 使用更少的num_workers
   - 关闭其他GPU程序

2. **模块导入错误**
   - 确认conda环境激活
   - 检查依赖安装: `pip install -r requirements.txt`

3. **数据集路径错误**
   - 检查`datasets/`目录
   - 确认数据集名称正确（t2m或kit）

4. **评估时间过长**
   - 这是正常的，完整评估需要5-9小时
   - 可以使用`--cal_mm`禁用多模态评估以加速

5. **结果与论文不符**
   - 检查数据集版本
   - 确认评估器版本
   - 查看随机种子设置

## 📚 相关文档

- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - 详细使用指南
- [README.md](README.md) - 项目主README
- 原始论文和代码仓库

## 🎓 引用

如果使用本评估系统，请引用：

```bibtex
@article{mardm2024,
  title={MARDM: Motion-Aware Residual Diffusion Model},
  author={...},
  journal={...},
  year={2024}
}
```

## 📞 支持

如有问题：
1. 查看 [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) 的常见问题部分
2. 检查GitHub Issues
3. 联系项目维护者

---

## 🔄 更新日志

### 2025-12-03
- ✅ 创建完整的评估系统
- ✅ 添加环境检查工具
- ✅ 实现性能分析功能
- ✅ 创建报告生成器
- ✅ 编写详细文档
- ✅ 验证系统完整性

---

**最后更新**: 2025-12-03  
**版本**: 1.0  
**状态**: ✅ 就绪可用

