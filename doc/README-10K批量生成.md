# 10K静态姿态批量生成系统

> 从10,000条文本描述自动生成静态3D姿态，输出PNG图像、JSON姿态点云和NPY文件

---

## 🎯 系统特性

✅ **多GPU并行**: 自动在GPU 1、2、3上并行生成  
✅ **智能命名**: 文件以动作文本命名 (`arching_back.png`)  
✅ **双目录结构**: 临时缓存和最终结果分离  
✅ **完整输出**: PNG + JSON + NPY 三种格式  
✅ **进度监控**: 实时查看生成进度和预计时间  
✅ **错误处理**: 单个失败不影响整体，自动跳过已完成  

---

## ⚡ 快速开始

### 1. 测试系统（推荐）

```bash
./test_batch_system.sh
```

用10条数据测试，约1分钟完成。

### 2. 启动生成

```bash
./run_10k_generation.sh
```

自动启动3个GPU并行生成，预计12-15小时完成。

### 3. 监控进度

```bash
./monitor_progress.sh
```

实时显示进度、文件数、预计时间。

---

## 📁 输出结构

```
SinglePose/                      # 最终结果目录
├── arching_back.png            # 3D可视化图像 (130KB)
├── arching_back.json           # JSON姿态点云 (2.5KB)
├── arching_back.npy            # NumPy数组 (400B)
├── standing_naturally.png
├── standing_naturally.json
├── standing_naturally.npy
└── ...                         # 共10,000个姿态 × 3种格式
```

---

## 📊 关键指标

| 项目 | 数值 |
|------|------|
| 输入数据 | 10,000条文本描述 |
| 输出文件 | 30,000个 (PNG+JSON+NPY) |
| GPU使用 | GPU 1、2、3 (并行) |
| 预计耗时 | 12-15小时 |
| 磁盘占用 | ~1.3GB |
| 单个姿态 | ~5秒 |

---

## 🗂️ 文件说明

### 核心脚本

| 文件 | 说明 |
|------|------|
| `run_10k_generation.sh` | ⭐ 主控脚本，一键启动 |
| `batch_generate_10k.py` | 核心生成脚本 |
| `monitor_progress.sh` | 进度监控脚本 |
| `test_batch_system.sh` | 快速测试脚本 |

### 文档

| 文件 | 说明 |
|------|------|
| `10K批量生成-快速开始.md` | 快速上手指南 |
| `10K批量生成使用说明.md` | 完整使用手册 |
| `10K批量生成系统-总结.md` | 系统总结 |
| `README-10K批量生成.md` | 本文件 |

---

## 🔍 监控命令

```bash
# 实时监控（推荐）
./monitor_progress.sh

# 查看日志
tail -f logs/gpu1.log
tail -f logs/gpu2.log
tail -f logs/gpu3.log

# 统计进度
ls SinglePose/*.npy | wc -l

# 查看GPU状态
nvidia-smi
```

---

## 🛠️ 常用操作

```bash
# 启动生成
./run_10k_generation.sh

# 停止所有
pkill -f batch_generate_10k.py

# 重新启动（跳过已完成）
./run_10k_generation.sh

# 清理临时文件
rm -rf SinglePose_temp/*
```

---

## 📋 输出格式

### PNG图像
- 1200×1200像素，DPI 150
- 3D骨架可视化
- 约130KB/个

### JSON文件
```json
{
  "caption": "A person arching back",
  "frame_index": 8,
  "num_joints": 22,
  "joints": [[x, y, z], ...],
  "joint_names": ["pelvis", "left_hip", ...]
}
```

### NPY文件
- 形状: (22, 3)
- 22个关节点的XYZ坐标
- 单位: 米

```python
import numpy as np
pose = np.load('SinglePose/arching_back.npy')
# pose.shape = (22, 3)
```

---

## ⚙️ 系统架构

```
10K文本描述
    ↓
自动分割为3份
    ↓
┌─────────┬─────────┬─────────┐
│  GPU 1  │  GPU 2  │  GPU 3  │
│ 3,333条 │ 3,333条 │ 3,334条 │
└─────────┴─────────┴─────────┘
    ↓         ↓         ↓
临时缓存  临时缓存  临时缓存
    ↓         ↓         ↓
    └─────────┴─────────┘
            ↓
      SinglePose/
    (PNG+JSON+NPY)
```

---

## 🐛 故障排查

### GPU内存不足

编辑 `run_10k_generation.sh`，减小批处理大小：
```bash
--batch_size 4  # 从8改为4
```

### 进程意外退出

查看日志并重新运行：
```bash
tail -100 logs/gpu1.log
./run_10k_generation.sh  # 会跳过已完成的
```

### 生成速度太慢

使用快速模式（编辑 `run_10k_generation.sh`）：
```bash
--sequence_length 8   # 从16改为8，速度×2
--time_steps 10       # 从18改为10
```

---

## 📚 详细文档

- **快速开始**: `10K批量生成-快速开始.md`
- **完整手册**: `10K批量生成使用说明.md`
- **系统总结**: `10K批量生成系统-总结.md`

---

## ✅ 验证结果

生成完成后，运行：

```bash
echo "NPY:  $(find SinglePose -name '*.npy' | wc -l) / 10000"
echo "JSON: $(find SinglePose -name '*.json' | wc -l) / 10000"
echo "PNG:  $(find SinglePose -name '*.png' | wc -l) / 10000"
du -sh SinglePose
```

预期结果：
- NPY: 10000
- JSON: 10000
- PNG: 10000
- 大小: ~1.3GB

---

## 🎉 应用场景

生成的数据可用于：

- 🤖 动作识别训练数据
- 🎨 3D动画参考库
- 🔬 姿态估计研究
- 📊 人体运动分析
- 🧠 机器学习数据集

---

## 💡 性能优化

### 提高速度

```bash
--sequence_length 8   # 序列长度减半，速度×2
--time_steps 10       # 扩散步数减少
--batch_size 16       # 批大小增加（需要更多显存）
```

### 提高质量

```bash
--sequence_length 32  # 序列长度增加
--time_steps 50       # 扩散步数增加
--cfg 5.0             # CFG强度增加
```

---

## 📞 技术支持

遇到问题？

1. 查看 `10K批量生成使用说明.md` 的故障排查章节
2. 检查 `logs/` 目录下的日志文件
3. 确认GPU状态: `nvidia-smi`
4. 验证磁盘空间: `df -h`

---

## 🚀 开始使用

```bash
# 1. 测试（1分钟）
./test_batch_system.sh

# 2. 启动（12-15小时）
./run_10k_generation.sh

# 3. 监控
./monitor_progress.sh
```

---

**系统版本**: 1.0  
**创建日期**: 2025年12月2日  
**状态**: ✅ 就绪  
**开始命令**: `./run_10k_generation.sh` 🚀

