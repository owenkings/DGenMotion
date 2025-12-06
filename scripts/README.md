# 评估脚本使用说明

本目录包含MARDM项目的评估辅助脚本。

## 📁 目录内容

### 评估脚本
- `run_full_evaluation.sh` - 一键运行完整评估（推荐）
- `comprehensive_evaluation.py` - 综合评估自动化脚本
- `performance_profiling.py` - 性能分析工具
- `generate_evaluation_report.py` - 报告生成器

### 其他脚本
- `monitor_progress.sh` - 进度监控
- `run_10k_generation.sh` - 批量生成
- `sample_single_pose.py` - 单帧姿态生成
- 等等...

### 文档
- `EVALUATION_README.md` - 详细评估指南

---

## 🚀 快速开始

### ⚠️ 重要说明

**所有脚本都应从项目根目录运行！**

```bash
# 正确做法 ✅
cd /data/tiany/MARDM
bash scripts/run_full_evaluation.sh t2m

# 错误做法 ❌
cd /data/tiany/MARDM/scripts
bash run_full_evaluation.sh t2m  # 路径会出错
```

---

## 📊 评估流程

### 方式1: 一键完整评估（最简单）

```bash
cd /data/tiany/MARDM
bash scripts/run_full_evaluation.sh t2m
```

**时间**: 5-9小时  
**包含**: 所有模型评估 + 性能分析 + 报告生成

### 方式2: 分步运行

```bash
cd /data/tiany/MARDM

# 1. AE模型评估（30-60分钟）
python evaluation_AE.py --name AE --dataset_name t2m

# 2. MARDM-SiT-XL评估（2-4小时）
python evaluation_MARDM.py --name MARDM_SiT_XL --model "MARDM-SiT-XL" --dataset_name t2m --cfg 4.5

# 3. MARDM-DDPM-XL评估（2-4小时）
python evaluation_MARDM.py --name MARDM_DDPM_XL --model "MARDM-DDPM-XL" --dataset_name t2m --cfg 4.5

# 4. 性能分析（30-60分钟）
python scripts/performance_profiling.py --dataset_name t2m

# 5. 生成报告（<1分钟）
python scripts/generate_evaluation_report.py --dataset_name t2m
```

---

## 🔧 脚本工作原理

### Shell脚本（.sh）

Shell脚本会自动切换到项目根目录：

```bash
# run_full_evaluation.sh 中的代码
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."  # 切换到项目根目录
```

所以你可以从任何位置运行：
```bash
bash /data/tiany/MARDM/scripts/run_full_evaluation.sh t2m
```

### Python脚本（.py）

Python脚本在开始时也会切换目录：

```python
# 在每个脚本开头
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
```

---

## 📂 输出位置

所有输出都在项目根目录：

```
/data/tiany/MARDM/
├── evaluation_results/    # 评估报告
│   ├── evaluation_report_*.md
│   ├── performance_profile_*.json
│   └── ...
├── logs/                  # 运行日志
│   ├── eval_ae_*.log
│   ├── eval_mardm_*.log
│   └── ...
└── checkpoints/           # 模型评估日志
    └── t2m/
        ├── AE/eval/eval.log
        ├── MARDM_SiT_XL/eval/eval.log
        └── MARDM_DDPM_XL/eval/eval.log
```

---

## 💡 常见问题

### Q1: 为什么脚本要在项目根目录运行？

**A**: 因为评估脚本需要：
- 导入 `models/` 和 `utils/` 模块
- 访问 `checkpoints/` 和 `datasets/` 目录
- 写入 `evaluation_results/` 和 `logs/` 目录

### Q2: 我可以在scripts目录直接运行吗？

**A**: 可以，但**不推荐**！脚本会自动切换目录，但可能有路径问题。

推荐做法：
```bash
cd /data/tiany/MARDM
bash scripts/run_full_evaluation.sh t2m
```

### Q3: 如何查看评估进度？

**A**: 使用日志文件：
```bash
# 查看最新日志
tail -f logs/eval_*.log

# 或使用监控脚本
bash scripts/monitor_progress.sh
```

### Q4: 如何后台运行？

**A**:
```bash
cd /data/tiany/MARDM
nohup bash scripts/run_full_evaluation.sh t2m > evaluation.log 2>&1 &

# 查看进度
tail -f evaluation.log
```

---

## 📚 详细文档

查看 `EVALUATION_README.md` 了解更多详情：
- 每个脚本的详细说明
- 评估指标解释
- 高级用法
- 故障排除

---

## 🎯 快速参考

| 任务 | 命令 | 时间 |
|------|------|------|
| 完整评估 | `bash scripts/run_full_evaluation.sh t2m` | 5-9小时 |
| 性能分析 | `python scripts/performance_profiling.py --dataset_name t2m` | 30-60分钟 |
| 生成报告 | `python scripts/generate_evaluation_report.py --dataset_name t2m` | <1分钟 |

**记住**: 始终从 `/data/tiany/MARDM` 运行！

---

**最后更新**: 2025-12-05

