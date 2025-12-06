#!/bin/bash
# MARDM项目完整评估脚本
# 运行所有评估任务并生成完整报告

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 切换到项目根目录
cd "$SCRIPT_DIR/.."

echo "========================================================================"
echo "MARDM项目完整评估"
echo "========================================================================"
echo "工作目录: $(pwd)"
echo "开始时间: $(date)"
echo ""

# 设置数据集
DATASET=${1:-"t2m"}  # 默认使用HumanML3D (t2m)
echo "数据集: $DATASET"
echo ""

# 创建结果目录
mkdir -p evaluation_results
mkdir -p logs

# 1. 环境检查
echo "========================================================================"
echo "步骤 1/5: 环境检查"
echo "========================================================================"
python -c "
import torch
import sys
print(f'Python版本: {sys.version}')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

# 2. 评估AE模型
echo "========================================================================"
echo "步骤 2/5: 评估AutoEncoder模型"
echo "========================================================================"
echo "这将需要约30-60分钟..."
python evaluation_AE.py \
    --name AE \
    --dataset_name $DATASET \
    --num_workers 4 \
    2>&1 | tee logs/eval_ae_${DATASET}_$(date +%Y%m%d_%H%M%S).log

echo "✓ AE评估完成"
echo ""

# 3. 评估MARDM-SiT-XL模型
echo "========================================================================"
echo "步骤 3/5: 评估MARDM-SiT-XL模型"
echo "========================================================================"
echo "这将需要约2-4小时..."

if [ "$DATASET" = "t2m" ]; then
    CFG_SCALE=4.5
else
    CFG_SCALE=2.5
fi

python evaluation_MARDM.py \
    --name MARDM_SiT_XL \
    --model "MARDM-SiT-XL" \
    --dataset_name $DATASET \
    --cfg $CFG_SCALE \
    --num_workers 4 \
    --time_steps 18 \
    2>&1 | tee logs/eval_mardm_sit_${DATASET}_$(date +%Y%m%d_%H%M%S).log

echo "✓ MARDM-SiT-XL评估完成"
echo ""

# 4. 评估MARDM-DDPM-XL模型
echo "========================================================================"
echo "步骤 4/5: 评估MARDM-DDPM-XL模型"
echo "========================================================================"
echo "这将需要约2-4小时..."

python evaluation_MARDM.py \
    --name MARDM_DDPM_XL \
    --model "MARDM-DDPM-XL" \
    --dataset_name $DATASET \
    --cfg $CFG_SCALE \
    --num_workers 4 \
    --time_steps 18 \
    2>&1 | tee logs/eval_mardm_ddpm_${DATASET}_$(date +%Y%m%d_%H%M%S).log

echo "✓ MARDM-DDPM-XL评估完成"
echo ""

# 5. 性能分析
echo "========================================================================"
echo "步骤 5/5: 性能分析"
echo "========================================================================"
echo "测量推理时间、内存使用等性能指标..."

python performance_profiling.py \
    --dataset_name $DATASET \
    2>&1 | tee logs/performance_profile_${DATASET}_$(date +%Y%m%d_%H%M%S).log

echo "✓ 性能分析完成"
echo ""

# 6. 生成综合报告
echo "========================================================================"
echo "生成综合评估报告"
echo "========================================================================"

python generate_evaluation_report.py \
    --dataset_name $DATASET \
    2>&1 | tee logs/report_generation_${DATASET}_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================================================"
echo "评估完成!"
echo "========================================================================"
echo "结束时间: $(date)"
echo ""
echo "评估结果保存在: ./evaluation_results/"
echo "日志文件保存在: ./logs/"
echo ""
echo "请查看 evaluation_results/ 目录中的报告文件"

