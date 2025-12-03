#!/bin/bash
# 测试批量生成系统 - 用10条数据快速测试

echo "=========================================="
echo "  批量生成系统测试"
echo "=========================================="
echo ""

# 配置
TEST_DIR="./test_batch_output"
TEMP_DIR="${TEST_DIR}/temp"
FINAL_DIR="${TEST_DIR}/final"
PYTHON="/data/conda-envs/MARDM/bin/python"

# 创建测试目录
mkdir -p $TEMP_DIR $FINAL_DIR

echo "📝 提取10条测试数据..."
sed -n '28,37p' 10K静态动作描述.md | sed 's/^[0-9]*\. //' > test_10_descriptions.txt

echo "✅ 测试数据:"
cat test_10_descriptions.txt | nl
echo ""

echo "🚀 开始测试生成（使用GPU 1）..."
echo ""

CUDA_VISIBLE_DEVICES=1 $PYTHON batch_generate_10k.py \
    --text_path test_10_descriptions.txt \
    --temp_dir $TEMP_DIR \
    --final_dir $FINAL_DIR \
    --gpu_id 0 \
    --batch_size 4 \
    --sequence_length 16

echo ""
echo "=========================================="
echo "✅ 测试完成！"
echo "=========================================="
echo ""

# 检查结果
NPY_COUNT=$(find $FINAL_DIR -name "*.npy" | wc -l)
JSON_COUNT=$(find $FINAL_DIR -name "*.json" | wc -l)
PNG_COUNT=$(find $FINAL_DIR -name "*.png" | wc -l)

echo "📊 生成结果:"
echo "   NPY文件:  $NPY_COUNT / 10"
echo "   JSON文件: $JSON_COUNT / 10"
echo "   PNG文件:  $PNG_COUNT / 10"
echo ""

if [ $NPY_COUNT -eq 10 ] && [ $JSON_COUNT -eq 10 ] && [ $PNG_COUNT -eq 10 ]; then
    echo "✅ 测试成功！所有文件都已生成"
else
    echo "⚠️  警告: 部分文件未生成"
fi

echo ""
echo "📁 生成的文件:"
ls -lh $FINAL_DIR | head -15
echo ""

echo "💡 查看结果:"
echo "   cd $FINAL_DIR"
echo "   ls -lh"
echo ""

echo "🧹 清理测试文件:"
echo "   rm -rf $TEST_DIR test_10_descriptions.txt"
echo ""

echo "=========================================="


