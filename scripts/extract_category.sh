#!/bin/bash
# extract_category.sh - 按关键词提取描述

KEYWORD=${1:-standing}
OUTPUT=${2:-${KEYWORD}_poses.txt}

echo "=========================================="
echo "  按类别提取描述工具"
echo "=========================================="
echo ""
echo "🔍 搜索关键词: '$KEYWORD'"
echo ""

grep "$KEYWORD" 10K静态动作描述.md | sed 's/^[0-9]*\. //' > $OUTPUT

COUNT=$(wc -l < $OUTPUT)
echo "✅ 提取了 $COUNT 条包含 '$KEYWORD' 的描述"
echo "📁 保存到: $OUTPUT"
echo ""

if [ $COUNT -gt 0 ]; then
    echo "📋 预览前10条:"
    head -10 $OUTPUT | nl
    echo ""
    
    read -p "是否立即生成这些姿态？(y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 开始生成..."
        python sample_single_pose.py \
            --text_path $OUTPUT \
            --dataset_name t2m \
            --repeat_times 2 \
            --save_image
    else
        echo "💡 稍后可以使用以下命令生成:"
        echo "   python sample_single_pose.py --text_path $OUTPUT --save_image"
    fi
else
    echo "❌ 未找到包含 '$KEYWORD' 的描述"
fi

echo ""
echo "=========================================="


