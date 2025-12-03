#!/bin/bash
# 单帧姿态生成 - 各种使用示例

echo "=========================================="
echo "  单帧3D姿态生成 - 使用示例集合"
echo "=========================================="
echo ""

# ============================================
# 示例1: 基础用法 - 生成举手姿态
# ============================================
echo "📝 示例1: 基础用法 - 生成举手姿态"
python sample_single_pose.py \
    --text_prompt "A person raising both hands" \
    --dataset_name t2m \
    --save_image

echo ""
echo "✅ 示例1完成！查看: generation/MARDM_t2m_single_pose/0/"
echo ""
read -p "按Enter继续下一个示例..."
echo ""

# ============================================
# 示例2: 生成多样性 - 同一动作的多个变体
# ============================================
echo "📝 示例2: 生成多样性 - 同一动作的5个不同版本"
python sample_single_pose.py \
    --text_prompt "A person waving hand" \
    --dataset_name t2m \
    --repeat_times 5 \
    --save_image \
    --save_json

echo ""
echo "✅ 示例2完成！生成了5个不同的挥手姿态"
echo ""
read -p "按Enter继续下一个示例..."
echo ""

# ============================================
# 示例3: 提取不同关键帧
# ============================================
echo "📝 示例3: 提取动作的不同关键帧"

echo "  → 第一帧（动作开始）"
python sample_single_pose.py \
    --text_prompt "A person jumping" \
    --dataset_name t2m \
    --frame_index 0 \
    --save_image

echo "  → 中间帧（动作高潮）"
python sample_single_pose.py \
    --text_prompt "A person jumping" \
    --dataset_name t2m \
    --frame_index -1 \
    --save_image

echo "  → 最后一帧（动作结束）"
python sample_single_pose.py \
    --text_prompt "A person jumping" \
    --dataset_name t2m \
    --frame_index -2 \
    --save_image

echo ""
echo "✅ 示例3完成！生成了跳跃动作的3个关键帧"
echo ""
read -p "按Enter继续下一个示例..."
echo ""

# ============================================
# 示例4: 批量生成多个不同姿态
# ============================================
echo "📝 示例4: 批量生成多个不同姿态"

# 创建文本文件
cat > /tmp/example_poses.txt << EOF
A person raising left hand
A person sitting on a chair
A person standing with arms crossed
A person kneeling down
A person pointing forward
EOF

echo "  文本文件内容:"
cat /tmp/example_poses.txt
echo ""

python sample_single_pose.py \
    --text_path /tmp/example_poses.txt \
    --dataset_name t2m \
    --repeat_times 2 \
    --save_image \
    --save_json

echo ""
echo "✅ 示例4完成！批量生成了5种不同姿态，每种2个变体"
echo ""
read -p "按Enter继续下一个示例..."
echo ""

# ============================================
# 示例5: 快速生成（短序列）
# ============================================
echo "📝 示例5: 快速生成模式（使用短序列）"
python sample_single_pose.py \
    --text_prompt "A person dancing" \
    --dataset_name t2m \
    --sequence_length 8 \
    --frame_index 4 \
    --repeat_times 5 \
    --save_image

echo ""
echo "✅ 示例5完成！使用8帧短序列快速生成"
echo ""
read -p "按Enter继续下一个示例..."
echo ""

# ============================================
# 示例6: 瑜伽/健身姿态
# ============================================
echo "📝 示例6: 瑜伽/健身姿态"

cat > /tmp/yoga_poses.txt << EOF
A person in tree pose
A person in warrior pose
A person stretching arms above head
A person in lunging position
EOF

python sample_single_pose.py \
    --text_path /tmp/yoga_poses.txt \
    --dataset_name t2m \
    --frame_index -1 \
    --repeat_times 3 \
    --save_image \
    --save_json

echo ""
echo "✅ 示例6完成！生成了4种瑜伽姿态"
echo ""

# ============================================
# 总结
# ============================================
echo ""
echo "=========================================="
echo "🎉 所有示例完成！"
echo "=========================================="
echo ""
echo "📁 生成的文件保存在:"
echo "   generation/MARDM_t2m_single_pose/"
echo ""
echo "📊 文件统计:"
find generation/MARDM_t2m_single_pose/ -name "*.npy" | wc -l | xargs echo "   NPY文件数量:"
find generation/MARDM_t2m_single_pose/ -name "*.json" | wc -l | xargs echo "   JSON文件数量:"
find generation/MARDM_t2m_single_pose/ -name "*.png" | wc -l | xargs echo "   PNG图片数量:"
echo ""
echo "🔍 查看示例结果:"
echo "   cd generation/MARDM_t2m_single_pose/"
echo "   ls -R"
echo ""
echo "💡 提示: 使用图片查看器打开PNG文件查看可视化结果"
echo "=========================================="

