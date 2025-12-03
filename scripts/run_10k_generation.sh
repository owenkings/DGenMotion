#!/bin/bash
# 批量生成10K静态姿态 - 多GPU并行
# 将10K数据分成3份，分别在GPU 1、2、3上运行

echo "=========================================="
echo "  10K静态姿态批量生成系统"
echo "=========================================="
echo ""

# 配置
DATASET_FILE="10K静态动作描述.md"
TEMP_DIR="./SinglePose_temp"
FINAL_DIR="./SinglePose"
PYTHON="/data/conda-envs/MARDM/bin/python"

# 创建目录
mkdir -p $TEMP_DIR $FINAL_DIR
mkdir -p ${TEMP_DIR}/gpu1 ${TEMP_DIR}/gpu2 ${TEMP_DIR}/gpu3
mkdir -p split_data

echo "📁 创建目录结构..."
echo "   临时目录: $TEMP_DIR"
echo "   最终目录: $FINAL_DIR"
echo ""

# 提取所有描述（跳过文件头）
echo "📝 提取10K条描述..."
sed -n '28,10027p' $DATASET_FILE | sed 's/^[0-9]*\. //' > all_10k_descriptions.txt

TOTAL_LINES=$(wc -l < all_10k_descriptions.txt)
echo "   总计: $TOTAL_LINES 条描述"
echo ""

# 计算每份的大小
LINES_PER_GPU=$((TOTAL_LINES / 3))
REMAINDER=$((TOTAL_LINES % 3))

echo "📊 数据分配:"
echo "   GPU 1: $LINES_PER_GPU 条"
echo "   GPU 2: $LINES_PER_GPU 条"
echo "   GPU 3: $((LINES_PER_GPU + REMAINDER)) 条"
echo ""

# 分割文件
echo "✂️  分割数据文件..."
head -n $LINES_PER_GPU all_10k_descriptions.txt > split_data/gpu1_data.txt
tail -n +$((LINES_PER_GPU + 1)) all_10k_descriptions.txt | head -n $LINES_PER_GPU > split_data/gpu2_data.txt
tail -n +$((LINES_PER_GPU * 2 + 1)) all_10k_descriptions.txt > split_data/gpu3_data.txt

echo "   GPU 1 数据: $(wc -l < split_data/gpu1_data.txt) 条"
echo "   GPU 2 数据: $(wc -l < split_data/gpu2_data.txt) 条"
echo "   GPU 3 数据: $(wc -l < split_data/gpu3_data.txt) 条"
echo ""

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "🚀 开始生成..."
echo "=========================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 在GPU 1、2、3上并行运行
echo "启动 GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON batch_generate_10k.py \
    --text_path split_data/gpu1_data.txt \
    --temp_dir ${TEMP_DIR}/gpu1 \
    --final_dir $FINAL_DIR \
    --gpu_id 0 \
    --batch_size 8 \
    --sequence_length 16 \
    > logs/gpu1.log 2>&1 &
PID1=$!
echo "   PID: $PID1"

echo "启动 GPU 2..."
CUDA_VISIBLE_DEVICES=2 $PYTHON batch_generate_10k.py \
    --text_path split_data/gpu2_data.txt \
    --temp_dir ${TEMP_DIR}/gpu2 \
    --final_dir $FINAL_DIR \
    --gpu_id 0 \
    --batch_size 8 \
    --sequence_length 16 \
    > logs/gpu2.log 2>&1 &
PID2=$!
echo "   PID: $PID2"

echo "启动 GPU 3..."
CUDA_VISIBLE_DEVICES=3 $PYTHON batch_generate_10k.py \
    --text_path split_data/gpu3_data.txt \
    --temp_dir ${TEMP_DIR}/gpu3 \
    --final_dir $FINAL_DIR \
    --gpu_id 0 \
    --batch_size 8 \
    --sequence_length 16 \
    > logs/gpu3.log 2>&1 &
PID3=$!
echo "   PID: $PID3"

echo ""
echo "✅ 所有GPU任务已启动"
echo ""
echo "📊 监控命令:"
echo "   查看GPU 1进度: tail -f logs/gpu1.log"
echo "   查看GPU 2进度: tail -f logs/gpu2.log"
echo "   查看GPU 3进度: tail -f logs/gpu3.log"
echo "   查看所有进度: tail -f logs/*.log"
echo ""
echo "🔍 检查进程:"
echo "   ps aux | grep batch_generate_10k.py"
echo ""
echo "💾 检查输出:"
echo "   ls -lh $FINAL_DIR | wc -l"
echo ""

# 等待所有进程完成
echo "⏳ 等待所有GPU任务完成..."
echo "   (这可能需要几个小时，请耐心等待)"
echo ""

wait $PID1
STATUS1=$?
echo "✅ GPU 1 完成 (退出码: $STATUS1)"

wait $PID2
STATUS2=$?
echo "✅ GPU 2 完成 (退出码: $STATUS2)"

wait $PID3
STATUS3=$?
echo "✅ GPU 3 完成 (退出码: $STATUS3)"

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "🎉 所有任务完成！"
echo "=========================================="
echo ""
echo "⏱️  总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""

# 统计结果
echo "📊 生成统计:"
NPY_COUNT=$(find $FINAL_DIR -name "*.npy" | wc -l)
JSON_COUNT=$(find $FINAL_DIR -name "*.json" | wc -l)
PNG_COUNT=$(find $FINAL_DIR -name "*.png" | wc -l)

echo "   NPY文件:  $NPY_COUNT 个"
echo "   JSON文件: $JSON_COUNT 个"
echo "   PNG文件:  $PNG_COUNT 个"
echo ""

# 检查是否有失败
if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ] || [ $STATUS3 -ne 0 ]; then
    echo "⚠️  警告: 部分GPU任务失败"
    echo "   GPU 1: $([ $STATUS1 -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
    echo "   GPU 2: $([ $STATUS2 -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
    echo "   GPU 3: $([ $STATUS3 -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
    echo ""
    echo "   请查看日志文件获取详细信息:"
    echo "   logs/gpu1.log, logs/gpu2.log, logs/gpu3.log"
fi

echo "📁 结果目录:"
echo "   最终结果: $FINAL_DIR"
echo "   临时文件: $TEMP_DIR"
echo ""

echo "💡 查看结果:"
echo "   ls $FINAL_DIR | head -20"
echo ""

# 显示磁盘使用
echo "💾 磁盘使用:"
du -sh $FINAL_DIR
du -sh $TEMP_DIR
echo ""

echo "=========================================="
echo "✅ 完成！"
echo "=========================================="


