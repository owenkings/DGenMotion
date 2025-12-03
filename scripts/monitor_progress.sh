#!/bin/bash
# 监控10K生成进度

FINAL_DIR="./SinglePose"

echo "=========================================="
echo "  10K姿态生成进度监控"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "  实时进度监控"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # 统计文件数量
    NPY_COUNT=$(find $FINAL_DIR -name "*.npy" 2>/dev/null | wc -l)
    JSON_COUNT=$(find $FINAL_DIR -name "*.json" 2>/dev/null | wc -l)
    PNG_COUNT=$(find $FINAL_DIR -name "*.png" 2>/dev/null | wc -l)
    
    TOTAL_EXPECTED=10000
    PROGRESS=$((NPY_COUNT * 100 / TOTAL_EXPECTED))
    
    echo "📊 生成进度:"
    echo "   NPY文件:  $NPY_COUNT / $TOTAL_EXPECTED ($PROGRESS%)"
    echo "   JSON文件: $JSON_COUNT / $TOTAL_EXPECTED"
    echo "   PNG文件:  $PNG_COUNT / $TOTAL_EXPECTED"
    echo ""
    
    # 进度条
    BAR_LENGTH=50
    FILLED=$((PROGRESS * BAR_LENGTH / 100))
    EMPTY=$((BAR_LENGTH - FILLED))
    printf "   ["
    printf "%${FILLED}s" | tr ' ' '='
    printf "%${EMPTY}s" | tr ' ' '-'
    printf "] %d%%\n" $PROGRESS
    echo ""
    
    # GPU进程状态
    echo "🖥️  GPU进程状态:"
    GPU_PROCS=$(ps aux | grep "batch_generate_10k.py" | grep -v grep | wc -l)
    if [ $GPU_PROCS -gt 0 ]; then
        echo "   运行中: $GPU_PROCS 个进程"
        ps aux | grep "batch_generate_10k.py" | grep -v grep | awk '{printf "   PID %s: GPU %s\n", $2, $NF}'
    else
        echo "   ⚠️  没有运行中的进程"
    fi
    echo ""
    
    # 最近生成的文件
    echo "📝 最近生成的5个文件:"
    find $FINAL_DIR -name "*.npy" -type f -printf '%T@ %p\n' 2>/dev/null | \
        sort -rn | head -5 | cut -d' ' -f2- | \
        xargs -I {} basename {} .npy | head -5 | nl
    echo ""
    
    # 磁盘使用
    echo "💾 磁盘使用:"
    if [ -d "$FINAL_DIR" ]; then
        du -sh $FINAL_DIR 2>/dev/null
    fi
    echo ""
    
    # 预计剩余时间（粗略估计）
    if [ $NPY_COUNT -gt 100 ]; then
        # 假设每个姿态平均5秒，3个GPU并行
        REMAINING=$((TOTAL_EXPECTED - NPY_COUNT))
        SECONDS_LEFT=$((REMAINING * 5 / 3))
        HOURS=$((SECONDS_LEFT / 3600))
        MINUTES=$(((SECONDS_LEFT % 3600) / 60))
        echo "⏱️  预计剩余时间: 约 ${HOURS}小时 ${MINUTES}分钟"
        echo ""
    fi
    
    echo "=========================================="
    echo "按 Ctrl+C 退出监控"
    echo "=========================================="
    
    # 每10秒刷新一次
    sleep 10
done


