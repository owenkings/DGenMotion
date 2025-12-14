#!/bin/bash
# 在远程服务器执行此脚本启动 TensorBoard

cd /data/tiany/MARDM

# 检查端口是否被占用
if lsof -Pi :6006 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口 6006 已被占用，正在终止旧进程..."
    pkill -f "tensorboard.*6006"
    sleep 2
fi

# 启动 TensorBoard
echo "🚀 启动 TensorBoard..."
nohup tensorboard \
    --logdir=checkpoints/t2m/FSQ_MARDM_DiT_XL/model \
    --port=6006 \
    --host=0.0.0.0 \
    > /tmp/tensorboard.log 2>&1 &

sleep 2

# 检查是否启动成功
if lsof -Pi :6006 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✅ TensorBoard 已成功启动"
    echo ""
    echo "📊 在本地机器执行以下命令建立 SSH 隧道:"
    echo "   ssh -L 6006:localhost:6006 $(whoami)@$(hostname -I | awk '{print $1}')"
    echo ""
    echo "   或者如果已知主机名:"
    echo "   ssh -L 6006:localhost:6006 $(whoami)@$(hostname)"
    echo ""
    echo "🌐 然后在本地浏览器访问: http://localhost:6006"
    echo ""
    echo "📝 查看日志: tail -f /tmp/tensorboard.log"
    echo "🛑 停止服务: pkill -f 'tensorboard.*6006'"
else
    echo "❌ TensorBoard 启动失败，查看日志: cat /tmp/tensorboard.log"
fi
