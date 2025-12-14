# 在本地查看远程 TensorBoard 指南

## 方法 1: SSH 端口转发（推荐）

### 步骤 1: 建立 SSH 隧道

在**本地机器**的终端执行：

```bash
ssh -L 6006:localhost:6006 username@remote_host
```

**示例**（根据你的实际情况修改）：
```bash
ssh -L 6006:localhost:6006 szya800@szy
```

或者如果你已经在 SSH 连接中，可以在**新的本地终端**窗口执行：

```bash
ssh -L 6006:localhost:6006 szya800@your_remote_ip
```

### 步骤 2: 在远程服务器启动 TensorBoard

在 SSH 连接的远程终端中：

```bash
cd /data/tiany/MARDM
tensorboard --logdir=checkpoints/t2m/FSQ_MARDM_DiT_XL/model --port=6006 --host=0.0.0.0
```

**注意**: `--host=0.0.0.0` 允许外部访问（通过 SSH 隧道）

### 步骤 3: 在本地浏览器访问

保持 SSH 隧道连接，然后在**本地浏览器**打开：

```
http://localhost:6006
```

---

## 方法 2: 后台运行 SSH 隧道（推荐用于长期使用）

### 在本地机器执行：

```bash
# 建立后台 SSH 隧道
ssh -f -N -L 6006:localhost:6006 username@remote_host

# 或者更详细的版本（包含日志）
ssh -f -N -L 6006:localhost:6006 username@remote_host -v > /tmp/ssh_tunnel.log 2>&1
```

**参数说明**：
- `-f`: 后台运行
- `-N`: 不执行远程命令，只做端口转发
- `-L 6006:localhost:6006`: 本地端口转发

### 查看隧道状态：

```bash
# 检查端口是否在监听
lsof -i :6006
# 或
netstat -an | grep 6006
```

### 停止隧道：

```bash
# 查找进程
ps aux | grep "ssh.*6006"

# 终止进程
pkill -f "ssh.*6006"
```

---

## 方法 3: 使用 VS Code 端口转发（最简单）

如果你使用 VS Code 连接远程服务器：

1. **VS Code 会自动检测** TensorBoard 在 6006 端口运行
2. 点击 VS Code 底部状态栏的**端口转发提示**
3. 或者手动添加：
   - 按 `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`)
   - 输入 "Forward a Port"
   - 输入端口号 `6006`
4. 点击生成的链接，自动在浏览器打开

---

## 方法 4: 使用 Jupyter Notebook 端口转发

如果你使用 Jupyter：

```bash
# 在远程服务器
jupyter notebook --port=8888 --no-browser --allow-root

# 在本地建立隧道
ssh -L 8888:localhost:8888 username@remote_host

# 然后在 Jupyter 中运行 TensorBoard
```

---

## 完整示例脚本

### 远程服务器端脚本 (`start_tensorboard.sh`)

```bash
#!/bin/bash
# 在远程服务器执行

cd /data/tiany/MARDM

# 检查端口是否被占用
if lsof -Pi :6006 -sTCP:LISTEN -t >/dev/null ; then
    echo "端口 6006 已被占用，正在终止旧进程..."
    pkill -f "tensorboard.*6006"
    sleep 2
fi

# 启动 TensorBoard
echo "启动 TensorBoard..."
nohup tensorboard \
    --logdir=checkpoints/t2m/FSQ_MARDM_DiT_XL/model \
    --port=6006 \
    --host=0.0.0.0 \
    > /tmp/tensorboard.log 2>&1 &

echo "TensorBoard 已启动"
echo "日志: tail -f /tmp/tensorboard.log"
echo ""
echo "在本地机器执行: ssh -L 6006:localhost:6006 username@remote_host"
echo "然后访问: http://localhost:6006"
```

### 本地机器脚本 (`connect_tensorboard.sh`)

```bash
#!/bin/bash
# 在本地机器执行

REMOTE_HOST="your_remote_host"  # 修改为你的远程主机
REMOTE_USER="szya800"            # 修改为你的用户名

echo "建立 SSH 隧道到 $REMOTE_USER@$REMOTE_HOST:6006"
echo "在浏览器访问: http://localhost:6006"
echo ""
echo "按 Ctrl+C 停止隧道"
echo ""

ssh -L 6006:localhost:6006 $REMOTE_USER@$REMOTE_HOST
```

---

## 故障排查

### 问题 1: 连接被拒绝

**原因**: TensorBoard 没有绑定到 `0.0.0.0`

**解决**: 启动时添加 `--host=0.0.0.0`

```bash
tensorboard --logdir=... --port=6006 --host=0.0.0.0
```

### 问题 2: 端口已被占用

**解决**: 使用其他端口

```bash
# 远程
tensorboard --logdir=... --port=6007 --host=0.0.0.0

# 本地隧道
ssh -L 6007:localhost:6007 username@remote_host
```

### 问题 3: 防火墙阻止

**解决**: 检查远程服务器防火墙设置，或使用 VPN

### 问题 4: SSH 配置优化

在 `~/.ssh/config` 中添加：

```
Host your_remote_alias
    HostName your_remote_ip
    User szya800
    LocalForward 6006 localhost:6006
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

然后直接 `ssh your_remote_alias` 即可自动建立端口转发。

---

## 快速命令参考

```bash
# 1. 在远程服务器启动 TensorBoard
cd /data/tiany/MARDM
tensorboard --logdir=checkpoints/t2m/FSQ_MARDM_DiT_XL/model --port=6006 --host=0.0.0.0

# 2. 在本地机器建立隧道（新终端窗口）
ssh -L 6006:localhost:6006 szya800@your_remote_ip

# 3. 在本地浏览器访问
# http://localhost:6006
```

---

**推荐**: 使用方法 2（后台 SSH 隧道）+ VS Code 端口转发，最简单可靠。

