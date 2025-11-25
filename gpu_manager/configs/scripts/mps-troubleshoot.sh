#!/bin/bash
# MPS 故障诊断脚本
# 用于诊断和修复 CUDA MPS 服务问题

echo "开始 MPS 故障诊断..."

# 检查 MPS 服务状态
echo "1. 检查 MPS 服务进程..."
MPS_PID=$(pgrep nvidia-cuda-mps)
if [ -n "$MPS_PID" ]; then
    echo "✓ MPS 服务正在运行 (PID: $MPS_PID)"
else
    echo "✗ MPS 服务未运行"
fi

# 检查 GPU 计算模式
echo "2. 检查 GPU 计算模式..."
COMPUTE_MODE=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader)
if [[ "$COMPUTE_MODE" == *"Exclusive_Process"* ]]; then
    echo "✓ GPU 处于独占进程模式"
else
    echo "✗ GPU 未处于独占进程模式: $COMPUTE_MODE"
fi

# 检查 MPS 控制连接
echo "3. 测试 MPS 控制连接..."
if timeout 5 nvidia-cuda-mps-control -d; then
    echo "✓ MPS 控制连接正常"
else
    echo "✗ MPS 控制连接失败"
fi

# 检查环境变量
echo "4. 检查 CUDA 环境变量..."
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "✓ CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    echo "⚠ CUDA_VISIBLE_DEVICES 未设置"
fi

# 检查客户端连接
echo "5. 检查 MPS 客户端..."
MPS_CLIENTS=$(ps aux | grep -i "cuda.*mps" | grep -v grep | wc -l)
if [ "$MPS_CLIENTS" -gt 0 ]; then
    echo "✓ 发现 $MPS_CLIENTS 个 MPS 客户端"
else
    echo "⚠ 未发现 MPS 客户端"
fi

# 常见问题修复
echo "6. 执行常见问题修复..."

# 重启 MPS 服务
echo "   重启 MPS 服务..."
sudo nvidia-smi -i 0 -c DEFAULT
sleep 2
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
sleep 2
sudo pkill -f nvidia-cuda-mps
sleep 2
sudo nvidia-cuda-mps-control -d

echo "MPS 故障诊断完成！"
echo "如果问题仍然存在，请检查:"
echo "  - NVIDIA 驱动版本"
echo "  - GPU 硬件兼容性"
echo "  - 系统日志 /var/log/syslog"