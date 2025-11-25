#!/bin/bash
# CUDA MPS 多进程服务配置脚本
# 用于配置和管理 NVIDIA MPS 服务

echo "正在配置 CUDA MPS 多进程服务..."

# 检查 NVIDIA 驱动是否安装
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: NVIDIA 驱动未安装或未正确配置"
    exit 1
fi

# 停止现有的 MPS 服务（如果正在运行）
echo "停止现有 MPS 服务..."
sudo nvidia-smi -i 0 -c DEFAULT
sudo timeout 10 nvidia-cuda-mps-control -d

# 启用 MPS 模式
echo "启用 MPS 模式..."
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# 启动 MPS 服务
echo "启动 MPS 服务..."
sudo nvidia-cuda-mps-control -d

# 配置 MPS 资源限制
echo "配置 MPS 资源限制..."
echo "limit_compute_units 8" | nvidia-cuda-mps-control
echo "limit_memory 1" | nvidia-cuda-mps-control

# 验证 MPS 状态
echo "验证 MPS 服务状态..."
nvidia-smi -i 0 -q | grep -A 5 -B 5 "MPS"

echo "CUDA MPS 配置完成！"
echo "使用 'nvidia-cuda-mps-control' 管理 MPS 服务"