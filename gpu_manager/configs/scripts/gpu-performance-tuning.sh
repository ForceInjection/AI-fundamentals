#!/bin/bash
# GPU 性能调优脚本
# 用于优化 GPU 性能和资源利用率

echo "正在执行 GPU 性能调优..."

# 检查 NVIDIA 驱动
echo "检查 NVIDIA 驱动状态..."
nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv

# 设置持久模式
echo "启用 GPU 持久模式..."
sudo nvidia-smi -pm 1

# 优化时钟频率（根据 GPU 型号调整）
echo "优化 GPU 时钟频率..."
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader)

case $GPU_MODEL in
    *"A100"*)
        sudo nvidia-smi -ac 1215,1410
        ;;
    *"V100"*)
        sudo nvidia-smi -ac 877,1530
        ;;
    *"H100"*)
        sudo nvidia-smi -ac 1350,1600
        ;;
    *)
        echo "使用默认时钟频率配置"
        ;;
esac

# 内存优化配置
echo "配置内存优化参数..."
sudo nvidia-smi -c 3  # 启用计算模式

# 功耗管理（可选）
echo "配置功耗管理..."
sudo nvidia-smi -pl 250  # 设置功耗限制为 250W

# 显示优化结果
echo "性能调优完成！当前配置："
nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.limit --format=csv

echo "使用 'nvidia-smi -q -d PERFORMANCE' 查看详细性能信息"