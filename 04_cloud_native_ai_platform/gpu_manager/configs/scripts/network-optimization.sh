#!/bin/bash
# GPU 网络优化配置脚本
# 用于优化 GPU 服务器的网络性能

echo "正在配置 GPU 网络优化参数..."

# 设置高性能网络参数
echo "配置高性能网络参数..."
sysctl -w net.core.rmem_max=268435456
sysctl -w net.core.wmem_max=268435456
sysctl -w net.core.rmem_default=268435456
sysctl -w net.core.wmem_default=268435456
sysctl -w net.ipv4.tcp_rmem="4096 87380 268435456"
sysctl -w net.ipv4.tcp_wmem="4096 65536 268435456"

# 启用 RDMA 优化（如果可用）
if command -v ibstat &> /dev/null; then
    echo "检测到 InfiniBand/RDMA，启用优化配置..."
    sysctl -w net.ipv4.tcp_mtu_probing=1
    sysctl -w net.ipv4.tcp_window_scaling=1
fi

echo "网络优化配置完成！"
echo "当前网络参数："
sysctl net.core.rmem_max net.core.wmem_max net.ipv4.tcp_rmem net.ipv4.tcp_wmem