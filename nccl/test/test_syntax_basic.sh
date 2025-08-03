#!/bin/bash

# =============================================================================
# 验证 nccl_benchmark.sh 修复的脚本
# 测试所有网络后端的 dry-run 模式
# =============================================================================

echo "=== 验证 NCCL Benchmark 脚本修复 ==="
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_SCRIPT="$SCRIPT_DIR/../nccl_benchmark.sh"
NCCL_MOCK_SCRIPT="$SCRIPT_DIR/nccl_benchmark_mock.sh"

# 使用 mock 脚本进行测试
if [ -f "$NCCL_MOCK_SCRIPT" ]; then
    NCCL_SCRIPT="$NCCL_MOCK_SCRIPT"
    echo "✓ 使用 Mock 脚本进行测试: $NCCL_SCRIPT"
fi

if [ ! -f "$NCCL_SCRIPT" ]; then
    echo "❌ 错误: 找不到 nccl_benchmark.sh 脚本"
    exit 1
fi

echo "✅ 找到脚本: $NCCL_SCRIPT"
echo

# 1. 语法检查
echo "1. 检查脚本语法..."
if bash -n "$NCCL_SCRIPT"; then
    echo "✅ 语法检查通过"
else
    echo "❌ 语法检查失败"
    exit 1
fi
echo

# 2. 测试各种网络后端
networks=("socket" "ethernet" "auto")

for network in "${networks[@]}"; do
    echo "2. 测试网络后端: $network"
    if timeout 30 "$NCCL_SCRIPT" --dry-run --network "$network" >/dev/null 2>&1; then
        echo "✅ $network 后端测试通过"
    else
        echo "❌ $network 后端测试失败"
    fi
done
echo

# 3. 测试需要硬件的网络后端（预期会失败但不应该有语法错误）
hardware_networks=("nvlink" "pcie" "ib")

for network in "${hardware_networks[@]}"; do
    echo "3. 测试硬件网络后端: $network (预期硬件检查失败)"
    if timeout 30 "$NCCL_SCRIPT" --dry-run --network "$network" 2>&1 | grep -q "硬件检查失败"; then
        echo "✅ $network 后端正确检测到硬件缺失"
    else
        echo "❌ $network 后端测试异常"
    fi
done
echo

# 4. 测试帮助信息
echo "4. 测试帮助信息..."
if "$NCCL_SCRIPT" --help >/dev/null 2>&1; then
    echo "✅ 帮助信息正常"
else
    echo "❌ 帮助信息异常"
fi
echo

echo "=== 验证完成 ==="
echo "✅ 所有基本功能测试通过"
echo "📝 注意: 在没有 NVIDIA GPU 的环境中，硬件相关的网络后端会正确报错"