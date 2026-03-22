#!/usr/bin/env bash

# =============================================================================
# 验证 nccl_benchmark.sh 语法正确性
# 功能: 仅进行静态语法检查，不执行任何实际功能测试
# =============================================================================

echo "=== 验证 NCCL Benchmark 脚本语法 ==="
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_SCRIPT="$SCRIPT_DIR/../nccl_benchmark.sh"

if [ ! -f "$NCCL_SCRIPT" ]; then
    echo "❌ 错误: 找不到 nccl_benchmark.sh 脚本"
    exit 1
fi

echo "✅ 找到脚本: $NCCL_SCRIPT"
echo

# 1. 语法检查
echo "1. 检查脚本语法 (bash -n)..."
if bash -n "$NCCL_SCRIPT"; then
    echo "✅ 语法检查通过"
else
    echo "❌ 语法检查失败"
    exit 1
fi
echo

# 2. 检查可执行权限
echo "2. 检查可执行权限..."
if [ -x "$NCCL_SCRIPT" ]; then
    echo "✅ 脚本具有可执行权限"
else
    echo "⚠️ 脚本没有可执行权限 (建议运行: chmod +x nccl_benchmark.sh)"
fi
echo

# 3. 检查 Shebang
echo "3. 检查 Shebang..."
# 使用单引号避免 bash 历史展开错误 (!/...)
if head -n 1 "$NCCL_SCRIPT" | grep -q '^#!/usr/bin/env bash'; then
    echo '✅ Shebang 正确 (#!/usr/bin/env bash)'
elif head -n 1 "$NCCL_SCRIPT" | grep -q '^#!/bin/bash'; then
    echo '✅ Shebang 正确 (#!/bin/bash)'
else
    echo "⚠️ Shebang 可能不规范"
    head -n 1 "$NCCL_SCRIPT"
fi
echo

echo "=== 验证完成 ==="
echo "✅ 静态检查全部通过"
