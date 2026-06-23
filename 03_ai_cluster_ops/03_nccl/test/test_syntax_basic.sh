#!/usr/bin/env bash
# nccl_benchmark.sh 语法检查

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/../nccl_benchmark.sh"

echo "=== 语法检查 ==="
[ ! -f "$SCRIPT" ] && { echo "[FAIL] 脚本不存在: $SCRIPT"; exit 1; }

if bash -n "$SCRIPT" 2>&1; then
    echo "[PASS] bash -n 语法检查通过"
else
    echo "[FAIL] bash -n 语法检查失败"
    exit 1
fi
