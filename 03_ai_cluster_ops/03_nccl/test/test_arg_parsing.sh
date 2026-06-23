#!/usr/bin/env bash
# nccl_benchmark.sh 参数解析回归测试
# 不检查退出码（环境依赖如 PyTorch 可能导致非零退出），只检查无参数解析错误

set -uo nounset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$SCRIPT_DIR/../nccl_benchmark.sh"

PASSED=0; FAILED=0
pass() { PASSED=$((PASSED + 1)); echo -e "\033[0;32m[PASS]\033[0m $1"; }
fail() { FAILED=$((FAILED + 1)); echo -e "\033[0;31m[FAIL]\033[0m $1"; }

echo "=== 参数解析测试 ==="
[ ! -f "$S" ] && { echo "脚本不存在: $S"; exit 1; }

# 参数解析错误的关键词（出现这些说明 flag 没被正确识别）
ERR_PATTERN="未知选项|unrecognized|invalid option"

# --help 必须成功
bash "$S" --help 2>&1 | grep -q "用法\|Usage\|帮助" && pass "--help" || fail "--help"

# 合法参数组合：不应出现参数解析错误
for combo in \
    "--size 1G --dry-run" \
    "--size 100M --dry-run" \
    "--time 10 --dry-run" \
    "--dry-run" \
    "--network auto --dry-run" \
    "--network nvlink --dry-run" \
    "--network ib --dry-run" \
    "--network pxn --dry-run"; do
    out=$(bash "$S" $combo 2>&1)
    if echo "$out" | grep -qiE "$ERR_PATTERN"; then
        fail "$combo — 参数解析错误"
    else
        pass "$combo"
    fi
done

# --size 缺参数：应该报错
out=$(bash "$S" --size 2>&1) || true
echo "$out" | grep -qi "需要参数" && pass "--size 缺参数报错" || fail "--size 缺参数报错"

# 无效网络后端：应该报错
out=$(bash "$S" --network invalid_xyz --dry-run 2>&1) || true
echo "$out" | grep -qi "无效" && pass "--network 无效后端报错" || fail "--network 无效后端报错"

echo ""; echo "=== 结果: $PASSED 通过, $FAILED 失败 ==="
[ "$FAILED" -eq 0 ] && exit 0 || exit 1
