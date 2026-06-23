#!/usr/bin/env bash
# NCCL Benchmark 配置管理器专项测试

set -uo pipefail

PASSED=0; FAILED=0
pass() { PASSED=$((PASSED + 1)); echo -e "\033[0;32m[PASS]\033[0m $1"; }
fail() { FAILED=$((FAILED + 1)); echo -e "\033[0;31m[FAIL]\033[0m $1"; }

# ── 被测试的配置函数（与 nccl_benchmark.sh 中一致）──
set_nccl_config() {
    local key="$1"; local value="$2"
    [ -z "$key" ] || [ -z "$value" ] && return 1
    export "NCCL_$key=$value"
    return 0
}

setup_network_config() {
    local network="${1:-auto}"
    case "$network" in
        ib|IB|ib_enable)      set_nccl_config "IB_DISABLE" "0"; set_nccl_config "NET" "IB" ;;
        ib_disable)            set_nccl_config "IB_DISABLE" "1" ;;
        nvlink|p2p_nvlink)    set_nccl_config "P2P_LEVEL" "NVL"; set_nccl_config "IB_DISABLE" "1"; set_nccl_config "NVLS_ENABLE" "1" ;;
        pcie|p2p_pcie)        set_nccl_config "P2P_LEVEL" "PIX"; set_nccl_config "NVLS_ENABLE" "0" ;;
        p2p_disable)          set_nccl_config "P2P_DISABLE" "1" ;;
        socket|socket_only)   set_nccl_config "P2P_DISABLE" "1"; set_nccl_config "IB_DISABLE" "1" ;;
        auto) : ;;
        *) return 1 ;;
    esac
    return 0
}

cache_system_info() {
    local gpu_count=0; local nvlink_count=0; local ib_available="false"
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    elif [ -n "${MOCK_GPU_COUNT:-}" ]; then
        gpu_count="$MOCK_GPU_COUNT"
    fi
    if [ -n "${MOCK_NVLINK_COUNT:-}" ] && [ "$MOCK_NVLINK_COUNT" -gt 0 ]; then
        nvlink_count="$MOCK_NVLINK_COUNT"
    elif [ "$gpu_count" -gt 1 ]; then
        nvlink_count=$(nvidia-smi nvlink --status 2>/dev/null | grep -c "GB/s" 2>/dev/null | tr -d ' \n\r\t' || echo "0")
        [ -z "$nvlink_count" ] && nvlink_count=0
    fi
    [ "${MOCK_IB_AVAILABLE:-false}" = "true" ] && ib_available="true"
    export SYSTEM_INFO_GPU_COUNT="$gpu_count"
    export SYSTEM_INFO_NVLINK_COUNT="$nvlink_count"
    export SYSTEM_INFO_IB_AVAILABLE="$ib_available"
    return 0
}

echo "=== 配置管理器测试 ==="

# ── 1. set_nccl_config ──
echo ""; echo "--- set_nccl_config ---"
set_nccl_config "DEBUG" "INFO" && [ "$NCCL_DEBUG" = "INFO" ] && pass "set_nccl_config 基本功能" || fail "set_nccl_config 基本功能"
set_nccl_config "IB_DISABLE" "1" && [ "$NCCL_IB_DISABLE" = "1" ] && pass "set_nccl_config 第二次调用" || fail "set_nccl_config 第二次调用"
set_nccl_config "" "val" && fail "空 key 应返回失败" || pass "空 key 正确拒绝"
set_nccl_config "KEY" "" && fail "空 value 应返回失败" || pass "空 value 正确拒绝"

# ── 2. setup_network_config ──
echo ""; echo "--- setup_network_config ---"

test_network() {
    local preset="$1" var="$2" expected="$3"
    unset "$var" 2>/dev/null || true
    setup_network_config "$preset"
    [ "${!var:-}" = "$expected" ] && pass "$preset: $var=$expected" || fail "$preset: $var 期望=$expected 实际=${!var:-}"
}

test_network "ib_enable"   "NCCL_IB_DISABLE" "0"
test_network "ib_disable"  "NCCL_IB_DISABLE" "1"
test_network "p2p_nvlink"  "NCCL_P2P_LEVEL"  "NVL"
test_network "p2p_nvlink"  "NCCL_IB_DISABLE" "1"
test_network "p2p_pcie"    "NCCL_P2P_LEVEL"  "PIX"
test_network "p2p_disable" "NCCL_P2P_DISABLE" "1"
test_network "socket_only" "NCCL_P2P_DISABLE" "1"
test_network "socket_only" "NCCL_IB_DISABLE" "1"
test_network "auto"        "NCCL_P2P_LEVEL"  ""   # auto 不做设置
setup_network_config "auto" && pass "auto 返回成功" || fail "auto 返回失败"
setup_network_config "invalid_backend" && fail "无效后端应返回失败" || pass "无效后端正确拒绝"

# ── 3. cache_system_info ──
echo ""; echo "--- cache_system_info ---"
cache_system_info
[ -n "${SYSTEM_INFO_GPU_COUNT:-}" ] && pass "cache_system_info: gpu_count 已设置" || fail "cache_system_info: gpu_count 未设置"
[ -n "${SYSTEM_INFO_NVLINK_COUNT:-}" ] && pass "cache_system_info: nvlink_count 已设置" || fail "cache_system_info: nvlink_count 未设置"

# ── 4. 带 Mock 环境的 cache ──
MOCK_GPU_COUNT=8 MOCK_NVLINK_COUNT=18 MOCK_IB_AVAILABLE=true cache_system_info
[ "${SYSTEM_INFO_GPU_COUNT}" = "8" ] && pass "Mock 8 GPU 正确" || fail "Mock 8 GPU: 实际=${SYSTEM_INFO_GPU_COUNT}"
[ "${SYSTEM_INFO_NVLINK_COUNT}" = "18" ] && pass "Mock 18 NVLink 正确" || fail "Mock 18 NVLink: 实际=${SYSTEM_INFO_NVLINK_COUNT}"
[ "${SYSTEM_INFO_IB_AVAILABLE}" = "true" ] && pass "Mock IB 可用正确" || fail "Mock IB: 实际=${SYSTEM_INFO_IB_AVAILABLE}"

# ── 报告 ──
echo ""; echo "=== 结果: $PASSED 通过, $FAILED 失败 ==="
[ "$FAILED" -eq 0 ] && exit 0 || exit 1
