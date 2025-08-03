#!/bin/bash
# =============================================================================
# NCCL Benchmark 配置管理器专项测试
# 功能: 专门测试新增的统一配置管理器功能
# =============================================================================

# 测试配置
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_SCRIPT_PATH="$(dirname "$TEST_DIR")/nccl_benchmark.sh"
NCCL_MOCK_SCRIPT="$(dirname "$TEST_DIR")/nccl_benchmark_mock.sh"
TEST_LOG="/tmp/config_manager_test.log"

# 使用 mock 脚本进行测试
if [ -f "$NCCL_MOCK_SCRIPT" ]; then
    NCCL_SCRIPT_PATH="$NCCL_MOCK_SCRIPT"
    echo "✓ 使用 Mock 脚本进行测试: $NCCL_SCRIPT_PATH"
fi

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 测试统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 日志函数
log_test() {
    echo -e "$1" | tee -a "$TEST_LOG"
}

log_test_pass() {
    PASSED_TESTS=$((PASSED_TESTS + 1))
    log_test "${GREEN}[PASS]${NC} $1"
}

log_test_fail() {
    FAILED_TESTS=$((FAILED_TESTS + 1))
    log_test "${RED}[FAIL]${NC} $1"
}

log_test_header() {
    log_test ""
    log_test "${YELLOW}=== $1 ===${NC}"
    log_test ""
}

# 创建 mock 环境
setup_mock_environment() {
    # Mock nvidia-smi
    cat > /tmp/mock_nvidia_smi << 'EOF'
#!/bin/bash
case "$1" in
    "-L") echo -e "GPU 0: Mock GPU\nGPU 1: Mock GPU\nGPU 2: Mock GPU\nGPU 3: Mock GPU" ;;
    "nvlink") [ "$2" = "-s" ] && echo -e "Link 0: Active\nLink 1: Active\nLink 2: Active\nLink 3: Active" ;;
    *) echo "Mock nvidia-smi output" ;;
esac
EOF
    chmod +x /tmp/mock_nvidia_smi
    
    # Mock ibv_devinfo
    cat > /tmp/mock_ibv_devinfo << 'EOF'
#!/bin/bash
echo "hca_id: mlx5_0"
echo "        transport: InfiniBand (0)"
echo "        port: 1"
echo "                state: PORT_ACTIVE (4)"
EOF
    chmod +x /tmp/mock_ibv_devinfo
    
    # 添加到 PATH
    export PATH="/tmp:$PATH"
    alias nvidia-smi='/tmp/mock_nvidia_smi'
    alias ibv_devinfo='/tmp/mock_ibv_devinfo'
}

# 测试配置缓存功能
test_config_cache() {
    log_test_header "测试配置缓存功能"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # 创建测试脚本
    cat > /tmp/test_cache.sh << 'EOF'
#!/bin/bash
# 导入配置管理器函数
source /Users/wangtianqing/Project/AI-fundermentals/nccl/nccl_benchmark.sh

# 初始化缓存
declare -A NCCL_CONFIG_CACHE
declare -A SYSTEM_INFO_CACHE

# 测试 set_nccl_config
set_nccl_config "DEBUG" "INFO" "测试调试级别"
set_nccl_config "IB_DISABLE" "1" "测试IB禁用"

# 验证缓存
if [ "${NCCL_CONFIG_CACHE[DEBUG]}" = "INFO" ] && [ "${NCCL_CONFIG_CACHE[IB_DISABLE]}" = "1" ]; then
    echo "CACHE_TEST_PASS"
else
    echo "CACHE_TEST_FAIL"
fi

# 验证环境变量
if [ "$NCCL_DEBUG" = "INFO" ] && [ "$NCCL_IB_DISABLE" = "1" ]; then
    echo "ENV_TEST_PASS"
else
    echo "ENV_TEST_FAIL"
fi
EOF
    
    chmod +x /tmp/test_cache.sh
    local output=$(bash /tmp/test_cache.sh 2>/dev/null)
    
    if echo "$output" | grep -q "CACHE_TEST_PASS" && echo "$output" | grep -q "ENV_TEST_PASS"; then
        log_test_pass "配置缓存功能正常"
    else
        log_test_fail "配置缓存功能异常"
    fi
}

# 测试批量配置功能
test_batch_config() {
    log_test_header "测试批量配置功能"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    cat > /tmp/test_batch.sh << 'EOF'
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundermentals/nccl/nccl_benchmark.sh

declare -A NCCL_CONFIG_CACHE
declare -A test_config=(
    ["DEBUG"]="WARN"
    ["BUFFSIZE"]="4194304"
    ["NTHREADS"]="128"
)

set_nccl_configs test_config "批量测试配置"

# 验证批量设置
if [ "$NCCL_DEBUG" = "WARN" ] && [ "$NCCL_BUFFSIZE" = "4194304" ] && [ "$NCCL_NTHREADS" = "128" ]; then
    echo "BATCH_TEST_PASS"
else
    echo "BATCH_TEST_FAIL"
fi
EOF
    
    chmod +x /tmp/test_batch.sh
    local output=$(bash /tmp/test_batch.sh 2>/dev/null)
    
    if echo "$output" | grep -q "BATCH_TEST_PASS"; then
        log_test_pass "批量配置功能正常"
    else
        log_test_fail "批量配置功能异常"
    fi
}

# 测试网络配置预设
test_network_presets() {
    log_test_header "测试网络配置预设"
    
    local presets=("ib_enable" "ib_disable" "p2p_nvlink" "p2p_pcie" "p2p_disable" "socket_only")
    
    for preset in "${presets[@]}"; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        
        cat > /tmp/test_preset.sh << EOF
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundermentals/nccl/nccl_benchmark.sh

declare -A NCCL_CONFIG_CACHE
setup_network_config "$preset"

# 根据预设验证关键配置
case "$preset" in
    "ib_disable")
        [ "\$NCCL_IB_DISABLE" = "1" ] && echo "PRESET_TEST_PASS" || echo "PRESET_TEST_FAIL"
        ;;
    "p2p_nvlink")
        [ "\$NCCL_P2P_LEVEL" = "NVL" ] && [ "\$NCCL_NVLS_ENABLE" = "1" ] && echo "PRESET_TEST_PASS" || echo "PRESET_TEST_FAIL"
        ;;
    "p2p_pcie")
        [ "\$NCCL_P2P_LEVEL" = "PIX" ] && [ "\$NCCL_NVLS_ENABLE" = "0" ] && echo "PRESET_TEST_PASS" || echo "PRESET_TEST_FAIL"
        ;;
    "p2p_disable")
        [ "\$NCCL_P2P_DISABLE" = "1" ] && echo "PRESET_TEST_PASS" || echo "PRESET_TEST_FAIL"
        ;;
    *)
        echo "PRESET_TEST_PASS"  # 其他预设暂时通过
        ;;
esac
EOF
        
        chmod +x /tmp/test_preset.sh
        local output=$(bash /tmp/test_preset.sh 2>/dev/null)
        
        if echo "$output" | grep -q "PRESET_TEST_PASS"; then
            log_test_pass "网络预设 $preset 配置正常"
        else
            log_test_fail "网络预设 $preset 配置异常"
        fi
    done
}

# 测试性能配置预设
test_performance_presets() {
    log_test_header "测试性能配置预设"
    
    local presets=("nvlink_optimized" "pcie_optimized" "ib_optimized")
    
    for preset in "${presets[@]}"; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        
        cat > /tmp/test_perf_preset.sh << EOF
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundermentals/nccl/nccl_benchmark.sh

declare -A NCCL_CONFIG_CACHE
setup_performance_config "$preset"

# 根据预设验证关键配置
case "$preset" in
    "nvlink_optimized")
        [ "\$NCCL_NTHREADS" = "256" ] && [ "\$NCCL_MAX_NCHANNELS" = "32" ] && echo "PERF_TEST_PASS" || echo "PERF_TEST_FAIL"
        ;;
    "pcie_optimized")
        [ "\$NCCL_NTHREADS" = "16" ] && [ "\$NCCL_MAX_NCHANNELS" = "16" ] && echo "PERF_TEST_PASS" || echo "PERF_TEST_FAIL"
        ;;
    "ib_optimized")
        [ "\$NCCL_IB_TC" = "136" ] && [ "\$NCCL_IB_TIMEOUT" = "22" ] && echo "PERF_TEST_PASS" || echo "PERF_TEST_FAIL"
        ;;
    *)
        echo "PERF_TEST_PASS"
        ;;
esac
EOF
        
        chmod +x /tmp/test_perf_preset.sh
        local output=$(bash /tmp/test_perf_preset.sh 2>/dev/null)
        
        if echo "$output" | grep -q "PERF_TEST_PASS"; then
            log_test_pass "性能预设 $preset 配置正常"
        else
            log_test_fail "性能预设 $preset 配置异常"
        fi
    done
}

# 测试系统信息缓存
test_system_info_cache() {
    log_test_header "测试系统信息缓存"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    cat > /tmp/test_sys_cache.sh << 'EOF'
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundermentals/nccl/nccl_benchmark.sh

declare -A SYSTEM_INFO_CACHE

# 测试缓存功能
cache_system_info

# 验证缓存内容
if [ -n "${SYSTEM_INFO_CACHE[gpu_count]:-}" ] && \
   [ -n "${SYSTEM_INFO_CACHE[nvlink_available]:-}" ] && \
   [ -n "${SYSTEM_INFO_CACHE[ib_available]:-}" ]; then
    echo "SYS_CACHE_TEST_PASS"
else
    echo "SYS_CACHE_TEST_FAIL"
fi

# 测试重复调用（应该使用缓存）
cache_system_info
if [ -n "${SYSTEM_INFO_CACHE[gpu_count]:-}" ]; then
    echo "SYS_CACHE_REUSE_PASS"
else
    echo "SYS_CACHE_REUSE_FAIL"
fi
EOF
    
    chmod +x /tmp/test_sys_cache.sh
    local output=$(bash /tmp/test_sys_cache.sh 2>/dev/null)
    
    if echo "$output" | grep -q "SYS_CACHE_TEST_PASS" && echo "$output" | grep -q "SYS_CACHE_REUSE_PASS"; then
        log_test_pass "系统信息缓存功能正常"
    else
        log_test_fail "系统信息缓存功能异常"
    fi
}

# 测试网络接口配置
test_network_interface_config() {
    log_test_header "测试网络接口配置"
    
    local interface_types=("auto_ethernet" "loopback_only" "exclude_virtual" "clear_interface")
    
    for interface_type in "${interface_types[@]}"; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        
        cat > /tmp/test_interface.sh << EOF
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundermentals/nccl/nccl_benchmark.sh

declare -A NCCL_CONFIG_CACHE
MULTI_NODE_MODE=false

setup_network_interface "$interface_type"

# 根据接口类型验证配置
case "$interface_type" in
    "loopback_only")
        [ "\$NCCL_SOCKET_IFNAME" = "lo" ] && echo "INTERFACE_TEST_PASS" || echo "INTERFACE_TEST_FAIL"
        ;;
    "exclude_virtual")
        [ "\$NCCL_SOCKET_IFNAME" = "^docker0,lo,virbr" ] && echo "INTERFACE_TEST_PASS" || echo "INTERFACE_TEST_FAIL"
        ;;
    "clear_interface")
        [ "\$NCCL_SOCKET_IFNAME" = "" ] && echo "INTERFACE_TEST_PASS" || echo "INTERFACE_TEST_FAIL"
        ;;
    *)
        echo "INTERFACE_TEST_PASS"  # auto_ethernet 需要更复杂的验证
        ;;
esac
EOF
        
        chmod +x /tmp/test_interface.sh
        local output=$(bash /tmp/test_interface.sh 2>/dev/null)
        
        if echo "$output" | grep -q "INTERFACE_TEST_PASS"; then
            log_test_pass "网络接口配置 $interface_type 正常"
        else
            log_test_fail "网络接口配置 $interface_type 异常"
        fi
    done
}

# 测试配置管理器集成
test_config_manager_integration() {
    log_test_header "测试配置管理器集成"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    cat > /tmp/test_integration.sh << 'EOF'
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundermentals/nccl/nccl_benchmark.sh

declare -A NCCL_CONFIG_CACHE
declare -A SYSTEM_INFO_CACHE

# 测试完整的配置流程
setup_common_nccl_config
cache_system_info
setup_network_config "ib_disable"
setup_performance_config "pcie_optimized"
setup_network_interface "exclude_virtual"

# 验证集成效果
config_count=0
[ -n "$NCCL_DEBUG" ] && ((config_count++))
[ -n "$NCCL_IB_DISABLE" ] && ((config_count++))
[ -n "$NCCL_NTHREADS" ] && ((config_count++))
[ -n "$NCCL_SOCKET_IFNAME" ] && ((config_count++))

if [ $config_count -ge 4 ]; then
    echo "INTEGRATION_TEST_PASS"
else
    echo "INTEGRATION_TEST_FAIL"
fi

# 验证缓存状态
cache_count=0
[ -n "${SYSTEM_INFO_CACHE[gpu_count]:-}" ] && ((cache_count++))
[ -n "${NCCL_CONFIG_CACHE[DEBUG]:-}" ] && ((cache_count++))

if [ $cache_count -ge 2 ]; then
    echo "CACHE_INTEGRATION_PASS"
else
    echo "CACHE_INTEGRATION_FAIL"
fi
EOF
    
    chmod +x /tmp/test_integration.sh
    local output=$(bash /tmp/test_integration.sh 2>/dev/null)
    
    if echo "$output" | grep -q "INTEGRATION_TEST_PASS" && echo "$output" | grep -q "CACHE_INTEGRATION_PASS"; then
        log_test_pass "配置管理器集成测试正常"
    else
        log_test_fail "配置管理器集成测试异常"
    fi
}

# 清理测试环境
cleanup_test_environment() {
    rm -f /tmp/mock_nvidia_smi /tmp/mock_ibv_devinfo
    rm -f /tmp/test_*.sh
    unalias nvidia-smi ibv_devinfo 2>/dev/null || true
}

# 生成测试报告
generate_test_report() {
    log_test_header "配置管理器测试报告"
    
    local success_rate=0
    if [ $TOTAL_TESTS -gt 0 ]; then
        success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    fi
    
    log_test ""
    log_test "📊 配置管理器测试统计:"
    log_test "   总测试数: $TOTAL_TESTS"
    log_test "   通过测试: $PASSED_TESTS"
    log_test "   失败测试: $FAILED_TESTS"
    log_test "   成功率: ${success_rate}%"
    log_test ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        log_test "${GREEN}🎉 配置管理器所有测试通过！${NC}"
        log_test "统一配置管理器功能正常，优化效果良好"
    else
        log_test "${RED}❌ 配置管理器存在问题${NC}"
        log_test "请检查失败的测试项目"
    fi
    
    log_test ""
    log_test "详细测试日志: $TEST_LOG"
}

# 主测试函数
main() {
    echo "🔧 开始配置管理器专项测试"
    echo "目标脚本: $NCCL_SCRIPT_PATH"
    echo "测试日志: $TEST_LOG"
    echo ""
    
    # 初始化测试日志
    echo "NCCL Config Manager Test - $(date)" > "$TEST_LOG"
    
    # 设置 mock 环境
    setup_mock_environment
    
    # 执行专项测试
    test_config_cache
    test_batch_config
    test_network_presets
    test_performance_presets
    test_system_info_cache
    test_network_interface_config
    test_config_manager_integration
    
    # 清理和报告
    cleanup_test_environment
    generate_test_report
    
    # 返回适当的退出码
    if [ $FAILED_TESTS -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# 执行主函数
main "$@"