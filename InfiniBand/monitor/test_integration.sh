#!/bin/bash

# 集成测试脚本 - 测试实际的多设备配置场景
# 这个脚本会创建模拟的 IB 环境来测试修复后的功能

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/ib_bandwidth_monitor.sh"

# 引入测试工具函数
source "$SCRIPT_DIR/test_utils.sh"

# 测试计数器初始化
init_test_counters

# 额外的颜色定义（补充test_utils.sh中的）
BLUE='\033[0;34m'

# 日志函数（使用test_utils.sh中的函数，但保持原有的格式）
log_test() {
    echo -e "${BLUE}[INTEGRATION TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    increment_pass_counter
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    increment_fail_counter
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# 创建模拟的 IB 环境
setup_mock_environment() {
    log_info "设置模拟 IB 环境"
    
    # 创建临时目录结构模拟 /sys/class/infiniband
    export MOCK_IB_DIR="/tmp/mock_ib_$$"
    mkdir -p "$MOCK_IB_DIR"
    
    # 创建模拟设备
    for device in mlx5_0 mlx5_1; do
        mkdir -p "$MOCK_IB_DIR/$device/ports"
        
        # 创建端口1（活跃）
        mkdir -p "$MOCK_IB_DIR/$device/ports/1/counters"
        echo "Active" > "$MOCK_IB_DIR/$device/ports/1/state"
        echo "100" > "$MOCK_IB_DIR/$device/ports/1/rate"
        echo "1000000" > "$MOCK_IB_DIR/$device/ports/1/counters/port_xmit_data"
        echo "2000000" > "$MOCK_IB_DIR/$device/ports/1/counters/port_rcv_data"
        echo "500" > "$MOCK_IB_DIR/$device/ports/1/counters/port_xmit_pkts"
        echo "600" > "$MOCK_IB_DIR/$device/ports/1/counters/port_rcv_pkts"
        
        # 创建端口2（对于 mlx5_0 不存在，对于 mlx5_1 存在但不活跃）
        if [ "$device" = "mlx5_1" ]; then
            mkdir -p "$MOCK_IB_DIR/$device/ports/2/counters"
            echo "Down" > "$MOCK_IB_DIR/$device/ports/2/state"
            echo "0" > "$MOCK_IB_DIR/$device/ports/2/rate"
        fi
    done
    
    log_info "模拟环境创建完成: $MOCK_IB_DIR"
}

# 清理模拟环境
cleanup_mock_environment() {
    if [ -n "$MOCK_IB_DIR" ] && [ -d "$MOCK_IB_DIR" ]; then
        rm -rf "$MOCK_IB_DIR"
        log_info "清理模拟环境: $MOCK_IB_DIR"
    fi
}

# 创建修改版的监控脚本用于测试
create_test_script() {
    local test_script="/tmp/test_monitor_$$"
    
    # 复制原脚本并修改路径
    cp "$MONITOR_SCRIPT" "$test_script"
    
    # 修改脚本中的路径引用，使其使用模拟环境
    sed -i.bak "s|/sys/class/infiniband|$MOCK_IB_DIR|g" "$test_script"
    
    # 创建一个包含模拟函数的临时文件
    cat > "/tmp/mock_functions_$$" << 'EOF'
#!/bin/bash
# 模拟命令函数
perfquery() { 
    echo "PortXmitData:1000000"
    echo "PortRcvData:2000000" 
    echo "PortXmitPkts:500"
    echo "PortRcvPkts:600"
}

# 模拟 timeout 命令
timeout() {
    shift  # 跳过超时时间参数
    "$@"   # 执行剩余的命令
}

ibstat() {
    local device="$1"
    local port="$2"
    
    if [ "$1" = "-l" ]; then
        echo "mlx5_0"
        echo "mlx5_1"
        return 0
    fi
    
    if [ -z "$device" ]; then
        echo "mlx5_0: Active"
        echo "mlx5_1: Active"
        return 0
    fi
    
    case "$device" in
        "mlx5_0")
            if [ -z "$port" ]; then
                echo "CA 'mlx5_0'"
                echo "        CA type: MT4123"
                echo "        Number of ports: 2"
                echo "        Firmware version: 20.31.1014"
                echo "        Hardware version: 0"
            else
                case "$port" in
                    "1")
                        echo "CA 'mlx5_0'"
                        echo "        Port 1:"
                        echo "                State: Active"
                        echo "                Physical state: LinkUp"
                        echo "                Rate: 100"
                        echo "                Base lid: 1"
                        ;;
                    "2")
                        echo "CA 'mlx5_0'"
                        echo "        Port 2:"
                        echo "                State: Down"
                        echo "                Physical state: Disabled"
                        echo "                Rate: 0"
                        echo "                Base lid: 0"
                        ;;
                    *)
                        return 1
                        ;;
                esac
            fi
            ;;
        "mlx5_1")
            if [ -z "$port" ]; then
                echo "CA 'mlx5_1'"
                echo "        CA type: MT4123"
                echo "        Number of ports: 1"
                echo "        Firmware version: 20.31.1014"
                echo "        Hardware version: 0"
            else
                case "$port" in
                    "1")
                        echo "CA 'mlx5_1'"
                        echo "        Port 1:"
                        echo "                State: Active"
                        echo "                Physical state: LinkUp"
                        echo "                Rate: 100"
                        echo "                Base lid: 2"
                        ;;
                    *)
                        return 1
                        ;;
                esac
            fi
            ;;
        *)
            return 1
            ;;
    esac
}

# 模拟依赖检查函数
check_dependencies() {
    return 0
}

export -f perfquery ibstat check_dependencies
EOF
    
    # 将模拟函数插入到脚本开头
    cat "/tmp/mock_functions_$$" "$test_script" > "${test_script}.new"
    mv "${test_script}.new" "$test_script"
    
    # 清理临时文件
    rm -f "/tmp/mock_functions_$$"
    
    chmod +x "$test_script"
    echo "$test_script"
}

# 测试多设备配置验证修复
test_multi_devices_validation_fix() {
    increment_test_counter
    log_test "测试多设备配置验证修复"
    
    local test_script
    test_script=$(create_test_script)
    
    # 测试原问题场景：mlx5_0:1,2;mlx5_1:1
    local output
    output=$("$test_script" --multi-devices "mlx5_0:1,2;mlx5_1:1" --time 1 --quiet 2>&1)
    local exit_code=$?
    
    # 检查是否不再报错"起始端口超出范围"
    if ! echo "$output" | grep -q "起始端口超出范围"; then
        log_pass "不再出现端口范围错误"
    else
        log_fail "仍然出现端口范围错误"
        echo "输出: $output"
    fi
    
    # 检查是否只输出一次端口2的警告
    local warning_count
    warning_count=$(echo "$output" | grep -c "mlx5_0:2" || true)
    if [ "$warning_count" -le 1 ]; then
        log_pass "端口警告不重复"
    else
        log_fail "端口警告重复 ($warning_count 次)"
    fi
    
    rm -f "$test_script" "${test_script}.bak"
}

# 测试原始问题修复：端口范围验证
test_port_range_validation_fix() {
    increment_test_counter
    log_test "测试端口范围验证修复"
    
    local test_script
    test_script=$(create_test_script)
    
    # 测试原始问题：mlx5_0:1,2 中的端口2超出范围
    local output
    output=$("$test_script" --multi-devices "mlx5_0:1,2;mlx5_1:1" --time 1 --quiet 2>&1)
    local exit_code=$?
    
    # 检查脚本是否成功运行（不因端口2而失败）
    if [ $exit_code -eq 0 ]; then
        log_pass "脚本成功处理超出范围的端口"
    else
        log_fail "脚本因端口范围问题失败 (退出码: $exit_code)"
        echo "输出: $output"
    fi
    
    # 检查是否正确跳过了无效端口2
    if echo "$output" | grep -q "跳过无效端口" || ! echo "$output" | grep -q "mlx5_0:2"; then
        log_pass "正确跳过无效端口"
    else
        log_fail "未正确跳过无效端口"
    fi
    
    rm -f "$test_script" "${test_script}.bak"
}

# 测试ANSI颜色代码混合问题修复
test_ansi_color_code_fix() {
    increment_test_counter
    log_test "测试ANSI颜色代码混合问题修复"
    
    local test_script
    test_script=$(create_test_script)
    
    # 运行脚本并检查是否有"command not found"错误
    local output
    output=$("$test_script" --multi-devices "mlx5_0:1;mlx5_1:1" --time 1 2>&1)
    
    # 检查是否没有"command not found"错误
    if ! echo "$output" | grep -q "command not found"; then
        log_pass "没有发现 'command not found' 错误"
    else
        log_fail "仍然存在 'command not found' 错误"
        echo "错误输出:"
        echo "$output" | grep "command not found"
    fi
    
    # 检查是否没有ANSI颜色代码混合问题
    if ! echo "$output" | grep -qE '\[[0-9;]+m[a-zA-Z_][a-zA-Z0-9_]*'; then
        log_pass "没有发现ANSI颜色代码混合问题"
    else
        log_fail "仍然存在ANSI颜色代码混合问题"
        echo "问题输出:"
        echo "$output" | grep -E '\[[0-9;]+m[a-zA-Z_][a-zA-Z0-9_]*'
    fi
    
    rm -f "$test_script" "${test_script}.bak"
}

# 测试safe_key转换修复
test_safe_key_conversion_fix() {
    increment_test_counter
    log_test "测试safe_key转换修复"
    
    local test_script
    test_script=$(create_test_script)
    
    # 测试包含特殊字符的设备端口组合
    local output
    output=$("$test_script" --multi-devices "mlx5_0:1;mlx5_1:1" --time 1 --quiet 2>&1)
    local exit_code=$?
    
    # 检查脚本是否正常运行（没有因为safe_key转换问题而失败）
    if [ $exit_code -eq 0 ]; then
        log_pass "safe_key转换正常工作"
    else
        log_fail "safe_key转换存在问题 (退出码: $exit_code)"
        echo "输出: $output"
    fi
    
    # 检查是否没有eval相关的错误
    if ! echo "$output" | grep -qE "(eval|syntax error|bad substitution)"; then
        log_pass "没有eval相关错误"
    else
        log_fail "存在eval相关错误"
        echo "错误输出:"
        echo "$output" | grep -E "(eval|syntax error|bad substitution)"
    fi
    
    rm -f "$test_script" "${test_script}.bak"
}

# 测试配置过滤功能
test_config_filtering() {
    increment_test_counter
    log_test "测试配置过滤功能"
    
    local test_script
    test_script=$(create_test_script)
    
    # 测试混合有效无效配置 - mlx5_0:1(有效),2(无效-关闭);mlx5_1:1(有效);invalid_device:1(无效设备)
    local output
    local exit_code
    
    # 使用超时控制运行测试，不使用quiet模式以便检查输出
    output=$(timeout_func 10 "$test_script --multi-devices 'mlx5_0:1,2;mlx5_1:1;invalid_device:1' --time 1" 2>&1)
    exit_code=$?
    
    # 如果超时，记录失败
    if [ $exit_code -eq 124 ]; then
        log_fail "配置过滤测试超时"
    else
        # 检查脚本是否成功运行（退出码为0）
        if [ $exit_code -eq 0 ]; then
            # 检查是否正确处理了有效和无效配置
            if echo "$output" | grep -q "有效的设备端口配置.*mlx5_0:1.*mlx5_1:1" || 
               (echo "$output" | grep -q "mlx5_0.*端口 1" && echo "$output" | grep -q "mlx5_1.*端口 1"); then
                log_pass "正确保留有效配置"
            else
                log_fail "配置过滤异常"
                echo "输出: $output"
            fi
        else
            log_fail "配置过滤测试失败 (退出码: $exit_code)"
            echo "输出: $output"
        fi
    fi
    
    rm -f "$test_script" "${test_script}.bak"
}

# 测试重复验证消除
test_duplicate_validation_elimination() {
    increment_test_counter
    log_test "测试重复验证消除"
    
    local test_script
    test_script=$(create_test_script)
    
    # 启用调试模式查看详细输出（添加超时控制）
    local output
    local exit_code
    
    output=$(timeout_func 10 "$test_script --multi-devices 'mlx5_0:1;mlx5_1:1' --time 1" 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        log_fail "重复验证消除测试超时"
    else
        # 检查验证信息是否只出现一次
        local validation_count
        validation_count=$(echo "$output" | grep -c "验证.*配置" || true)
        if [ "$validation_count" -le 2 ]; then  # 允许一些合理的验证信息
            log_pass "验证过程不重复"
        else
            log_fail "验证过程重复 ($validation_count 次)"
        fi
    fi
    
    rm -f "$test_script" "${test_script}.bak"
}

# 测试边界情况
test_edge_cases() {
    increment_test_counter
    log_test "测试边界情况"
    
    local test_script
    test_script=$(create_test_script)
    
    # 测试空配置（添加超时控制）
    local exit_code
    timeout_func 5 "$test_script --multi-devices '' --quiet >/dev/null 2>&1"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_pass "空配置正确处理"
    else
        log_fail "空配置处理异常"
    fi
    
    # 测试只有无效配置（添加超时控制）
    timeout_func 5 "$test_script --multi-devices 'invalid:1;another_invalid:2' --quiet >/dev/null 2>&1"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_pass "全无效配置正确处理"
    else
        log_fail "全无效配置处理异常"
    fi
    
    # 测试格式错误的配置（添加超时控制）
    timeout_func 5 "$test_script --multi-devices 'malformed::config' --quiet >/dev/null 2>&1"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_pass "格式错误配置正确处理"
    else
        log_fail "格式错误配置处理异常"
    fi
    
    rm -f "$test_script" "${test_script}.bak"
}

# 测试向后兼容性
test_backward_compatibility() {
    increment_test_counter
    log_test "测试向后兼容性"
    
    local test_script
    test_script=$(create_test_script)
    
    # 测试单设备单端口（原有功能）
    local output
    local exit_code
    
    output=$(timeout_func 10 "$test_script --device mlx5_0 --port 1 --time 1 --quiet" 2>&1)
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        log_fail "单设备单端口兼容性测试超时"
    elif [ $exit_code -eq 0 ]; then
        log_pass "单设备单端口兼容性"
    else
        log_fail "单设备单端口兼容性"
    fi
    
    # 测试多端口（原有功能）
    output=$(timeout_func 10 "$test_script --device mlx5_0 --ports '1' --time 1 --quiet" 2>&1)
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        log_fail "多端口兼容性测试超时"
    elif [ $exit_code -eq 0 ]; then
        log_pass "多端口兼容性"
    else
        log_fail "多端口兼容性"
    fi
    
    rm -f "$test_script" "${test_script}.bak"
}

# 主测试函数
main() {
    echo "========================================"
    echo "IB 带宽监控脚本 - 集成测试"
    echo "========================================"
    echo
    
    # 设置模拟环境
    setup_mock_environment
    
    # 设置清理陷阱
    trap cleanup_mock_environment EXIT
    
    # 运行集成测试
    echo "=== 原始问题修复验证 ==="
    test_multi_devices_validation_fix
    test_port_range_validation_fix
    test_ansi_color_code_fix
    test_safe_key_conversion_fix
    
    echo
    echo "=== 功能完整性测试 ==="
    test_config_filtering
    test_duplicate_validation_elimination
    test_edge_cases
    test_backward_compatibility
    
    echo
    # 使用test_utils.sh中的函数显示测试结果汇总
    if print_test_summary "集成测试"; then
        exit 0
    else
        exit 1
    fi
}

# 运行主函数
main "$@"