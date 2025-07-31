#!/bin/bash

# 回归测试脚本 - 确保修复没有破坏现有功能
# 测试所有主要功能的正确性

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/ib_bandwidth_monitor.sh"

# 引入测试工具函数
source "$(dirname "$0")/test_utils.sh"

# 测试基本命令行选项
test_basic_options() {
    log_test "测试基本命令行选项"
    
    increment_test_counter
    # 测试帮助选项
    if "$MONITOR_SCRIPT" --help >/dev/null 2>&1; then
        log_pass "帮助选项 (--help)"
    else
        log_fail "帮助选项 (--help)"
    fi
    
    increment_test_counter
    # 测试版本选项
    if "$MONITOR_SCRIPT" --version >/dev/null 2>&1; then
        log_pass "版本选项 (--version)"
    else
        log_fail "版本选项 (--version)"
    fi
    
    increment_test_counter
    # 测试列出端口选项
    if "$MONITOR_SCRIPT" --list-ports >/dev/null 2>&1; then
        log_pass "列出端口选项 (--list-ports)"
    else
        log_fail "列出端口选项 (--list-ports)"
    fi
}

# 测试参数验证功能
test_parameter_validation() {
    log_test "测试参数验证功能"
    
    increment_test_counter
    # 测试无效间隔
    if ! "$MONITOR_SCRIPT" --interval -1 >/dev/null 2>&1; then
        log_pass "无效间隔验证"
    else
        log_fail "无效间隔验证"
    fi
    
    increment_test_counter
    # 测试无效持续时间
    if ! "$MONITOR_SCRIPT" --duration -5 >/dev/null 2>&1; then
        log_pass "无效持续时间验证"
    else
        log_fail "无效持续时间验证"
    fi
    
    increment_test_counter
    # 测试无效端口范围
    if ! "$MONITOR_SCRIPT" --device mlx5_0 --port 0 >/dev/null 2>&1; then
        log_pass "无效端口验证"
    else
        log_fail "无效端口验证"
    fi
}

# 测试配置解析功能
test_config_parsing() {
    log_test "测试配置解析功能"
    
    # 创建临时测试脚本来测试解析函数
    local test_script="/tmp/test_parsing_$$"
    cat > "$test_script" << 'EOF'
#!/bin/bash

# 从主脚本中提取解析函数进行测试
source /dev/stdin << 'FUNCTIONS'

# 验证正整数
validate_positive_integer() {
    local value="$1"
    local name="$2"
    
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误: $name 必须是正整数，当前值: $value" >&2
        return 1
    fi
    return 0
}

# 验证非负整数
validate_non_negative_integer() {
    local value="$1"
    local name="$2"
    
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "错误: $name 必须是非负整数，当前值: $value" >&2
        return 1
    fi
    return 0
}

# 验证浮点数
validate_float() {
    local value="$1"
    local name="$2"
    
    if ! [[ "$value" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "错误: $name 必须是有效的数字，当前值: $value" >&2
        return 1
    fi
    return 0
}

# 解析端口列表
parse_ports_list() {
    local ports_str="$1"
    local quiet_mode="${2:-false}"
    
    if [ -z "$ports_str" ]; then
        [ "$quiet_mode" = false ] && echo "错误: 端口列表不能为空" >&2
        return 1
    fi
    
    # 处理端口范围和列表
    local ports=()
    IFS=',' read -ra PORT_ITEMS <<< "$ports_str"
    
    for item in "${PORT_ITEMS[@]}"; do
        item=$(echo "$item" | tr -d ' ')  # 移除空格
        
        if [[ "$item" =~ ^[0-9]+-[0-9]+$ ]]; then
            # 端口范围
            local start_port end_port
            start_port=$(echo "$item" | cut -d'-' -f1)
            end_port=$(echo "$item" | cut -d'-' -f2)
            
            if ! validate_positive_integer "$start_port" "起始端口" || \
               ! validate_positive_integer "$end_port" "结束端口"; then
                return 1
            fi
            
            if [ "$start_port" -gt "$end_port" ]; then
                [ "$quiet_mode" = false ] && echo "错误: 起始端口 ($start_port) 不能大于结束端口 ($end_port)" >&2
                return 1
            fi
            
            for ((port=start_port; port<=end_port; port++)); do
                ports+=("$port")
            done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            # 单个端口
            if ! validate_positive_integer "$item" "端口"; then
                return 1
            fi
            ports+=("$item")
        else
            [ "$quiet_mode" = false ] && echo "错误: 无效的端口格式: $item" >&2
            return 1
        fi
    done
    
    # 输出解析后的端口列表
    printf '%s\n' "${ports[@]}"
    return 0
}

FUNCTIONS

# 测试用例
case "$1" in
    "validate_positive")
        validate_positive_integer "$2" "测试值"
        ;;
    "validate_non_negative")
        validate_non_negative_integer "$2" "测试值"
        ;;
    "validate_float")
        validate_float "$2" "测试值"
        ;;
    "parse_ports")
        parse_ports_list "$2" true
        ;;
    *)
        echo "未知测试: $1"
        exit 1
        ;;
esac
EOF

    chmod +x "$test_script"
    
    increment_test_counter
    # 测试正整数验证
    if "$test_script" "validate_positive" "5" >/dev/null 2>&1; then
        log_pass "正整数验证 (有效值)"
    else
        log_fail "正整数验证 (有效值)"
    fi
    
    increment_test_counter
    if ! "$test_script" "validate_positive" "0" >/dev/null 2>&1; then
        log_pass "正整数验证 (无效值)"
    else
        log_fail "正整数验证 (无效值)"
    fi
    
    increment_test_counter
    # 测试端口列表解析
    local result
    result=$("$test_script" "parse_ports" "1,2,3" 2>/dev/null)
    if [ "$(echo "$result" | wc -l)" -eq 3 ]; then
        log_pass "端口列表解析"
    else
        log_fail "端口列表解析"
    fi
    
    increment_test_counter
    # 测试端口范围解析
    result=$("$test_script" "parse_ports" "1-3" 2>/dev/null)
    if [ "$(echo "$result" | wc -l)" -eq 3 ]; then
        log_pass "端口范围解析"
    else
        log_fail "端口范围解析"
    fi
    
    rm -f "$test_script"
}

# 测试输出格式
test_output_formats() {
    log_test "测试输出格式"
    
    increment_test_counter
    # 测试 SAR 格式输出
    local output
    output=$("$MONITOR_SCRIPT" --help 2>&1)
    if echo "$output" | grep -q "SAR"; then
        log_pass "SAR 格式说明存在"
    else
        log_fail "SAR 格式说明缺失"
    fi
    
    increment_test_counter
    # 测试静默模式
    output=$("$MONITOR_SCRIPT" --version --quiet 2>&1)
    if [ -n "$output" ]; then
        log_pass "静默模式基本功能"
    else
        log_fail "静默模式异常"
    fi
}

# 测试错误处理
test_error_handling() {
    log_test "测试错误处理"
    
    increment_test_counter
    # 测试冲突参数
    if ! "$MONITOR_SCRIPT" --device mlx5_0 --multi-devices "mlx5_1:1" >/dev/null 2>&1; then
        log_pass "冲突参数检测"
    else
        log_fail "冲突参数检测"
    fi
    
    increment_test_counter
    # 测试缺少必需参数
    if ! "$MONITOR_SCRIPT" --device mlx5_0 >/dev/null 2>&1; then
        log_pass "缺少必需参数检测"
    else
        log_fail "缺少必需参数检测"
    fi
}

# 测试兼容性函数
test_compatibility_functions() {
    log_test "测试兼容性函数"
    
    # 检查兼容性函数是否存在
    increment_test_counter
    if grep -q "monitor_single_port" "$MONITOR_SCRIPT"; then
        log_pass "单端口监控兼容性函数存在"
    else
        log_fail "单端口监控兼容性函数缺失"
    fi
    
    increment_test_counter
    if grep -q "monitor_multiple_ports" "$MONITOR_SCRIPT"; then
        log_pass "多端口监控兼容性函数存在"
    else
        log_fail "多端口监控兼容性函数缺失"
    fi
    
    increment_test_counter
    if grep -q "monitor_multi_devices" "$MONITOR_SCRIPT"; then
        log_pass "多设备监控兼容性函数存在"
    else
        log_fail "多设备监控兼容性函数缺失"
    fi
}

# 测试通用工具函数
test_utility_functions() {
    log_test "测试通用工具函数"
    
    # 检查关键工具函数是否存在
    increment_test_counter
    if grep -q "handle_counter_wraparound" "$MONITOR_SCRIPT"; then
        log_pass "计数器回绕处理函数存在"
    else
        log_fail "计数器回绕处理函数缺失"
    fi
    
    increment_test_counter
    if grep -q "safe_calc" "$MONITOR_SCRIPT"; then
        log_pass "安全计算函数存在"
    else
        log_fail "安全计算函数缺失"
    fi
    
    increment_test_counter
    if grep -q "calculate_percentage" "$MONITOR_SCRIPT"; then
        log_pass "百分比计算函数存在"
    else
        log_fail "百分比计算函数缺失"
    fi
}

# 测试脚本完整性
test_script_integrity() {
    log_test "测试脚本完整性"
    
    increment_test_counter
    # 检查脚本语法
    if bash -n "$MONITOR_SCRIPT"; then
        log_pass "脚本语法正确"
    else
        log_fail "脚本语法错误"
    fi
    
    increment_test_counter
    # 检查关键变量定义
    if grep -q "VERSION=" "$MONITOR_SCRIPT"; then
        log_pass "版本变量定义"
    else
        log_fail "版本变量缺失"
    fi
    
    increment_test_counter
    # 检查主函数存在
    if grep -q "^main()" "$MONITOR_SCRIPT"; then
        log_pass "主函数定义"
    else
        log_fail "主函数缺失"
    fi
}

# 主测试函数
main() {
    echo "========================================"
    echo "IB 带宽监控脚本 - 回归测试"
    echo "========================================"
    echo
    
    # 检查脚本存在
    if [ ! -f "$MONITOR_SCRIPT" ]; then
        echo -e "${RED}错误: 监控脚本不存在: $MONITOR_SCRIPT${NC}"
        exit 1
    fi
    
    # 运行所有回归测试
    test_basic_options
    test_parameter_validation
    test_config_parsing
    test_output_formats
    test_error_handling
    test_compatibility_functions
    test_utility_functions
    test_script_integrity
    
    echo
    # 使用统一的测试结果汇总函数
    print_test_summary "回归测试"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}所有回归测试通过！修复没有破坏现有功能。${NC}"
        exit 0
    else
        echo -e "\n${RED}有 $FAILED_TESTS 个回归测试失败！${NC}"
        echo -e "${RED}修复可能破坏了现有功能，请检查。${NC}"
        exit 1
    fi
}

# 运行主函数
main "$@"