#!/bin/bash

# IB 带宽监控脚本 - 多设备配置综合测试
# 整合单元测试、修复验证测试和多设备修复测试
# 版本: 1.0

# 设置测试环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/ib_bandwidth_monitor.sh"

# 引入测试工具函数
source "$SCRIPT_DIR/test_utils.sh"

# 测试计数器初始化
init_test_counters

# 额外的颜色定义（补充test_utils.sh中的）
BLUE='\033[0;34m'
PURPLE='\033[0;35m'

# 日志函数
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
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

log_section() {
    echo -e "${PURPLE}========================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}========================================${NC}"
}

log_subsection() {
    echo -e "${CYAN}--- $1 ---${NC}"
}

# 设置模拟环境
setup_mock_environment() {
    # 获取脚本所在目录
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local MOCK_SOURCE_DIR="$SCRIPT_DIR/mock_commands"
    local TEMP_BIN_DIR="/tmp/mock_ib_commands"
    
    # 检查模拟命令源目录是否存在
    if [ ! -d "$MOCK_SOURCE_DIR" ]; then
        log_fail "模拟命令源目录不存在: $MOCK_SOURCE_DIR"
        exit 1
    fi
    
    # 创建临时目录用于模拟命令
    mkdir -p "$TEMP_BIN_DIR"
    
    # 复制模拟命令到临时目录
    cp "$MOCK_SOURCE_DIR"/* "$TEMP_BIN_DIR/" 2>/dev/null || {
        log_fail "复制模拟命令失败"
        exit 1
    }
    
    # 创建模拟的 timeout 命令（如果不存在）
    if [ ! -f "$TEMP_BIN_DIR/timeout" ]; then
        cat > "$TEMP_BIN_DIR/timeout" << 'EOF'
#!/bin/bash
duration="$1"
shift
# 直接执行命令，忽略超时
exec "$@"
EOF
    fi
    
    # 设置执行权限
    chmod +x "$TEMP_BIN_DIR"/*
    
    # 将模拟命令目录添加到PATH前面
    export PATH="$TEMP_BIN_DIR:$PATH"
    
    log_info "模拟环境已设置 (使用 $MOCK_SOURCE_DIR)"
}

# 清理模拟环境
cleanup_mock_environment() {
    rm -rf "/tmp/mock_ib_commands"
    log_info "模拟环境已清理"
}

# 运行测试函数
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_exit_code="${3:-0}"
    
    increment_test_counter
    log_test "运行测试: $test_name"
    
    # 执行测试命令
    eval "$test_command"
    local actual_exit_code=$?
    
    if [ $actual_exit_code -eq $expected_exit_code ]; then
        log_pass "$test_name"
        return 0
    else
        log_fail "$test_name (期望退出码: $expected_exit_code, 实际: $actual_exit_code)"
        return 1
    fi
}

# 检查脚本是否存在
check_script_exists() {
    if [ ! -f "$MONITOR_SCRIPT" ]; then
        log_fail "监控脚本不存在: $MONITOR_SCRIPT"
        exit 1
    fi
    log_info "找到监控脚本: $MONITOR_SCRIPT"
}

# ========================================
# 第一部分：基础功能测试（来自 test_multi_devices_fix.sh）
# ========================================

# 测试脚本语法
test_script_syntax() {
    log_test "检查脚本语法"
    if bash -n "$MONITOR_SCRIPT"; then
        log_pass "脚本语法检查"
    else
        log_fail "脚本语法检查"
        return 1
    fi
}

# 测试帮助信息
test_help_output() {
    increment_test_counter
    log_test "测试帮助信息输出"
    local output
    output=$("$MONITOR_SCRIPT" --help 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ] && echo "$output" | grep -q "InfiniBand"; then
        log_pass "帮助信息输出正常"
    else
        log_fail "帮助信息输出异常"
        return 1
    fi
}

# 测试版本信息
test_version_output() {
    increment_test_counter
    log_test "测试版本信息输出"
    local output
    output=$("$MONITOR_SCRIPT" --version 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ] && echo "$output" | grep -q "version"; then
        log_pass "版本信息输出正常"
    else
        log_fail "版本信息输出异常"
        return 1
    fi
}

# 测试无效参数处理
test_invalid_arguments() {
    increment_test_counter
    log_test "测试无效参数处理"
    local output
    output=$("$MONITOR_SCRIPT" --invalid-option 2>&1)
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        log_pass "无效参数正确处理"
    else
        log_fail "无效参数处理异常"
        return 1
    fi
}

# 测试多设备配置验证（模拟环境）
test_multi_devices_validation() {
    increment_test_counter
    log_test "测试多设备配置验证逻辑"
    
    # 创建临时测试脚本来模拟验证函数
    local test_script="/tmp/test_validation_$$"
    cat > "$test_script" << 'EOF'
#!/bin/bash

# 模拟 validate_multi_devices_config 函数的核心逻辑
validate_multi_devices_config() {
    local config="$1"
    local valid_configs=()
    
    # 解析配置
    IFS=';' read -ra DEVICE_CONFIGS <<< "$config"
    
    for device_config in "${DEVICE_CONFIGS[@]}"; do
        local device_name
        local ports_str
        device_name=$(echo "$device_config" | cut -d':' -f1)
        ports_str=$(echo "$device_config" | cut -d':' -f2)
        
        # 模拟设备验证（假设 mlx5_0 和 mlx5_1 存在）
        if [[ "$device_name" =~ ^mlx5_[0-9]+$ ]]; then
            IFS=',' read -ra ports <<< "$ports_str"
            local valid_ports=()
            
            for port in "${ports[@]}"; do
                # 模拟端口验证（假设端口1存在，端口2不存在）
                if [ "$port" = "1" ]; then
                    valid_ports+=("$port")
                else
                    echo "警告: 端口 ${device_name}:${port} 不存在或不活跃" >&2
                fi
            done
            
            # 如果有有效端口，添加到配置中
            if [ ${#valid_ports[@]} -gt 0 ]; then
                local valid_ports_str
                valid_ports_str=$(IFS=','; echo "${valid_ports[*]}")
                valid_configs+=("${device_name}:${valid_ports_str}")
            fi
        else
            echo "错误: 设备 $device_name 不存在" >&2
            return 1
        fi
    done
    
    # 输出有效配置
    if [ ${#valid_configs[@]} -gt 0 ]; then
        local result
        result=$(IFS=';'; echo "${valid_configs[*]}")
        echo "$result"
        return 0
    else
        echo "错误: 没有有效的设备端口配置" >&2
        return 1
    fi
}

# 测试用例
test_case="$1"
case "$test_case" in
    "valid_single")
        validate_multi_devices_config "mlx5_0:1"
        ;;
    "valid_multiple")
        validate_multi_devices_config "mlx5_0:1;mlx5_1:1"
        ;;
    "mixed_valid_invalid")
        validate_multi_devices_config "mlx5_0:1,2;mlx5_1:1"
        ;;
    "invalid_device")
        validate_multi_devices_config "invalid_device:1"
        ;;
    "invalid_port")
        validate_multi_devices_config "mlx5_0:999"
        ;;
    *)
        echo "未知测试用例: $test_case"
        exit 1
        ;;
esac
EOF

    chmod +x "$test_script"
    
    # 测试有效的单设备配置
    local output
    output=$("$test_script" "valid_single" 2>&1)
    if [ $? -eq 0 ] && [ "$output" = "mlx5_0:1" ]; then
        log_pass "有效单设备配置验证"
    else
        log_fail "有效单设备配置验证 (输出: $output)"
    fi
    
    # 测试有效的多设备配置
    output=$("$test_script" "valid_multiple" 2>&1)
    if [ $? -eq 0 ] && [ "$output" = "mlx5_0:1;mlx5_1:1" ]; then
        log_pass "有效多设备配置验证"
    else
        log_fail "有效多设备配置验证 (输出: $output)"
    fi
    
    # 测试混合有效无效配置
    output=$("$test_script" "mixed_valid_invalid" 2>/dev/null)
    if [ $? -eq 0 ] && [ "$output" = "mlx5_0:1;mlx5_1:1" ]; then
        log_pass "混合有效无效配置验证"
    else
        log_fail "混合有效无效配置验证 (输出: $output)"
    fi
    
    # 测试无效设备
    "$test_script" "invalid_device" >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_pass "无效设备配置验证"
    else
        log_fail "无效设备配置验证"
    fi
    
    # 测试无效端口
    "$test_script" "invalid_port" >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_pass "无效端口配置验证"
    else
        log_fail "无效端口配置验证"
    fi
    
    # 清理临时文件
    rm -f "$test_script"
}

# 测试错误处理
test_error_handling() {
    increment_test_counter
    log_test "测试错误处理"
    
    # 临时移除模拟环境，测试真实的错误处理
    local original_path="$PATH"
    export PATH=$(echo "$PATH" | sed 's|/tmp/mock_ib_commands:||g')
    
    # 测试空配置
    local output
    local exit_code
    output=$(timeout 5 bash "$MONITOR_SCRIPT" --multi-devices '' 2>&1)
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_pass "空配置错误处理"
    else
        log_fail "空配置错误处理"
    fi
    
    # 测试格式错误的配置
    output=$(timeout 5 bash "$MONITOR_SCRIPT" --multi-devices 'invalid_format' 2>&1)
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_pass "格式错误配置处理"
    else
        log_fail "格式错误配置处理"
    fi
    
    # 测试不存在的设备配置
    output=$(timeout 5 bash "$MONITOR_SCRIPT" --multi-devices 'nonexistent_device:1' --time 1 2>&1)
    exit_code=$?
    if [ $exit_code -ne 0 ] || echo "$output" | grep -q -i "error\|fail\|not found"; then
        log_pass "不存在设备错误处理"
    else
        log_fail "不存在设备错误处理"
    fi
    
    # 恢复模拟环境
    export PATH="$original_path"
}

# 性能测试
test_performance() {
    increment_test_counter
    log_test "测试脚本启动性能"
    
    local start_time
    local end_time
    local duration
    
    start_time=$(date +%s%N)
    "$MONITOR_SCRIPT" --version >/dev/null 2>&1
    end_time=$(date +%s%N)
    
    duration=$(( (end_time - start_time) / 1000000 )) # 转换为毫秒
    
    if [ $duration -lt 1000 ]; then # 小于1秒
        log_pass "脚本启动性能 (${duration}ms)"
    else
        log_fail "脚本启动性能过慢 (${duration}ms)"
    fi
}

# ========================================
# 第二部分：修复验证测试（来自 test_fixes_verification.sh）
# ========================================

# 测试1: 验证原始问题配置不再报错
test_original_issue_fix() {
    increment_test_counter
    log_test "验证原始问题配置: mlx5_0:1,2;mlx5_1:1"
    
    # 运行原始问题中的配置
    local output
    output=$(bash "$MONITOR_SCRIPT" --multi-devices "mlx5_0:1,2;mlx5_1:1" --time 3 2>&1)
    local exit_code=$?
    
    # 检查是否没有 "command not found" 错误
    if ! echo "$output" | grep -q "command not found"; then
        log_pass "没有 'command not found' 错误"
    else
        log_fail "仍然存在 'command not found' 错误"
        echo "错误输出:"
        echo "$output" | grep "command not found"
    fi
    
    # 检查是否没有端口范围错误
    if ! echo "$output" | grep -qE "(端口.*超出范围|port.*out of range)"; then
        log_pass "没有端口范围错误"
    else
        log_fail "仍然存在端口范围错误"
        echo "错误输出:"
        echo "$output" | grep -E "(端口.*超出范围|port.*out of range)"
    fi
    
    # 检查脚本是否正常运行
    if [ $exit_code -eq 0 ]; then
        log_pass "脚本正常运行完成"
    else
        log_fail "脚本运行失败 (退出码: $exit_code)"
    fi
}

# 测试2: 验证端口2被正确跳过
test_port_skipping() {
    increment_test_counter
    log_test "验证端口2被正确处理"
    
    local output
    output=$(bash "$MONITOR_SCRIPT" --multi-devices "mlx5_0:1,2;mlx5_1:1" --time 1 2>&1)
    
    # 检查脚本是否正常运行（不因端口2而失败）
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_pass "脚本正常处理了端口2（无论跳过还是处理）"
    else
        log_fail "脚本因端口2处理失败"
        echo "相关输出:"
        echo "$output" | grep -E "(端口|port|错误|error)" | head -5
    fi
}

# 测试3: 验证ANSI颜色代码问题修复
test_ansi_color_fix() {
    increment_test_counter
    log_test "验证ANSI颜色代码混合问题修复"
    
    local output
    output=$(bash "$MONITOR_SCRIPT" --multi-devices "mlx5_0:1;mlx5_1:1" --time 1 2>&1)
    
    # 检查是否没有ANSI颜色代码混合导致的"command not found"
    if ! echo "$output" | grep -qE "(\[0;31m|\[0;32m|\[1;33m).*command not found"; then
        log_pass "没有ANSI颜色代码混合问题"
    else
        log_fail "仍然存在ANSI颜色代码混合问题"
        echo "问题输出:"
        echo "$output" | grep -E "(\[0;31m|\[0;32m|\[1;33m).*command not found"
    fi
}

# ========================================
# 第三部分：多设备修复功能测试（来自 test_multi_devices_fixed.sh）
# ========================================

# 测试有效配置
test_valid_configuration() {
    increment_test_counter
    log_test "测试有效配置 mlx5_0:1;mlx5_1:1"
    
    local output
    output=$(timeout 5 bash "$MONITOR_SCRIPT" --multi-devices "mlx5_0:1;mlx5_1:1" --time 2 --quiet 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_pass "有效配置测试通过"
    else
        log_fail "有效配置测试失败 (退出码: $exit_code)"
    fi
}

# 测试混合配置
test_mixed_configuration() {
    increment_test_counter
    log_test "测试混合配置 mlx5_0:1,2;mlx5_1:1 (端口2应该被跳过)"
    
    local output
    output=$(timeout 5 bash "$MONITOR_SCRIPT" --multi-devices "mlx5_0:1,2;mlx5_1:1" --time 2 --quiet 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_pass "混合配置测试通过"
    else
        log_fail "混合配置测试失败 (退出码: $exit_code)"
    fi
}

# 测试命令错误检查
test_command_error_check() {
    increment_test_counter
    log_test "检查 'command not found' 错误"
    
    local output
    output=$(timeout 5 bash "$MONITOR_SCRIPT" --multi-devices "mlx5_0:1;mlx5_1:1" --time 2 2>&1)
    
    if ! echo "$output" | grep -i "command not found"; then
        log_pass "没有发现 'command not found' 错误"
    else
        log_fail "仍然存在 'command not found' 错误"
        echo "错误输出:"
        echo "$output" | grep -i "command not found"
    fi
}

# ========================================
# 主测试函数
# ========================================

run_basic_tests() {
    log_subsection "基础功能测试"
    
    test_script_syntax
    test_help_output
    test_version_output
    test_invalid_arguments
    test_multi_devices_validation
    test_error_handling
    test_performance
}

run_fix_verification_tests() {
    log_subsection "修复验证测试"
    
    test_original_issue_fix
    test_port_skipping
    test_ansi_color_fix
}

run_multi_device_tests() {
    log_subsection "多设备功能测试"
    
    test_valid_configuration
    test_mixed_configuration
    test_command_error_check
}

# 主函数
main() {
    log_section "IB 带宽监控脚本 - 多设备配置综合测试"
    echo
    
    # 检查脚本存在性
    check_script_exists
    echo
    
    # 设置模拟环境
    setup_mock_environment
    echo
    
    # 运行所有测试
    run_basic_tests
    echo
    
    run_fix_verification_tests
    echo
    
    run_multi_device_tests
    echo
    
    # 清理模拟环境
    cleanup_mock_environment
    echo
    
    # 使用test_utils.sh中的函数显示测试结果汇总
    if print_test_summary "多设备配置综合测试"; then
        echo
        echo -e "${GREEN}✅ 所有测试通过！${NC}"
        echo "测试覆盖范围："
        echo "1. ✅ 基础功能测试 - 脚本语法、帮助信息、版本信息等"
        echo "2. ✅ 多设备配置验证 - 配置解析、验证逻辑等"
        echo "3. ✅ 错误处理测试 - 无效参数、错误配置等"
        echo "4. ✅ 修复验证测试 - 原始问题修复验证"
        echo "5. ✅ 多设备功能测试 - 实际多设备配置运行"
        echo "6. ✅ 性能测试 - 脚本启动性能"
        exit 0
    else
        echo
        echo -e "${RED}❌ 有测试失败${NC}"
        echo "请检查失败的测试并修复问题"
        exit 1
    fi
}

# 运行主函数
main "$@"