#!/bin/bash

# 测试工具函数库
# 提供通用的测试辅助函数

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试计数器初始化
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 通用的超时函数
# 用法: timeout_func <超时秒数> <命令>
# 返回值: 命令的退出码，如果超时则返回124
timeout_func() {
    local timeout_duration="$1"
    shift
    local cmd="$*"
    
    # 在后台运行命令
    eval "$cmd" &
    local cmd_pid=$!
    
    # 等待指定时间
    local count=0
    while [ $count -lt $timeout_duration ]; do
        if ! kill -0 $cmd_pid 2>/dev/null; then
            wait $cmd_pid
            return $?
        fi
        sleep 1
        ((count++))
    done
    
    # 超时，杀死进程
    kill -TERM $cmd_pid 2>/dev/null
    sleep 1
    kill -KILL $cmd_pid 2>/dev/null
    return 124  # timeout exit code
}

# 日志函数
log_test() {
    echo -e "${YELLOW}[测试]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[通过]${NC} $1"
    ((PASSED_TESTS++))
}

log_fail() {
    echo -e "${RED}[失败]${NC} $1"
    ((FAILED_TESTS++))
}

log_info() {
    echo -e "[信息] $1"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查文件是否存在且可执行
is_executable() {
    [ -f "$1" ] && [ -x "$1" ]
}

# 创建临时文件的安全函数
create_temp_file() {
    local prefix="${1:-test}"
    mktemp "/tmp/${prefix}.XXXXXX"
}

# 清理临时文件
cleanup_temp_files() {
    local pattern="${1:-/tmp/test.*}"
    rm -f $pattern 2>/dev/null || true
}

# 验证退出码
check_exit_code() {
    local actual="$1"
    local expected="$2"
    local description="$3"
    
    if [ "$actual" -eq "$expected" ]; then
        log_pass "$description (退出码: $actual)"
        return 0
    else
        log_fail "$description (期望退出码: $expected, 实际: $actual)"
        return 1
    fi
}

# 验证输出包含指定内容
check_output_contains() {
    local output="$1"
    local expected="$2"
    local description="$3"
    
    if echo "$output" | grep -q "$expected"; then
        log_pass "$description"
        return 0
    else
        log_fail "$description (未找到: $expected)"
        return 1
    fi
}

# 验证输出不包含指定内容
check_output_not_contains() {
    local output="$1"
    local unexpected="$2"
    local description="$3"
    
    if ! echo "$output" | grep -q "$unexpected"; then
        log_pass "$description"
        return 0
    else
        log_fail "$description (意外找到: $unexpected)"
        return 1
    fi
}

# 测试计数器初始化
init_test_counters() {
    TOTAL_TESTS=0
    PASSED_TESTS=0
    FAILED_TESTS=0
}

# 增加测试计数
increment_test_counter() {
    ((TOTAL_TESTS++))
}

# 增加通过计数
increment_pass_counter() {
    ((PASSED_TESTS++))
}

# 增加失败计数
increment_fail_counter() {
    ((FAILED_TESTS++))
}

# 打印测试结果汇总
print_test_summary() {
    local test_name="${1:-测试}"
    
    echo
    echo "========================================"
    echo "${test_name}结果汇总"
    echo "========================================"
    echo "总测试数: $TOTAL_TESTS"
    echo -e "通过: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "失败: ${RED}$FAILED_TESTS${NC}"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}所有${test_name}通过！${NC}"
        return 0
    else
        echo -e "\n${RED}有 $FAILED_TESTS 个${test_name}失败${NC}"
        return 1
    fi
}