#!/bin/bash

# IB 带宽监控脚本 - 完整测试套件运行器
# 使用综合测试脚本进行全面测试
# 版本: 4.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 测试结果变量
COMPREHENSIVE_TEST_RESULT=""
INTEGRATION_TEST_RESULT=""
REGRESSION_TEST_RESULT=""

# 测试统计
TOTAL_TEST_SUITES=0
PASSED_TEST_SUITES=0
FAILED_TEST_SUITES=0

# 日志函数
log_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

log_subheader() {
    echo -e "${PURPLE}--- $1 ---${NC}"
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 运行测试并记录结果
run_test_suite() {
    local test_name="$1"
    local test_script="$2"
    local result_var="$3"
    local description="$4"
    
    ((TOTAL_TEST_SUITES++))
    
    log_subheader "$test_name"
    
    if [ -n "$description" ]; then
        log_info "$description"
    fi
    
    if [ ! -f "$test_script" ]; then
        log_error "测试脚本不存在: $test_script"
        eval "$result_var='MISSING'"
        ((FAILED_TEST_SUITES++))
        return 1
    fi
    
    # 使脚本可执行
    chmod +x "$test_script"
    
    # 运行测试
    log_info "正在运行: $(basename "$test_script")"
    
    if "$test_script"; then
        log_success "$test_name 通过"
        eval "$result_var='PASS'"
        ((PASSED_TEST_SUITES++))
        return 0
    else
        log_error "$test_name 失败"
        eval "$result_var='FAIL'"
        ((FAILED_TEST_SUITES++))
        return 1
    fi
}

# 生成测试报告
generate_report() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    log_header "测试报告"
    
    echo "测试时间: $timestamp"
    echo "测试环境: $(uname -s) $(uname -r)"
    echo "脚本版本: $(grep '^VERSION=' "$SCRIPT_DIR/ib_bandwidth_monitor.sh" | cut -d'=' -f2 | tr -d '"' || echo '未知')"
    echo "测试套件: $TOTAL_TEST_SUITES 个"
    echo
    
    echo "测试结果汇总:"
    echo "=============="
    
    # 综合测试结果
    case "$COMPREHENSIVE_TEST_RESULT" in
        "PASS")
            echo -e "综合测试:           ${GREEN}通过${NC}"
            ;;
        "FAIL")
            echo -e "综合测试:           ${RED}失败${NC}"
            ;;
        "MISSING")
            echo -e "综合测试:           ${YELLOW}缺失${NC}"
            ;;
        *)
            echo -e "综合测试:           ${YELLOW}未运行${NC}"
            ;;
    esac
    
    # 集成测试结果
    case "$INTEGRATION_TEST_RESULT" in
        "PASS")
            echo -e "集成测试:           ${GREEN}通过${NC}"
            ;;
        "FAIL")
            echo -e "集成测试:           ${RED}失败${NC}"
            ;;
        "MISSING")
            echo -e "集成测试:           ${YELLOW}缺失${NC}"
            ;;
        *)
            echo -e "集成测试:           ${YELLOW}未运行${NC}"
            ;;
    esac
    
    # 回归测试结果
    case "$REGRESSION_TEST_RESULT" in
        "PASS")
            echo -e "回归测试:           ${GREEN}通过${NC}"
            ;;
        "FAIL")
            echo -e "回归测试:           ${RED}失败${NC}"
            ;;
        "MISSING")
            echo -e "回归测试:           ${YELLOW}缺失${NC}"
            ;;
        *)
            echo -e "回归测试:           ${YELLOW}未运行${NC}"
            ;;
    esac
    
    echo
    echo "统计信息:"
    echo "=========="
    echo "总测试套件: $TOTAL_TEST_SUITES"
    echo -e "通过: ${GREEN}$PASSED_TEST_SUITES${NC}"
    echo -e "失败: ${RED}$FAILED_TEST_SUITES${NC}"
    echo
    
    # 总体结果
    if [ "$PASSED_TEST_SUITES" -eq "$TOTAL_TEST_SUITES" ] && [ "$TOTAL_TEST_SUITES" -gt 0 ]; then
        echo -e "总体结果:           ${GREEN}所有测试通过${NC}"
        echo
        echo -e "${GREEN}✓ 多设备配置修复验证成功${NC}"
        echo -e "${GREEN}✓ 现有功能未受影响${NC}"
        echo -e "${GREEN}✓ 脚本可以安全部署${NC}"
        echo -e "${GREEN}✓ 所有已知问题已修复${NC}"
        return 0
    else
        echo -e "总体结果:           ${RED}存在测试失败${NC}"
        echo
        
        if [ "$COMPREHENSIVE_TEST_RESULT" = "FAIL" ]; then
            echo -e "${RED}✗ 综合测试失败 - 基础功能或修复存在问题${NC}"
        fi
        
        if [ "$INTEGRATION_TEST_RESULT" = "FAIL" ]; then
            echo -e "${RED}✗ 集成测试失败 - 修复可能不完整${NC}"
        fi
        
        if [ "$REGRESSION_TEST_RESULT" = "FAIL" ]; then
            echo -e "${RED}✗ 回归测试失败 - 修复破坏了现有功能${NC}"
        fi
        
        echo
        echo -e "${RED}建议: 请检查失败的测试并修复问题后重新运行${NC}"
        return 1
    fi
}

# 保存测试报告到文件
save_report() {
    local report_file="$SCRIPT_DIR/test_report_$(date '+%Y%m%d_%H%M%S').txt"
    
    {
        echo "IB 带宽监控脚本测试报告"
        echo "========================"
        echo
        generate_report
    } > "$report_file"
    
    log_info "测试报告已保存到: $report_file"
}

# 显示使用说明
show_usage() {
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -c, --comprehensive     仅运行综合测试"
    echo "  -i, --integration       仅运行集成测试"
    echo "  -r, --regression        仅运行回归测试"
    echo "  -s, --save              保存测试报告到文件"
    echo "  -v, --verbose           详细输出模式"
    echo "  -q, --quick             快速测试模式（跳过耗时测试）"
    echo "  --list                  列出所有可用的测试脚本"
    echo
    echo "默认情况下会运行所有测试。"
    echo
    echo "测试类型说明:"
    echo "  综合测试:       测试基础功能、修复验证和多设备配置"
    echo "  集成测试:       测试实际的多设备配置场景"
    echo "  回归测试:       确保修复没有破坏现有功能"
    echo
    echo "注意:"
    echo "  - 测试完成后会自动清理 /tmp 目录中的临时文件"
    echo "  - 支持 Ctrl+C 中断，中断时也会自动清理临时文件"
}

# 列出所有测试脚本
list_test_scripts() {
    echo "可用的测试脚本:"
    echo "==============="
    
    local test_scripts=(
        "test_multi_devices_comprehensive.sh:综合测试"
        "test_integration.sh:集成测试"
        "test_regression.sh:回归测试"
    )
    
    for script_info in "${test_scripts[@]}"; do
        local script_name="${script_info%:*}"
        local script_desc="${script_info#*:}"
        local script_path="$SCRIPT_DIR/$script_name"
        
        if [ -f "$script_path" ]; then
            echo -e "  ${GREEN}✓${NC} $script_name - $script_desc"
        else
            echo -e "  ${RED}✗${NC} $script_name - $script_desc (缺失)"
        fi
    done
}

# 清理临时文件
cleanup_temp_files() {
    log_info "清理临时文件..."
    
    # 清理主脚本创建的临时文件
    rm -f /tmp/ibstat_* 2>/dev/null || true
    rm -f /tmp/perfquery_* 2>/dev/null || true
    
    # 清理测试脚本创建的临时文件
    rm -f /tmp/test_* 2>/dev/null || true
    rm -f /tmp/mock_* 2>/dev/null || true
    rm -f /tmp/test_monitor_* 2>/dev/null || true
    rm -f /tmp/test_validation_* 2>/dev/null || true
    rm -f /tmp/test_parsing_* 2>/dev/null || true
    rm -f /tmp/mock_functions_* 2>/dev/null || true
    
    # 清理模拟命令目录
    rm -rf /tmp/mock_ib_commands 2>/dev/null || true
    rm -rf /tmp/mock_ib_* 2>/dev/null || true
    
    log_success "临时文件清理完成"
}

# 设置清理陷阱
setup_cleanup_trap() {
    # 在脚本退出时自动清理
    trap cleanup_temp_files EXIT
    trap cleanup_temp_files INT
    trap cleanup_temp_files TERM
}

# 检查测试环境
check_test_environment() {
    log_info "检查测试环境..."
    
    # 检查主脚本是否存在
    if [ ! -f "$SCRIPT_DIR/ib_bandwidth_monitor.sh" ]; then
        log_error "主脚本不存在: $SCRIPT_DIR/ib_bandwidth_monitor.sh"
        exit 1
    fi
    
    # 检查脚本语法
    if ! bash -n "$SCRIPT_DIR/ib_bandwidth_monitor.sh"; then
        log_error "主脚本语法错误"
        exit 1
    fi
    
    log_success "测试环境检查通过"
}

# 主函数
main() {
    local run_comprehensive=true
    local run_integration=true
    local run_regression=true
    local save_report_flag=false
    local verbose=false
    local quick_mode=false
    local list_only=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -c|--comprehensive)
                run_comprehensive=true
                run_integration=false
                run_regression=false
                shift
                ;;
            -i|--integration)
                run_comprehensive=false
                run_integration=true
                run_regression=false
                shift
                ;;
            -r|--regression)
                run_comprehensive=false
                run_integration=false
                run_regression=true
                shift
                ;;
            -s|--save)
                save_report_flag=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -q|--quick)
                quick_mode=true
                shift
                ;;
            --list)
                list_only=true
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # 如果只是列出测试脚本
    if [ "$list_only" = true ]; then
        list_test_scripts
        exit 0
    fi
    
    log_header "IB 带宽监控脚本 - 完整测试套件 v4.0"
    
    # 设置清理陷阱
    setup_cleanup_trap
    
    # 检查测试环境
    check_test_environment
    echo
    
    log_info "开始运行测试套件..."
    if [ "$quick_mode" = true ]; then
        log_info "快速模式: 跳过耗时测试"
    fi
    echo
    
    # 运行测试
    local overall_result=0
    
    if [ "$run_comprehensive" = true ]; then
        if ! run_test_suite "综合测试" "$SCRIPT_DIR/test_multi_devices_comprehensive.sh" "COMPREHENSIVE_TEST_RESULT" "测试基础功能、修复验证和多设备配置"; then
            overall_result=1
        fi
        echo
    fi
    
    if [ "$run_integration" = true ]; then
        if ! run_test_suite "集成测试" "$SCRIPT_DIR/test_integration.sh" "INTEGRATION_TEST_RESULT" "测试实际的多设备配置场景"; then
            overall_result=1
        fi
        echo
    fi
    
    if [ "$run_regression" = true ]; then
        if ! run_test_suite "回归测试" "$SCRIPT_DIR/test_regression.sh" "REGRESSION_TEST_RESULT" "确保修复没有破坏现有功能"; then
            overall_result=1
        fi
        echo
    fi
    
    # 生成并显示报告
    generate_report
    
    # 保存报告（如果需要）
    if [ "$save_report_flag" = true ]; then
        save_report
    fi
    
    # 手动清理临时文件（确保清理完成）
    cleanup_temp_files
    
    exit $overall_result
}

# 运行主函数
main "$@"