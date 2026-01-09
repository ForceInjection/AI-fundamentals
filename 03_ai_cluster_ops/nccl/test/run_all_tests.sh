#!/bin/bash
# =============================================================================
# NCCL Benchmark 完整测试套件运行器
# 功能: 统一管理和运行所有测试脚本，提供完整的测试报告
# =============================================================================

# 版本信息
VERSION="2.0"
SCRIPT_NAME="NCCL Benchmark Test Suite"

# 测试配置
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_SCRIPT_PATH="$(dirname "$TEST_DIR")/nccl_benchmark.sh"
NCCL_MOCK_SCRIPT="$TEST_DIR/nccl_benchmark_mock.sh"
TEST_RESULTS_DIR="/tmp/nccl_test_results_$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$TEST_RESULTS_DIR/test_suite.log"

# Mock 配置
MOCK_SCENARIO=""
USE_MOCK_MODE=true

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 测试统计
TOTAL_TEST_SUITES=0
PASSED_TEST_SUITES=0
FAILED_TEST_SUITES=0
SKIPPED_TEST_SUITES=0

# 测试模式
QUICK_MODE=false
VERBOSE_MODE=false
PERFORMANCE_MODE=false
INTEGRATION_MODE=false

# 日志函数
log() {
    # 确保日志目录存在
    mkdir -p "$(dirname "$MAIN_LOG")" 2>/dev/null || true
    echo -e "$1" | tee -a "$MAIN_LOG"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_header() {
    log ""
    log "${PURPLE}=== $1 ===${NC}"
    log ""
}

log_suite_result() {
    local suite_name="$1"
    local result="$2"
    local details="$3"
    
    case "$result" in
        "PASS")
            PASSED_TEST_SUITES=$((PASSED_TEST_SUITES + 1))
            log "${GREEN}[SUITE-PASS]${NC} $suite_name $details"
            ;;
        "FAIL")
            FAILED_TEST_SUITES=$((FAILED_TEST_SUITES + 1))
            log "${RED}[SUITE-FAIL]${NC} $suite_name $details"
            ;;
        "SKIP")
            SKIPPED_TEST_SUITES=$((SKIPPED_TEST_SUITES + 1))
            log "${YELLOW}[SUITE-SKIP]${NC} $suite_name $details"
            ;;
    esac
}

# 显示帮助信息
show_help() {
    cat << EOF
$SCRIPT_NAME v$VERSION

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -v, --version           显示版本信息
  --verbose               详细输出模式
  -q, --quick             快速测试模式 (跳过性能测试)
  -p, --performance       仅运行性能测试
  -i, --integration       仅运行集成测试
  --list                  列出所有可用的测试套件
  --suite SUITE_NAME      运行指定的测试套件
  --mock-scenario SCENARIO 设置 Mock 测试场景
  --no-mock               禁用 Mock 模式，使用原始脚本

测试套件:
  syntax              语法和基础功能验证
  config              配置管理器测试
  mock                Mock 环境测试
  nvlink              NVLink 计数测试
  dns                 DNS 解析逻辑测试
  optimization        优化级别测试
  network-fix         网络配置修复验证
  pxn                 PXN 模式功能测试
  performance         性能基准测试
  integration         集成测试

Mock 场景:
  single_gpu              单 GPU 环境
  multi_gpu_nvlink        多 GPU + NVLink 环境
  multi_gpu_pcie          多 GPU + PCIe 环境
  cluster_ib              集群 + InfiniBand 环境

示例:
  $0                      # 运行所有测试套件 (默认 Mock 模式)
  $0 --quick              # 快速测试模式
  $0 --suite syntax       # 仅运行语法测试
  $0 --performance        # 仅运行性能测试
  $0 --verbose            # 详细输出模式
  $0 --mock-scenario=multi_gpu_nvlink  # 使用特定 Mock 场景
  $0 --no-mock            # 禁用 Mock，使用真实环境

EOF
}

# 显示版本信息
show_version() {
    echo "$SCRIPT_NAME v$VERSION"
    echo "NCCL Benchmark 脚本完整测试套件"
}

# 列出所有测试套件
list_test_suites() {
    log_header "可用测试套件"
    
    local suites=(
        "syntax:语法和基础功能验证:test_syntax_basic.sh"
        "config:配置管理器测试:test_config_manager.sh"
        "mock:Mock 环境测试:test_mock_environment.sh"
        "nvlink:NVLink 计数测试:test_nvlink_count.sh"
        "dns:DNS 解析逻辑测试:test_dns_resolution.sh"
        "optimization:优化级别测试:test_optimization_levels.sh"
        "network-fix:网络配置修复验证:test_network_config_fix.sh"
        "pxn:PXN 模式功能测试:test_pxn_mode.sh"
        "performance:性能基准测试:test_performance_benchmark.sh"
    )
    
    for suite in "${suites[@]}"; do
        IFS=':' read -r name desc script <<< "$suite"
        local status="❓"
        if [ -f "$TEST_DIR/$script" ]; then
            status="✅"
        else
            status="❌"
        fi
        log "$status $name - $desc ($script)"
    done
}

# 初始化测试环境
setup_test_environment() {
    log_header "初始化测试环境"
    
    # 创建测试结果目录
    mkdir -p "$TEST_RESULTS_DIR"
    
    # 检查原始脚本
    if [ ! -f "$NCCL_SCRIPT_PATH" ]; then
        log_error "原始脚本不存在: $NCCL_SCRIPT_PATH"
        exit 1
    fi
    
    # 根据配置选择测试脚本
    if [ "$USE_MOCK_MODE" = true ]; then
        # 检查增强版 mock 脚本
        if [ ! -f "$NCCL_MOCK_SCRIPT" ]; then
            log_error "增强版 Mock 脚本不存在: $NCCL_MOCK_SCRIPT"
            exit 1
        fi
        
        # 确保 mock 脚本可执行
        chmod +x "$NCCL_MOCK_SCRIPT" 2>/dev/null || {
            log_error "无法设置 mock 脚本执行权限"
            exit 1
        }
        
        # 构建 mock 脚本参数
        NCCL_TEST_CMD=("$NCCL_MOCK_SCRIPT")
        if [ -n "$MOCK_SCENARIO" ]; then
            NCCL_TEST_CMD+=("--mock-scenario=$MOCK_SCENARIO")
        fi
        
        log_success "测试环境初始化完成 (Mock 模式)"
        log_info "Mock 脚本: $NCCL_MOCK_SCRIPT"
        if [ -n "$MOCK_SCENARIO" ]; then
            log_info "Mock 场景: $MOCK_SCENARIO"
        fi
    else
        # 使用原始脚本
        NCCL_TEST_CMD=("$NCCL_SCRIPT_PATH")
        log_success "测试环境初始化完成 (原始脚本模式)"
    fi
    
    log_info "测试结果目录: $TEST_RESULTS_DIR"
    log_info "原始脚本: $NCCL_SCRIPT_PATH"
    log_info "测试命令: ${NCCL_TEST_CMD[*]}"
    log_info "主日志文件: $MAIN_LOG"
}

# 运行单个测试套件
run_test_suite() {
    local suite_name="$1"
    local script_name="$2"
    local description="$3"
    
    TOTAL_TEST_SUITES=$((TOTAL_TEST_SUITES + 1))
    
    log_header "运行测试套件: $suite_name"
    log_info "描述: $description"
    log_info "脚本: $script_name"
    
    local script_path="$TEST_DIR/$script_name"
    local suite_log="$TEST_RESULTS_DIR/${suite_name}_test.log"
    
    # 检查测试脚本是否存在
    if [ ! -f "$script_path" ]; then
        log_suite_result "$suite_name" "SKIP" "(脚本不存在: $script_name)"
        return 0
    fi
    
    # 确保脚本可执行
    chmod +x "$script_path" 2>/dev/null || true
    
    # 运行测试
    local start_time=$(date +%s)
    local exit_code=0
    
    if [ "$VERBOSE_MODE" = true ]; then
        log_info "开始执行测试套件..."
        bash "$script_path" 2>&1 | tee "$suite_log"
        exit_code=${PIPESTATUS[0]}
    else
        bash "$script_path" > "$suite_log" 2>&1
        exit_code=$?
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 分析测试结果
    local test_summary=""
    if [ -f "$suite_log" ]; then
        local passed_count=$(grep -c "\[PASS\]\|\[SUCCESS\]\|✅" "$suite_log" 2>/dev/null || echo "0")
        local failed_count=$(grep -c "\[FAIL\]\|\[ERROR\]\|❌" "$suite_log" 2>/dev/null || echo "0")
        test_summary="(${passed_count} passed, ${failed_count} failed, ${duration}s)"
    else
        test_summary="(no log generated, ${duration}s)"
    fi
    
    # 记录结果
    if [ $exit_code -eq 0 ]; then
        log_suite_result "$suite_name" "PASS" "$test_summary"
    else
        log_suite_result "$suite_name" "FAIL" "$test_summary"
        if [ "$VERBOSE_MODE" = false ]; then
            log_warning "查看详细错误信息: $suite_log"
        fi
    fi
    
    return $exit_code
}

# 运行语法测试
run_syntax_tests() {
    if [ "$QUICK_MODE" = true ] || [ "$PERFORMANCE_MODE" = true ]; then
        return 0
    fi
    
    run_test_suite "syntax" "test_syntax_basic.sh" "语法和基础功能验证"
}

# 运行配置测试
run_config_tests() {
    if [ "$PERFORMANCE_MODE" = true ]; then
        return 0
    fi
    
    run_test_suite "config" "test_config_manager.sh" "配置管理器功能测试"
}

# 运行 Mock 测试
run_mock_tests() {
    if [ "$QUICK_MODE" = true ] || [ "$PERFORMANCE_MODE" = true ]; then
        return 0
    fi
    
    run_test_suite "mock" "test_mock_environment.sh" "Mock 环境功能测试"
}

# 运行 NVLink 测试
run_nvlink_tests() {
    if [ "$PERFORMANCE_MODE" = true ]; then
        return 0
    fi
    
    run_test_suite "nvlink" "test_nvlink_count.sh" "NVLink 计数测试"
}

# 运行 DNS 测试
run_dns_tests() {
    if [ "$QUICK_MODE" = true ] || [ "$PERFORMANCE_MODE" = true ]; then
        return 0
    fi
    
    run_test_suite "dns" "test_dns_resolution.sh" "DNS 解析逻辑测试"
}

# 运行优化级别测试
run_optimization_tests() {
    if [ "$PERFORMANCE_MODE" = true ]; then
        return 0
    fi
    
    run_test_suite "optimization" "test_optimization_levels.sh" "优化级别测试"
}

# 运行网络配置修复测试
run_network_fix_tests() {
    if [ "$PERFORMANCE_MODE" = true ]; then
        return 0
    fi
    
    # 设置测试模式环境变量
    export TEST_MODE=true
    run_test_suite "network-fix" "test_network_config_fix.sh" "网络配置修复验证测试"
    unset TEST_MODE
}

# 运行 PXN 模式测试
run_pxn_tests() {
    if [ "$PERFORMANCE_MODE" = true ]; then
        return 0
    fi
    
    run_test_suite "pxn" "test_pxn_mode.sh" "PXN 模式功能测试"
}

# 运行性能测试
run_performance_tests() {
    if [ "$QUICK_MODE" = true ]; then
        return 0
    fi
    
    run_test_suite "performance" "test_performance_benchmark.sh" "性能基准测试"
}



# 运行集成测试
run_integration_tests() {
    if [ "$QUICK_MODE" = true ] || [ "$PERFORMANCE_MODE" = true ]; then
        return 0
    fi
    
    log_header "集成测试"
    log_info "运行 NCCL Benchmark 脚本的实际测试..."
    
    local integration_log="$TEST_RESULTS_DIR/integration_test.log"
    local exit_code=0
    
    # 测试不同的网络后端
    local backends=("auto" "socket" "ethernet")
    local integration_passed=0
    local integration_total=0
    
    for backend in "${backends[@]}"; do
        integration_total=$((integration_total + 1))
        log_info "测试网络后端: $backend"
        
        if timeout 60 "${NCCL_TEST_CMD[@]}" --dry-run --network "$backend" >> "$integration_log" 2>&1; then
            log_success "网络后端 $backend 测试通过"
            integration_passed=$((integration_passed + 1))
        else
            log_error "网络后端 $backend 测试失败"
        fi
    done
    
    TOTAL_TEST_SUITES=$((TOTAL_TEST_SUITES + 1))
    if [ $integration_passed -eq $integration_total ]; then
        log_suite_result "integration" "PASS" "($integration_passed/$integration_total backends)"
    else
        log_suite_result "integration" "FAIL" "($integration_passed/$integration_total backends)"
    fi
}

# 生成测试报告
generate_test_report() {
    log_header "测试报告"
    
    local total_tests=$((PASSED_TEST_SUITES + FAILED_TEST_SUITES + SKIPPED_TEST_SUITES))
    local success_rate=0
    
    if [ $total_tests -gt 0 ]; then
        success_rate=$(( (PASSED_TEST_SUITES * 100) / total_tests ))
    fi
    
    log_info "测试套件统计:"
    log_info "  总计: $total_tests"
    log_info "  通过: $PASSED_TEST_SUITES"
    log_info "  失败: $FAILED_TEST_SUITES"
    log_info "  跳过: $SKIPPED_TEST_SUITES"
    log_info "  成功率: ${success_rate}%"
    
    # 生成详细报告文件
    local report_file="$TEST_RESULTS_DIR/test_report.md"
    cat > "$report_file" << EOF
# NCCL Benchmark 测试报告

**测试时间**: $(date)
**测试版本**: $VERSION
**目标脚本**: $NCCL_SCRIPT_PATH

## 测试统计

- **总计**: $total_tests 个测试套件
- **通过**: $PASSED_TEST_SUITES 个
- **失败**: $FAILED_TEST_SUITES 个
- **跳过**: $SKIPPED_TEST_SUITES 个
- **成功率**: ${success_rate}%

## 测试结果详情

EOF
    
    # 添加各个测试套件的详细结果
    for log_file in "$TEST_RESULTS_DIR"/*_test.log; do
        if [ -f "$log_file" ]; then
            local suite_name=$(basename "$log_file" _test.log)
            echo "### $suite_name 测试套件" >> "$report_file"
            echo '```' >> "$report_file"
            tail -20 "$log_file" >> "$report_file"
            echo '```' >> "$report_file"
            echo "" >> "$report_file"
        fi
    done
    
    log_success "详细测试报告已生成: $report_file"
    
    # 返回适当的退出码
    if [ $FAILED_TEST_SUITES -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                show_version
                exit 0
                ;;
            --verbose)
                VERBOSE_MODE=true
                shift
                ;;
            -q|--quick)
                QUICK_MODE=true
                shift
                ;;
            -p|--performance)
                PERFORMANCE_MODE=true
                shift
                ;;
            -i|--integration)
                INTEGRATION_MODE=true
                shift
                ;;
            --list)
                list_test_suites
                exit 0
                ;;
            --suite)
                if [ -z "$2" ]; then
                    log_error "--suite 选项需要参数"
                    exit 1
                fi
                SINGLE_SUITE="$2"
                shift 2
                ;;
            --mock-scenario)
                if [ -z "$2" ]; then
                    log_error "--mock-scenario 选项需要参数"
                    exit 1
                fi
                MOCK_SCENARIO="$2"
                USE_MOCK_MODE=true
                shift 2
                ;;
            --mock-scenario=*)
                MOCK_SCENARIO="${1#*=}"
                USE_MOCK_MODE=true
                shift
                ;;
            --no-mock)
                USE_MOCK_MODE=false
                shift
                ;;
            *)
                log_error "未知选项: $1"
                echo "使用 '$0 --help' 查看帮助信息"
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    # 解析参数
    parse_arguments "$@"
    
    # 初始化环境
    setup_test_environment
    
    log_header "开始 NCCL Benchmark 测试套件"
    log_info "测试模式: $([ "$QUICK_MODE" = true ] && echo "快速" || echo "完整")"
    log_info "详细输出: $([ "$VERBOSE_MODE" = true ] && echo "启用" || echo "禁用")"
    log_info "Mock 模式: $([ "$USE_MOCK_MODE" = true ] && echo "启用" || echo "禁用")"
    if [ "$USE_MOCK_MODE" = true ] && [ -n "$MOCK_SCENARIO" ]; then
        log_info "Mock 场景: $MOCK_SCENARIO"
    fi
    
    # 运行指定的单个测试套件
    if [ -n "${SINGLE_SUITE:-}" ]; then
        case "$SINGLE_SUITE" in
            syntax) run_syntax_tests ;;
            config) run_config_tests ;;
            mock) run_mock_tests ;;
            nvlink) run_nvlink_tests ;;
            dns) run_dns_tests ;;
            optimization) run_optimization_tests ;;
            network-fix) run_network_fix_tests ;;
            pxn) run_pxn_tests ;;
            performance) run_performance_tests ;;
            integration) run_integration_tests ;;
            *)
                log_error "未知的测试套件: $SINGLE_SUITE"
                list_test_suites
                exit 1
                ;;
        esac
    elif [ "$PERFORMANCE_MODE" = true ]; then
        # 仅运行性能测试
        run_performance_tests
    elif [ "$INTEGRATION_MODE" = true ]; then
        # 仅运行集成测试
        run_integration_tests
    else
        # 运行所有适用的测试套件
        run_syntax_tests
        run_config_tests
        run_nvlink_tests
        run_mock_tests
        run_dns_tests
        run_optimization_tests
        run_network_fix_tests
        run_pxn_tests
        run_performance_tests
        run_integration_tests
    fi
    
    # 生成测试报告
    generate_test_report
    local report_exit_code=$?
    
    log_header "测试套件完成"
    if [ $report_exit_code -eq 0 ]; then
        log_success "所有测试套件执行完成，无失败项"
    else
        log_error "测试套件执行完成，存在失败项"
    fi
    
    exit $report_exit_code
}

# 运行主函数
main "$@"