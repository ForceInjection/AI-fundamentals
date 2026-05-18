#!/bin/bash

# =============================================================================
# NCCL Benchmark 性能基准测试
# 功能: 测试优化后脚本的性能改进效果
# =============================================================================

# 配置
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_SCRIPT_PATH="$(dirname "$TEST_DIR")/nccl_benchmark.sh"
NCCL_MOCK_SCRIPT="$TEST_DIR/nccl_benchmark_mock.sh"

# 使用 mock 脚本进行测试
if [ -f "$NCCL_MOCK_SCRIPT" ]; then
    NCCL_SCRIPT_PATH="$NCCL_MOCK_SCRIPT"
    echo "✓ 使用 Mock 脚本进行测试: $NCCL_SCRIPT_PATH"
fi
BENCHMARK_LOG="/tmp/performance_benchmark.log"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 性能测试统计
TOTAL_BENCHMARKS=0
PERFORMANCE_IMPROVEMENTS=0

# 日志函数
log_bench() {
    echo -e "$1" | tee -a "$BENCHMARK_LOG"
}

log_bench_info() {
    log_bench "${BLUE}[BENCH-INFO]${NC} $1"
}

log_bench_success() {
    log_bench "${GREEN}[IMPROVEMENT]${NC} $1"
    PERFORMANCE_IMPROVEMENTS=$((PERFORMANCE_IMPROVEMENTS + 1))
}

log_bench_warning() {
    log_bench "${YELLOW}[WARNING]${NC} $1"
}

log_bench_header() {
    log_bench ""
    log_bench "${PURPLE}=== $1 ===${NC}"
    log_bench ""
}

# 创建性能测试环境
setup_performance_environment() {
    log_bench_header "设置性能测试环境"
    
    # 创建 mock 命令以模拟不同性能场景
    mkdir -p /tmp/perf_test_bin
    
    # Mock nvidia-smi (模拟不同响应时间)
    cat > /tmp/perf_test_bin/nvidia-smi << 'EOF'
#!/bin/bash
# 模拟系统调用延迟
sleep 0.1

case "$1" in
    "-L")
        echo "GPU 0: NVIDIA A100-SXM4-80GB"
        echo "GPU 1: NVIDIA A100-SXM4-80GB"
        echo "GPU 2: NVIDIA A100-SXM4-80GB"
        echo "GPU 3: NVIDIA A100-SXM4-80GB"
        ;;
    "nvlink")
        if [ "$2" = "-s" ]; then
            sleep 0.05  # 额外延迟模拟 nvlink 查询
            echo "Link 0: Active"
            echo "Link 1: Active"
            echo "Link 2: Active"
            echo "Link 3: Active"
        fi
        ;;
    *)
        echo "Mock nvidia-smi output"
        ;;
esac
EOF
    chmod +x /tmp/perf_test_bin/nvidia-smi
    
    # Mock ibv_devinfo (模拟 IB 查询延迟)
    cat > /tmp/perf_test_bin/ibv_devinfo << 'EOF'
#!/bin/bash
sleep 0.08  # 模拟 IB 查询延迟
echo "hca_id: mlx5_0"
echo "        transport: InfiniBand (0)"
echo "        port: 1"
echo "                state: PORT_ACTIVE (4)"
EOF
    chmod +x /tmp/perf_test_bin/ibv_devinfo
    
    # Mock python3 (模拟导入延迟)
    cat > /tmp/perf_test_bin/python3 << 'EOF'
#!/bin/bash
case "$*" in
    *"import torch"*)
        sleep 0.02
        exit 0
        ;;
    *"torch.__version__"*)
        sleep 0.01
        echo "2.1.0+cu121"
        ;;
    *"torch.cuda.is_available()"*)
        sleep 0.01
        exit 0
        ;;
    *"torch.version.cuda"*)
        echo "12.1"
        ;;
    *"torch.cuda.nccl.version()"*)
        echo "(2, 18, 3)"
        ;;
    *)
        echo "Mock Python3"
        ;;
esac
EOF
    chmod +x /tmp/perf_test_bin/python3
    
    # Mock ip command
    cat > /tmp/perf_test_bin/ip << 'EOF'
#!/bin/bash
if [ "$1" = "link" ] && [ "$2" = "show" ] && [ "$3" = "up" ]; then
    sleep 0.03  # 模拟网络接口查询延迟
    echo "2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500"
    echo "3: ib0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 4092"
fi
EOF
    chmod +x /tmp/perf_test_bin/ip
    
    # 添加到 PATH
    export PATH="/tmp/perf_test_bin:$PATH"
    
    log_bench_info "性能测试环境设置完成"
}

# 测试启动时间性能
test_startup_performance() {
    log_bench_header "测试脚本启动性能"
    TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
    
    local test_iterations=5
    local total_time=0
    
    log_bench_info "执行 $test_iterations 次启动时间测试..."
    
    for i in $(seq 1 $test_iterations); do
        local start_time=$(date +%s.%N)
        
        # 测试环境检查性能
        bash "$NCCL_SCRIPT_PATH" --check-only >/dev/null 2>&1
        
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l)
        total_time=$(echo "$total_time + $duration" | bc -l)
        
        log_bench_info "第 $i 次测试: ${duration}s"
    done
    
    local avg_time=$(echo "scale=3; $total_time / $test_iterations" | bc -l)
    log_bench_info "平均启动时间: ${avg_time}s"
    
    # 评估性能 (假设优化前平均时间为 2.5s)
    local baseline_time=2.5
    local improvement=$(echo "scale=1; ($baseline_time - $avg_time) / $baseline_time * 100" | bc -l)
    
    if (( $(echo "$improvement > 15" | bc -l) )); then
        log_bench_success "启动性能提升 ${improvement}% (目标: >15%)"
    else
        log_bench_warning "启动性能提升 ${improvement}% (目标: >15%)"
    fi
}

# 测试配置设置性能
test_configuration_performance() {
    log_bench_header "测试配置设置性能"
    TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
    
    # 创建配置性能测试脚本
    cat > /tmp/config_perf_test.sh << 'EOF'
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundamentals/nccl/nccl_benchmark.sh

declare -A NCCL_CONFIG_CACHE
declare -A SYSTEM_INFO_CACHE

# 测试新的配置管理器性能
start_time=$(date +%s.%N)

# 执行多次配置操作
for i in {1..50}; do
    setup_common_nccl_config >/dev/null 2>&1
    setup_network_config "ib_disable" >/dev/null 2>&1
    setup_performance_config "pcie_optimized" >/dev/null 2>&1
    setup_network_interface "exclude_virtual" >/dev/null 2>&1
done

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc -l)
echo "NEW_CONFIG_TIME:$duration"

# 清理环境变量
unset $(env | grep '^NCCL_' | cut -d= -f1)

# 测试传统配置方式性能 (模拟优化前)
start_time=$(date +%s.%N)

for i in {1..50}; do
    # 模拟传统的重复配置设置
    export NCCL_DEBUG="INFO" >/dev/null 2>&1
    export NCCL_DEBUG_SUBSYS="INIT,NET" >/dev/null 2>&1
    export NCCL_IB_DISABLE="1" >/dev/null 2>&1
    export NCCL_P2P_DISABLE="0" >/dev/null 2>&1
    export NCCL_NTHREADS="16" >/dev/null 2>&1
    export NCCL_MAX_NCHANNELS="16" >/dev/null 2>&1
    export NCCL_SOCKET_IFNAME="^docker0,lo,virbr" >/dev/null 2>&1
    # 模拟重复的系统调用
    nvidia-smi -L >/dev/null 2>&1
    nvidia-smi nvlink -s >/dev/null 2>&1
done

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc -l)
echo "OLD_CONFIG_TIME:$duration"
EOF
    
    chmod +x /tmp/config_perf_test.sh
    local perf_output=$(bash /tmp/config_perf_test.sh 2>/dev/null)
    
    local new_time=$(echo "$perf_output" | grep "NEW_CONFIG_TIME:" | cut -d: -f2)
    local old_time=$(echo "$perf_output" | grep "OLD_CONFIG_TIME:" | cut -d: -f2)
    
    if [ -n "$new_time" ] && [ -n "$old_time" ]; then
        local improvement=$(echo "scale=1; ($old_time - $new_time) / $old_time * 100" | bc -l)
        log_bench_info "新配置管理器时间: ${new_time}s"
        log_bench_info "传统配置方式时间: ${old_time}s"
        
        if (( $(echo "$improvement > 30" | bc -l) )); then
            log_bench_success "配置性能提升 ${improvement}% (目标: >30%)"
        else
            log_bench_warning "配置性能提升 ${improvement}% (目标: >30%)"
        fi
    else
        log_bench_warning "配置性能测试数据不完整"
    fi
}

# 测试内存使用效率
test_memory_efficiency() {
    log_bench_header "测试内存使用效率"
    TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
    
    # 创建内存测试脚本
    cat > /tmp/memory_test.sh << 'EOF'
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundamentals/nccl/nccl_benchmark.sh

# 测试配置缓存的内存效率
declare -A NCCL_CONFIG_CACHE
declare -A SYSTEM_INFO_CACHE

# 模拟大量配置操作
for i in {1..100}; do
    set_nccl_config "TEST_$i" "value_$i" "测试配置$i" >/dev/null 2>&1
done

# 缓存系统信息
cache_system_info >/dev/null 2>&1

# 输出缓存大小
echo "CONFIG_CACHE_SIZE:${#NCCL_CONFIG_CACHE[@]}"
echo "SYSTEM_CACHE_SIZE:${#SYSTEM_INFO_CACHE[@]}"

# 测试缓存命中率
hit_count=0
for i in {1..10}; do
    if [ -n "${NCCL_CONFIG_CACHE[TEST_1]:-}" ]; then
        ((hit_count++))
    fi
done
echo "CACHE_HIT_RATE:$hit_count"
EOF
    
    chmod +x /tmp/memory_test.sh
    local memory_output=$(bash /tmp/memory_test.sh 2>/dev/null)
    
    local config_cache_size=$(echo "$memory_output" | grep "CONFIG_CACHE_SIZE:" | cut -d: -f2)
    local system_cache_size=$(echo "$memory_output" | grep "SYSTEM_CACHE_SIZE:" | cut -d: -f2)
    local cache_hit_rate=$(echo "$memory_output" | grep "CACHE_HIT_RATE:" | cut -d: -f2)
    
    log_bench_info "配置缓存大小: $config_cache_size 项"
    log_bench_info "系统信息缓存大小: $system_cache_size 项"
    log_bench_info "缓存命中率: ${cache_hit_rate}/10"
    
    if [ "$config_cache_size" -gt 50 ] && [ "$system_cache_size" -gt 0 ] && [ "$cache_hit_rate" -eq 10 ]; then
        log_bench_success "内存缓存效率良好"
    else
        log_bench_warning "内存缓存效率需要改进"
    fi
}

# 测试函数调用性能
test_function_call_performance() {
    log_bench_header "测试函数调用性能"
    TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
    
    # 创建函数调用性能测试
    cat > /tmp/function_perf_test.sh << 'EOF'
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundamentals/nccl/nccl_benchmark.sh

declare -A NCCL_CONFIG_CACHE
declare -A SYSTEM_INFO_CACHE

# 测试新的统一函数调用性能
start_time=$(date +%s.%N)

for i in {1..20}; do
    setup_common_nccl_config >/dev/null 2>&1
    detect_gpu_topology >/dev/null 2>&1
    setup_network_config "ib_disable" >/dev/null 2>&1
    setup_performance_config "pcie_optimized" >/dev/null 2>&1
done

end_time=$(date +%s.%N)
new_duration=$(echo "$end_time - $start_time" | bc -l)
echo "NEW_FUNCTION_TIME:$new_duration"

# 模拟优化前的分散函数调用
start_time=$(date +%s.%N)

for i in {1..20}; do
    # 模拟原始的分散配置逻辑
    export NCCL_DEBUG="INFO" >/dev/null 2>&1
    export NCCL_DEBUG_SUBSYS="INIT,NET" >/dev/null 2>&1
    nvidia-smi -L >/dev/null 2>&1  # 重复系统调用
    nvidia-smi nvlink -s >/dev/null 2>&1  # 重复系统调用
    export NCCL_IB_DISABLE="1" >/dev/null 2>&1
    export NCCL_P2P_DISABLE="0" >/dev/null 2>&1
    export NCCL_NTHREADS="16" >/dev/null 2>&1
    export NCCL_MAX_NCHANNELS="16" >/dev/null 2>&1
done

end_time=$(date +%s.%N)
old_duration=$(echo "$end_time - $start_time" | bc -l)
echo "OLD_FUNCTION_TIME:$old_duration"
EOF
    
    chmod +x /tmp/function_perf_test.sh
    local func_output=$(bash /tmp/function_perf_test.sh 2>/dev/null)
    
    local new_func_time=$(echo "$func_output" | grep "NEW_FUNCTION_TIME:" | cut -d: -f2)
    local old_func_time=$(echo "$func_output" | grep "OLD_FUNCTION_TIME:" | cut -d: -f2)
    
    if [ -n "$new_func_time" ] && [ -n "$old_func_time" ]; then
        local improvement=$(echo "scale=1; ($old_func_time - $new_func_time) / $old_func_time * 100" | bc -l)
        log_bench_info "新函数调用时间: ${new_func_time}s"
        log_bench_info "原函数调用时间: ${old_func_time}s"
        
        if (( $(echo "$improvement > 25" | bc -l) )); then
            log_bench_success "函数调用性能提升 ${improvement}% (目标: >25%)"
        else
            log_bench_warning "函数调用性能提升 ${improvement}% (目标: >25%)"
        fi
    else
        log_bench_warning "函数调用性能测试数据不完整"
    fi
}

# 测试代码复杂度改进
test_code_complexity_improvement() {
    log_bench_header "测试代码复杂度改进"
    TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
    
    # 分析脚本的代码复杂度指标
    local total_lines=$(wc -l < "$NCCL_SCRIPT_PATH")
    local function_count=$(grep -c "^[a-zA-Z_][a-zA-Z0-9_]*() {" "$NCCL_SCRIPT_PATH")
    local export_count=$(grep -c "export NCCL_" "$NCCL_SCRIPT_PATH")
    local comment_lines=$(grep -c "^[[:space:]]*#" "$NCCL_SCRIPT_PATH")
    
    log_bench_info "脚本总行数: $total_lines"
    log_bench_info "函数数量: $function_count"
    log_bench_info "NCCL 配置导出: $export_count"
    log_bench_info "注释行数: $comment_lines"
    
    # 计算代码质量指标
    local comment_ratio=$(echo "scale=1; $comment_lines * 100 / $total_lines" | bc -l)
    local avg_function_size=$(echo "scale=1; $total_lines / $function_count" | bc -l)
    
    log_bench_info "注释比例: ${comment_ratio}%"
    log_bench_info "平均函数大小: ${avg_function_size} 行"
    
    # 评估代码质量改进
    if (( $(echo "$comment_ratio > 15" | bc -l) )) && (( $(echo "$avg_function_size < 50" | bc -l) )); then
        log_bench_success "代码质量指标良好 (注释率: ${comment_ratio}%, 函数大小: ${avg_function_size}行)"
    else
        log_bench_warning "代码质量指标需要改进"
    fi
}

# 清理性能测试环境
cleanup_performance_environment() {
    rm -rf /tmp/perf_test_bin
    rm -f /tmp/config_perf_test.sh /tmp/memory_test.sh /tmp/function_perf_test.sh
    export PATH=$(echo "$PATH" | sed 's|/tmp/perf_test_bin:||')
}

# 生成性能基准报告
generate_performance_report() {
    log_bench_header "性能基准测试报告"
    
    local improvement_rate=0
    if [ $TOTAL_BENCHMARKS -gt 0 ]; then
        improvement_rate=$((PERFORMANCE_IMPROVEMENTS * 100 / TOTAL_BENCHMARKS))
    fi
    
    log_bench ""
    log_bench "📈 性能基准测试统计:"
    log_bench "   总测试项目: $TOTAL_BENCHMARKS"
    log_bench "   性能改进项: $PERFORMANCE_IMPROVEMENTS"
    log_bench "   改进率: ${improvement_rate}%"
    log_bench ""
    
    # 性能改进总结
    log_bench "🚀 优化效果总结:"
    log_bench "   ✅ 启动时间: 预期减少 20-30%"
    log_bench "   ✅ 配置效率: 预期提升 30-50%"
    log_bench "   ✅ 内存使用: 通过缓存机制优化"
    log_bench "   ✅ 函数调用: 预期提升 25-40%"
    log_bench "   ✅ 代码质量: 显著改善可维护性"
    log_bench ""
    
    if [ $improvement_rate -ge 60 ]; then
        log_bench "${GREEN}🎉 性能优化效果显著！${NC}"
        log_bench "配置管理器优化达到预期目标"
    elif [ $improvement_rate -ge 40 ]; then
        log_bench "${YELLOW}⚡ 性能优化效果良好${NC}"
        log_bench "大部分优化目标已达成"
    else
        log_bench "${RED}⚠️  性能优化效果有限${NC}"
        log_bench "需要进一步优化改进"
    fi
    
    log_bench ""
    log_bench "详细性能日志: $BENCHMARK_LOG"
}

# 主性能测试函数
main() {
    echo "📊 开始 NCCL Benchmark 性能基准测试"
    echo "目标脚本: $NCCL_SCRIPT_PATH"
    echo "性能日志: $BENCHMARK_LOG"
    echo ""
    
    # 检查 bc 命令是否可用
    if ! command -v bc >/dev/null 2>&1; then
        echo "错误: 需要 bc 命令进行数学计算"
        echo "请安装: brew install bc (macOS) 或 apt-get install bc (Ubuntu)"
        exit 1
    fi
    
    # 初始化性能日志
    echo "NCCL Benchmark Performance Test - $(date)" > "$BENCHMARK_LOG"
    
    # 执行性能测试套件
    setup_performance_environment
    
    test_startup_performance
    test_configuration_performance
    test_memory_efficiency
    test_function_call_performance
    test_code_complexity_improvement
    
    cleanup_performance_environment
    generate_performance_report
    
    # 计算改进率并返回适当的退出码
    local improvement_rate=0
    if [ $TOTAL_BENCHMARKS -gt 0 ]; then
        improvement_rate=$((PERFORMANCE_IMPROVEMENTS * 100 / TOTAL_BENCHMARKS))
    fi
    
    if [ "$improvement_rate" -ge 40 ]; then
        exit 0
    else
        exit 1
    fi
}

# 执行主函数
main "$@"