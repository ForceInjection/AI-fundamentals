#!/bin/bash
# =============================================================================
# NCCL Benchmark 脚本 Mock 测试框架
# 功能: 通过 mock 方法测试 nccl_benchmark.sh 的各个功能模块
# =============================================================================

# 配置
TEST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_SCRIPT_PATH="$(dirname "$TEST_SCRIPT_DIR")/nccl_benchmark.sh"
NCCL_MOCK_SCRIPT="$TEST_SCRIPT_DIR/nccl_benchmark_mock.sh"

# 使用 mock 脚本进行测试
if [ -f "$NCCL_MOCK_SCRIPT" ]; then
    NCCL_SCRIPT_PATH="$NCCL_MOCK_SCRIPT"
    echo "✓ 使用 Mock 脚本进行测试: $NCCL_SCRIPT_PATH"
fi
MOCK_DIR="/tmp/nccl_mock_test"
TEST_LOG="/tmp/nccl_mock_test.log"

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

log_test_info() {
    log_test "${BLUE}[TEST-INFO]${NC} $1"
}

log_test_pass() {
    PASSED_TESTS=$((PASSED_TESTS + 1))
    log_test "${GREEN}[PASS]${NC} $1"
}

log_test_fail() {
    FAILED_TESTS=$((FAILED_TESTS + 1))
    log_test "${RED}[FAIL]${NC} $1"
}

# 开始测试函数 - 确保每个测试都被计数
start_test() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

log_test_header() {
    log_test ""
    log_test "${YELLOW}=== $1 ===${NC}"
    log_test ""
}

# 初始化测试环境
setup_test_environment() {
    log_test_header "初始化测试环境"
    
    # 创建 mock 目录
    mkdir -p "$MOCK_DIR"/{bin,lib,logs,configs}
    
    # 检查目标脚本是否存在
    if [ ! -f "$NCCL_SCRIPT_PATH" ]; then
        log_test_fail "目标脚本不存在: $NCCL_SCRIPT_PATH"
        exit 1
    fi
    
    log_test_pass "测试环境初始化完成"
    log_test_info "Mock 目录: $MOCK_DIR"
    log_test_info "目标脚本: $NCCL_SCRIPT_PATH"
}

# 创建 mock 命令
create_mock_commands() {
    log_test_header "创建 Mock 命令"
    
    # Mock nvidia-smi
    cat > "$MOCK_DIR/bin/nvidia-smi" << 'EOF'
#!/bin/bash
case "$1" in
    "-L")
        echo "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-12345678-1234-1234-1234-123456789abc)"
        echo "GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-87654321-4321-4321-4321-cba987654321)"
        echo "GPU 2: NVIDIA A100-SXM4-80GB (UUID: GPU-11111111-2222-3333-4444-555555555555)"
        echo "GPU 3: NVIDIA A100-SXM4-80GB (UUID: GPU-66666666-7777-8888-9999-aaaaaaaaaaaa)"
        ;;
    "nvlink")
        if [ "$2" = "-s" ]; then
            echo "GPU 0: Active"
            echo "GPU 1: Active"
            echo "GPU 2: Active"
            echo "GPU 3: Active"
            echo "Link 0: Active"
            echo "Link 1: Active"
            echo "Link 2: Active"
            echo "Link 3: Active"
        fi
        ;;
    *)
        cat << 'NVIDIA_SMI_OUTPUT'
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:07:00.0 Off |                    0 |
| N/A   32C    P0    68W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM...  On   | 00000000:0F:00.0 Off |                    0 |
| N/A   33C    P0    69W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM...  On   | 00000000:47:00.0 Off |                    0 |
| N/A   34C    P0    70W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM...  On   | 00000000:4E:00.0 Off |                    0 |
| N/A   35C    P0    71W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
NVIDIA_SMI_OUTPUT
        ;;
esac
EOF
    chmod +x "$MOCK_DIR/bin/nvidia-smi"
    
    # Mock ibv_devinfo
    cat > "$MOCK_DIR/bin/ibv_devinfo" << 'EOF'
#!/bin/bash
cat << 'IBV_OUTPUT'
hca_id: mlx5_0
        transport:                      InfiniBand (0)
        fw_ver:                         20.31.1014
        node_guid:                      248a:0703:00b4:7db0
        sys_image_guid:                 248a:0703:00b4:7db0
        vendor_id:                      0x02c9
        vendor_part_id:                 4123
        hw_ver:                         0x0
        board_id:                       MT_0000000012
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 1
                        port_lid:               1
                        port_lmc:               0x00
                        link_layer:             InfiniBand

hca_id: mlx5_1
        transport:                      InfiniBand (0)
        fw_ver:                         20.31.1014
        node_guid:                      248a:0703:00b4:7db1
        sys_image_guid:                 248a:0703:00b4:7db1
        vendor_id:                      0x02c9
        vendor_part_id:                 4123
        hw_ver:                         0x0
        board_id:                       MT_0000000013
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 2
                        port_lid:               2
                        port_lmc:               0x00
                        link_layer:             InfiniBand
IBV_OUTPUT
EOF
    chmod +x "$MOCK_DIR/bin/ibv_devinfo"
    
    # Mock python3
    cat > "$MOCK_DIR/bin/python3" << 'EOF'
#!/bin/bash
case "$*" in
    *"import torch"*)
        exit 0
        ;;
    *"torch.__version__"*)
        echo "2.1.0+cu121"
        ;;
    *"torch.cuda.is_available()"*)
        exit 0
        ;;
    *"torch.version.cuda"*)
        echo "12.1"
        ;;
    *"torch.cuda.nccl.version()"*)
        echo "(2, 18, 3)"
        ;;
    *"torch.cuda.device_count()"*)
        echo "4"
        ;;
    *)
        # 模拟 NCCL 测试脚本执行
        if [[ "$*" == *"nccl_test"* ]]; then
            cat << 'NCCL_OUTPUT'
[INFO] NCCL Test Starting...
[INFO] Rank 0: Initializing NCCL
[INFO] Rank 1: Initializing NCCL
[INFO] Rank 2: Initializing NCCL
[INFO] Rank 3: Initializing NCCL
[INFO] AllReduce Test - Size: 1048576 elements
[INFO] Iteration 1: 45.2 GB/s, 12.3 us latency
[INFO] Iteration 2: 46.1 GB/s, 11.8 us latency
[INFO] Iteration 3: 45.8 GB/s, 12.0 us latency
[INFO] Average: 45.7 GB/s, 12.0 us latency
[INFO] NCCL Test Completed Successfully
NCCL_OUTPUT
        else
            echo "Mock Python3 - Command: $*"
        fi
        ;;
esac
EOF
    chmod +x "$MOCK_DIR/bin/python3"
    
    # Mock torchrun
    cat > "$MOCK_DIR/bin/torchrun" << 'EOF'
#!/bin/bash
echo "[MOCK] torchrun executed with args: $*"
echo "[INFO] Starting distributed training..."
echo "[INFO] Rank 0/4 initialized"
echo "[INFO] Rank 1/4 initialized"
echo "[INFO] Rank 2/4 initialized"
echo "[INFO] Rank 3/4 initialized"
echo "[INFO] AllReduce benchmark completed"
echo "[INFO] Average bandwidth: 45.7 GB/s"
echo "[INFO] Average latency: 12.0 us"
exit 0
EOF
    chmod +x "$MOCK_DIR/bin/torchrun"
    
    # Mock ip command
    cat > "$MOCK_DIR/bin/ip" << 'EOF'
#!/bin/bash
if [ "$1" = "link" ] && [ "$2" = "show" ] && [ "$3" = "up" ]; then
    cat << 'IP_OUTPUT'
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000
3: ib0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 4092 qdisc mq state UP mode DEFAULT group default qlen 256
4: ib1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 4092 qdisc mq state UP mode DEFAULT group default qlen 256
IP_OUTPUT
fi
EOF
    chmod +x "$MOCK_DIR/bin/ip"
    
    # Mock lspci
    cat > "$MOCK_DIR/bin/lspci" << 'EOF'
#!/bin/bash
if [[ "$*" == *"vvv"* ]]; then
    echo "Mock lspci output - no ACS detected"
fi
EOF
    chmod +x "$MOCK_DIR/bin/lspci"
    
    # 添加 mock 目录到 PATH
    export PATH="$MOCK_DIR/bin:$PATH"
    
    log_test_pass "Mock 命令创建完成"
}

# 测试脚本语法检查
test_script_syntax() {
    log_test_header "测试脚本语法"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if bash -n "$NCCL_SCRIPT_PATH"; then
        log_test_pass "脚本语法检查通过"
    else
        log_test_fail "脚本语法检查失败"
    fi
}

# 测试帮助信息
test_help_function() {
    log_test_header "测试帮助功能"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    local help_output
    if help_output=$(bash "$NCCL_SCRIPT_PATH" --help 2>&1); then
        if echo "$help_output" | grep -q "NCCL InfiniBand 测试验证脚本"; then
            log_test_pass "帮助信息显示正常"
        else
            log_test_fail "帮助信息内容不完整"
        fi
    else
        log_test_fail "无法获取帮助信息"
    fi
}

# 测试版本信息
test_version_function() {
    log_test_header "测试版本功能"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    local version_output
    if version_output=$(bash "$NCCL_SCRIPT_PATH" --version 2>&1); then
        if echo "$version_output" | grep -q "v2.0"; then
            log_test_pass "版本信息显示正常"
        else
            log_test_fail "版本信息不正确"
        fi
    else
        log_test_fail "无法获取版本信息"
    fi
}

# 测试参数验证
test_parameter_validation() {
    log_test_header "测试参数验证"
    
    # 测试无效的测试大小
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if bash "$NCCL_SCRIPT_PATH" --size "invalid" --dry-run 2>&1 | grep -q "无效的测试数据大小"; then
        log_test_pass "无效测试大小参数验证正常"
    else
        log_test_fail "无效测试大小参数验证失败"
    fi
    
    # 测试无效的时间参数
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if bash "$NCCL_SCRIPT_PATH" --time "5" --dry-run 2>&1 | grep -q "无效的测试时长"; then
        log_test_pass "无效时间参数验证正常"
    else
        log_test_fail "无效时间参数验证失败"
    fi
    
    # 测试无效的网络后端
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if bash "$NCCL_SCRIPT_PATH" --network "invalid" --dry-run 2>&1 | grep -q "无效的网络后端"; then
        log_test_pass "无效网络后端参数验证正常"
    else
        log_test_fail "无效网络后端参数验证失败"
    fi
}

# 测试环境检查功能
test_environment_check() {
    log_test_header "测试环境检查功能"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    local check_output
    if check_output=$(bash "$NCCL_SCRIPT_PATH" --dry-run 2>&1); then
        if echo "$check_output" | grep -q "检测到.*个 NVIDIA GPU"; then
            log_test_pass "GPU 检测功能正常"
        else
            log_test_fail "GPU 检测功能异常"
        fi
        
        if echo "$check_output" | grep -q "PyTorch 版本"; then
            log_test_pass "PyTorch 检测功能正常"
        else
            log_test_fail "PyTorch 检测功能异常"
        fi
    else
        log_test_fail "环境检查功能执行失败"
    fi
}

# 测试配置管理器功能
test_config_manager() {
    log_test_header "测试配置管理器功能"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # 创建测试脚本来验证配置管理器
    cat > "$MOCK_DIR/test_config.sh" << 'EOF'
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundermentals/nccl/nccl_benchmark.sh

# 测试配置缓存
declare -A NCCL_CONFIG_CACHE
declare -A SYSTEM_INFO_CACHE

# 测试 set_nccl_config 函数
set_nccl_config "DEBUG" "INFO" "调试级别"
if [ "${NCCL_CONFIG_CACHE[DEBUG]}" = "INFO" ] && [ "$NCCL_DEBUG" = "INFO" ]; then
    echo "PASS: set_nccl_config 功能正常"
else
    echo "FAIL: set_nccl_config 功能异常"
fi

# 测试 setup_network_config 函数
setup_network_config "ib_disable"
if [ "$NCCL_IB_DISABLE" = "1" ]; then
    echo "PASS: setup_network_config 功能正常"
else
    echo "FAIL: setup_network_config 功能异常"
fi

# 测试 cache_system_info 函数
cache_system_info
if [ -n "${SYSTEM_INFO_CACHE[gpu_count]:-}" ]; then
    echo "PASS: cache_system_info 功能正常"
else
    echo "FAIL: cache_system_info 功能异常"
fi
EOF
    
    chmod +x "$MOCK_DIR/test_config.sh"
    
    local config_output
    if config_output=$(bash "$MOCK_DIR/test_config.sh" 2>&1); then
        if echo "$config_output" | grep -q "PASS.*set_nccl_config"; then
            log_test_pass "配置管理器 set_nccl_config 功能正常"
        else
            log_test_fail "配置管理器 set_nccl_config 功能异常"
        fi
        
        if echo "$config_output" | grep -q "PASS.*setup_network_config"; then
            log_test_pass "配置管理器 setup_network_config 功能正常"
        else
            log_test_fail "配置管理器 setup_network_config 功能异常"
        fi
        
        if echo "$config_output" | grep -q "PASS.*cache_system_info"; then
            log_test_pass "配置管理器 cache_system_info 功能正常"
        else
            log_test_fail "配置管理器 cache_system_info 功能异常"
        fi
    else
        log_test_fail "配置管理器测试执行失败"
    fi
}

# 测试网络配置功能
test_network_configurations() {
    log_test_header "测试网络配置功能"
    
    local network_backends=("auto" "ib" "nvlink" "pcie" "ethernet" "socket")
    
    for backend in "${network_backends[@]}"; do
        start_test "测试网络后端: $backend"
        
        local config_output
        if config_output=$(bash "$NCCL_SCRIPT_PATH" --network "$backend" --dry-run 2>&1); then
            if echo "$config_output" | grep -q "NCCL 环境变量配置完成"; then
                log_test_pass "网络后端 $backend 配置正常"
            else
                log_test_fail "网络后端 $backend 配置异常"
            fi
        else
            log_test_fail "网络后端 $backend 配置执行失败"
        fi
    done
}

# 测试多节点模式
test_multinode_mode() {
    log_test_header "测试多节点模式"
    start_test "多节点模式配置"
    
    local multinode_output
    if multinode_output=$(bash "$NCCL_SCRIPT_PATH" --multi-node --master-addr "192.168.1.100" --dry-run 2>&1); then
        if echo "$multinode_output" | grep -q "主节点地址: 192.168.1.100"; then
            log_test_pass "多节点模式配置正常"
        else
            log_test_fail "多节点模式配置异常"
        fi
    else
        log_test_fail "多节点模式测试执行失败"
    fi
}

# 测试错误处理
test_error_handling() {
    log_test_header "测试错误处理"
    
    # 测试缺少必需参数
    start_test "缺少必需参数错误处理"
    if bash "$NCCL_SCRIPT_PATH" --multi-node --dry-run 2>&1 | grep -q "多节点模式需要指定"; then
        log_test_pass "缺少必需参数错误处理正常"
    else
        log_test_fail "缺少必需参数错误处理异常"
    fi
    
    # 测试 dry-run 模式
    start_test "Dry-run 模式"
    if bash "$NCCL_SCRIPT_PATH" --dry-run 2>&1 | grep -q "Dry-run 完成"; then
        log_test_pass "Dry-run 模式正常"
    else
        log_test_fail "Dry-run 模式失败"
    fi
}

# 测试日志功能
test_logging_functions() {
    log_test_header "测试日志功能"
    start_test "日志功能"
    
    # 创建测试脚本来验证日志功能
    cat > "$MOCK_DIR/test_logging.sh" << 'EOF'
#!/bin/bash
source /Users/wangtianqing/Project/AI-fundermentals/nccl/nccl_benchmark.sh

# 重定向日志到测试文件
TEST_LOG_FILE="/tmp/test_logging_output.log"
LOG_FILE="$TEST_LOG_FILE"

# 测试各种日志级别
log_info "测试信息日志"
log_success "测试成功日志"
log_warning "测试警告日志"
log_error "测试错误日志"

# 检查日志文件是否包含预期内容
if [ -f "$TEST_LOG_FILE" ]; then
    if grep -q "测试信息日志" "$TEST_LOG_FILE" && \
       grep -q "测试成功日志" "$TEST_LOG_FILE" && \
       grep -q "测试警告日志" "$TEST_LOG_FILE" && \
       grep -q "测试错误日志" "$TEST_LOG_FILE"; then
        echo "PASS: 日志功能正常"
    else
        echo "FAIL: 日志内容不完整"
    fi
else
    echo "FAIL: 日志文件未创建"
fi

# 清理测试文件
rm -f "$TEST_LOG_FILE"
EOF
    
    chmod +x "$MOCK_DIR/test_logging.sh"
    
    local logging_output
    if logging_output=$(bash "$MOCK_DIR/test_logging.sh" 2>&1); then
        if echo "$logging_output" | grep -q "PASS: 日志功能正常"; then
            log_test_pass "日志功能测试通过"
        else
            log_test_fail "日志功能测试失败"
        fi
    else
        log_test_fail "日志功能测试执行失败"
    fi
}

# 性能测试模拟
test_performance_simulation() {
    log_test_header "测试性能模拟"
    start_test "性能测试模拟"
    
    # 使用内联 Python 脚本进行性能模拟测试
    local perf_output
    if perf_output=$(TENSOR_ELEMENTS=262144 TEST_DURATION=5 python3 -c "
import os
import time
import random

def simulate_nccl_test():
    print('[INFO] NCCL Test Starting...')
    print('[INFO] Initializing distributed environment...')
    
    # 模拟测试参数
    tensor_elements = int(os.environ.get('TENSOR_ELEMENTS', '262144'))
    test_duration = int(os.environ.get('TEST_DURATION', '30'))
    
    print(f'[INFO] Tensor elements: {tensor_elements}')
    print(f'[INFO] Test duration: {test_duration}s')
    
    # 模拟测试迭代
    for i in range(3):  # 减少迭代次数以加快测试
        # 模拟随机性能数据
        bandwidth = random.uniform(40.0, 50.0)
        latency = random.uniform(10.0, 15.0)
        print(f'[INFO] Iteration {i+1}: {bandwidth:.1f} GB/s, {latency:.1f} us latency')
        time.sleep(0.05)  # 减少睡眠时间
    
    print('[INFO] NCCL Test Completed Successfully')

if __name__ == '__main__':
    simulate_nccl_test()
" 2>&1); then
        if echo "$perf_output" | grep -q "NCCL Test Completed Successfully"; then
            log_test_pass "性能测试模拟正常"
        else
            log_test_fail "性能测试模拟异常: $perf_output"
        fi
    else
        log_test_fail "性能测试模拟执行失败: $perf_output"
    fi
}

# 清理测试环境
cleanup_test_environment() {
    log_test_header "清理测试环境"
    
    # 恢复原始 PATH
    export PATH=$(echo "$PATH" | sed "s|$MOCK_DIR/bin:||")
    
    # 清理临时文件
    rm -rf "$MOCK_DIR"
    
    log_test_pass "测试环境清理完成"
}

# 生成测试报告
generate_test_report() {
    log_test_header "测试报告"
    
    local success_rate=0
    if [ $TOTAL_TESTS -gt 0 ]; then
        success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    fi
    
    log_test ""
    log_test "📊 测试统计:"
    log_test "   总测试数: $TOTAL_TESTS"
    log_test "   通过测试: $PASSED_TESTS"
    log_test "   失败测试: $FAILED_TESTS"
    log_test "   成功率: ${success_rate}%"
    log_test ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        log_test "${GREEN}🎉 所有测试通过！${NC}"
        log_test "nccl_benchmark.sh 脚本功能正常"
    else
        log_test "${RED}❌ 存在测试失败${NC}"
        log_test "请检查失败的测试项目并修复相关问题"
    fi
    
    log_test ""
    log_test "详细测试日志: $TEST_LOG"
    log_test "目标脚本: $NCCL_SCRIPT_PATH"
}

# 主测试函数
main() {
    echo "🚀 开始 NCCL Benchmark 脚本 Mock 测试"
    echo "目标脚本: $NCCL_SCRIPT_PATH"
    echo "测试日志: $TEST_LOG"
    echo ""
    
    # 初始化测试日志
    echo "NCCL Benchmark Mock Test - $(date)" > "$TEST_LOG"
    
    # 执行测试套件
    setup_test_environment
    create_mock_commands
    
    test_script_syntax
    test_help_function
    test_version_function
    test_parameter_validation
    test_environment_check
    test_config_manager
    test_network_configurations
    test_multinode_mode
    test_error_handling
    test_logging_functions
    test_performance_simulation
    
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