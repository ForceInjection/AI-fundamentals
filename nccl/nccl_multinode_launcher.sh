#!/bin/bash
# =============================================================================
# NCCL 多节点测试启动脚本
# =============================================================================
# 
# 功能: 简化多节点 NCCL 测试的启动过程
# 用法: ./nccl_multinode_launcher.sh <node_rank> <master_addr> [options]
#
# 示例:
#   # 主节点 (node_rank=0)
#   ./nccl_multinode_launcher.sh 0 192.168.1.100
#   
#   # 工作节点 (node_rank=1)
#   ./nccl_multinode_launcher.sh 1 192.168.1.100
#
# =============================================================================

# 全局变量
TEMP_FILES=()
CHILD_PIDS=()

# 清理函数
cleanup() {
    local exit_code=$?
    
    echo ""
    echo "正在清理资源..."
    
    # 终止子进程
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "终止进程: $pid"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        fi
    done
    
    # 清理临时文件
    for file in "${TEMP_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "删除临时文件: $file"
            rm -f "$file"
        fi
    done
    
    if [ $exit_code -ne 0 ]; then
        echo "脚本异常退出 (退出码: $exit_code)"
    else
        echo "清理完成"
    fi
    
    exit $exit_code
}

# 信号处理
trap cleanup EXIT
trap 'echo "收到中断信号，正在清理..."; exit 130' INT TERM

# 添加临时文件到清理列表
add_temp_file() {
    TEMP_FILES+=("$1")
}

# 添加子进程到清理列表
add_child_pid() {
    CHILD_PIDS+=("$1")
}

# 脚本信息
SCRIPT_NAME="nccl_multinode_launcher.sh"
VERSION="1.0.0"

# 默认配置
DEFAULT_WORLD_SIZE=4
DEFAULT_NPROC_PER_NODE=2
DEFAULT_MASTER_PORT=29500
DEFAULT_NETWORK="ib"
DEFAULT_TEST_SIZE="50M"
DEFAULT_TEST_DURATION=90

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
$SCRIPT_NAME v$VERSION - NCCL 多节点测试启动脚本

用法: $0 <node_rank> <master_addr> [选项]

必需参数:
  node_rank               节点编号 (0=主节点, 1,2,3...=工作节点)
  master_addr             主节点IP地址

可选参数:
  -w, --world-size SIZE   总GPU数量 (默认: $DEFAULT_WORLD_SIZE)
  -n, --nproc-per-node N  每节点GPU数 (默认: $DEFAULT_NPROC_PER_NODE)
  -p, --master-port PORT  主节点端口 (默认: $DEFAULT_MASTER_PORT)
  --network BACKEND       网络后端 (默认: $DEFAULT_NETWORK)
  -s, --size SIZE         测试数据大小 (默认: $DEFAULT_TEST_SIZE)
  -t, --time SECONDS      测试时长 (默认: $DEFAULT_TEST_DURATION)
  -h, --help              显示此帮助信息

网络后端选项:
  ib                      InfiniBand (推荐)
  ethernet                以太网
  socket                  Socket (调试用)

示例:
  # 2节点4GPU集群测试
  # 主节点 (192.168.1.100):
  $0 0 192.168.1.100 -w 4 -n 2
  
  # 工作节点 (192.168.1.101):
  $0 1 192.168.1.100 -w 4 -n 2
  
  # 4节点8GPU大规模测试
  # 主节点:
  $0 0 192.168.1.100 -w 8 -n 2 -s 100M -t 120
  
  # 工作节点1:
  $0 1 192.168.1.100 -w 8 -n 2 -s 100M -t 120

注意事项:
  1. 所有节点必须使用相同的参数配置
  2. 建议先启动主节点，再启动工作节点
  3. 确保所有节点网络连通且时间同步
  4. 需要在每个节点手动运行此脚本

EOF
}

# 解析命令行参数
parse_arguments() {
    # 首先检查是否有帮助参数
    for arg in "$@"; do
        if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
            show_help
            exit 0
        fi
    done
    
    if [ $# -lt 2 ]; then
        log_error "缺少必需参数"
        echo "用法: $0 <node_rank> <master_addr> [选项]"
        echo "使用 '$0 --help' 查看详细帮助"
        exit 1
    fi
    
    NODE_RANK="$1"
    MASTER_ADDR="$2"
    shift 2
    
    # 设置默认值
    WORLD_SIZE="$DEFAULT_WORLD_SIZE"
    NPROC_PER_NODE="$DEFAULT_NPROC_PER_NODE"
    MASTER_PORT="$DEFAULT_MASTER_PORT"
    NETWORK="$DEFAULT_NETWORK"
    TEST_SIZE="$DEFAULT_TEST_SIZE"
    TEST_DURATION="$DEFAULT_TEST_DURATION"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -w|--world-size)
                WORLD_SIZE="$2"
                shift 2
                ;;
            -n|--nproc-per-node)
                NPROC_PER_NODE="$2"
                shift 2
                ;;
            -p|--master-port)
                MASTER_PORT="$2"
                shift 2
                ;;
            --network)
                NETWORK="$2"
                shift 2
                ;;
            -s|--size)
                TEST_SIZE="$2"
                shift 2
                ;;
            -t|--time)
                TEST_DURATION="$2"
                shift 2
                ;;
            *)
                log_error "未知选项: $1"
                echo "使用 '$0 --help' 查看帮助信息"
                exit 1
                ;;
        esac
    done
    
    # 验证参数
    if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]]; then
        log_error "节点编号必须是非负整数"
        exit 1
    fi
    
    if ! [[ "$WORLD_SIZE" =~ ^[0-9]+$ ]] || [ "$WORLD_SIZE" -lt 2 ]; then
        log_error "总GPU数量必须是大于等于2的整数"
        exit 1
    fi
    
    if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [ "$NPROC_PER_NODE" -lt 1 ]; then
        log_error "每节点GPU数必须是正整数"
        exit 1
    fi
    
    # 计算节点数量
    NNODES=$((WORLD_SIZE / NPROC_PER_NODE))
    if [ $((NNODES * NPROC_PER_NODE)) -ne "$WORLD_SIZE" ]; then
        log_error "总GPU数量 ($WORLD_SIZE) 必须能被每节点GPU数 ($NPROC_PER_NODE) 整除"
        exit 1
    fi
    
    if [ "$NODE_RANK" -ge "$NNODES" ]; then
        log_error "节点编号 ($NODE_RANK) 必须小于总节点数 ($NNODES)"
        exit 1
    fi
}

# 显示配置信息
show_config() {
    log_header "多节点测试配置"
    log_info "节点信息:"
    log_info "  当前节点编号: $NODE_RANK"
    log_info "  节点角色: $([ "$NODE_RANK" -eq 0 ] && echo "主节点 (Master)" || echo "工作节点 (Worker)")"
    log_info "  主节点地址: $MASTER_ADDR"
    log_info "  主节点端口: $MASTER_PORT"
    
    log_info ""
    log_info "集群配置:"
    log_info "  总节点数: $NNODES"
    log_info "  总GPU数: $WORLD_SIZE"
    log_info "  每节点GPU数: $NPROC_PER_NODE"
    
    log_info ""
    log_info "测试配置:"
    log_info "  网络后端: $NETWORK"
    log_info "  测试数据大小: $TEST_SIZE"
    log_info "  测试时长: ${TEST_DURATION}秒"
}

# 检查依赖
check_dependencies() {
    log_header "依赖检查"
    
    local missing_deps=()
    
    # 检查必需的命令
    local required_commands=("python3" "ping" "nvidia-smi")
    
    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            log_success "✓ $cmd 可用"
        else
            log_error "✗ $cmd 不可用"
            missing_deps+=("$cmd")
        fi
    done
    
    # 检查 Python 模块
    log_info "检查 Python 模块..."
    local python_modules=("torch" "torch.distributed")
    
    for module in "${python_modules[@]}"; do
        if python3 -c "import $module" 2>/dev/null; then
            log_success "✓ Python 模块 $module 可用"
        else
            log_error "✗ Python 模块 $module 不可用"
            missing_deps+=("python3-$module")
        fi
    done
    
    # 检查 CUDA
    if command -v nvcc >/dev/null 2>&1; then
        local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        log_success "✓ CUDA 可用 (版本: $cuda_version)"
    else
        log_warning "⚠ nvcc 不可用，可能影响某些功能"
    fi
    
    # 检查网络工具
    local network_tools=("ss" "netstat" "ip")
    local network_tool_found=false
    
    for tool in "${network_tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_success "✓ 网络工具 $tool 可用"
            network_tool_found=true
            break
        fi
    done
    
    if [ "$network_tool_found" = false ]; then
        log_warning "⚠ 未找到网络诊断工具 (ss/netstat/ip)"
    fi
    
    # 如果有缺失的依赖，退出
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "缺少以下依赖:"
        for dep in "${missing_deps[@]}"; do
            log_error "  - $dep"
        done
        log_info "请安装缺失的依赖后重试"
        exit 1
    fi
    
    log_success "所有依赖检查通过"
}

# 检查环境
check_environment() {
    log_header "环境检查"
    
    # 检查 nccl_benchmark.sh 脚本
    if [ ! -f "./nccl_benchmark.sh" ]; then
        log_error "找不到 nccl_benchmark.sh 脚本"
        log_info "请确保在正确的目录下运行此脚本"
        exit 1
    fi
    log_success "找到 nccl_benchmark.sh 脚本"
    
    # 检查脚本权限
    if [ ! -x "./nccl_benchmark.sh" ]; then
        log_warning "nccl_benchmark.sh 脚本没有执行权限，正在添加..."
        chmod +x "./nccl_benchmark.sh"
        log_success "已添加执行权限"
    fi
    
    # 检查网络连通性
    log_info "检查与主节点的网络连通性..."
    if ping -c 1 -W 3 "$MASTER_ADDR" >/dev/null 2>&1; then
        log_success "网络连通性正常"
    else
        log_error "无法连接到主节点: $MASTER_ADDR"
        log_info "请检查网络配置和防火墙设置"
        exit 1
    fi
    
    # 检查端口可用性
    if [ "$NODE_RANK" -eq 0 ]; then
        log_info "检查主节点端口 $MASTER_PORT 可用性..."
        if command -v ss >/dev/null 2>&1; then
            if ss -tuln | grep -q ":$MASTER_PORT "; then
                log_warning "端口 $MASTER_PORT 已被占用"
                log_info "如果测试失败，请尝试使用其他端口"
            else
                log_success "端口 $MASTER_PORT 可用"
            fi
        elif command -v netstat >/dev/null 2>&1; then
            if netstat -tuln | grep -q ":$MASTER_PORT "; then
                log_warning "端口 $MASTER_PORT 已被占用"
                log_info "如果测试失败，请尝试使用其他端口"
            else
                log_success "端口 $MASTER_PORT 可用"
            fi
        else
            log_warning "无法检查端口状态 (缺少 ss/netstat 工具)"
        fi
    fi
    
    # 检查 GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        log_success "检测到 $gpu_count 个 GPU"
        if [ "$gpu_count" -lt "$NPROC_PER_NODE" ]; then
            log_error "GPU数量 ($gpu_count) 少于配置的每节点GPU数 ($NPROC_PER_NODE)"
            exit 1
        fi
        
        # 检查 GPU 状态
        log_info "检查 GPU 状态..."
        if nvidia-smi >/dev/null 2>&1; then
            log_success "所有 GPU 状态正常"
        else
            log_error "GPU 状态检查失败"
            exit 1
        fi
    else
        log_warning "nvidia-smi 不可用，无法检查GPU"
    fi
    
    # 检查磁盘空间
    local available_space=$(df /tmp | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 1048576 ]; then  # 1GB in KB
        log_warning "临时目录空间不足 (< 1GB)，可能影响测试"
    else
        log_success "磁盘空间充足"
    fi
}

# 等待其他节点
wait_for_coordination() {
    if [ "$NODE_RANK" -eq 0 ]; then
        log_info "主节点等待 5 秒后开始测试..."
        sleep 5
    else
        local wait_time=$((NODE_RANK * 10 + 10))
        log_info "工作节点等待 $wait_time 秒后开始测试..."
        sleep $wait_time
    fi
}

# 启动测试
start_test() {
    log_header "启动 NCCL 多节点测试"
    
    # 设置环境变量
    export WORLD_SIZE="$WORLD_SIZE"
    export NODE_RANK="$NODE_RANK"
    export NPROC_PER_NODE="$NPROC_PER_NODE"
    export MASTER_ADDR="$MASTER_ADDR"
    export MASTER_PORT="$MASTER_PORT"
    
    log_info "设置环境变量:"
    log_info "  WORLD_SIZE=$WORLD_SIZE"
    log_info "  NODE_RANK=$NODE_RANK"
    log_info "  NPROC_PER_NODE=$NPROC_PER_NODE"
    log_info "  MASTER_ADDR=$MASTER_ADDR"
    log_info "  MASTER_PORT=$MASTER_PORT"
    
    # 启动测试
    log_info ""
    log_info "启动测试命令:"
    local test_cmd="./nccl_benchmark.sh -m --master-addr $MASTER_ADDR --master-port $MASTER_PORT --network $NETWORK -s $TEST_SIZE -t $TEST_DURATION"
    log_info "$test_cmd"
    
    log_info ""
    log_success "开始执行测试..."
    
    # 执行测试
    if $test_cmd; then
        log_success "测试执行完成"
    else
        log_error "测试执行失败"
        exit 1
    fi
}

# 主函数
main() {
    echo "=== NCCL 多节点测试启动脚本 v$VERSION ==="
    echo ""
    
    # 解析参数
    parse_arguments "$@"
    
    # 检查依赖
    check_dependencies
    
    # 显示配置
    show_config
    
    # 检查环境
    check_environment
    
    # 等待协调
    wait_for_coordination
    
    # 启动测试
    start_test
    
    log_success "多节点测试脚本执行完成"
}

# 执行主函数
main "$@"