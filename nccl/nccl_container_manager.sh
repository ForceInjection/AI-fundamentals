#!/bin/bash
# =============================================================================
# NCCL 容器化测试脚本
# 功能: 使用 Docker 容器运行 NCCL 单节点测试
# 作者: Grissom
# 版本: 2.0
# 
# 注意: 多节点测试请使用 Kubernetes 方案 (./k8s/deploy.sh)
# =============================================================================

# 脚本配置
SCRIPT_NAME="NCCL Container Test"
VERSION="2.0"
CONTAINER_NAME="nccl-test"
IMAGE_NAME="nccl-test:latest"

# 默认参数
GPU_COUNT="all"
TEST_SIZE="1M"
TEST_DURATION=30
NETWORK_BACKEND="auto"
OPTIMIZATION_LEVEL="balanced"
CLEANUP=true
INTERACTIVE=false
LOG_LEVEL="INFO"
DRY_RUN=false
MULTI_NODE=false
MASTER_ADDR=""
MASTER_PORT="29500"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

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
    echo -e "${PURPLE}=== $1 ===${NC}"
    echo ""
}

# 显示帮助信息
show_help() {
    cat << EOF
$SCRIPT_NAME v$VERSION

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -v, --version           显示版本信息
  -i, --interactive       交互模式 (进入容器 shell)
  -g, --gpus COUNT        指定 GPU 数量 [默认: all]
                          all     - 使用所有可用 GPU
                          N       - 使用 N 个 GPU (例如: 2, 4, 8)
                          0,1,2   - 指定特定 GPU ID
  -s, --size SIZE         测试数据大小 [默认: 1M]
  -t, --time SECONDS      测试持续时间 [默认: 30]
  --network BACKEND       网络后端 [默认: auto]
                          auto     - 自动检测并选择最佳网络 (按NCCL优先级)
                                   单节点: NVLink > PCIe P2P > 共享内存 > 网络传输
                                   多节点: InfiniBand > PXN > 以太网
                          pxn      - 强制使用 PXN 模式 (多节点专用高性能通信)
                          ib       - 强制使用 InfiniBand/RoCE
                          nvlink   - 强制使用 NVLink (单节点多GPU)
                          pcie     - 强制使用 PCIe P2P (单节点多GPU)
                          shm      - 强制使用共享内存 (单节点多GPU)
                          ethernet - 强制使用以太网 (TCP/IP)
                          socket   - 强制使用 Socket 传输
  --optimization LEVEL    优化级别 [默认: balanced] (仅适用于 NVLink 和 PXN)
                          conservative - 保守配置 (稳定性优先)
                          balanced     - 平衡配置 (推荐)
                          aggressive   - 激进配置 (最大性能)
  -m, --multi-node        启用多节点模式 (需要配合 --master-addr)
  --master-addr ADDR      主节点地址 (多节点模式必需)
  --master-port PORT      主节点端口 [默认: 29500]
  --log-level LEVEL       日志级别 [默认: INFO]
  --dry-run               Dry-run 模式：检查环境、配置变量但不执行测试
  --no-cleanup            测试后不清理容器
  --container-name NAME   自定义容器名称 [默认: nccl-test]
  --image-name NAME       自定义镜像名称 [默认: nccl-test:latest]

注意事项:
  • 容器强制以 privileged + host network 模式运行
  • Host Network: 直接访问主机网络设备，无 Docker 网络层开销
  • privileged 模式自动提供设备访问，无需手动挂载 /dev、/sys、/proc
  • 支持完整的 InfiniBand 和 NVLink 设备访问
  • --gpus 参数在 privileged 模式下仍然必要 (用于 NVIDIA 运行时初始化)
  • 启用 GPUDirect RDMA 和高性能网络传输优化
  • 调用 nccl_benchmark.sh 进行 NCCL 环境配置和测试

示例:
  # Dry-run 模式 (检查环境和配置但不执行测试)
  $0 --dry-run --gpus all --size 100M --network auto
  
  # 单节点测试 (使用所有 GPU)
  $0 --gpus all --size 100M --time 60
  
  # 单节点测试 (使用 4 个 GPU，NVLink 后端)
  $0 --gpus 4 --size 1G --network nvlink --optimization balanced
  
  # 单节点测试 (使用 PCIe P2P 通信)
  $0 --gpus 2 --size 100M --network pcie
  
  # 单节点测试 (使用共享内存通信)
  $0 --gpus 2 --size 10M --network shm
  
  # PXN 模式多节点测试 (主节点)
  $0 --multi-node --master-addr 192.168.1.100 --network pxn --optimization balanced --gpus 4 --size 1G
  
  # PXN 模式多节点测试 (工作节点)
  $0 --multi-node --master-addr 192.168.1.100 --network pxn --optimization balanced --gpus 4 --size 1G
  
  # 交互模式 (调试用)
  $0 --interactive
  
  # 自定义配置
  $0 --gpus 2 --size 500M --time 120 --network ib --log-level DEBUG

多节点测试:
  对于多节点 NCCL 测试，请使用 Kubernetes 方案:
  
  # 部署多节点测试到 Kubernetes
  ./k8s/deploy.sh deploy --world-size 8 --gpus-per-node 4
  
  # 查看测试状态
  ./k8s/deploy.sh status
  
  # 查看测试日志
  ./k8s/deploy.sh logs
  
  # 清理资源
  ./k8s/deploy.sh cleanup

前置条件:
  • 安装 Docker 和 NVIDIA Container Toolkit
  • 确保 GPU 驱动正常工作
  • 镜像已预先构建 (docker build -t nccl-test:latest .)
  • 需要 root 权限或 Docker 组成员身份
  • 确保 nccl_benchmark.sh 脚本在当前目录

EOF
}

# 显示版本信息
show_version() {
    echo "$SCRIPT_NAME v$VERSION"
}

# 检查前置条件
check_prerequisites() {
    log_header "检查前置条件"
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装"
        exit 1
    fi
    log_success "Docker 可用"
    
    # 检查 Docker 服务
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未运行"
        exit 1
    fi
    log_success "Docker 服务正常"
    
    # 检查 NVIDIA Container Toolkit (使用自己构建的镜像)
    if docker image inspect "$IMAGE_NAME" &> /dev/null; then
        if ! docker run --rm --gpus all "$IMAGE_NAME" nvidia-smi &> /dev/null; then
            log_error "NVIDIA Container Toolkit 不可用"
            log_info "请安装 NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            exit 1
        fi
        log_success "NVIDIA Container Toolkit 可用 (使用镜像: $IMAGE_NAME)"
    else
        log_warning "镜像 $IMAGE_NAME 不存在，跳过 NVIDIA Container Toolkit 检查"
        log_info "请确保镜像已预先构建并包含 nvidia-smi"
    fi
    
    # 检查 GPU
    local gpu_count=$(nvidia-smi -L | wc -l)
    if [ "$gpu_count" -eq 0 ]; then
        log_error "未检测到 GPU"
        exit 1
    fi
    log_success "检测到 $gpu_count 个 GPU"
}

# 检查镜像是否存在
check_image() {
    if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
        log_error "镜像 $IMAGE_NAME 不存在"
        log_info "请先构建镜像: docker build -t $IMAGE_NAME ."
        exit 1
    else
        log_success "镜像 $IMAGE_NAME 已存在"
    fi
}

# 清理容器
cleanup() {
    if [ "$CLEANUP" = true ]; then
        log_info "清理容器..."
        docker stop "$CONTAINER_NAME" &> /dev/null || true
        docker rm "$CONTAINER_NAME" &> /dev/null || true
    fi
}

# 运行交互模式
run_interactive() {
    log_header "启动交互模式"
    
    cleanup
    
    local gpu_option=""
    if [ "$GPU_COUNT" = "all" ]; then
        gpu_option="--gpus all"
    else
        gpu_option="--gpus $GPU_COUNT"
    fi
    
    log_info "启动容器: $CONTAINER_NAME (privileged 模式)"
    log_info "网络模式: Host (直接访问主机网络设备)"
    log_info "Privileged 模式: 自动获得完整系统设备访问权限"
    
    docker run -it --rm \
        $gpu_option \
        --privileged \
        --name "$CONTAINER_NAME" \
        --network host \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -e NCCL_DEBUG="$LOG_LEVEL" \
        "$IMAGE_NAME" \
        /bin/bash
}

# 运行 NCCL 测试
run_nccl_test() {
    log_header "运行 NCCL 测试"
    
    cleanup
    
    # 设置 GPU 选项
    local gpu_option=""
    if [ "$GPU_COUNT" = "all" ]; then
        gpu_option="--gpus all"
    elif [[ "$GPU_COUNT" =~ ^[0-9]+$ ]]; then
        gpu_option="--gpus $GPU_COUNT"
    else
        gpu_option="--gpus \"device=$GPU_COUNT\""
    fi
    
    log_info "容器配置:"
    log_info "  镜像: $IMAGE_NAME"
    log_info "  容器名: $CONTAINER_NAME"
    log_info "  GPU: $GPU_COUNT"
    log_info "  测试大小: $TEST_SIZE"
    log_info "  测试时长: $TEST_DURATION 秒"
    log_info "  网络后端: $NETWORK_BACKEND"
    log_info "  优化级别: $OPTIMIZATION_LEVEL"
    log_info "  多节点模式: $([ "$MULTI_NODE" = true ] && echo "启用" || echo "禁用")"
    if [ "$MULTI_NODE" = true ]; then
        log_info "  主节点地址: $MASTER_ADDR:$MASTER_PORT"
    fi
    log_info "  运行模式: $([ "$DRY_RUN" = true ] && echo "Dry-run (仅检查环境和配置)" || echo "正常测试模式")"
    log_info "  网络模式: Host (直接访问主机网络设备)"
    log_info "  容器模式: Privileged (完整设备访问)"
    
    # 构建 nccl_benchmark.sh 参数
    local nccl_test_args=()
    [ "$TEST_SIZE" != "1M" ] && nccl_test_args+=("-s" "$TEST_SIZE")
    [ "$TEST_DURATION" != "30" ] && nccl_test_args+=("-t" "$TEST_DURATION")
    [ "$NETWORK_BACKEND" != "auto" ] && nccl_test_args+=("--network" "$NETWORK_BACKEND")
    [ "$OPTIMIZATION_LEVEL" != "balanced" ] && nccl_test_args+=("--optimization-level" "$OPTIMIZATION_LEVEL")
    [ "$MULTI_NODE" = true ] && nccl_test_args+=("--multi-node")
    [ -n "$MASTER_ADDR" ] && nccl_test_args+=("--master-addr" "$MASTER_ADDR")
    [ "$MASTER_PORT" != "29500" ] && nccl_test_args+=("--master-port" "$MASTER_PORT")
    [ "$DRY_RUN" = true ] && nccl_test_args+=("--dry-run")
    
    # 启动容器并运行 nccl_benchmark.sh
    if [ "$DRY_RUN" = true ]; then
        log_info "启动 Dry-run 容器 (privileged + host network 模式)..."
        log_info "Dry-run 模式: 仅检查环境、配置变量，不执行实际 NCCL 测试"
    else
        log_info "启动测试容器 (privileged + host network 模式)..."
    fi
    log_info "Host Network: 直接访问主机网络设备，无网络层开销"
    log_info "Privileged 模式: 自动获得完整系统设备访问权限"
    log_info "挂载主机 /tmp 目录: 容器内 NCCL 输出直接保存到主机"
    log_info "调用容器内 nccl_benchmark.sh 进行 NCCL 环境配置$([ "$DRY_RUN" = true ] && echo "检查" || echo "和测试")"
    
    docker run --rm \
        $gpu_option \
        --privileged \
        --name "$CONTAINER_NAME" \
        --network host \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "/tmp:/tmp" \
        -e NCCL_DEBUG="$LOG_LEVEL" \
        "$IMAGE_NAME" \
        bash -c "cd /workspace/nccl_test && ./nccl_benchmark.sh ${nccl_test_args[*]}"
    
    # 检查并报告保存的文件 (仅在非 dry-run 模式下执行)
    if [ "$DRY_RUN" != true ]; then
        if [ -f "/tmp/nccl_test_output.log" ]; then
            log_success "NCCL 原始输出已保存到: /tmp/nccl_test_output.log"
            log_info "查看完整 NCCL 原始输出: cat /tmp/nccl_test_output.log"
            
            # 简单分析保存的原始输出
            local file_size=$(wc -l < "/tmp/nccl_test_output.log" 2>/dev/null || echo "0")
            if [ "$file_size" -gt 10 ]; then
                log_info "原始输出文件大小: $file_size 行"
                # 检查是否包含真实的 NCCL 日志
                if grep -q "NCCL INFO" "/tmp/nccl_test_output.log" 2>/dev/null; then
                    log_success "✅ 检测到真实的 NCCL 原始日志"
                    # 显示网络类型信息
                    local net_info=$(grep -E "NCCL INFO.*NET/" "/tmp/nccl_test_output.log" 2>/dev/null | head -3)
                    if [ -n "$net_info" ]; then
                        log_info "🔍 NCCL 网络信息预览:"
                        echo "$net_info" | while read line; do
                            log_info "    $line"
                        done
                    fi
                else
                    log_warning "⚠️  文件不包含 NCCL 原始日志，可能是错误信息"
                fi
            else
                log_warning "⚠️  原始输出文件过小 ($file_size 行)，可能测试失败"
            fi
        else
            log_warning "⚠️  未能保存 NCCL 原始输出文件到 /tmp/nccl_test_output.log"
        fi
    else
        log_info "Dry-run 模式: 跳过输出文件检查和分析"
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
            -i|--interactive)
                INTERACTIVE=true
                shift
                ;;
            -g|--gpus)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "选项 $1 需要一个参数"
                    exit 1
                fi
                GPU_COUNT="$2"
                shift 2
                ;;
            -s|--size)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "选项 $1 需要一个参数"
                    exit 1
                fi
                TEST_SIZE="$2"
                shift 2
                ;;
            -t|--time)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "选项 $1 需要一个参数"
                    exit 1
                fi
                TEST_DURATION="$2"
                shift 2
                ;;
            --network)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "选项 $1 需要一个参数"
                    exit 1
                fi
                NETWORK_BACKEND="$2"
                shift 2
                ;;
            --optimization)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "选项 $1 需要一个参数"
                    exit 1
                fi
                OPTIMIZATION_LEVEL="$2"
                shift 2
                ;;
            -m|--multi-node)
                MULTI_NODE=true
                shift
                ;;
            --master-addr)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "选项 $1 需要一个参数"
                    exit 1
                fi
                MASTER_ADDR="$2"
                shift 2
                ;;
            --master-port)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "选项 $1 需要一个参数"
                    exit 1
                fi
                MASTER_PORT="$2"
                shift 2
                ;;
            --log-level)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "选项 $1 需要一个参数"
                    exit 1
                fi
                LOG_LEVEL="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --no-cleanup)
                CLEANUP=false
                shift
                ;;
            --container-name)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "选项 $1 需要一个参数"
                    exit 1
                fi
                CONTAINER_NAME="$2"
                shift 2
                ;;
            --image-name)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "选项 $1 需要一个参数"
                    exit 1
                fi
                IMAGE_NAME="$2"
                shift 2
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
    log_header "$SCRIPT_NAME v$VERSION"
    
    # 解析参数
    parse_arguments "$@"
    
    # 检查前置条件
    check_prerequisites
    
    # 检查镜像
    check_image
    
    # 运行模式
    if [ "$INTERACTIVE" = true ]; then
        run_interactive
    else
        run_nccl_test
    fi
    
    log_success "操作完成"
}

# 设置清理陷阱
trap cleanup EXIT

# 运行主函数
main "$@"