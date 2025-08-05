#!/bin/bash
# =============================================================================
# NCCL Kubernetes 部署脚本
# 功能: 在 Kubernetes 集群中部署和管理 NCCL 多节点测试
# =============================================================================

set -e

# 脚本配置
SCRIPT_NAME="NCCL Kubernetes Deployer"
VERSION="1.0"
NAMESPACE="default"
STS_NAME="nccl-multinode"
REPLICAS="2"
GPUS_PER_NODE="2"
NETWORK_BACKEND="auto"
OPTIMIZATION_LEVEL="balanced"
TEST_SIZE="100M"
TEST_DURATION="60"
SIMULATION_MODE="false"

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

log_warn() {
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
    cat << 'EOF'
NCCL 多节点测试部署脚本

用法:
    ./deploy.sh [选项] <操作>

操作:
    deploy          部署 NCCL 多节点测试 (StatefulSet)
    status          查看部署状态
    logs            查看测试日志
    cleanup         清理所有资源
    help            显示此帮助信息
    version         显示版本信息

选项:
    -n, --namespace <namespace>    指定 Kubernetes 命名空间 (默认: default)
    -r, --replicas <number>        指定副本数 (默认: 2)
    -g, --gpus <number>            每节点 GPU 数量 (默认: 2)
    --network <backend>            网络后端 (默认: auto)
                                   auto - 自动检测 (InfiniBand > PXN > 以太网)
                                   pxn - PXN 模式 (多节点专用高性能通信)
                                   ib - InfiniBand
                                   ethernet - 以太网
    --optimization <level>         优化级别 (默认: balanced)
                                   conservative - 保守配置 (稳定性优先)
                                   balanced - 平衡配置 (推荐)
                                   aggressive - 激进配置 (最大性能)
    -s, --size <size>              测试数据大小 (默认: 100M)
    -t, --time <seconds>           测试持续时间 (默认: 60)
    --simulation                   启用模拟模式 (跳过 GPU 检查)
    -h, --help                     显示帮助信息
    -v, --version                  显示版本信息

示例:
    # 在默认命名空间部署 2 个副本 (自动网络检测)
    ./deploy.sh deploy

    # 在指定命名空间部署 4 个副本
    ./deploy.sh -n nccl-test -r 4 deploy
    
    # PXN 模式高性能部署
    ./deploy.sh -n nccl-test -r 4 -g 4 --network pxn --optimization balanced -s 1G -t 120 deploy
    
    # PXN 激进模式部署 (最大性能)
    ./deploy.sh -n nccl-test -r 2 -g 8 --network pxn --optimization aggressive -s 500M -t 60 deploy

    # 模拟模式部署 (用于测试配置，无需 GPU)
    ./deploy.sh --simulation -r 2 -g 4 --network pxn --optimization balanced deploy

    # 查看状态
    ./deploy.sh -n nccl-test status

    # 查看日志
    ./deploy.sh -n nccl-test logs

    # 清理资源
    ./deploy.sh -n nccl-test cleanup

注意:
    • 总 GPU 数 = 副本数 × 每节点GPU数量
    • StatefulSet 提供稳定的 Pod 命名和有序启动
    • ConfigMap 统一管理 NCCL 配置参数
    • 所有资源都使用 app=nccl-multinode 标签进行管理

EOF
}

# 检查前置条件
check_prerequisites() {
    log_header "检查前置条件"
    
    # 检查 kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl 未安装"
        exit 1
    fi
    log_success "kubectl 可用"
    
    # 检查集群连接
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到 Kubernetes 集群"
        exit 1
    fi
    log_success "Kubernetes 集群连接正常"
    
    # 检查命名空间
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warn "命名空间 $NAMESPACE 不存在，将创建"
        kubectl create namespace "$NAMESPACE"
        log_success "命名空间 $NAMESPACE 已创建"
    else
        log_success "命名空间 $NAMESPACE 存在"
    fi
    
    # 检查 GPU 节点 (除非在模拟模式下)
    if [ "$SIMULATION_MODE" = "true" ]; then
        log_warn "模拟模式: 跳过 GPU 节点检查"
        log_info "注意: 在模拟模式下，Pod 可能无法正常运行 NCCL 测试"
    else
        local gpu_nodes=$(kubectl get nodes -l gpu=on --no-headers | wc -l)
        if [ "$gpu_nodes" -eq 0 ]; then
            log_error "集群中没有 GPU 节点"
            log_info "请确保节点已安装 Hami，或使用 --simulation 模式进行配置测试"
            exit 1
        fi
        log_success "检测到 $gpu_nodes 个 GPU 节点"
    fi
    
    # 检查配置文件
    if [ ! -f "./nccl-multinode-sts.yaml" ]; then
        log_error "找不到 StatefulSet 配置文件: nccl-multinode-sts.yaml"
        log_info "请确保在正确的目录下运行此脚本"
        exit 1
    fi
    log_success "StatefulSet 配置文件存在"
}

# 部署测试 (StatefulSet)
deploy_test() {
    log_header "部署 NCCL 多节点测试 (StatefulSet)"
    
    # 检查配置文件
    local required_files=("nccl-multinode-sts.yaml" "nccl-service.yaml" "nccl-configmap.yaml" "nccl-rbac.yaml")
    for file in "${required_files[@]}"; do
        if [ ! -f "./$file" ]; then
            log_error "找不到配置文件: $file"
            exit 1
        fi
    done
    
    log_info "配置信息:"
    log_info "  • 命名空间: $NAMESPACE"
    log_info "  • 副本数: $REPLICAS"
    log_info "  • 每节点 GPU 数量: $GPUS_PER_NODE"
    log_info "  • 总 GPU 数量: $((REPLICAS * GPUS_PER_NODE))"
    log_info "  • 网络后端: $NETWORK_BACKEND"
    log_info "  • 优化级别: $OPTIMIZATION_LEVEL"
    log_info "  • 测试数据大小: $TEST_SIZE"
    log_info "  • 测试持续时间: ${TEST_DURATION}秒"
    if [ "$SIMULATION_MODE" = "true" ]; then
        log_warn "  • 模拟模式: 已启用 (仅用于配置测试)"
    fi
    log_info ""
    
    # 应用配置 (按依赖顺序)
    log_info "应用 RBAC 配置..."
    sed -e "s/\$NAMESPACE/$NAMESPACE/g" ./nccl-rbac.yaml | kubectl apply -f - -n "$NAMESPACE"
    
    log_info "应用 ConfigMap..."
    kubectl apply -f ./nccl-configmap.yaml -n "$NAMESPACE"
    
    log_info "应用 Service..."
    kubectl apply -f ./nccl-service.yaml -n "$NAMESPACE"
    
    log_info "应用 StatefulSet (副本数: $REPLICAS)..."
    # 使用 sed 动态设置副本数、WORLD_SIZE 和测试参数
    local world_size=$((REPLICAS * GPUS_PER_NODE))
    sed -e "s/\$REPLICAS/$REPLICAS/g" \
        -e "s/\$WORLD_SIZE/$world_size/g" \
        -e "s/\$NPROC_PER_NODE/$GPUS_PER_NODE/g" \
        -e "s/\$NETWORK_BACKEND/$NETWORK_BACKEND/g" \
        -e "s/\$OPTIMIZATION_LEVEL/$OPTIMIZATION_LEVEL/g" \
        -e "s/\$TEST_SIZE/$TEST_SIZE/g" \
        -e "s/\$TEST_DURATION/$TEST_DURATION/g" \
        ./nccl-multinode-sts.yaml | kubectl apply -f - -n "$NAMESPACE"
    
    log_success "NCCL 测试已部署到 Kubernetes (StatefulSet)"
    log_info "StatefulSet 优势:"
    log_info "  • 稳定的 Pod 命名 (nccl-multinode-0, nccl-multinode-1, ...)"
    log_info "  • 可预测的 NODE_RANK 分配"
    log_info "  • 有序的启动和停止"
    log_info "  • 统一的配置管理 (ConfigMap)"
    log_info ""
    log_info "使用以下命令查看状态:"
    log_info "  kubectl get statefulsets -n $NAMESPACE"
    log_info "  kubectl get pods -n $NAMESPACE"
    log_info "  kubectl get configmaps -n $NAMESPACE"
}

# 查看状态
show_status() {
    log_header "查看 NCCL 测试状态"
    
    log_info "StatefulSet 状态:"
    kubectl get statefulsets -n "$NAMESPACE" -l app=nccl-multinode 2>/dev/null || log_warn "未找到 StatefulSet"
    
    echo ""
    log_info "Pod 状态:"
    kubectl get pods -n "$NAMESPACE" -l app=nccl-multinode -o wide 2>/dev/null || log_warn "未找到 Pod"
    
    echo ""
    log_info "Service 状态:"
    kubectl get services -n "$NAMESPACE" -l app=nccl-multinode 2>/dev/null || log_warn "未找到 Service"
}

# 查看日志
show_logs() {
    log_header "NCCL 测试日志"
    
    # 获取所有相关的 Pod
    PODS=$(kubectl get pods -n "$NAMESPACE" -l app=nccl-multinode -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
    
    if [ -z "$PODS" ]; then
        log_warn "未找到任何 Pod"
        return 1
    fi
    
    for pod in $PODS; do
        log_info "Pod: $pod"
        echo "----------------------------------------"
        kubectl logs "$pod" -n "$NAMESPACE" --tail=50
        echo ""
    done
}

# 清理资源
cleanup_resources() {
    log_header "清理 NCCL 测试资源"
    
    log_info "删除 StatefulSet..."
    kubectl delete statefulset -l app=nccl-multinode -n "$NAMESPACE" --ignore-not-found=true
    
    log_info "删除 Service..."
    kubectl delete service -l app=nccl-multinode -n "$NAMESPACE" --ignore-not-found=true
    
    log_info "删除 ConfigMap..."
    kubectl delete configmap -l app=nccl-multinode -n "$NAMESPACE" --ignore-not-found=true
    
    log_info "删除 RBAC 资源..."
    kubectl delete serviceaccount,role,rolebinding -l app=nccl-multinode -n "$NAMESPACE" --ignore-not-found=true
    
    log_info "等待 Pod 完全删除..."
    kubectl wait --for=delete pod -l app=nccl-multinode -n "$NAMESPACE" --timeout=60s 2>/dev/null || true
    
    log_success "清理完成"
}

# 主函数
main() {
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            deploy)
                ACTION="deploy"
                shift
                ;;
            status)
                ACTION="status"
                shift
                ;;
            logs)
                ACTION="logs"
                shift
                ;;
            cleanup)
                ACTION="cleanup"
                shift
                ;;
            -r|--replicas)
            REPLICAS="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --network)
            NETWORK_BACKEND="$2"
            shift 2
            ;;
        --optimization)
            OPTIMIZATION_LEVEL="$2"
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
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --simulation)
                SIMULATION_MODE="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                echo "$SCRIPT_NAME v$VERSION"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查必需的参数
    if [ -z "$ACTION" ]; then
        log_error "请指定操作"
        show_help
        exit 1
    fi
    
    # 执行相应的操作
    case $ACTION in
        deploy)
            check_prerequisites
            deploy_test
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        cleanup)
            cleanup_resources
            ;;
        *)
            log_error "未知操作: $ACTION"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"