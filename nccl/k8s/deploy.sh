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
JOB_NAME="nccl-multinode-test"

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

用法: $0 [命令] [选项]

命令:
  deploy      部署 NCCL 测试到 Kubernetes
  status      查看测试状态
  logs        查看测试日志
  cleanup     清理测试资源
  help        显示此帮助信息

选项:
  -n, --namespace NAMESPACE    指定命名空间 [默认: default]
  -j, --job-name NAME         指定作业名称 [默认: nccl-multinode-test]
  -w, --world-size SIZE       总 GPU 数量 [默认: 4]
  -g, --gpus-per-node COUNT   每节点 GPU 数量 [默认: 2]
  -s, --test-size SIZE        测试数据大小 [默认: 100M]
  -t, --test-time SECONDS     测试时长 [默认: 60]
  --network BACKEND           网络后端 [默认: ib]
  --image IMAGE               Docker 镜像 [默认: nccl-test:latest]

示例:
  # 部署默认配置的测试
  $0 deploy
  
  # 部署自定义配置
  $0 deploy --world-size 8 --gpus-per-node 4 --test-size 1G
  
  # 查看测试状态
  $0 status
  
  # 查看测试日志
  $0 logs
  
  # 清理资源
  $0 cleanup

前置条件:
  • Kubernetes 集群已配置
  • kubectl 已安装并配置
  • 集群支持 GPU (NVIDIA Device Plugin)
  • 节点支持 InfiniBand (如果使用 IB 网络)
  • Docker 镜像已推送到可访问的仓库

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
        log_warning "命名空间 $NAMESPACE 不存在，将创建"
        kubectl create namespace "$NAMESPACE"
        log_success "命名空间 $NAMESPACE 已创建"
    else
        log_success "命名空间 $NAMESPACE 存在"
    fi
    
    # 检查 GPU 节点
    local gpu_nodes=$(kubectl get nodes -l nvidia.com/gpu.present=true --no-headers | wc -l)
    if [ "$gpu_nodes" -eq 0 ]; then
        log_error "集群中没有 GPU 节点"
        log_info "请确保节点已安装 NVIDIA Device Plugin"
        exit 1
    fi
    log_success "检测到 $gpu_nodes 个 GPU 节点"
    
    # 检查配置文件
    if [ ! -f "k8s/nccl-multinode-job.yaml" ]; then
        log_error "找不到 Kubernetes 配置文件"
        log_info "请确保在正确的目录下运行此脚本"
        exit 1
    fi
    log_success "Kubernetes 配置文件存在"
}

# 部署测试
deploy_test() {
    log_header "部署 NCCL 多节点测试"
    
    # 应用配置
    log_info "应用 ConfigMap..."
    kubectl apply -f k8s/nccl-configmap.yaml -n "$NAMESPACE"
    
    log_info "应用 Service..."
    kubectl apply -f k8s/nccl-service.yaml -n "$NAMESPACE"
    
    log_info "应用 Job..."
    kubectl apply -f k8s/nccl-multinode-job.yaml -n "$NAMESPACE"
    
    log_success "NCCL 测试已部署到 Kubernetes"
    log_info "使用以下命令查看状态:"
    log_info "  kubectl get jobs -n $NAMESPACE"
    log_info "  kubectl get pods -n $NAMESPACE"
}

# 查看状态
show_status() {
    log_header "NCCL 测试状态"
    
    echo "Job 状态:"
    kubectl get jobs -n "$NAMESPACE" -l app=nccl-test
    
    echo ""
    echo "Pod 状态:"
    kubectl get pods -n "$NAMESPACE" -l app=nccl-test
    
    echo ""
    echo "Service 状态:"
    kubectl get services -n "$NAMESPACE" -l app=nccl-test
}

# 查看日志
show_logs() {
    log_header "NCCL 测试日志"
    
    local pods=$(kubectl get pods -n "$NAMESPACE" -l app=nccl-test --no-headers -o custom-columns=":metadata.name")
    
    if [ -z "$pods" ]; then
        log_warning "没有找到运行中的 Pod"
        return
    fi
    
    for pod in $pods; do
        echo ""
        log_info "Pod: $pod"
        echo "----------------------------------------"
        kubectl logs "$pod" -n "$NAMESPACE" --tail=50
    done
}

# 清理资源
cleanup_resources() {
    log_header "清理 NCCL 测试资源"
    
    log_info "删除 Job..."
    kubectl delete job "$JOB_NAME" -n "$NAMESPACE" --ignore-not-found=true
    
    log_info "删除 Service..."
    kubectl delete service nccl-master-service -n "$NAMESPACE" --ignore-not-found=true
    
    log_info "删除 ConfigMap..."
    kubectl delete configmap nccl-config -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "资源清理完成"
}

# 解析参数
parse_arguments() {
    COMMAND=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            deploy|status|logs|cleanup|help)
                COMMAND="$1"
                shift
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -j|--job-name)
                JOB_NAME="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                echo "使用 '$0 help' 查看帮助信息"
                exit 1
                ;;
        esac
    done
    
    if [ -z "$COMMAND" ]; then
        log_error "请指定命令"
        echo "使用 '$0 help' 查看帮助信息"
        exit 1
    fi
}

# 主函数
main() {
    log_header "$SCRIPT_NAME v$VERSION"
    
    # 解析参数
    parse_arguments "$@"
    
    # 执行命令
    case $COMMAND in
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
        help)
            show_help
            ;;
        *)
            log_error "未知命令: $COMMAND"
            exit 1
            ;;
    esac
    
    log_success "操作完成"
}

# 运行主函数
main "$@"