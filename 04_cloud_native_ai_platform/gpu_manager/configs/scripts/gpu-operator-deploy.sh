#!/bin/bash
# GPU Operator 一键部署脚本
# 支持在Kubernetes集群中自动部署NVIDIA GPU Operator

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量
GPU_OPERATOR_VERSION="v23.9.1"
HELM_CHART_VERSION="v23.9.1"
NAMESPACE="gpu-operator"
NFD_ENABLED="true"
MIG_STRATEGY="single"
TIME_SLICING_ENABLED="false"
DRIVER_VERSION="535.129.03"
CUDA_VERSION="12.2"

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "命令 $1 未找到，请先安装"
        return 1
    fi
}

# 检查先决条件
check_prerequisites() {
    log_info "检查部署先决条件..."
    
    # 检查必要命令
    check_command kubectl
    check_command helm
    
    # 检查Kubernetes连接
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到Kubernetes集群"
        return 1
    fi
    
    # 检查Kubernetes版本
    local k8s_version=$(kubectl version --short --client | grep -oE 'v[0-9]+\.[0-9]+' | head -1)
    log_info "Kubernetes版本: $k8s_version"
    
    # 检查GPU节点
    local gpu_nodes=$(kubectl get nodes -l feature.node.kubernetes.io/pci-0300_10de.present=true --no-headers 2>/dev/null | wc -l || echo "0")
    if [[ $gpu_nodes -eq 0 ]]; then
        log_warn "未检测到GPU节点，请确保集群中有GPU节点"
    else
        log_info "检测到 $gpu_nodes 个GPU节点"
    fi
    
    log_info "✅ 先决条件检查完成"
}

# 安装Node Feature Discovery (NFD)
install_nfd() {
    if [[ "$NFD_ENABLED" != "true" ]]; then
        log_info "跳过NFD安装"
        return 0
    fi
    
    log_info "安装Node Feature Discovery..."
    
    # 检查NFD是否已安装
    if kubectl get deployment -n node-feature-discovery nfd-master &> /dev/null; then
        log_warn "NFD已经安装，跳过"
        return 0
    fi
    
    # 添加NFD Helm仓库
    helm repo add nfd https://kubernetes-sigs.github.io/node-feature-discovery/charts
    helm repo update
    
    # 创建命名空间
    kubectl create namespace node-feature-discovery --dry-run=client -o yaml | kubectl apply -f -
    
    # 安装NFD
    helm upgrade --install nfd nfd/node-feature-discovery \
        --namespace node-feature-discovery \
        --set master.tolerations[0].key=node-role.kubernetes.io/master \
        --set master.tolerations[0].operator=Exists \
        --set master.tolerations[0].effect=NoSchedule \
        --set worker.tolerations[0].operator=Exists \
        --set worker.tolerations[0].effect=NoSchedule
    
    # 等待NFD部署完成
    log_info "等待NFD部署完成..."
    kubectl wait --for=condition=available --timeout=300s deployment/nfd-master -n node-feature-discovery
    
    log_info "✅ NFD安装完成"
}

# 安装GPU Operator
install_gpu_operator() {
    log_info "安装GPU Operator..."
    
    # 添加NVIDIA Helm仓库
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
    helm repo update
    
    # 创建命名空间
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # 准备Helm values
    local helm_values=""
    
    # MIG配置
    if [[ "$MIG_STRATEGY" != "none" ]]; then
        helm_values="$helm_values --set mig.strategy=$MIG_STRATEGY"
    fi
    
    # 时间切片配置
    if [[ "$TIME_SLICING_ENABLED" == "true" ]]; then
        helm_values="$helm_values --set devicePlugin.config.name=time-slicing-config"
    fi
    
    # 驱动版本配置
    if [[ -n "$DRIVER_VERSION" ]]; then
        helm_values="$helm_values --set driver.version=$DRIVER_VERSION"
    fi
    
    # CUDA版本配置
    if [[ -n "$CUDA_VERSION" ]]; then
        helm_values="$helm_values --set toolkit.version=$CUDA_VERSION"
    fi
    
    # 安装GPU Operator
    log_info "执行Helm安装..."
    helm upgrade --install gpu-operator nvidia/gpu-operator \
        --namespace $NAMESPACE \
        --version $HELM_CHART_VERSION \
        --set operator.defaultRuntime=containerd \
        --set driver.enabled=true \
        --set toolkit.enabled=true \
        --set devicePlugin.enabled=true \
        --set dcgmExporter.enabled=true \
        --set gfd.enabled=true \
        --set migManager.enabled=true \
        --set nodeStatusExporter.enabled=true \
        --set gds.enabled=false \
        --set vgpuManager.enabled=false \
        --set vgpuDeviceManager.enabled=false \
        --set sandboxWorkloads.enabled=false \
        --set vfioManager.enabled=false \
        $helm_values
    
    log_info "✅ GPU Operator安装完成"
}

# 等待部署完成
wait_for_deployment() {
    log_info "等待GPU Operator部署完成..."
    
    # 等待所有Pod就绪
    local timeout=600
    local elapsed=0
    local interval=10
    
    while [[ $elapsed -lt $timeout ]]; do
        local ready_pods=$(kubectl get pods -n $NAMESPACE --no-headers | grep -c "Running\|Completed" || echo "0")
        local total_pods=$(kubectl get pods -n $NAMESPACE --no-headers | wc -l || echo "0")
        
        log_debug "Pod状态: $ready_pods/$total_pods 就绪"
        
        if [[ $ready_pods -eq $total_pods && $total_pods -gt 0 ]]; then
            log_info "✅ 所有Pod已就绪"
            break
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    if [[ $elapsed -ge $timeout ]]; then
        log_warn "⚠️ 部署超时，请检查Pod状态"
        kubectl get pods -n $NAMESPACE
        return 1
    fi
}

# 验证GPU资源
verify_gpu_resources() {
    log_info "验证GPU资源注册..."
    
    # 检查节点GPU资源
    local gpu_nodes=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.capacity."nvidia.com/gpu" != null) | .metadata.name' 2>/dev/null || echo "")
    
    if [[ -z "$gpu_nodes" ]]; then
        log_warn "⚠️ 未检测到GPU资源，可能需要等待更长时间"
        return 1
    fi
    
    log_info "GPU节点列表:"
    echo "$gpu_nodes" | while read -r node; do
        if [[ -n "$node" ]]; then
            local gpu_count=$(kubectl get node $node -o jsonpath='{.status.capacity.nvidia\.com/gpu}' 2>/dev/null || echo "0")
            log_info "  - $node: $gpu_count GPU(s)"
        fi
    done
    
    log_info "✅ GPU资源验证完成"
}

# 部署测试工作负载
deploy_test_workload() {
    log_info "部署GPU测试工作负载..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
  namespace: default
spec:
  restartPolicy: Never
  containers:
  - name: gpu-test
    image: nvidia/cuda:12.2-runtime-ubuntu20.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
  tolerations:
  - operator: Exists
    effect: NoSchedule
EOF
    
    # 等待Pod完成
    log_info "等待测试Pod完成..."
    kubectl wait --for=condition=complete --timeout=120s pod/gpu-test -n default || true
    
    # 显示测试结果
    log_info "测试结果:"
    kubectl logs gpu-test -n default || log_warn "无法获取测试日志"
    
    # 清理测试Pod
    kubectl delete pod gpu-test -n default --ignore-not-found=true
    
    log_info "✅ 测试工作负载完成"
}

# 配置时间切片
configure_time_slicing() {
    if [[ "$TIME_SLICING_ENABLED" != "true" ]]; then
        log_info "跳过时间切片配置"
        return 0
    fi
    
    log_info "配置GPU时间切片..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: $NAMESPACE
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4
EOF
    
    # 重启device plugin
    kubectl delete pods -n $NAMESPACE -l app=nvidia-device-plugin-daemonset
    
    log_info "✅ 时间切片配置完成"
}

# 显示部署状态
show_status() {
    log_info "GPU Operator部署状态:"
    
    echo "=== Helm Release ==="
    helm list -n $NAMESPACE
    
    echo ""
    echo "=== Pod状态 ==="
    kubectl get pods -n $NAMESPACE -o wide
    
    echo ""
    echo "=== GPU节点资源 ==="
    kubectl get nodes -o custom-columns="NAME:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu" | grep -v "<none>" || echo "未找到GPU资源"
    
    echo ""
    echo "=== GPU设备信息 ==="
    kubectl describe nodes | grep -A 5 "nvidia.com/gpu" || echo "未找到GPU设备信息"
}

# 卸载GPU Operator
uninstall_gpu_operator() {
    log_info "卸载GPU Operator..."
    
    # 删除测试工作负载
    kubectl delete pod gpu-test -n default --ignore-not-found=true
    
    # 卸载Helm release
    helm uninstall gpu-operator -n $NAMESPACE || true
    
    # 删除命名空间
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    
    # 卸载NFD（可选）
    if [[ "$NFD_ENABLED" == "true" ]]; then
        log_info "卸载Node Feature Discovery..."
        helm uninstall nfd -n node-feature-discovery || true
        kubectl delete namespace node-feature-discovery --ignore-not-found=true
    fi
    
    log_info "✅ GPU Operator卸载完成"
}

# 显示帮助信息
show_help() {
    echo "GPU Operator部署脚本"
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  install     安装GPU Operator"
    echo "  uninstall   卸载GPU Operator"
    echo "  status      显示部署状态"
    echo "  test        部署测试工作负载"
    echo "  verify      验证GPU资源"
    echo ""
    echo "选项:"
    echo "  --version VERSION        GPU Operator版本 (默认: $GPU_OPERATOR_VERSION)"
    echo "  --namespace NAMESPACE    命名空间 (默认: $NAMESPACE)"
    echo "  --driver-version VERSION 驱动版本 (默认: $DRIVER_VERSION)"
    echo "  --cuda-version VERSION   CUDA版本 (默认: $CUDA_VERSION)"
    echo "  --enable-mig             启用MIG支持"
    echo "  --enable-time-slicing    启用时间切片"
    echo "  --disable-nfd            禁用NFD安装"
    echo ""
    echo "示例:"
    echo "  $0 install                           # 标准安装"
    echo "  $0 install --enable-mig              # 启用MIG支持"
    echo "  $0 install --enable-time-slicing     # 启用时间切片"
    echo "  $0 status                            # 查看状态"
    echo "  $0 uninstall                         # 卸载"
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version)
                GPU_OPERATOR_VERSION="$2"
                HELM_CHART_VERSION="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --driver-version)
                DRIVER_VERSION="$2"
                shift 2
                ;;
            --cuda-version)
                CUDA_VERSION="$2"
                shift 2
                ;;
            --enable-mig)
                MIG_STRATEGY="mixed"
                shift
                ;;
            --enable-time-slicing)
                TIME_SLICING_ENABLED="true"
                shift
                ;;
            --disable-nfd)
                NFD_ENABLED="false"
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    local command=$1
    shift
    
    # 解析剩余参数
    parse_args "$@"
    
    case $command in
        "install")
            check_prerequisites
            install_nfd
            install_gpu_operator
            configure_time_slicing
            wait_for_deployment
            verify_gpu_resources
            deploy_test_workload
            show_status
            ;;
        "uninstall")
            uninstall_gpu_operator
            ;;
        "status")
            show_status
            ;;
        "test")
            deploy_test_workload
            ;;
        "verify")
            verify_gpu_resources
            ;;
        "help"|"--help"|"h")
            show_help
            ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"