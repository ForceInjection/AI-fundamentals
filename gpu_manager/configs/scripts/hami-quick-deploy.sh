#!/bin/bash
# HAMi快速部署脚本
# 自动化部署HAMi GPU管理系统

set -e

# 配置变量
HAMI_VERSION="v2.3.9"
KUBERNETES_VERSION="1.28"
NAMESPACE="hami-system"
HELM_CHART_URL="https://project-hami.github.io/HAMi"
CONFIG_DIR="/tmp/hami-config"
LOG_FILE="/var/log/hami-deploy.log"
VERBOSE=false
DRY_RUN=false
SKIP_PREREQ=false

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log() {
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
    log "INFO: $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
    log "WARN: $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
    log "ERROR: $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
    log "SUCCESS: $*"
}

# 检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "命令 '$1' 未找到，请先安装"
        return 1
    fi
    return 0
}

# 检查先决条件
check_prerequisites() {
    if [[ "$SKIP_PREREQ" == "true" ]]; then
        log_info "跳过先决条件检查"
        return 0
    fi
    
    log_info "检查先决条件..."
    
    # 检查必要命令
    local required_commands=("kubectl" "helm" "docker")
    for cmd in "${required_commands[@]}"; do
        if ! check_command "$cmd"; then
            return 1
        fi
    done
    
    # 检查Kubernetes连接
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到Kubernetes集群"
        log_info "请确保:"
        log_info "  1. kubectl已正确配置"
        log_info "  2. 集群可访问"
        log_info "  3. 具有管理员权限"
        return 1
    fi
    
    # 检查Kubernetes版本
    local k8s_version
    k8s_version=$(kubectl version --client -o json | jq -r '.clientVersion.gitVersion' | sed 's/v//')
    local major_version=$(echo "$k8s_version" | cut -d. -f1)
    local minor_version=$(echo "$k8s_version" | cut -d. -f2)
    
    if [[ "$major_version" -lt 1 ]] || [[ "$major_version" -eq 1 && "$minor_version" -lt 20 ]]; then
        log_warn "Kubernetes版本 ($k8s_version) 可能不兼容，建议使用1.20+"
    fi
    
    # 检查GPU节点
    local gpu_nodes
    gpu_nodes=$(kubectl get nodes -l accelerator=nvidia -o name 2>/dev/null | wc -l)
    if [[ "$gpu_nodes" -eq 0 ]]; then
        log_warn "未找到标记为GPU的节点，请确保节点已正确标记"
        log_info "可以使用以下命令标记GPU节点:"
        log_info "  kubectl label nodes <node-name> accelerator=nvidia"
    else
        log_success "找到 $gpu_nodes 个GPU节点"
    fi
    
    # 检查NVIDIA驱动
    log_info "检查GPU节点上的NVIDIA驱动..."
    kubectl get nodes -l accelerator=nvidia -o name 2>/dev/null | while read -r node; do
        node_name=$(echo "$node" | cut -d/ -f2)
        if kubectl debug "$node_name" -it --image=nvidia/cuda:12.4-base-ubuntu22.04 -- nvidia-smi &> /dev/null; then
            log_success "节点 $node_name 上的NVIDIA驱动正常"
        else
            log_warn "节点 $node_name 上的NVIDIA驱动可能有问题"
        fi
    done
    
    log_success "先决条件检查完成"
    return 0
}

# 安装Helm Chart
install_helm_chart() {
    log_info "安装HAMi Helm Chart..."
    
    # 添加Helm仓库
    log_info "添加HAMi Helm仓库..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] helm repo add hami $HELM_CHART_URL"
    else
        helm repo add hami "$HELM_CHART_URL" || {
            log_error "添加Helm仓库失败"
            return 1
        }
        helm repo update
    fi
    
    # 创建命名空间
    log_info "创建命名空间 $NAMESPACE..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] kubectl create namespace $NAMESPACE"
    else
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    # 准备values文件
    local values_file="$CONFIG_DIR/hami-values.yaml"
    mkdir -p "$CONFIG_DIR"
    
    cat > "$values_file" << EOF
# HAMi配置文件
scheduler:
  kubeScheduler:
    imageTag: "v$KUBERNETES_VERSION.5"
  extender:
    image: "projecthami/hami:$HAMI_VERSION"

devicePlugin:
  image: "projecthami/hami:$HAMI_VERSION"
  env:
    - name: NODE_NAME
      valueFrom:
        fieldRef:
          fieldPath: spec.nodeName
    - name: NVIDIA_VISIBLE_DEVICES
      value: all
    - name: NVIDIA_DRIVER_CAPABILITIES
      value: utility

monitor:
  enabled: true
  serviceMonitor:
    enabled: true

resources:
  scheduler:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 512Mi
  devicePlugin:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 512Mi

nodeSelector:
  accelerator: nvidia

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  - key: CriticalAddonsOnly
    operator: Exists
  - effect: NoSchedule
    key: node.kubernetes.io/not-ready
    operator: Exists

affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: accelerator
          operator: In
          values:
          - nvidia
EOF
    
    # 安装HAMi
    log_info "安装HAMi到命名空间 $NAMESPACE..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] helm install hami hami/hami --namespace $NAMESPACE --values $values_file"
        log_info "配置文件内容:"
        cat "$values_file"
    else
        helm install hami hami/hami \
            --namespace "$NAMESPACE" \
            --values "$values_file" \
            --wait \
            --timeout 10m || {
            log_error "HAMi安装失败"
            return 1
        }
    fi
    
    log_success "HAMi安装完成"
    return 0
}

# 手动部署（不使用Helm）
manual_deploy() {
    log_info "手动部署HAMi组件..."
    
    # 创建命名空间
    log_info "创建命名空间..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] 创建命名空间和RBAC"
    else
        kubectl apply -f - << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: $NAMESPACE
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: hami-device-plugin
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: hami-device-plugin
rules:
- apiGroups: [""]
  resources: ["nodes", "pods"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: hami-device-plugin
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: hami-device-plugin
subjects:
- kind: ServiceAccount
  name: hami-device-plugin
  namespace: $NAMESPACE
EOF
    fi
    
    # 部署设备插件
    log_info "部署HAMi设备插件..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] 部署设备插件DaemonSet"
    else
        kubectl apply -f - << EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: hami-device-plugin
  namespace: $NAMESPACE
spec:
  selector:
    matchLabels:
      app: hami-device-plugin
  template:
    metadata:
      labels:
        app: hami-device-plugin
    spec:
      serviceAccountName: hami-device-plugin
      hostNetwork: true
      hostPID: true
      containers:
      - name: hami-device-plugin
        image: projecthami/hami:$HAMI_VERSION
        command:
        - /usr/bin/hami-device-plugin
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        - name: NVIDIA_VISIBLE_DEVICES
          value: all
        securityContext:
          privileged: true
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
        - name: dev
          mountPath: /dev
        - name: sys
          mountPath: /sys
        - name: proc
          mountPath: /proc
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
      - name: dev
        hostPath:
          path: /dev
      - name: sys
        hostPath:
          path: /sys
      - name: proc
        hostPath:
          path: /proc
      nodeSelector:
        accelerator: nvidia
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
EOF
    fi
    
    log_success "HAMi手动部署完成"
    return 0
}

# 验证部署
verify_deployment() {
    log_info "验证HAMi部署..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] 跳过部署验证"
        return 0
    fi
    
    # 等待Pod就绪
    log_info "等待HAMi组件启动..."
    kubectl wait --for=condition=Ready pod -l app=hami-device-plugin -n "$NAMESPACE" --timeout=300s || {
        log_error "HAMi设备插件启动超时"
        return 1
    }
    
    # 检查Pod状态
    log_info "检查HAMi组件状态..."
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # 检查GPU资源
    log_info "检查GPU资源注册..."
    local gpu_resources
    gpu_resources=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.allocatable."hami.io/gpu"?) | .metadata.name + ": " + .status.allocatable."hami.io/gpu"')
    
    if [[ -n "$gpu_resources" ]]; then
        log_success "GPU资源已注册:"
        echo "$gpu_resources"
    else
        log_warn "未检测到HAMi GPU资源，可能需要等待或检查配置"
    fi
    
    # 检查调度器
    if kubectl get pods -n "$NAMESPACE" -l app=hami-scheduler &> /dev/null; then
        log_info "检查HAMi调度器状态..."
        kubectl get pods -n "$NAMESPACE" -l app=hami-scheduler
    fi
    
    log_success "HAMi部署验证完成"
    return 0
}

# 部署测试工作负载
deploy_test_workload() {
    log_info "部署测试工作负载..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] 跳过测试工作负载部署"
        return 0
    fi
    
    # 创建测试Pod
    kubectl apply -f - << EOF
apiVersion: v1
kind: Pod
metadata:
  name: hami-test-pod
  namespace: default
spec:
  containers:
  - name: gpu-test
    image: nvidia/cuda:12.4-runtime-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        hami.io/gpu: 1
        hami.io/gpumem: 1024
        hami.io/gpucores: 50
  restartPolicy: Never
EOF
    
    # 等待Pod完成
    log_info "等待测试Pod完成..."
    kubectl wait --for=condition=Ready pod/hami-test-pod --timeout=120s || {
        log_warn "测试Pod启动超时，检查日志:"
        kubectl describe pod hami-test-pod
        return 1
    }
    
    # 查看测试结果
    log_info "测试Pod输出:"
    kubectl logs hami-test-pod
    
    # 清理测试Pod
    kubectl delete pod hami-test-pod
    
    log_success "测试工作负载完成"
    return 0
}

# 生成使用指南
generate_usage_guide() {
    local guide_file="$CONFIG_DIR/hami-usage-guide.md"
    
    cat > "$guide_file" << EOF
# HAMi使用指南

## 基本概念

HAMi支持GPU切分和共享，主要资源类型：
- \`hami.io/gpu\`: GPU数量
- \`hami.io/gpumem\`: GPU内存（MB）
- \`hami.io/gpucores\`: GPU计算资源（百分比）

## 使用示例

### 1. 基本GPU使用
\`\`\`yaml
resources:
  limits:
    hami.io/gpu: 1
    hami.io/gpumem: 2048  # 2GB
    hami.io/gpucores: 50  # 50%
\`\`\`

### 2. GPU切分示例
\`\`\`yaml
# Pod 1 - 使用50%的GPU
resources:
  limits:
    hami.io/gpu: 1
    hami.io/gpumem: 4096
    hami.io/gpucores: 50

# Pod 2 - 使用另外50%的GPU
resources:
  limits:
    hami.io/gpu: 1
    hami.io/gpumem: 4096
    hami.io/gpucores: 50
\`\`\`

### 3. 多GPU使用
\`\`\`yaml
resources:
  limits:
    hami.io/gpu: 2
    hami.io/gpumem: 8192
    hami.io/gpucores: 100
\`\`\`

## 监控命令

\`\`\`bash
# 查看GPU资源
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.hami\.io/gpu

# 查看HAMi组件状态
kubectl get pods -n $NAMESPACE

# 查看GPU使用情况
kubectl top nodes
\`\`\`

## 故障排查

\`\`\`bash
# 查看设备插件日志
kubectl logs -n $NAMESPACE -l app=hami-device-plugin

# 查看调度器日志
kubectl logs -n $NAMESPACE -l app=hami-scheduler

# 检查GPU节点标签
kubectl get nodes --show-labels | grep accelerator
\`\`\`
EOF
    
    log_success "使用指南已生成: $guide_file"
}

# 清理部署
cleanup_deployment() {
    log_info "清理HAMi部署..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] 清理HAMi部署"
        return 0
    fi
    
    # 删除Helm release
    if helm list -n "$NAMESPACE" | grep -q hami; then
        log_info "删除Helm release..."
        helm uninstall hami -n "$NAMESPACE"
    fi
    
    # 删除手动部署的资源
    kubectl delete daemonset hami-device-plugin -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete clusterrolebinding hami-device-plugin 2>/dev/null || true
    kubectl delete clusterrole hami-device-plugin 2>/dev/null || true
    kubectl delete serviceaccount hami-device-plugin -n "$NAMESPACE" 2>/dev/null || true
    
    # 删除命名空间
    kubectl delete namespace "$NAMESPACE" 2>/dev/null || true
    
    log_success "HAMi清理完成"
}

# 显示帮助信息
show_help() {
    cat << EOF
HAMi快速部署脚本

用法: $0 [选项] [命令]

命令:
  install         安装HAMi（默认）
  manual          手动部署（不使用Helm）
  verify          验证部署
  test            部署测试工作负载
  cleanup         清理部署
  guide           生成使用指南

选项:
  -h, --help              显示此帮助信息
  -v, --verbose           详细输出
  -n, --namespace NAME    指定命名空间（默认: hami-system）
  --version VERSION       指定HAMi版本（默认: $HAMI_VERSION）
  --dry-run              仅显示将要执行的操作
  --skip-prereq          跳过先决条件检查
  --helm-url URL         指定Helm Chart URL
  --config-dir DIR       指定配置目录

示例:
  $0 install                    # 安装HAMi
  $0 --dry-run install          # 预览安装过程
  $0 manual                     # 手动部署
  $0 verify                     # 验证部署
  $0 test                       # 运行测试
  $0 cleanup                    # 清理部署
EOF
}

# 主函数
main() {
    local command="install"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --version)
                HAMI_VERSION="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-prereq)
                SKIP_PREREQ=true
                shift
                ;;
            --helm-url)
                HELM_CHART_URL="$2"
                shift 2
                ;;
            --config-dir)
                CONFIG_DIR="$2"
                shift 2
                ;;
            install|manual|verify|test|cleanup|guide)
                command="$1"
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 创建日志目录
    mkdir -p "$(dirname "$LOG_FILE")"
    
    log_info "开始HAMi部署脚本"
    log_info "命令: $command"
    log_info "版本: $HAMI_VERSION"
    log_info "命名空间: $NAMESPACE"
    
    # 执行命令
    case $command in
        install)
            check_prerequisites
            install_helm_chart
            verify_deployment
            generate_usage_guide
            ;;
        manual)
            check_prerequisites
            manual_deploy
            verify_deployment
            generate_usage_guide
            ;;
        verify)
            verify_deployment
            ;;
        test)
            deploy_test_workload
            ;;
        cleanup)
            cleanup_deployment
            ;;
        guide)
            generate_usage_guide
            ;;
    esac
    
    log_success "HAMi部署脚本完成"
}

# 执行主函数
main "$@"