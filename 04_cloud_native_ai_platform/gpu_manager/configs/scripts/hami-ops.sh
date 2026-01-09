#!/bin/bash

# HAMi (GPU 资源管理平台) 运维脚本
# 提供 HAMi 的部署、监控、维护和故障排查功能

set -e

# 配置参数
HAMI_NAMESPACE="${1:-hami-system}"
ACTION="${2:-status}"  # status|deploy|upgrade|cleanup|monitor|troubleshoot

# 日志函数
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 检查依赖
check_dependencies() {
    local deps=("kubectl" "helm" "jq")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log "错误: 缺少依赖 $dep"
            exit 1
        fi
    done
}

# 检查 HAMi 状态
check_hami_status() {
    log "检查 HAMi 系统状态..."
    
    # 检查命名空间
    if ! kubectl get namespace "$HAMI_NAMESPACE" &> /dev/null; then
        log "HAMi 命名空间 $HAMI_NAMESPACE 不存在"
        return 1
    fi
    
    # 检查所有 Pod 状态
    local pods=$(kubectl get pods -n "$HAMI_NAMESPACE" -o json)
    local total_pods=$(echo "$pods" | jq '.items | length')
    local running_pods=$(echo "$pods" | jq '.items[] | select(.status.phase == "Running") | .metadata.name' | wc -l)
    
    log "HAMi Pod 状态: $running_pods/$total_pods 运行中"
    
    # 显示有问题的 Pod
    local problem_pods=$(echo "$pods" | jq -r '.items[] | select(.status.phase != "Running") | "\(.metadata.name): \(.status.phase)"')
    if [ -n "$problem_pods" ]; then
        log "有问题的 Pod:"
        echo "$problem_pods"
    fi
    
    # 检查服务状态
    log "检查 HAMi 服务..."
    kubectl get svc -n "$HAMI_NAMESPACE"
    
    # 检查 GPU 资源
    log "检查 GPU 资源分配..."
    kubectl get nodes -o json | jq '.items[] | select(.status.allocatable["nvidia.com/gpu"] != null) | {name: .metadata.name, gpu: .status.allocatable["nvidia.com/gpu"]}'
}

# 部署 HAMi
deploy_hami() {
    log "部署 HAMi 系统..."
    
    # 创建命名空间
    kubectl create namespace "$HAMI_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # 添加 HAMi Helm 仓库
    if ! helm repo list | grep -q "hami-repo"; then
        helm repo add hami-repo https://hami.github.io/charts
        helm repo update
    fi
    
    # 部署 HAMi
    helm upgrade --install hami hami-repo/hami \
        --namespace "$HAMI_NAMESPACE" \
        --set scheduler.enabled=true \
        --set devicePlugin.enabled=true \
        --set monitoring.enabled=true \
        --set webhook.enabled=true \
        --wait
    
    log "HAMi 部署完成"
}

# 升级 HAMi
upgrade_hami() {
    log "升级 HAMi 系统..."
    
    # 更新 Helm 仓库
    helm repo update hami-repo
    
    # 获取最新版本
    local latest_version=$(helm search repo hami-repo/hami --versions | head -2 | tail -1 | awk '{print $2}')
    log "最新 HAMi 版本: $latest_version"
    
    # 执行升级
    helm upgrade hami hami-repo/hami \
        --namespace "$HAMI_NAMESPACE" \
        --version "$latest_version" \
        --reuse-values \
        --wait
    
    log "HAMi 升级完成"
}

# 清理 HAMi
cleanup_hami() {
    log "清理 HAMi 系统..."
    
    read -p "确定要删除 HAMi 系统吗？(y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        log "取消清理操作"
        return
    fi
    
    # 卸载 HAMi
    helm uninstall hami --namespace "$HAMI_NAMESPACE"
    
    # 删除命名空间（可选）
    read -p "是否删除命名空间 $HAMI_NAMESPACE？(y/N): " delete_ns
    if [[ "$delete_ns" == "y" || "$delete_ns" == "Y" ]]; then
        kubectl delete namespace "$HAMI_NAMESPACE"
        log "命名空间 $HAMI_NAMESPACE 已删除"
    fi
    
    log "HAMi 清理完成"
}

# 监控 HAMi
monitor_hami() {
    log "监控 HAMi 系统..."
    
    # 检查资源使用情况
    echo "=== HAMi 资源使用情况 ==="
    kubectl top pods -n "$HAMI_NAMESPACE"
    
    echo "=== GPU 节点资源 ==="
    kubectl describe nodes -l nvidia.com/gpu.product | grep -A5 -B5 "Allocatable:"
    
    # 检查事件
    echo "=== 最近事件 ==="
    kubectl get events -n "$HAMI_NAMESPACE" --sort-by='.lastTimestamp' | tail -10
    
    # 检查调度器状态
    echo "=== 调度器状态 ==="
    kubectl logs -n "$HAMI_NAMESPACE" -l app.kubernetes.io/component=scheduler --tail=20
}

# 故障排查
troubleshoot_hami() {
    log "HAMi 故障排查..."
    
    # 检查所有组件状态
    echo "=== 组件状态检查 ==="
    check_hami_status
    
    # 检查日志
    echo "=== 关键组件日志 ==="
    local components=("scheduler" "device-plugin" "webhook" "monitoring")
    for comp in "${components[@]}"; do
        echo "--- $comp 日志 ---"
        kubectl logs -n "$HAMI_NAMESPACE" -l "app.kubernetes.io/component=$comp" --tail=10 2>/dev/null || echo "没有找到 $comp 日志"
    done
    
    # 检查网络策略
    echo "=== 网络策略检查 ==="
    kubectl get networkpolicy -n "$HAMI_NAMESPACE"
    
    # 检查存储
    echo "=== 存储检查 ==="
    kubectl get pvc -n "$HAMI_NAMESPACE"
}

# 主函数
main() {
    check_dependencies
    
    case "$ACTION" in
        "status")
            check_hami_status
            ;;
        "deploy")
            deploy_hami
            ;;
        "upgrade")
            upgrade_hami
            ;;
        "cleanup")
            cleanup_hami
            ;;
        "monitor")
            monitor_hami
            ;;
        "troubleshoot")
            troubleshoot_hami
            ;;
        *)
            log "错误: 不支持的操作: $ACTION"
            echo "可用操作: status, deploy, upgrade, cleanup, monitor, troubleshoot"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"