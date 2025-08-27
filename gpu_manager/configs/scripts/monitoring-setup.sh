#!/bin/bash

# GPU 监控系统设置脚本
# 用于自动化部署和配置 GPU 监控系统

set -euo pipefail

# 配置变量
MONITORING_NAMESPACE="gpu-monitoring"
PROMETHEUS_VERSION="v2.45.0"
GRAFANA_VERSION="10.0.0"
NODE_EXPORTER_VERSION="v1.6.0"
DCGM_EXPORTER_VERSION="3.1.8-3.1.5-ubuntu20.04"
HELM_REPO_PROMETHEUS="https://prometheus-community.github.io/helm-charts"
HELM_REPO_GRAFANA="https://grafana.github.io/helm-charts"

# 日志配置
LOG_FILE="/tmp/monitoring-setup.log"
LOG_LEVEL="INFO"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖项..."
    
    # 检查 kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl 未安装"
        return 1
    fi
    
    # 检查 helm
    if ! command -v helm &> /dev/null; then
        log_error "helm 未安装"
        return 1
    fi
    
    # 检查 Kubernetes 连接
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到 Kubernetes 集群"
        return 1
    fi
    
    log_success "依赖项检查通过"
}

# 创建命名空间
create_namespace() {
    log_info "创建监控命名空间: $MONITORING_NAMESPACE"
    
    kubectl create namespace "$MONITORING_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "命名空间创建完成"
}

# 添加 Helm 仓库
add_helm_repos() {
    log_info "添加 Helm 仓库..."
    
    helm repo add prometheus-community "$HELM_REPO_PROMETHEUS"
    helm repo add grafana "$HELM_REPO_GRAFANA"
    helm repo update
    
    log_success "Helm 仓库添加完成"
}

# 部署 Prometheus
deploy_prometheus() {
    log_info "部署 Prometheus..."
    
    # 创建 Prometheus 配置
    cat > prometheus-values.yaml << 'EOF'
prometheus:
  prometheusSpec:
    retention: 30d
    storageSpec:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi
    additionalScrapeConfigs:
      - job_name: 'dcgm-exporter'
        static_configs:
          - targets: ['dcgm-exporter:9400']
        scrape_interval: 15s
        metrics_path: /metrics
      - job_name: 'node-exporter'
        kubernetes_sd_configs:
          - role: endpoints
        relabel_configs:
          - source_labels: [__meta_kubernetes_endpoints_name]
            regex: node-exporter
            action: keep

grafana:
  enabled: true
  adminPassword: admin123
  persistence:
    enabled: true
    size: 10Gi
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
      - name: 'gpu-dashboards'
        orgId: 1
        folder: 'GPU Monitoring'
        type: file
        disableDeletion: false
        editable: true
        options:
          path: /var/lib/grafana/dashboards/gpu-dashboards

nodeExporter:
  enabled: true
  
kubeStateMetrics:
  enabled: true

alertmanager:
  enabled: true
  alertmanagerSpec:
    storage:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 10Gi
EOF
    
    # 部署 Prometheus Stack
    helm upgrade --install prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace "$MONITORING_NAMESPACE" \
        --values prometheus-values.yaml \
        --version "51.2.0"
    
    log_success "Prometheus 部署完成"
}

# 部署 DCGM Exporter
deploy_dcgm_exporter() {
    log_info "部署 DCGM Exporter..."
    
    # 创建 DCGM Exporter DaemonSet
    cat > dcgm-exporter.yaml << EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
  namespace: $MONITORING_NAMESPACE
  labels:
    app: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    metadata:
      labels:
        app: dcgm-exporter
    spec:
      nodeSelector:
        accelerator: nvidia
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:$DCGM_EXPORTER_VERSION
        ports:
        - containerPort: 9400
          name: metrics
        env:
        - name: DCGM_EXPORTER_LISTEN
          value: ":9400"
        - name: DCGM_EXPORTER_KUBERNETES
          value: "true"
        securityContext:
          runAsNonRoot: false
          runAsUser: 0
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      hostNetwork: true
      hostPID: true
---
apiVersion: v1
kind: Service
metadata:
  name: dcgm-exporter
  namespace: $MONITORING_NAMESPACE
  labels:
    app: dcgm-exporter
spec:
  selector:
    app: dcgm-exporter
  ports:
  - port: 9400
    targetPort: 9400
    name: metrics
EOF
    
    kubectl apply -f dcgm-exporter.yaml
    
    log_success "DCGM Exporter 部署完成"
}

# 配置 GPU 监控告警规则
configure_gpu_alerts() {
    log_info "配置 GPU 监控告警规则..."
    
    cat > gpu-alerts.yaml << EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gpu-alerts
  namespace: $MONITORING_NAMESPACE
  labels:
    prometheus: kube-prometheus
    role: alert-rules
spec:
  groups:
  - name: gpu.rules
    rules:
    - alert: GPUHighUtilization
      expr: DCGM_FI_DEV_GPU_UTIL > 90
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "GPU utilization is high"
        description: "GPU {{ \$labels.gpu }} on node {{ \$labels.instance }} has utilization above 90% for more than 5 minutes."
    
    - alert: GPUHighMemoryUsage
      expr: (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL) * 100 > 85
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "GPU memory usage is high"
        description: "GPU {{ \$labels.gpu }} on node {{ \$labels.instance }} has memory usage above 85% for more than 5 minutes."
    
    - alert: GPUHighTemperature
      expr: DCGM_FI_DEV_GPU_TEMP > 80
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "GPU temperature is high"
        description: "GPU {{ \$labels.gpu }} on node {{ \$labels.instance }} temperature is above 80°C for more than 2 minutes."
    
    - alert: GPUDown
      expr: up{job="dcgm-exporter"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "GPU exporter is down"
        description: "DCGM exporter on node {{ \$labels.instance }} has been down for more than 1 minute."
EOF
    
    kubectl apply -f gpu-alerts.yaml
    
    log_success "GPU 告警规则配置完成"
}

# 导入 Grafana 仪表板
import_grafana_dashboards() {
    log_info "导入 Grafana GPU 仪表板..."
    
    # 等待 Grafana Pod 就绪
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=grafana -n "$MONITORING_NAMESPACE" --timeout=300s
    
    # 获取 Grafana Pod 名称
    GRAFANA_POD=$(kubectl get pods -n "$MONITORING_NAMESPACE" -l app.kubernetes.io/name=grafana -o jsonpath='{.items[0].metadata.name}')
    
    # 复制仪表板文件到 Grafana Pod
    if [ -f "../monitoring/grafana-gpu-dashboard.json" ]; then
        kubectl cp "../monitoring/grafana-gpu-dashboard.json" "$MONITORING_NAMESPACE/$GRAFANA_POD:/var/lib/grafana/dashboards/gpu-dashboards/" || true
        log_success "GPU 仪表板导入完成"
    else
        log_warn "GPU 仪表板文件不存在，跳过导入"
    fi
}

# 配置服务监控
configure_service_monitors() {
    log_info "配置服务监控..."
    
    cat > gpu-service-monitor.yaml << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
  namespace: $MONITORING_NAMESPACE
  labels:
    app: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
EOF
    
    kubectl apply -f gpu-service-monitor.yaml
    
    log_success "服务监控配置完成"
}

# 验证部署
verify_deployment() {
    log_info "验证监控系统部署..."
    
    # 检查 Pod 状态
    log_info "检查 Pod 状态..."
    kubectl get pods -n "$MONITORING_NAMESPACE"
    
    # 检查服务状态
    log_info "检查服务状态..."
    kubectl get svc -n "$MONITORING_NAMESPACE"
    
    # 检查 DCGM Exporter 指标
    log_info "检查 DCGM Exporter 指标..."
    DCGM_POD=$(kubectl get pods -n "$MONITORING_NAMESPACE" -l app=dcgm-exporter -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [ -n "$DCGM_POD" ]; then
        kubectl exec -n "$MONITORING_NAMESPACE" "$DCGM_POD" -- curl -s localhost:9400/metrics | head -10 || true
    fi
    
    # 显示访问信息
    log_info "监控系统访问信息:"
    echo "Prometheus: kubectl port-forward -n $MONITORING_NAMESPACE svc/prometheus-stack-kube-prom-prometheus 9090:9090"
    echo "Grafana: kubectl port-forward -n $MONITORING_NAMESPACE svc/prometheus-stack-grafana 3000:80"
    echo "Grafana 默认用户名/密码: admin/admin123"
    
    log_success "监控系统部署验证完成"
}

# 清理函数
cleanup() {
    log_info "清理临时文件..."
    rm -f prometheus-values.yaml dcgm-exporter.yaml gpu-alerts.yaml gpu-service-monitor.yaml
    log_success "清理完成"
}

# 卸载监控系统
uninstall_monitoring() {
    log_info "卸载监控系统..."
    
    # 卸载 Helm releases
    helm uninstall prometheus-stack -n "$MONITORING_NAMESPACE" || true
    
    # 删除 DCGM Exporter
    kubectl delete -f dcgm-exporter.yaml || true
    
    # 删除告警规则
    kubectl delete -f gpu-alerts.yaml || true
    
    # 删除服务监控
    kubectl delete -f gpu-service-monitor.yaml || true
    
    # 删除命名空间
    kubectl delete namespace "$MONITORING_NAMESPACE" || true
    
    log_success "监控系统卸载完成"
}

# 显示帮助信息
show_help() {
    cat << EOF
GPU 监控系统设置脚本

用法: $0 [选项]

选项:
  install         安装完整的 GPU 监控系统
  uninstall       卸载 GPU 监控系统
  verify          验证监控系统部署状态
  prometheus      仅部署 Prometheus
  dcgm            仅部署 DCGM Exporter
  alerts          仅配置告警规则
  dashboards      仅导入 Grafana 仪表板
  cleanup         清理临时文件
  help            显示此帮助信息

环境变量:
  MONITORING_NAMESPACE    监控命名空间 (默认: gpu-monitoring)
  PROMETHEUS_VERSION      Prometheus 版本 (默认: v2.45.0)
  GRAFANA_VERSION         Grafana 版本 (默认: 10.0.0)
  DCGM_EXPORTER_VERSION   DCGM Exporter 版本 (默认: 3.1.8-3.1.5-ubuntu20.04)

示例:
  $0 install
  $0 verify
  MONITORING_NAMESPACE=monitoring $0 install

EOF
}

# 主函数
main() {
    local action=${1:-help}
    
    log_info "开始执行 GPU 监控系统设置: $action"
    
    case $action in
        install)
            check_dependencies
            create_namespace
            add_helm_repos
            deploy_prometheus
            deploy_dcgm_exporter
            configure_gpu_alerts
            configure_service_monitors
            import_grafana_dashboards
            verify_deployment
            ;;
        uninstall)
            uninstall_monitoring
            ;;
        verify)
            verify_deployment
            ;;
        prometheus)
            check_dependencies
            create_namespace
            add_helm_repos
            deploy_prometheus
            ;;
        dcgm)
            check_dependencies
            create_namespace
            deploy_dcgm_exporter
            ;;
        alerts)
            configure_gpu_alerts
            ;;
        dashboards)
            import_grafana_dashboards
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知选项: $action"
            show_help
            exit 1
            ;;
    esac
    
    log_success "GPU 监控系统设置完成"
}

# 信号处理
trap cleanup EXIT

# 执行主函数
main "$@"