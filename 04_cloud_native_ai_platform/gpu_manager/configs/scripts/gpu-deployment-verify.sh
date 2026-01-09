#!/bin/bash

# GPU 部署验证脚本
# 用于验证 GPU 环境部署的完整性和功能性

set -euo pipefail

# 配置变量
VERIFY_LOG="/tmp/gpu-deployment-verify-$(date +%Y%m%d-%H%M%S).log"
REPORT_FILE="/tmp/gpu-deployment-report-$(date +%Y%m%d-%H%M%S).md"
TEST_NAMESPACE="gpu-verify-test"
TEST_TIMEOUT=300
CLEANUP_ON_EXIT=true

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 测试结果统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$VERIFY_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$VERIFY_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$VERIFY_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$VERIFY_LOG"
}

log_test() {
    echo -e "${PURPLE}[TEST]${NC} $*" | tee -a "$VERIFY_LOG"
}

log_result() {
    local status=$1
    shift
    case $status in
        PASS)
            echo -e "${GREEN}[PASS]${NC} $*" | tee -a "$VERIFY_LOG"
            ((PASSED_TESTS++))
            ;;
        FAIL)
            echo -e "${RED}[FAIL]${NC} $*" | tee -a "$VERIFY_LOG"
            ((FAILED_TESTS++))
            ;;
        SKIP)
            echo -e "${CYAN}[SKIP]${NC} $*" | tee -a "$VERIFY_LOG"
            ((SKIPPED_TESTS++))
            ;;
    esac
    ((TOTAL_TESTS++))
}

# 执行测试函数
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_result="${3:-0}"
    
    log_test "运行测试: $test_name"
    
    if eval "$test_command" >> "$VERIFY_LOG" 2>&1; then
        if [ "$expected_result" = "0" ]; then
            log_result PASS "$test_name"
            return 0
        else
            log_result FAIL "$test_name - 期望失败但成功了"
            return 1
        fi
    else
        if [ "$expected_result" = "1" ]; then
            log_result PASS "$test_name - 期望失败"
            return 0
        else
            log_result FAIL "$test_name"
            return 1
        fi
    fi
}

# 检查系统依赖
check_system_dependencies() {
    log_info "=== 检查系统依赖 ==="
    
    # 检查必要命令
    local required_commands=("nvidia-smi" "docker" "kubectl")
    
    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            log_result PASS "命令 '$cmd' 可用"
        else
            log_result FAIL "命令 '$cmd' 不可用"
        fi
    done
}

# 验证 NVIDIA 驱动
verify_nvidia_driver() {
    log_info "=== 验证 NVIDIA 驱动 ==="
    
    # 检查驱动版本
    run_test "NVIDIA 驱动版本检查" "nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1"
    
    # 检查 GPU 设备
    run_test "GPU 设备检测" "nvidia-smi -L"
    
    # 检查 GPU 状态
    run_test "GPU 状态查询" "nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv"
    
    # 检查 CUDA 版本
    if command -v nvcc &> /dev/null; then
        run_test "CUDA 编译器版本" "nvcc --version"
    else
        log_result SKIP "CUDA 编译器未安装"
    fi
}

# 验证 Docker GPU 支持
verify_docker_gpu() {
    log_info "=== 验证 Docker GPU 支持 ==="
    
    # 检查 Docker 版本
    run_test "Docker 版本检查" "docker --version"
    
    # 检查 NVIDIA Container Runtime
    if docker info 2>/dev/null | grep -q "nvidia"; then
        log_result PASS "NVIDIA Container Runtime 已配置"
    else
        log_result FAIL "NVIDIA Container Runtime 未配置"
    fi
    
    # 测试 GPU 容器
    log_test "运行 GPU 测试容器"
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi >> "$VERIFY_LOG" 2>&1; then
        log_result PASS "GPU 容器测试"
    else
        log_result FAIL "GPU 容器测试"
    fi
}

# 验证 Kubernetes GPU 支持
verify_kubernetes_gpu() {
    log_info "=== 验证 Kubernetes GPU 支持 ==="
    
    # 检查 kubectl 连接
    run_test "Kubernetes 集群连接" "kubectl cluster-info --request-timeout=10s"
    
    # 检查 GPU 节点
    log_test "检查 GPU 节点"
    local gpu_nodes
    gpu_nodes=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.capacity."nvidia.com/gpu" != null) | .metadata.name' 2>/dev/null || echo "")
    
    if [ -n "$gpu_nodes" ]; then
        log_result PASS "发现 GPU 节点: $(echo "$gpu_nodes" | tr '\n' ' ')"
        
        # 检查每个 GPU 节点的资源
        while IFS= read -r node; do
            if [ -n "$node" ]; then
                log_test "检查节点 $node 的 GPU 资源"
                kubectl describe node "$node" | grep -A 5 "nvidia.com/gpu" >> "$VERIFY_LOG" 2>&1
                log_result PASS "节点 $node GPU 资源检查"
            fi
        done <<< "$gpu_nodes"
    else
        log_result FAIL "未发现 GPU 节点"
    fi
    
    # 检查 GPU Operator
    if kubectl get pods -n gpu-operator &> /dev/null; then
        log_test "检查 GPU Operator 状态"
        local operator_ready
        operator_ready=$(kubectl get pods -n gpu-operator --no-headers | awk '{print $3}' | grep -v "Running\|Completed" | wc -l)
        
        if [ "$operator_ready" -eq 0 ]; then
            log_result PASS "GPU Operator 运行正常"
        else
            log_result FAIL "GPU Operator 存在异常 Pod"
        fi
    else
        log_result SKIP "GPU Operator 未安装"
    fi
    
    # 检查 HAMi
    if kubectl get pods -n hami-system &> /dev/null; then
        log_test "检查 HAMi 状态"
        local hami_ready
        hami_ready=$(kubectl get pods -n hami-system --no-headers | awk '{print $3}' | grep -v "Running\|Completed" | wc -l)
        
        if [ "$hami_ready" -eq 0 ]; then
            log_result PASS "HAMi 运行正常"
        else
            log_result FAIL "HAMi 存在异常 Pod"
        fi
    else
        log_result SKIP "HAMi 未安装"
    fi
}

# 运行 GPU 工作负载测试
run_gpu_workload_test() {
    log_info "=== 运行 GPU 工作负载测试 ==="
    
    # 创建测试命名空间
    kubectl create namespace "$TEST_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - >> "$VERIFY_LOG" 2>&1
    
    # 创建简单的 GPU 测试 Pod
    cat > /tmp/gpu-test-pod.yaml << EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test-pod
  namespace: $TEST_NAMESPACE
spec:
  restartPolicy: Never
  containers:
  - name: gpu-test
    image: nvidia/cuda:11.8-base-ubuntu20.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
EOF
    
    # 部署测试 Pod
    log_test "部署 GPU 测试 Pod"
    if kubectl apply -f /tmp/gpu-test-pod.yaml >> "$VERIFY_LOG" 2>&1; then
        log_result PASS "GPU 测试 Pod 部署"
        
        # 等待 Pod 完成
        log_test "等待 GPU 测试 Pod 完成"
        if kubectl wait --for=condition=Ready pod/gpu-test-pod -n "$TEST_NAMESPACE" --timeout="${TEST_TIMEOUT}s" >> "$VERIFY_LOG" 2>&1; then
            log_result PASS "GPU 测试 Pod 就绪"
            
            # 检查 Pod 日志
            log_test "检查 GPU 测试 Pod 日志"
            kubectl logs gpu-test-pod -n "$TEST_NAMESPACE" >> "$VERIFY_LOG" 2>&1
            log_result PASS "GPU 测试 Pod 日志获取"
        else
            log_result FAIL "GPU 测试 Pod 超时"
        fi
    else
        log_result FAIL "GPU 测试 Pod 部署失败"
    fi
    
    # 创建 GPU 计算测试
    cat > /tmp/gpu-compute-test.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-compute-test
  namespace: $TEST_NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: gpu-compute
        image: nvidia/cuda:11.8-devel-ubuntu20.04
        command: ["/bin/bash"]
        args:
        - "-c"
        - |
          echo "编译 CUDA 测试程序..."
          cat > /tmp/gpu_test.cu << 'CUDA_EOF'
          #include <stdio.h>
          #include <cuda_runtime.h>
          
          __global__ void vectorAdd(float *a, float *b, float *c, int n) {
              int i = blockDim.x * blockIdx.x + threadIdx.x;
              if (i < n) {
                  c[i] = a[i] + b[i];
              }
          }
          
          int main() {
              int n = 1000000;
              size_t size = n * sizeof(float);
              
              float *h_a = (float*)malloc(size);
              float *h_b = (float*)malloc(size);
              float *h_c = (float*)malloc(size);
              
              for (int i = 0; i < n; i++) {
                  h_a[i] = i;
                  h_b[i] = i * 2;
              }
              
              float *d_a, *d_b, *d_c;
              cudaMalloc(&d_a, size);
              cudaMalloc(&d_b, size);
              cudaMalloc(&d_c, size);
              
              cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
              cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
              
              int threadsPerBlock = 256;
              int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
              
              vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
              
              cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
              
              printf("GPU 计算测试完成\n");
              printf("验证结果: h_c[0] = %.2f (期望: 0.00)\n", h_c[0]);
              printf("验证结果: h_c[999999] = %.2f (期望: 2999997.00)\n", h_c[999999]);
              
              free(h_a); free(h_b); free(h_c);
              cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
              
              return 0;
          }
          CUDA_EOF
          
          nvcc -o /tmp/gpu_test /tmp/gpu_test.cu
          /tmp/gpu_test
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
EOF
    
    # 部署计算测试 Job
    log_test "部署 GPU 计算测试 Job"
    if kubectl apply -f /tmp/gpu-compute-test.yaml >> "$VERIFY_LOG" 2>&1; then
        log_result PASS "GPU 计算测试 Job 部署"
        
        # 等待 Job 完成
        log_test "等待 GPU 计算测试完成"
        if kubectl wait --for=condition=complete job/gpu-compute-test -n "$TEST_NAMESPACE" --timeout="${TEST_TIMEOUT}s" >> "$VERIFY_LOG" 2>&1; then
            log_result PASS "GPU 计算测试完成"
            
            # 获取 Job 日志
            local job_pod
            job_pod=$(kubectl get pods -n "$TEST_NAMESPACE" -l job-name=gpu-compute-test -o jsonpath='{.items[0].metadata.name}')
            if [ -n "$job_pod" ]; then
                kubectl logs "$job_pod" -n "$TEST_NAMESPACE" >> "$VERIFY_LOG" 2>&1
                log_result PASS "GPU 计算测试日志获取"
            fi
        else
            log_result FAIL "GPU 计算测试超时"
        fi
    else
        log_result FAIL "GPU 计算测试 Job 部署失败"
    fi
}

# 验证监控系统
verify_monitoring() {
    log_info "=== 验证监控系统 ==="
    
    # 检查 Prometheus
    if kubectl get pods -n monitoring -l app=prometheus &> /dev/null; then
        log_test "检查 Prometheus 状态"
        local prometheus_ready
        prometheus_ready=$(kubectl get pods -n monitoring -l app=prometheus --no-headers | awk '{print $3}' | grep -v "Running" | wc -l)
        
        if [ "$prometheus_ready" -eq 0 ]; then
            log_result PASS "Prometheus 运行正常"
        else
            log_result FAIL "Prometheus 存在异常"
        fi
    else
        log_result SKIP "Prometheus 未安装"
    fi
    
    # 检查 DCGM Exporter
    if kubectl get pods -A -l app=dcgm-exporter &> /dev/null; then
        log_test "检查 DCGM Exporter 状态"
        local dcgm_ready
        dcgm_ready=$(kubectl get pods -A -l app=dcgm-exporter --no-headers | awk '{print $4}' | grep -v "Running" | wc -l)
        
        if [ "$dcgm_ready" -eq 0 ]; then
            log_result PASS "DCGM Exporter 运行正常"
        else
            log_result FAIL "DCGM Exporter 存在异常"
        fi
    else
        log_result SKIP "DCGM Exporter 未安装"
    fi
    
    # 检查 Grafana
    if kubectl get pods -n monitoring -l app=grafana &> /dev/null; then
        log_test "检查 Grafana 状态"
        local grafana_ready
        grafana_ready=$(kubectl get pods -n monitoring -l app=grafana --no-headers | awk '{print $3}' | grep -v "Running" | wc -l)
        
        if [ "$grafana_ready" -eq 0 ]; then
            log_result PASS "Grafana 运行正常"
        else
            log_result FAIL "Grafana 存在异常"
        fi
    else
        log_result SKIP "Grafana 未安装"
    fi
}

# 性能基准测试
run_performance_benchmark() {
    log_info "=== 运行性能基准测试 ==="
    
    # GPU 内存带宽测试
    cat > /tmp/gpu-bandwidth-test.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-bandwidth-test
  namespace: $TEST_NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: bandwidth-test
        image: nvidia/cuda:11.8-devel-ubuntu20.04
        command: ["/bin/bash"]
        args:
        - "-c"
        - |
          echo "运行 GPU 内存带宽测试..."
          /usr/local/cuda/extras/demo_suite/bandwidthTest
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
EOF
    
    log_test "运行 GPU 内存带宽测试"
    if kubectl apply -f /tmp/gpu-bandwidth-test.yaml >> "$VERIFY_LOG" 2>&1; then
        if kubectl wait --for=condition=complete job/gpu-bandwidth-test -n "$TEST_NAMESPACE" --timeout="${TEST_TIMEOUT}s" >> "$VERIFY_LOG" 2>&1; then
            local bandwidth_pod
            bandwidth_pod=$(kubectl get pods -n "$TEST_NAMESPACE" -l job-name=gpu-bandwidth-test -o jsonpath='{.items[0].metadata.name}')
            if [ -n "$bandwidth_pod" ]; then
                kubectl logs "$bandwidth_pod" -n "$TEST_NAMESPACE" >> "$VERIFY_LOG" 2>&1
                log_result PASS "GPU 内存带宽测试"
            fi
        else
            log_result FAIL "GPU 内存带宽测试超时"
        fi
    else
        log_result FAIL "GPU 内存带宽测试部署失败"
    fi
}

# 生成验证报告
generate_report() {
    log_info "=== 生成验证报告 ==="
    
    cat > "$REPORT_FILE" << EOF
# GPU 部署验证报告

**生成时间**: $(date)
**验证环境**: $(hostname)
**Kubernetes 版本**: $(kubectl version --short --client 2>/dev/null | grep Client || echo "未知")

## 测试结果摘要

- **总测试数**: $TOTAL_TESTS
- **通过测试**: $PASSED_TESTS
- **失败测试**: $FAILED_TESTS
- **跳过测试**: $SKIPPED_TESTS
- **成功率**: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%

## 系统信息

### NVIDIA 驱动信息
\`\`\`
$(nvidia-smi 2>/dev/null || echo "NVIDIA 驱动未安装或不可用")
\`\`\`

### Docker 信息
\`\`\`
$(docker version 2>/dev/null || echo "Docker 未安装或不可用")
\`\`\`

### Kubernetes 集群信息
\`\`\`
$(kubectl cluster-info 2>/dev/null || echo "Kubernetes 集群不可用")
\`\`\`

### GPU 节点信息
\`\`\`
$(kubectl get nodes -o wide 2>/dev/null | grep -E "NAME|gpu" || echo "无 GPU 节点信息")
\`\`\`

## 详细日志

详细的验证日志请查看: \`$VERIFY_LOG\`

## 建议

EOF
    
    # 根据测试结果添加建议
    if [ $FAILED_TESTS -eq 0 ]; then
        cat >> "$REPORT_FILE" << EOF
✅ **所有测试通过**: GPU 环境部署验证成功，系统运行正常。

### 后续步骤
1. 可以开始部署生产工作负载
2. 建议定期运行此验证脚本确保系统稳定性
3. 监控 GPU 使用率和性能指标
EOF
    else
        cat >> "$REPORT_FILE" << EOF
❌ **存在失败测试**: 发现 $FAILED_TESTS 个问题需要解决。

### 问题排查建议
1. 检查详细日志文件: \`$VERIFY_LOG\`
2. 验证 NVIDIA 驱动安装
3. 检查 Docker GPU 运行时配置
4. 确认 Kubernetes GPU 插件状态
5. 检查网络和存储配置

### 常见问题解决
- **NVIDIA 驱动问题**: 重新安装或更新驱动
- **Docker GPU 问题**: 检查 nvidia-container-runtime 配置
- **Kubernetes 问题**: 验证 GPU Operator 或设备插件状态
EOF
    fi
    
    log_success "验证报告已生成: $REPORT_FILE"
}

# 清理测试资源
cleanup_test_resources() {
    if [ "$CLEANUP_ON_EXIT" = "true" ]; then
        log_info "清理测试资源..."
        
        # 删除测试命名空间
        kubectl delete namespace "$TEST_NAMESPACE" --ignore-not-found=true >> "$VERIFY_LOG" 2>&1
        
        # 删除临时文件
        rm -f /tmp/gpu-test-pod.yaml /tmp/gpu-compute-test.yaml /tmp/gpu-bandwidth-test.yaml
        
        log_success "测试资源清理完成"
    fi
}

# 显示帮助信息
show_help() {
    cat << EOF
GPU 部署验证脚本

用法: $0 [选项]

选项:
  --full              运行完整验证（默认）
  --basic             仅运行基础验证
  --workload          仅运行工作负载测试
  --monitoring        仅验证监控系统
  --performance       仅运行性能测试
  --no-cleanup        测试后不清理资源
  --namespace NAME    使用指定的测试命名空间
  --timeout SECONDS   设置测试超时时间
  --help              显示此帮助信息

环境变量:
  TEST_NAMESPACE      测试命名空间（默认: gpu-verify-test）
  TEST_TIMEOUT        测试超时时间（默认: 300 秒）
  CLEANUP_ON_EXIT     是否清理测试资源（默认: true）

示例:
  $0                           # 运行完整验证
  $0 --basic                   # 仅基础验证
  $0 --no-cleanup              # 保留测试资源
  $0 --namespace my-test       # 使用自定义命名空间
  TEST_TIMEOUT=600 $0          # 设置 10 分钟超时

输出文件:
  - 验证日志: $VERIFY_LOG
  - 验证报告: $REPORT_FILE

EOF
}

# 主函数
main() {
    local run_basic=false
    local run_workload=false
    local run_monitoring=false
    local run_performance=false
    local run_full=true
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --basic)
                run_basic=true
                run_full=false
                shift
                ;;
            --workload)
                run_workload=true
                run_full=false
                shift
                ;;
            --monitoring)
                run_monitoring=true
                run_full=false
                shift
                ;;
            --performance)
                run_performance=true
                run_full=false
                shift
                ;;
            --no-cleanup)
                CLEANUP_ON_EXIT=false
                shift
                ;;
            --namespace)
                TEST_NAMESPACE="$2"
                shift 2
                ;;
            --timeout)
                TEST_TIMEOUT="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    log_info "开始 GPU 部署验证"
    log_info "验证日志: $VERIFY_LOG"
    log_info "验证报告: $REPORT_FILE"
    
    # 运行验证测试
    if [ "$run_full" = "true" ] || [ "$run_basic" = "true" ]; then
        check_system_dependencies
        verify_nvidia_driver
        verify_docker_gpu
        verify_kubernetes_gpu
    fi
    
    if [ "$run_full" = "true" ] || [ "$run_workload" = "true" ]; then
        run_gpu_workload_test
    fi
    
    if [ "$run_full" = "true" ] || [ "$run_monitoring" = "true" ]; then
        verify_monitoring
    fi
    
    if [ "$run_full" = "true" ] || [ "$run_performance" = "true" ]; then
        run_performance_benchmark
    fi
    
    # 生成报告
    generate_report
    
    # 显示结果摘要
    echo
    log_info "=== 验证结果摘要 ==="
    log_info "总测试数: $TOTAL_TESTS"
    log_info "通过测试: $PASSED_TESTS"
    log_info "失败测试: $FAILED_TESTS"
    log_info "跳过测试: $SKIPPED_TESTS"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "GPU 部署验证成功！"
        exit 0
    else
        log_error "GPU 部署验证发现问题，请检查日志和报告"
        exit 1
    fi
}

# 信号处理
trap cleanup_test_resources EXIT

# 执行主函数
main "$@"