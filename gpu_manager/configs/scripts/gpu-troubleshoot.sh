#!/bin/bash
# GPU故障排查脚本
# 提供全面的GPU问题诊断和修复建议

set -e

# 配置变量
LOG_FILE="/var/log/gpu-troubleshoot.log"
REPORT_FILE="/tmp/gpu-diagnostic-report-$(date +%Y%m%d-%H%M%S).txt"
VERBOSE=false
AUTO_FIX=false

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
    echo "[$timestamp] [$level] $message" >> "$REPORT_FILE"
}

log_info() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[INFO]${NC} $*"
    fi
    log "INFO" "$*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
    log "WARN" "$*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
    log "ERROR" "$*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
    log "SUCCESS" "$*"
}

# 初始化报告文件
init_report() {
    cat > "$REPORT_FILE" << EOF
================================================================================
GPU诊断报告
生成时间: $(date)
主机名: $(hostname)
操作系统: $(uname -a)
================================================================================

EOF
}

# 检查NVIDIA驱动
check_nvidia_driver() {
    log_info "检查NVIDIA驱动..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi命令未找到，NVIDIA驱动可能未安装"
        echo "修复建议:"
        echo "  1. 安装NVIDIA驱动: sudo apt install nvidia-driver-xxx"
        echo "  2. 或使用官方安装程序: https://www.nvidia.com/drivers"
        return 1
    fi
    
    local driver_version
    if driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1); then
        log_success "NVIDIA驱动已安装，版本: $driver_version"
        
        # 检查驱动版本是否过旧
        local major_version=$(echo "$driver_version" | cut -d. -f1)
        if [[ "$major_version" -lt 470 ]]; then
            log_warn "驱动版本较旧 ($driver_version)，建议升级到470+版本"
        fi
    else
        log_error "无法获取驱动版本信息"
        return 1
    fi
    
    return 0
}

# 检查GPU硬件
check_gpu_hardware() {
    log_info "检查GPU硬件..."
    
    local gpu_count
    if gpu_count=$(nvidia-smi --list-gpus | wc -l); then
        if [[ "$gpu_count" -eq 0 ]]; then
            log_error "未检测到GPU设备"
            echo "可能原因:"
            echo "  1. GPU未正确安装"
            echo "  2. PCIe连接问题"
            echo "  3. 电源供应不足"
            echo "  4. BIOS设置问题"
            return 1
        else
            log_success "检测到 $gpu_count 个GPU设备"
        fi
    else
        log_error "无法查询GPU设备列表"
        return 1
    fi
    
    # 检查每个GPU的状态
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.max_limit \
        --format=csv,noheader | while IFS=',' read -r index name pci_id memory power; do
        log_info "GPU $index: $name (PCI: $pci_id, Memory: ${memory}MB, Max Power: ${power}W)"
    done
    
    return 0
}

# 检查GPU温度和功耗
check_gpu_thermal() {
    log_info "检查GPU温度和功耗..."
    
    local issues_found=false
    
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,fan.speed \
        --format=csv,noheader,nounits | while IFS=',' read -r index temp power fan; do
        
        # 清理空格
        temp=$(echo "$temp" | tr -d ' ')
        power=$(echo "$power" | tr -d ' ')
        fan=$(echo "$fan" | tr -d ' ')
        
        log_info "GPU $index: 温度=${temp}°C, 功耗=${power}W, 风扇=${fan}%"
        
        # 检查温度
        if [[ "$temp" != "[Not Supported]" ]]; then
            if [[ "$temp" -gt 85 ]]; then
                log_error "GPU $index 温度过高: ${temp}°C"
                echo "修复建议:"
                echo "  1. 检查机箱散热"
                echo "  2. 清理GPU风扇和散热器"
                echo "  3. 检查热硅脂"
                echo "  4. 降低GPU负载"
                issues_found=true
            elif [[ "$temp" -gt 75 ]]; then
                log_warn "GPU $index 温度较高: ${temp}°C"
            fi
        fi
        
        # 检查风扇
        if [[ "$fan" != "[Not Supported]" && "$fan" -eq 0 && "$temp" -gt 60 ]]; then
            log_error "GPU $index 风扇未运行，但温度为 ${temp}°C"
            echo "修复建议:"
            echo "  1. 检查风扇连接"
            echo "  2. 检查风扇控制设置"
            echo "  3. 可能需要更换风扇"
            issues_found=true
        fi
    done
    
    if [[ "$issues_found" == "false" ]]; then
        log_success "GPU温度和功耗正常"
    fi
    
    return 0
}

# 检查GPU内存
check_gpu_memory() {
    log_info "检查GPU内存..."
    
    local issues_found=false
    
    nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free \
        --format=csv,noheader,nounits | while IFS=',' read -r index used total free; do
        
        # 清理空格
        used=$(echo "$used" | tr -d ' ')
        total=$(echo "$total" | tr -d ' ')
        free=$(echo "$free" | tr -d ' ')
        
        local usage_percent=$((used * 100 / total))
        
        log_info "GPU $index: 内存使用 ${used}MB/${total}MB (${usage_percent}%)"
        
        if [[ "$usage_percent" -gt 95 ]]; then
            log_error "GPU $index 内存使用率过高: ${usage_percent}%"
            echo "修复建议:"
            echo "  1. 终止不必要的GPU进程"
            echo "  2. 减少批处理大小"
            echo "  3. 使用内存优化技术"
            echo "  4. 检查内存泄漏"
            issues_found=true
        elif [[ "$usage_percent" -gt 85 ]]; then
            log_warn "GPU $index 内存使用率较高: ${usage_percent}%"
        fi
    done
    
    # 检查ECC错误
    if nvidia-smi --query-gpu=ecc.errors.corrected.total,ecc.errors.uncorrected.total \
        --format=csv,noheader,nounits 2>/dev/null | grep -v "\[Not Supported\]" > /dev/null; then
        
        nvidia-smi --query-gpu=index,ecc.errors.corrected.total,ecc.errors.uncorrected.total \
            --format=csv,noheader,nounits | while IFS=',' read -r index corrected uncorrected; do
            
            corrected=$(echo "$corrected" | tr -d ' ')
            uncorrected=$(echo "$uncorrected" | tr -d ' ')
            
            if [[ "$corrected" != "[Not Supported]" && "$corrected" -gt 0 ]]; then
                log_warn "GPU $index 检测到 $corrected 个可纠正ECC错误"
            fi
            
            if [[ "$uncorrected" != "[Not Supported]" && "$uncorrected" -gt 0 ]]; then
                log_error "GPU $index 检测到 $uncorrected 个不可纠正ECC错误"
                echo "修复建议:"
                echo "  1. 这可能表示硬件问题"
                echo "  2. 考虑更换GPU"
                echo "  3. 联系硬件供应商"
                issues_found=true
            fi
        done
    fi
    
    if [[ "$issues_found" == "false" ]]; then
        log_success "GPU内存状态正常"
    fi
    
    return 0
}

# 检查GPU进程
check_gpu_processes() {
    log_info "检查GPU进程..."
    
    local processes
    if processes=$(nvidia-smi pmon -c 1 2>/dev/null); then
        if echo "$processes" | grep -q "No running processes found"; then
            log_info "当前没有GPU进程运行"
        else
            log_info "当前GPU进程:"
            echo "$processes" | tail -n +3
        fi
    else
        log_warn "无法获取GPU进程信息"
    fi
    
    # 检查僵尸进程
    local zombie_processes
    if zombie_processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null); then
        
        if [[ -n "$zombie_processes" ]]; then
            echo "$zombie_processes" | while IFS=',' read -r pid name memory; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    log_warn "发现僵尸GPU进程: PID=$pid, Name=$name"
                    if [[ "$AUTO_FIX" == "true" ]]; then
                        log_info "尝试清理僵尸进程..."
                        nvidia-smi --gpu-reset -i all 2>/dev/null || true
                    fi
                fi
            done
        fi
    fi
    
    return 0
}

# 检查CUDA环境
check_cuda_environment() {
    log_info "检查CUDA环境..."
    
    # 检查CUDA工具包
    if command -v nvcc &> /dev/null; then
        local cuda_version
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1)
        log_success "CUDA工具包已安装，版本: $cuda_version"
    else
        log_warn "CUDA工具包未安装或nvcc不在PATH中"
        echo "修复建议:"
        echo "  1. 安装CUDA工具包"
        echo "  2. 设置正确的PATH环境变量"
    fi
    
    # 检查CUDA库
    if [[ -d "/usr/local/cuda" ]]; then
        log_info "CUDA安装目录: /usr/local/cuda"
        if [[ -f "/usr/local/cuda/lib64/libcudart.so" ]]; then
            log_success "CUDA运行时库存在"
        else
            log_warn "CUDA运行时库未找到"
        fi
    else
        log_warn "CUDA安装目录未找到"
    fi
    
    # 检查环境变量
    if [[ -n "$CUDA_HOME" ]]; then
        log_info "CUDA_HOME: $CUDA_HOME"
    else
        log_warn "CUDA_HOME环境变量未设置"
    fi
    
    if echo "$LD_LIBRARY_PATH" | grep -q cuda; then
        log_info "LD_LIBRARY_PATH包含CUDA路径"
    else
        log_warn "LD_LIBRARY_PATH可能未包含CUDA库路径"
    fi
    
    return 0
}

# 检查容器运行时
check_container_runtime() {
    log_info "检查容器GPU支持..."
    
    # 检查Docker
    if command -v docker &> /dev/null; then
        log_info "Docker已安装"
        
        # 检查nvidia-docker
        if command -v nvidia-docker &> /dev/null; then
            log_success "nvidia-docker已安装"
        elif docker info 2>/dev/null | grep -q "nvidia"; then
            log_success "Docker支持NVIDIA GPU"
        else
            log_warn "Docker可能不支持NVIDIA GPU"
            echo "修复建议:"
            echo "  1. 安装nvidia-container-toolkit"
            echo "  2. 重启Docker服务"
        fi
    fi
    
    # 检查containerd
    if command -v containerd &> /dev/null; then
        log_info "containerd已安装"
        
        if [[ -f "/etc/containerd/config.toml" ]]; then
            if grep -q "nvidia" "/etc/containerd/config.toml"; then
                log_success "containerd配置支持NVIDIA GPU"
            else
                log_warn "containerd可能未配置NVIDIA GPU支持"
            fi
        fi
    fi
    
    return 0
}

# 检查Kubernetes GPU支持
check_kubernetes_gpu() {
    log_info "检查Kubernetes GPU支持..."
    
    if command -v kubectl &> /dev/null; then
        log_info "kubectl已安装"
        
        # 检查节点GPU资源
        if kubectl get nodes -o json 2>/dev/null | jq -r '.items[].status.allocatable."nvidia.com/gpu"' | grep -v null > /dev/null; then
            log_success "Kubernetes节点支持GPU资源"
            
            # 显示GPU资源分配
            kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu 2>/dev/null || true
        else
            log_warn "Kubernetes节点未检测到GPU资源"
            echo "修复建议:"
            echo "  1. 安装NVIDIA Device Plugin"
            echo "  2. 检查节点标签和污点"
            echo "  3. 验证GPU Operator安装"
        fi
        
        # 检查GPU相关Pod
        if kubectl get pods -A -o json 2>/dev/null | jq -r '.items[] | select(.spec.containers[]?.resources.limits."nvidia.com/gpu"?) | .metadata.name' | head -5; then
            log_info "发现使用GPU的Pod"
        fi
    else
        log_info "kubectl未安装，跳过Kubernetes检查"
    fi
    
    return 0
}

# 运行GPU基准测试
run_gpu_benchmark() {
    log_info "运行GPU基准测试..."
    
    # 简单的GPU计算测试
    if command -v nvidia-smi &> /dev/null; then
        log_info "运行nvidia-smi测试..."
        if nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -q "Performance State"; then
            log_success "GPU性能状态查询正常"
        else
            log_warn "无法查询GPU性能状态"
        fi
    fi
    
    # 如果有CUDA样例，运行简单测试
    if [[ -f "/usr/local/cuda/samples/bin/x86_64/linux/release/deviceQuery" ]]; then
        log_info "运行CUDA设备查询测试..."
        if /usr/local/cuda/samples/bin/x86_64/linux/release/deviceQuery | grep -q "Result = PASS"; then
            log_success "CUDA设备查询测试通过"
        else
            log_error "CUDA设备查询测试失败"
        fi
    fi
    
    return 0
}

# 自动修复常见问题
auto_fix_issues() {
    if [[ "$AUTO_FIX" != "true" ]]; then
        return 0
    fi
    
    log_info "尝试自动修复常见问题..."
    
    # 重置GPU状态
    log_info "重置GPU状态..."
    nvidia-smi --gpu-reset -i all 2>/dev/null || log_warn "GPU重置失败"
    
    # 清理GPU内存
    log_info "清理GPU内存..."
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || log_warn "清理系统缓存失败"
    
    # 重启相关服务
    if systemctl is-active --quiet docker; then
        log_info "重启Docker服务..."
        systemctl restart docker || log_warn "Docker重启失败"
    fi
    
    log_success "自动修复完成"
}

# 生成修复建议
generate_recommendations() {
    log_info "生成修复建议..."
    
    cat >> "$REPORT_FILE" << EOF

================================================================================
修复建议总结
================================================================================

常见问题修复步骤:

1. 驱动问题:
   - 更新NVIDIA驱动到最新版本
   - 确保驱动与CUDA版本兼容
   - 重启系统使驱动生效

2. 硬件问题:
   - 检查GPU电源连接
   - 验证PCIe插槽连接
   - 检查系统电源供应是否足够

3. 温度问题:
   - 清理GPU散热器和风扇
   - 检查机箱通风
   - 考虑降低GPU频率

4. 内存问题:
   - 终止不必要的GPU进程
   - 优化应用程序内存使用
   - 考虑增加系统内存

5. 容器问题:
   - 安装nvidia-container-toolkit
   - 配置容器运行时支持GPU
   - 重启容器服务

6. Kubernetes问题:
   - 部署NVIDIA Device Plugin
   - 配置节点标签和污点
   - 验证GPU资源分配

EOF
}

# 显示帮助信息
show_help() {
    cat << EOF
GPU故障排查脚本

用法: $0 [选项]

选项:
  -h, --help          显示此帮助信息
  -v, --verbose       详细输出
  -a, --auto-fix      自动修复常见问题
  -f, --full          执行完整诊断
  --driver            仅检查驱动
  --hardware          仅检查硬件
  --thermal           仅检查温度
  --memory            仅检查内存
  --processes         仅检查进程
  --cuda              仅检查CUDA环境
  --container         仅检查容器支持
  --kubernetes        仅检查Kubernetes支持
  --benchmark         运行基准测试
  --report FILE       指定报告文件路径

示例:
  $0 --full --verbose              # 完整诊断，详细输出
  $0 --driver --cuda               # 仅检查驱动和CUDA
  $0 --auto-fix --thermal          # 检查温度并自动修复
EOF
}

# 主函数
main() {
    # 初始化
    init_report
    
    local run_all=true
    local checks=()
    
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
            -a|--auto-fix)
                AUTO_FIX=true
                shift
                ;;
            -f|--full)
                run_all=true
                shift
                ;;
            --driver)
                checks+=("driver")
                run_all=false
                shift
                ;;
            --hardware)
                checks+=("hardware")
                run_all=false
                shift
                ;;
            --thermal)
                checks+=("thermal")
                run_all=false
                shift
                ;;
            --memory)
                checks+=("memory")
                run_all=false
                shift
                ;;
            --processes)
                checks+=("processes")
                run_all=false
                shift
                ;;
            --cuda)
                checks+=("cuda")
                run_all=false
                shift
                ;;
            --container)
                checks+=("container")
                run_all=false
                shift
                ;;
            --kubernetes)
                checks+=("kubernetes")
                run_all=false
                shift
                ;;
            --benchmark)
                checks+=("benchmark")
                run_all=false
                shift
                ;;
            --report)
                REPORT_FILE="$2"
                shift 2
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    log_info "开始GPU故障排查..."
    log_info "报告文件: $REPORT_FILE"
    
    # 执行检查
    if [[ "$run_all" == "true" ]]; then
        check_nvidia_driver
        check_gpu_hardware
        check_gpu_thermal
        check_gpu_memory
        check_gpu_processes
        check_cuda_environment
        check_container_runtime
        check_kubernetes_gpu
        run_gpu_benchmark
    else
        for check in "${checks[@]}"; do
            case $check in
                driver) check_nvidia_driver ;;
                hardware) check_gpu_hardware ;;
                thermal) check_gpu_thermal ;;
                memory) check_gpu_memory ;;
                processes) check_gpu_processes ;;
                cuda) check_cuda_environment ;;
                container) check_container_runtime ;;
                kubernetes) check_kubernetes_gpu ;;
                benchmark) run_gpu_benchmark ;;
            esac
        done
    fi
    
    # 自动修复
    auto_fix_issues
    
    # 生成建议
    generate_recommendations
    
    log_success "GPU故障排查完成"
    log_info "详细报告已保存到: $REPORT_FILE"
    
    echo ""
    echo "=== 诊断摘要 ==="
    echo "报告文件: $REPORT_FILE"
    echo "日志文件: $LOG_FILE"
    echo ""
    echo "如需更多帮助，请查看详细报告或联系技术支持。"
}

# 执行主函数
main "$@"