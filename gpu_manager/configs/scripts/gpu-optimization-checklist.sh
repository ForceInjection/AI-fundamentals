#!/bin/bash
# GPU优化检查清单脚本
# 提供全面的GPU性能优化检查和建议

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

# 配置变量
OUTPUT_DIR="gpu_optimization_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_DIR/optimization_check.log"
REPORT_FILE="$OUTPUT_DIR/optimization_report.md"
CHECK_TYPE="all"
FIX_ISSUES=false
VERBOSE=false

# 检查结果统计
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0
OPTIMIZATION_SUGGESTIONS=()

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
    ((WARNING_CHECKS++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    ((FAILED_CHECKS++))
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a "$LOG_FILE"
    ((PASSED_CHECKS++))
}

log_check() {
    echo -e "${BLUE}[CHECK]${NC} $1" | tee -a "$LOG_FILE"
    ((TOTAL_CHECKS++))
}

log_section() {
    echo -e "${PURPLE}\n=== $1 ===${NC}" | tee -a "$LOG_FILE"
}

log_suggestion() {
    echo -e "${CYAN}[建议]${NC} $1" | tee -a "$LOG_FILE"
    OPTIMIZATION_SUGGESTIONS+=("$1")
}

# 创建输出目录
setup_output_directory() {
    mkdir -p "$OUTPUT_DIR"
    log_info "优化检查结果将保存到: $OUTPUT_DIR"
}

# 检查GPU驱动
check_gpu_driver() {
    log_section "GPU驱动检查"
    
    log_check "检查NVIDIA驱动是否安装"
    if command -v nvidia-smi &> /dev/null; then
        local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        log_success "NVIDIA驱动已安装，版本: $driver_version"
        
        # 检查驱动版本是否较新
        local major_version=$(echo "$driver_version" | cut -d. -f1)
        if [[ $major_version -ge 470 ]]; then
            log_success "驱动版本较新，支持最新功能"
        elif [[ $major_version -ge 450 ]]; then
            log_warn "驱动版本较旧，建议升级到最新版本"
            log_suggestion "升级NVIDIA驱动到最新版本以获得更好的性能和功能支持"
        else
            log_error "驱动版本过旧，可能影响性能和兼容性"
            log_suggestion "立即升级NVIDIA驱动，当前版本过旧"
        fi
    else
        log_error "NVIDIA驱动未安装或nvidia-smi不可用"
        log_suggestion "安装NVIDIA驱动程序"
        return 1
    fi
    
    # 检查驱动持久化模式
    log_check "检查GPU持久化模式"
    local persistence_mode=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader,nounits | head -1)
    if [[ "$persistence_mode" == "Enabled" ]]; then
        log_success "GPU持久化模式已启用"
    else
        log_warn "GPU持久化模式未启用"
        log_suggestion "启用GPU持久化模式: sudo nvidia-smi -pm 1"
        
        if [[ "$FIX_ISSUES" == "true" ]]; then
            log_info "尝试启用GPU持久化模式..."
            if sudo nvidia-smi -pm 1 &>/dev/null; then
                log_success "GPU持久化模式已启用"
            else
                log_error "无法启用GPU持久化模式"
            fi
        fi
    fi
}

# 检查CUDA环境
check_cuda_environment() {
    log_section "CUDA环境检查"
    
    log_check "检查CUDA是否安装"
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_success "CUDA已安装，版本: $cuda_version"
        
        # 检查CUDA版本兼容性
        local major_version=$(echo "$cuda_version" | cut -d. -f1)
        if [[ $major_version -ge 11 ]]; then
            log_success "CUDA版本较新，支持最新功能"
        elif [[ $major_version -ge 10 ]]; then
            log_warn "CUDA版本较旧，建议升级"
            log_suggestion "考虑升级到CUDA 11.x或更新版本"
        else
            log_error "CUDA版本过旧"
            log_suggestion "升级CUDA到支持的版本"
        fi
    else
        log_warn "CUDA未安装或nvcc不在PATH中"
        log_suggestion "安装CUDA Toolkit或将nvcc添加到PATH"
    fi
    
    # 检查cuDNN
    log_check "检查cuDNN库"
    if ldconfig -p | grep -q libcudnn; then
        log_success "cuDNN库已安装"
    else
        log_warn "cuDNN库未找到"
        log_suggestion "安装cuDNN库以获得更好的深度学习性能"
    fi
    
    # 检查CUDA库路径
    log_check "检查CUDA库路径"
    if [[ -n "$CUDA_HOME" ]] || [[ -n "$CUDA_PATH" ]]; then
        log_success "CUDA环境变量已设置"
    else
        log_warn "CUDA环境变量未设置"
        log_suggestion "设置CUDA_HOME和LD_LIBRARY_PATH环境变量"
    fi
}

# 检查GPU硬件配置
check_gpu_hardware() {
    log_section "GPU硬件配置检查"
    
    # GPU基本信息
    log_check "检查GPU设备信息"
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    log_info "检测到 $gpu_count 个GPU设备"
    
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap,power.limit --format=csv,noheader | while IFS=',' read -r index name memory compute_cap power_limit; do
        log_info "GPU $index: $name"
        log_info "  内存: $memory"
        log_info "  计算能力: $compute_cap"
        log_info "  功耗限制: $power_limit"
        
        # 检查计算能力
        local major_cc=$(echo "$compute_cap" | cut -d. -f1)
        if [[ $major_cc -ge 7 ]]; then
            log_success "GPU $index 计算能力较新 ($compute_cap)"
        elif [[ $major_cc -ge 6 ]]; then
            log_warn "GPU $index 计算能力较旧 ($compute_cap)"
            log_suggestion "考虑升级到更新的GPU架构"
        else
            log_error "GPU $index 计算能力过旧 ($compute_cap)"
            log_suggestion "GPU架构过旧，建议升级硬件"
        fi
        
        # 检查内存大小
        local memory_gb=$(echo "$memory" | sed 's/[^0-9]//g')
        memory_gb=$((memory_gb / 1024))
        if [[ $memory_gb -ge 16 ]]; then
            log_success "GPU $index 内存充足 (${memory_gb}GB)"
        elif [[ $memory_gb -ge 8 ]]; then
            log_warn "GPU $index 内存中等 (${memory_gb}GB)"
            log_suggestion "对于大型模型，考虑使用更大内存的GPU"
        else
            log_warn "GPU $index 内存较小 (${memory_gb}GB)"
            log_suggestion "内存较小，可能限制大型模型训练"
        fi
    done
    
    # 检查GPU拓扑
    log_check "检查GPU拓扑结构"
    if nvidia-smi topo -m &>/dev/null; then
        nvidia-smi topo -m > "$OUTPUT_DIR/gpu_topology.txt"
        log_success "GPU拓扑信息已保存"
        
        # 分析拓扑结构
        if grep -q "NV" "$OUTPUT_DIR/gpu_topology.txt"; then
            log_success "检测到NVLink连接"
        else
            log_warn "未检测到NVLink连接"
            log_suggestion "对于多GPU训练，NVLink可以显著提升性能"
        fi
    else
        log_warn "无法获取GPU拓扑信息"
    fi
}

# 检查GPU性能设置
check_gpu_performance_settings() {
    log_section "GPU性能设置检查"
    
    # 检查GPU时钟频率
    log_check "检查GPU时钟频率"
    nvidia-smi --query-gpu=index,clocks.gr,clocks.max.gr,clocks.mem,clocks.max.mem --format=csv,noheader | while IFS=',' read -r index gr_clock max_gr_clock mem_clock max_mem_clock; do
        log_info "GPU $index 当前时钟: 图形 $gr_clock, 内存 $mem_clock"
        log_info "GPU $index 最大时钟: 图形 $max_gr_clock, 内存 $max_mem_clock"
        
        # 检查是否运行在最大频率
        local gr_ratio=$(echo "scale=2; ${gr_clock%% *} * 100 / ${max_gr_clock%% *}" | bc 2>/dev/null || echo "0")
        if (( $(echo "$gr_ratio > 90" | bc -l 2>/dev/null || echo "0") )); then
            log_success "GPU $index 运行在较高频率"
        else
            log_warn "GPU $index 未运行在最大频率"
            log_suggestion "检查GPU功耗限制和温度限制"
        fi
    done
    
    # 检查功耗限制
    log_check "检查GPU功耗设置"
    nvidia-smi --query-gpu=index,power.draw,power.limit,power.max_limit --format=csv,noheader | while IFS=',' read -r index power_draw power_limit max_power_limit; do
        log_info "GPU $index 功耗: 当前 $power_draw, 限制 $power_limit, 最大 $max_power_limit"
        
        # 检查功耗限制是否合理
        local current_power=$(echo "$power_draw" | sed 's/[^0-9.]//g')
        local limit_power=$(echo "$power_limit" | sed 's/[^0-9.]//g')
        local max_power=$(echo "$max_power_limit" | sed 's/[^0-9.]//g')
        
        if (( $(echo "$limit_power < $max_power" | bc -l 2>/dev/null || echo "0") )); then
            log_warn "GPU $index 功耗限制低于最大值"
            log_suggestion "考虑提高功耗限制: sudo nvidia-smi -pl $max_power"
        else
            log_success "GPU $index 功耗限制已优化"
        fi
    done
    
    # 检查GPU模式
    log_check "检查GPU应用时钟模式"
    nvidia-smi --query-gpu=index,clocks_throttle_reasons.active --format=csv,noheader | while IFS=',' read -r index throttle_reasons; do
        if [[ "$throttle_reasons" == *"Not Active"* ]]; then
            log_success "GPU $index 无时钟限制"
        else
            log_warn "GPU $index 存在时钟限制: $throttle_reasons"
            log_suggestion "检查温度、功耗和其他限制因素"
        fi
    done
}

# 检查内存优化
check_memory_optimization() {
    log_section "内存优化检查"
    
    # 检查GPU内存使用
    log_check "检查GPU内存使用情况"
    nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader | while IFS=',' read -r index total used free; do
        local total_mb=$(echo "$total" | sed 's/[^0-9]//g')
        local used_mb=$(echo "$used" | sed 's/[^0-9]//g')
        local usage_percent=$(echo "scale=1; $used_mb * 100 / $total_mb" | bc 2>/dev/null || echo "0")
        
        log_info "GPU $index 内存使用: $used / $total ($usage_percent%)"
        
        if (( $(echo "$usage_percent > 90" | bc -l 2>/dev/null || echo "0") )); then
            log_warn "GPU $index 内存使用率过高"
            log_suggestion "考虑减少批次大小或使用梯度累积"
        elif (( $(echo "$usage_percent < 30" | bc -l 2>/dev/null || echo "0") )); then
            log_warn "GPU $index 内存利用率较低"
            log_suggestion "可以增加批次大小以提高GPU利用率"
        else
            log_success "GPU $index 内存使用合理"
        fi
    done
    
    # 检查系统内存
    log_check "检查系统内存"
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    local used_mem=$(free -g | awk '/^Mem:/{print $3}')
    local mem_usage_percent=$(echo "scale=1; $used_mem * 100 / $total_mem" | bc 2>/dev/null || echo "0")
    
    log_info "系统内存使用: ${used_mem}GB / ${total_mem}GB ($mem_usage_percent%)"
    
    if (( $(echo "$mem_usage_percent > 90" | bc -l 2>/dev/null || echo "0") )); then
        log_warn "系统内存使用率过高"
        log_suggestion "增加系统内存或优化数据加载"
    elif [[ $total_mem -lt 32 ]]; then
        log_warn "系统内存较小 (${total_mem}GB)"
        log_suggestion "对于大型模型训练，建议至少32GB系统内存"
    else
        log_success "系统内存充足"
    fi
    
    # 检查交换分区
    log_check "检查交换分区使用"
    local swap_used=$(free -g | awk '/^Swap:/{print $3}')
    if [[ $swap_used -gt 0 ]]; then
        log_warn "检测到交换分区使用 (${swap_used}GB)"
        log_suggestion "交换分区使用可能影响性能，考虑增加物理内存"
    else
        log_success "未使用交换分区"
    fi
}

# 检查容器和编排环境
check_container_environment() {
    log_section "容器环境检查"
    
    # 检查Docker
    log_check "检查Docker环境"
    if command -v docker &> /dev/null; then
        log_success "Docker已安装"
        
        # 检查NVIDIA Container Toolkit
        if docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi &>/dev/null; then
            log_success "NVIDIA Container Toolkit工作正常"
        else
            log_error "NVIDIA Container Toolkit未正确配置"
            log_suggestion "安装和配置NVIDIA Container Toolkit"
        fi
    else
        log_info "Docker未安装（跳过容器检查）"
    fi
    
    # 检查Kubernetes
    log_check "检查Kubernetes环境"
    if command -v kubectl &> /dev/null; then
        log_success "kubectl已安装"
        
        # 检查GPU资源
        if kubectl get nodes -o json 2>/dev/null | grep -q "nvidia.com/gpu"; then
            log_success "Kubernetes集群支持GPU资源"
        else
            log_warn "Kubernetes集群未检测到GPU资源"
            log_suggestion "安装GPU Operator或配置GPU设备插件"
        fi
    else
        log_info "kubectl未安装（跳过Kubernetes检查）"
    fi
}

# 检查深度学习框架
check_ml_frameworks() {
    log_section "深度学习框架检查"
    
    # 检查PyTorch
    log_check "检查PyTorch"
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        local pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        log_success "PyTorch已安装，版本: $pytorch_version"
        
        # 检查CUDA支持
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log_success "PyTorch CUDA支持正常"
            local gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
            log_info "PyTorch检测到 $gpu_count 个GPU"
        else
            log_error "PyTorch CUDA支持异常"
            log_suggestion "重新安装支持CUDA的PyTorch版本"
        fi
    else
        log_info "PyTorch未安装"
    fi
    
    # 检查TensorFlow
    log_check "检查TensorFlow"
    if python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" 2>/dev/null; then
        local tf_version=$(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
        log_success "TensorFlow已安装，版本: $tf_version"
        
        # 检查GPU支持
        if python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null | grep -q -v "0"; then
            log_success "TensorFlow GPU支持正常"
            local gpu_count=$(python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null)
            log_info "TensorFlow检测到 $gpu_count 个GPU"
        else
            log_error "TensorFlow GPU支持异常"
            log_suggestion "安装支持GPU的TensorFlow版本"
        fi
    else
        log_info "TensorFlow未安装"
    fi
}

# 检查系统优化
check_system_optimization() {
    log_section "系统优化检查"
    
    # 检查CPU性能模式
    log_check "检查CPU性能模式"
    if command -v cpupower &> /dev/null; then
        local cpu_governor=$(cpupower frequency-info -p 2>/dev/null | grep "current policy" | awk '{print $NF}' || echo "unknown")
        if [[ "$cpu_governor" == "performance" ]]; then
            log_success "CPU运行在性能模式"
        else
            log_warn "CPU未运行在性能模式 (当前: $cpu_governor)"
            log_suggestion "设置CPU为性能模式: sudo cpupower frequency-set -g performance"
        fi
    else
        log_info "cpupower工具未安装，跳过CPU检查"
    fi
    
    # 检查NUMA配置
    log_check "检查NUMA配置"
    if command -v numactl &> /dev/null; then
        local numa_nodes=$(numactl --hardware | grep "available:" | awk '{print $2}')
        log_info "检测到 $numa_nodes 个NUMA节点"
        
        if [[ $numa_nodes -gt 1 ]]; then
            log_warn "多NUMA节点系统"
            log_suggestion "考虑NUMA亲和性优化，特别是对于多GPU系统"
        else
            log_success "单NUMA节点系统"
        fi
    else
        log_info "numactl未安装，跳过NUMA检查"
    fi
    
    # 检查文件系统
    log_check "检查文件系统类型"
    local fs_type=$(df -T . | tail -1 | awk '{print $2}')
    log_info "当前文件系统: $fs_type"
    
    if [[ "$fs_type" == "ext4" ]] || [[ "$fs_type" == "xfs" ]]; then
        log_success "使用高性能文件系统"
    else
        log_warn "文件系统可能影响I/O性能"
        log_suggestion "考虑使用ext4或xfs文件系统以获得更好的性能"
    fi
    
    # 检查I/O调度器
    log_check "检查I/O调度器"
    local disk_device=$(df . | tail -1 | awk '{print $1}' | sed 's/[0-9]*$//')
    if [[ -f "/sys/block/$(basename $disk_device)/queue/scheduler" ]]; then
        local scheduler=$(cat "/sys/block/$(basename $disk_device)/queue/scheduler" | grep -o '\[.*\]' | tr -d '[]')
        log_info "当前I/O调度器: $scheduler"
        
        if [[ "$scheduler" == "noop" ]] || [[ "$scheduler" == "none" ]] || [[ "$scheduler" == "mq-deadline" ]]; then
            log_success "I/O调度器配置合理"
        else
            log_warn "I/O调度器可能不是最优选择"
            log_suggestion "对于SSD，考虑使用noop或none调度器"
        fi
    fi
}

# 检查网络优化
check_network_optimization() {
    log_section "网络优化检查"
    
    # 检查InfiniBand
    log_check "检查InfiniBand支持"
    if command -v ibstat &> /dev/null; then
        if ibstat &>/dev/null; then
            log_success "InfiniBand设备可用"
            local ib_devices=$(ibstat | grep "CA '" | wc -l)
            log_info "检测到 $ib_devices 个InfiniBand设备"
        else
            log_info "InfiniBand设备未激活"
        fi
    else
        log_info "InfiniBand工具未安装"
    fi
    
    # 检查网络接口
    log_check "检查高速网络接口"
    local high_speed_interfaces=$(ip link show | grep -E "(10000|25000|40000|100000)" | wc -l)
    if [[ $high_speed_interfaces -gt 0 ]]; then
        log_success "检测到高速网络接口"
    else
        log_warn "未检测到高速网络接口"
        log_suggestion "对于分布式训练，高速网络可以显著提升性能"
    fi
    
    # 检查网络优化参数
    log_check "检查网络缓冲区大小"
    local rmem_max=$(cat /proc/sys/net/core/rmem_max 2>/dev/null || echo "0")
    local wmem_max=$(cat /proc/sys/net/core/wmem_max 2>/dev/null || echo "0")
    
    if [[ $rmem_max -ge 134217728 ]] && [[ $wmem_max -ge 134217728 ]]; then
        log_success "网络缓冲区大小已优化"
    else
        log_warn "网络缓冲区大小较小"
        log_suggestion "增加网络缓冲区大小以提升网络性能"
    fi
}

# 生成优化报告
generate_optimization_report() {
    log_section "生成优化报告"
    
    cat > "$REPORT_FILE" << EOF
# GPU优化检查报告

## 检查概览
- 检查时间: $(date)
- 总检查项: $TOTAL_CHECKS
- 通过: $PASSED_CHECKS
- 警告: $WARNING_CHECKS
- 失败: $FAILED_CHECKS
- 成功率: $(echo "scale=1; $PASSED_CHECKS * 100 / $TOTAL_CHECKS" | bc 2>/dev/null || echo "N/A")%

## 系统环境
- 操作系统: $(uname -a)
- NVIDIA驱动版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 2>/dev/null || echo "N/A")
EOF
    
    if command -v nvcc &> /dev/null; then
        echo "- CUDA版本: $(nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-)" >> "$REPORT_FILE"
    fi
    
    cat >> "$REPORT_FILE" << EOF

## GPU设备信息
\`\`\`
$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv 2>/dev/null || echo "GPU信息获取失败")
\`\`\`

## 优化建议

EOF
    
    if [[ ${#OPTIMIZATION_SUGGESTIONS[@]} -gt 0 ]]; then
        echo "基于检查结果，以下是主要的优化建议：" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        
        local suggestion_num=1
        for suggestion in "${OPTIMIZATION_SUGGESTIONS[@]}"; do
            echo "$suggestion_num. $suggestion" >> "$REPORT_FILE"
            ((suggestion_num++))
        done
    else
        echo "恭喜！您的GPU环境已经很好地优化了。" >> "$REPORT_FILE"
    fi
    
    cat >> "$REPORT_FILE" << EOF

## 详细检查日志

详细的检查日志请查看: \`optimization_check.log\`

## 快速优化脚本

基于检查结果，您可以运行以下命令进行快速优化：

\`\`\`bash
# 启用GPU持久化模式
sudo nvidia-smi -pm 1

# 设置最大功耗限制（根据您的GPU调整）
# sudo nvidia-smi -pl <max_power_limit>

# 设置CPU性能模式
sudo cpupower frequency-set -g performance

# 优化网络参数（可选）
echo 134217728 | sudo tee /proc/sys/net/core/rmem_max
echo 134217728 | sudo tee /proc/sys/net/core/wmem_max
\`\`\`

**注意**: 请根据您的具体环境和需求调整这些设置。
EOF
    
    log_info "✅ 优化报告生成完成: $REPORT_FILE"
}

# 显示检查摘要
show_summary() {
    echo
    echo -e "${WHITE}=== GPU优化检查摘要 ===${NC}"
    echo -e "总检查项: ${BLUE}$TOTAL_CHECKS${NC}"
    echo -e "通过: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "警告: ${YELLOW}$WARNING_CHECKS${NC}"
    echo -e "失败: ${RED}$FAILED_CHECKS${NC}"
    
    local success_rate=$(echo "scale=1; $PASSED_CHECKS * 100 / $TOTAL_CHECKS" | bc 2>/dev/null || echo "0")
    echo -e "成功率: ${WHITE}$success_rate%${NC}"
    
    if [[ ${#OPTIMIZATION_SUGGESTIONS[@]} -gt 0 ]]; then
        echo
        echo -e "${CYAN}主要优化建议:${NC}"
        local count=1
        for suggestion in "${OPTIMIZATION_SUGGESTIONS[@]:0:5}"; do
            echo -e "  $count. $suggestion"
            ((count++))
        done
        
        if [[ ${#OPTIMIZATION_SUGGESTIONS[@]} -gt 5 ]]; then
            echo -e "  ... 更多建议请查看详细报告"
        fi
    fi
    
    echo
    echo -e "详细报告: ${BLUE}$REPORT_FILE${NC}"
    echo -e "检查日志: ${BLUE}$LOG_FILE${NC}"
}

# 显示帮助信息
show_help() {
    echo "GPU优化检查清单脚本"
    echo "用法: $0 [选项] [检查类型]"
    echo ""
    echo "检查类型:"
    echo "  driver     驱动检查"
    echo "  cuda       CUDA环境检查"
    echo "  hardware   硬件配置检查"
    echo "  performance 性能设置检查"
    echo "  memory     内存优化检查"
    echo "  container  容器环境检查"
    echo "  framework  深度学习框架检查"
    echo "  system     系统优化检查"
    echo "  network    网络优化检查"
    echo "  all        所有检查（默认）"
    echo ""
    echo "选项:"
    echo "  --output-dir DIR    输出目录（默认: gpu_optimization_TIMESTAMP）"
    echo "  --fix              尝试自动修复发现的问题"
    echo "  --verbose          详细输出"
    echo "  --help             显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                      # 运行所有检查"
    echo "  $0 driver cuda          # 只检查驱动和CUDA"
    echo "  $0 --fix performance    # 检查性能设置并尝试修复"
    echo "  $0 --verbose all        # 详细模式运行所有检查"
}

# 解析命令行参数
parse_arguments() {
    local check_types=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output-dir)
                OUTPUT_DIR="$2"
                LOG_FILE="$OUTPUT_DIR/optimization_check.log"
                REPORT_FILE="$OUTPUT_DIR/optimization_report.md"
                shift 2
                ;;
            --fix)
                FIX_ISSUES=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            driver|cuda|hardware|performance|memory|container|framework|system|network|all)
                check_types+=("$1")
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 如果没有指定检查类型，默认为all
    if [[ ${#check_types[@]} -eq 0 ]]; then
        check_types=("all")
    fi
    
    CHECK_TYPE="${check_types[*]}"
}

# 主函数
main() {
    # 设置输出目录
    setup_output_directory
    
    log_info "开始GPU优化检查..."
    log_info "检查类型: $CHECK_TYPE"
    
    # 根据检查类型运行相应检查
    for check_type in $CHECK_TYPE; do
        case $check_type in
            "driver")
                check_gpu_driver
                ;;
            "cuda")
                check_cuda_environment
                ;;
            "hardware")
                check_gpu_hardware
                ;;
            "performance")
                check_gpu_performance_settings
                ;;
            "memory")
                check_memory_optimization
                ;;
            "container")
                check_container_environment
                ;;
            "framework")
                check_ml_frameworks
                ;;
            "system")
                check_system_optimization
                ;;
            "network")
                check_network_optimization
                ;;
            "all")
                check_gpu_driver
                check_cuda_environment
                check_gpu_hardware
                check_gpu_performance_settings
                check_memory_optimization
                check_container_environment
                check_ml_frameworks
                check_system_optimization
                check_network_optimization
                ;;
        esac
    done
    
    # 生成报告
    generate_optimization_report
    
    # 显示摘要
    show_summary
    
    log_info "🎉 GPU优化检查完成！"
}

# 解析参数并运行主函数
parse_arguments "$@"
main