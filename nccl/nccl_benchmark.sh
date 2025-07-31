#!/bin/bash
# =============================================================================
# NCCL InfiniBand 测试验证脚本
# 功能: 专注于 NCCL 分布式通信测试，验证 InfiniBand 网络性能
# 作者: Grissom
# 版本: 2.0
# 
# 说明: 
#   - 此脚本专注于 NCCL 测试，不重复 ib_health_check.sh 的功能
#   - 建议先运行 ib_health_check.sh 确保 IB 网络正常
#   - 可配合 ib_bandwidth_monitor.sh 监控测试期间的网络性能
# =============================================================================

# 版本信息
VERSION="2.0"
SCRIPT_NAME="NCCL IB Test"

# 全局变量
LOG_FILE="/tmp/nccl_test_$(date +%Y%m%d_%H%M%S).log"
ERROR_COUNT=0
WARNING_COUNT=0
QUIET_MODE=false
TEST_SIZE="1M"  # 测试数据大小: 1M, 10M, 100M, 1G
TEST_DURATION=30  # 测试持续时间(秒)
MULTI_NODE_MODE=false
MASTER_ADDR="localhost"
MASTER_PORT="29500"
NETWORK_BACKEND="auto"  # 网络后端: auto, ib, ethernet, socket

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_info() {
    [ "$QUIET_MODE" = false ] && log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    WARNING_COUNT=$((WARNING_COUNT + 1))
    log "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    ERROR_COUNT=$((ERROR_COUNT + 1))
    log "${RED}[ERROR]${NC} $1"
}

log_header() {
    [ "$QUIET_MODE" = false ] && log ""
    [ "$QUIET_MODE" = false ] && log "${PURPLE}=== $1 ===${NC}"
    [ "$QUIET_MODE" = false ] && log ""
}

# 显示帮助信息
show_help() {
    cat << EOF
NCCL InfiniBand 测试验证脚本 v${VERSION}

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -v, --version           显示版本信息
  -q, --quiet             静默模式 (仅输出关键信息)
  -s, --size SIZE         测试数据大小 (1M, 10M, 100M, 1G, 10G) [默认: 1M]
                          1M  = 约 1MB  (262K 元素)
                          10M = 约 10MB (2.6M 元素) 
                          100M= 约 100MB(26M 元素)
                          1G  = 约 1GB  (268M 元素)
                          10G = 约 10GB (2.7B 元素)
  -t, --time SECONDS      测试持续时间 (秒) [默认: 30]
  -m, --multi-node        多节点模式
  --master-addr ADDR      主节点地址 [默认: localhost]
  --master-port PORT      主节点端口 [默认: 29500]
  --network BACKEND       指定网络后端 [默认: auto]
                          auto     - 自动检测并选择最佳网络 (按NCCL优先级)
                                   单节点: NVLink > PCIe P2P > 共享内存 > 网络传输
                                   多节点: InfiniBand > 以太网
                          ib       - 强制使用 InfiniBand/RoCE
                          nvlink   - 强制使用 NVLink (单节点多GPU)
                          ethernet - 强制使用以太网 (TCP/IP)
                          socket   - 强制使用 Socket 传输
  --check-only            仅检查 NCCL 环境，不运行测试
  --env-only              仅设置环境变量并显示配置

功能:
  • 检查 NCCL 和 PyTorch 环境
  • 按 NCCL 优先级自动检测并配置最佳通信路径
  • 运行 AllReduce 性能测试
  • 分析网络通信效率
  • 生成详细的测试报告

测试模式:
  单节点模式 (默认): 测试单机多 GPU 之间的 NCCL 通信
    • 自动检测优先级: NVLink > PCIe P2P > 共享内存 > 网络传输(IB > 以太网)
    • 推荐网络后端: auto (自动选择) > nvlink > ib > ethernet > socket
    • 主要测试 GPU 间高速通信和本地网络栈
    • 适用于单机训练和推理场景
  
  多节点模式 (-m): 测试跨节点的 NCCL 通信 (需要在每个节点运行)
    • 自动检测优先级: InfiniBand > 以太网
    • 推荐网络后端: auto (自动选择) > ib > ethernet > socket
    • 主要测试网络带宽和延迟
    • 适用于分布式训练场景

前置条件:
  • 建议先运行 ib_health_check.sh 确保 IB 网络正常
  • 可配合 ib_bandwidth_monitor.sh 监控测试期间的网络性能

示例:
  # 基础测试 (推荐使用 auto 模式)
  $0                                    # 自动检测最佳通信路径 (推荐)
  $0 --check-only                      # 仅检查环境
  $0 --env-only                        # 仅显示环境配置
  
  # 单节点测试 (auto 模式会按优先级自动选择)
  $0 --network auto -s 1G -t 60        # 自动选择最佳路径，1GB 数据，60秒 (推荐)
  $0 --network nvlink -s 1G            # 强制使用 NVLink (如果可用)
  $0 --network pcie -s 1G              # 强制使用 PCIe P2P 通信
  $0 --network ib -s 10M               # 强制使用 InfiniBand
  $0 --network ethernet -s 1M          # 以太网兼容性测试
  $0 --network socket -s 1M            # Socket 调试模式
  
  # 多节点测试 (auto 模式会优先选择 IB)
  $0 -m --master-addr 192.168.1.100    # 自动选择网络 (推荐)
  $0 -m --master-addr 192.168.1.100 --network ib     # 强制使用 InfiniBand
  $0 -m --master-addr 192.168.1.100 --network ethernet # 强制使用以太网
  $0 -m --master-addr 192.168.1.100 -s 100M -t 120   # 大数据长时间测试

EOF
}

# 显示版本信息
show_version() {
    echo "$SCRIPT_NAME v$VERSION"
    echo "专注于 NCCL InfiniBand 通信测试"
}

# 验证参数
validate_arguments() {
    log_header "验证参数配置"
    
    local validation_failed=false
    
    # 验证测试数据大小
    if [[ ! "$TEST_SIZE" =~ ^[0-9]+[MG]?$ ]]; then
        log_error "无效的测试数据大小: $TEST_SIZE"
        log_info "支持的格式: 数字 + 可选单位 (M/G)，例如: 50M, 1G, 100"
        validation_failed=true
    else
        log_success "测试数据大小: $TEST_SIZE"
    fi
    
    # 验证测试时长
    if [[ ! "$TEST_DURATION" =~ ^[0-9]+$ ]] || [ "$TEST_DURATION" -lt 10 ] || [ "$TEST_DURATION" -gt 3600 ]; then
        log_error "无效的测试时长: $TEST_DURATION"
        log_info "测试时长必须是 10-3600 秒之间的整数"
        validation_failed=true
    else
        log_success "测试时长: ${TEST_DURATION}秒"
    fi
    
    # 验证多节点配置
    if [ "$MULTI_NODE_MODE" = true ]; then
        if [ -z "$MASTER_ADDR" ]; then
            log_error "多节点模式需要指定 --master-addr 参数"
            validation_failed=true
        else
            # 验证 IP 地址格式
            if [[ "$MASTER_ADDR" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
                log_success "主节点地址: $MASTER_ADDR"
            else
                log_warning "主节点地址格式可能不正确: $MASTER_ADDR"
            fi
        fi
        
        if [ -n "$MASTER_PORT" ]; then
            if [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] && [ "$MASTER_PORT" -ge 1024 ] && [ "$MASTER_PORT" -le 65535 ]; then
                log_success "主节点端口: $MASTER_PORT"
            else
                log_error "无效的端口号: $MASTER_PORT (必须是 1024-65535 之间的整数)"
                validation_failed=true
            fi
        fi
    fi
    
    # 验证网络后端
    case "$NETWORK_BACKEND" in
        auto|ib|nvlink|pcie|ethernet|socket)
            log_success "网络后端: $NETWORK_BACKEND"
            ;;
        *)
            log_error "无效的网络后端: $NETWORK_BACKEND"
            log_info "支持的网络后端: auto, ib, nvlink, pcie, ethernet, socket"
            validation_failed=true
            ;;
    esac
    
    # 检查互斥选项
    local exclusive_count=0
    [ "$CHECK_ONLY" = true ] && ((exclusive_count++))
    [ "$ENV_ONLY" = true ] && ((exclusive_count++))
    
    if [ $exclusive_count -gt 1 ]; then
        log_error "不能同时使用 --check-only 和 --env-only 选项"
        validation_failed=true
    fi
    
    if [ "$validation_failed" = true ]; then
        log_error "参数验证失败，请检查并修正参数"
        exit 1
    fi
    
    log_success "所有参数验证通过"
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
            -q|--quiet)
                QUIET_MODE=true
                shift
                ;;
            -s|--size)
                if [ -z "$2" ]; then
                    log_error "--size 选项需要参数"
                    exit 1
                fi
                TEST_SIZE="$2"
                shift 2
                ;;
            -t|--time)
                if [ -z "$2" ]; then
                    log_error "--time 选项需要参数"
                    exit 1
                fi
                TEST_DURATION="$2"
                shift 2
                ;;
            -m|--multi-node)
                MULTI_NODE_MODE=true
                shift
                ;;
            --master-addr)
                if [ -z "$2" ]; then
                    log_error "--master-addr 选项需要参数"
                    exit 1
                fi
                MASTER_ADDR="$2"
                shift 2
                ;;
            --master-port)
                if [ -z "$2" ]; then
                    log_error "--master-port 选项需要参数"
                    exit 1
                fi
                MASTER_PORT="$2"
                shift 2
                ;;
            --network)
                if [ -z "$2" ]; then
                    log_error "--network 选项需要参数"
                    exit 1
                fi
                NETWORK_BACKEND="$2"
                shift 2
                ;;
            --check-only)
                CHECK_ONLY=true
                shift
                ;;
            --env-only)
                ENV_ONLY=true
                shift
                ;;
            *)
                log_error "未知选项: $1"
                echo "使用 '$0 --help' 查看帮助信息"
                exit 1
                ;;
        esac
    done
    
    # 验证参数
    validate_arguments
}

# 检查 NCCL 相关依赖
check_nccl_dependencies() {
    log_header "检查 NCCL 环境依赖"
    
    local deps_ok=true
    
    # 检查 Python3
    if command -v python3 >/dev/null 2>&1; then
        log_success "Python3 可用"
    else
        log_error "Python3 未安装"
        deps_ok=false
    fi
    
    # 检查 PyTorch 和 NCCL
    if python3 -c "import torch" 2>/dev/null; then
        local torch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        log_success "PyTorch 版本: $torch_version"
        
        # 检查 CUDA 支持
        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            local cuda_version=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
            log_success "CUDA 支持可用，版本: $cuda_version"
        else
            log_error "PyTorch CUDA 支持不可用"
            deps_ok=false
        fi
        
        # 检查 NCCL
        if python3 -c "import torch; torch.cuda.nccl.version()" 2>/dev/null; then
            local nccl_version=$(python3 -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null)
            log_success "NCCL 版本: $nccl_version"
        else
            log_warning "无法获取 NCCL 版本信息"
        fi
    else
        log_error "PyTorch 未安装或不可用"
        deps_ok=false
    fi
    
    # 检查 NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi &>/dev/null; then
            local gpu_count=$(nvidia-smi -L | wc -l)
            log_success "检测到 $gpu_count 个 NVIDIA GPU"
            if [ "$QUIET_MODE" = false ]; then
                nvidia-smi -L | while read line; do
                    log_info "  $line"
                done
            fi
        else
            log_error "nvidia-smi 执行失败"
            deps_ok=false
        fi
    else
        log_error "nvidia-smi 命令不可用"
        deps_ok=false
    fi
    
    # 简单检查 InfiniBand 可用性
    if command -v ibstat >/dev/null 2>&1; then
        if ibstat &>/dev/null; then
            log_success "InfiniBand 设备可用"
        else
            log_warning "InfiniBand 设备不可用 (可能影响性能)"
        fi
    else
        log_warning "ibstat 命令不可用 (建议先运行 ib_health_check.sh)"
    fi
    
    if [ "$deps_ok" = true ]; then
        log_success "NCCL 环境依赖检查通过"
        return 0
    else
        log_error "NCCL 环境依赖检查失败"
        return 1
    fi
}

# 智能检测网络类型并设置 NCCL 环境变量
# 功能说明：
# 1. 根据用户选择的网络后端配置 NCCL 环境变量
# 2. 支持自动检测、强制 InfiniBand、以太网和 Socket 传输
# 3. 配置 GPUDirect RDMA 和性能优化参数
setup_nccl_env() {
    log_header "配置 NCCL 环境变量"
    
    log_info "用户选择的网络后端: $NETWORK_BACKEND"
    
    # ========== 根据网络后端选择配置策略 ==========
    case "$NETWORK_BACKEND" in
        "auto")
            setup_auto_network
            ;;
        "ib")
            setup_infiniband_network
            ;;
        "nvlink")
            setup_nvlink_network
            ;;
        "pcie")
            setup_pcie_network
            ;;
        "ethernet")
            setup_ethernet_network
            ;;
        "socket")
            setup_socket_network
            ;;
        *)
            log_error "未知的网络后端: $NETWORK_BACKEND"
            return 1
            ;;
    esac
    
    # ========== 通用 NCCL 调试配置 ==========
    if [ "$QUIET_MODE" = false ]; then
        export NCCL_DEBUG=INFO
        export NCCL_DEBUG_SUBSYS=INIT,NET
        log_info "启用 NCCL 调试信息 (DEBUG=INFO, SUBSYS=INIT,NET)"
    else
        export NCCL_DEBUG=WARN
        export NCCL_DEBUG_SUBSYS=ALL
        log_info "设置 NCCL 调试级别为 WARN"
    fi
    
    # 多节点配置
    if [ "$MULTI_NODE_MODE" = true ]; then
        export NCCL_SOCKET_IFNAME=^docker0,lo
        log_info "多节点模式: 排除 docker0 和 lo 接口"
    fi
    
    # 显示最终配置摘要
    display_nccl_config_summary
}

# 自动检测网络配置 - 按照 NCCL 优先级
setup_auto_network() {
    log_info "自动检测网络环境 (按 NCCL 优先级: NVLink > PCIe P2P > 共享内存 > 网络传输)..."
    
    # ========== 第一优先级：检测 NVLink (仅单节点) ==========
    if [ "$MULTI_NODE_MODE" = false ]; then
        local nvlink_available=false
        local nvlink_active=false
        if command -v nvidia-smi >/dev/null 2>&1; then
            local gpu_count=$(nvidia-smi -L | wc -l)
            if [ "$gpu_count" -gt 1 ]; then
                # 方法1：检测活跃的NVLink连接（动态状态）- 这是最可靠的方法
                if nvidia-smi nvlink --status &>/dev/null; then
                    # 检测显示带宽的NVLink（如 "26.562 GB/s"）
                    local nvlink_count=$(nvidia-smi nvlink --status | grep -c "GB/s" 2>/dev/null || echo "0")
                    # 清理可能的空格和换行符
                    nvlink_count=$(echo "$nvlink_count" | tr -d ' \n\r\t')
                    # 确保是数字
                    if [[ "$nvlink_count" =~ ^[0-9]+$ ]] && [ "$nvlink_count" -gt 0 ]; then
                        nvlink_available=true
                        nvlink_active=true
                        # 获取平均带宽信息
                        local avg_bandwidth=$(nvidia-smi nvlink --status | grep "GB/s" | head -1 | grep -oE "[0-9]+\.[0-9]+ GB/s" | head -1)
                        log_success "检测到 $nvlink_count 个活跃的 NVLink 连接 (带宽: $avg_bandwidth)"
                        log_success "自动选择 NVLink 网络 (最高优先级)"
                        configure_nvlink_settings
                        return 0
                    fi
                fi
                
                # 方法2：检测GPU拓扑中的NVLink硬件（静态拓扑）- 仅作为备选检测
                local topo_output=$(nvidia-smi topo -m 2>/dev/null || echo "")
                if [ -n "$topo_output" ] && echo "$topo_output" | grep -qE "NV[0-9]+"; then
                    nvlink_available=true
                    local nvlink_connections=$(echo "$topo_output" | grep -oE "NV[0-9]+" | sort -u | tr '\n' ' ')
                    log_info "检测到 NVLink 硬件拓扑: $nvlink_connections"
                    log_warning "NVLink 硬件可用但当前未激活，可能被其他进程占用或需要GPU负载触发"
                    log_info "继续检测 PCIe P2P 作为备选方案..."
                    # 不直接返回，继续检测 PCIe P2P
                fi
            fi
        fi
        log_info "NVLink 检测: 硬件$([ "$nvlink_available" = true ] && echo "可用" || echo "不可用"), 激活状态$([ "$nvlink_active" = true ] && echo "活跃" || echo "未激活")"
    else
        log_info "多节点模式: 跳过 NVLink 检测"
    fi
    
    # ========== 第二优先级：检测 PCIe P2P (仅单节点) ==========
    if [ "$MULTI_NODE_MODE" = false ]; then
        local p2p_available=false
        if command -v nvidia-smi >/dev/null 2>&1; then
            local gpu_count=$(nvidia-smi -L | wc -l)
            if [ "$gpu_count" -gt 1 ]; then
                # 简单检测：如果有多个GPU且没有NVLink，假设有PCIe P2P
                p2p_available=true
                log_info "检测到多GPU环境，PCIe P2P 可能可用"
                log_success "自动选择 PCIe P2P 通信 (第二优先级)"
                configure_pcie_p2p_settings
                return 0
            fi
        fi
        log_info "PCIe P2P 检测: $([ "$p2p_available" = true ] && echo "可用" || echo "不可用")"
    else
        log_info "多节点模式: 跳过 PCIe P2P 检测"
    fi
    
    # ========== 第三优先级：共享内存 (仅单节点) ==========
    if [ "$MULTI_NODE_MODE" = false ]; then
        log_info "单节点模式: 共享内存通信可用"
        log_success "自动选择共享内存通信 (第三优先级)"
        configure_shm_settings
        return 0
    fi
    
    # ========== 第四优先级：网络传输 (InfiniBand > 以太网) ==========
    log_info "检测网络传输选项..."
    
    # 检测 InfiniBand 设备
    local has_ib=false
    local network_type="未知"
    local is_roce=false
    
    if command -v ibv_devinfo >/dev/null 2>&1; then
        local link_layer=$(ibv_devinfo | grep "link_layer:" | head -1 | awk '{print $2}')
        if [ "$link_layer" = "Ethernet" ]; then
            has_ib=true
            is_roce=true
            network_type="RoCE (Ethernet over IB)"
            log_info "检测到 RoCE 环境"
        elif [ "$link_layer" = "InfiniBand" ]; then
            has_ib=true
            network_type="原生 InfiniBand"
            log_info "检测到原生 IB 环境"
        fi
    fi
    
    if [ "$has_ib" = true ]; then
        log_success "自动选择 InfiniBand 网络 (网络传输最高优先级)"
        configure_infiniband_settings "$is_roce" "$network_type"
    else
        log_warning "未检测到 InfiniBand 设备，回退到以太网"
        log_success "自动选择以太网传输 (网络传输备选)"
        configure_ethernet_settings
    fi
}

# 强制使用 InfiniBand 网络
setup_infiniband_network() {
    log_info "强制使用 InfiniBand 网络..."
    
    # 检测 IB 类型
    local is_roce=false
    local network_type="InfiniBand (强制)"
    
    if command -v ibv_devinfo >/dev/null 2>&1; then
        local link_layer=$(ibv_devinfo | grep "link_layer:" | head -1 | awk '{print $2}')
        if [ "$link_layer" = "Ethernet" ]; then
            is_roce=true
            network_type="RoCE (强制)"
        elif [ "$link_layer" = "InfiniBand" ]; then
            network_type="原生 InfiniBand (强制)"
        fi
    else
        log_warning "无法检测 IB 设备信息，使用默认 IB 配置"
    fi
    
    configure_infiniband_settings "$is_roce" "$network_type"
}

# 强制使用 NVLink 传输
setup_nvlink_network() {
    log_info "强制使用 NVLink 传输..."
    
    # NVLink 仅适用于单节点多GPU场景
    if [ "$MULTI_NODE_MODE" = true ]; then
        log_error "NVLink 仅支持单节点多GPU模式，不支持多节点"
        log_info "建议使用 --network ib 或 --network ethernet 进行多节点通信"
        return 1
    fi
    
    configure_nvlink_settings
}

# 强制使用 PCIe P2P 传输
setup_pcie_network() {
    log_info "强制使用 PCIe P2P 传输..."
    
    # PCIe P2P 仅适用于单节点多GPU场景
    if [ "$MULTI_NODE_MODE" = true ]; then
        log_error "PCIe P2P 仅支持单节点多GPU模式，不支持多节点"
        log_info "建议使用 --network ib 或 --network ethernet 进行多节点通信"
        return 1
    fi
    
    configure_pcie_p2p_settings
}

# 强制使用以太网
setup_ethernet_network() {
    log_info "强制使用以太网 (TCP/IP)..."
    configure_ethernet_settings
}

# 强制使用 Socket 传输
setup_socket_network() {
    log_info "强制使用 Socket 传输..."
    configure_socket_settings
}

# 配置 InfiniBand 相关设置
configure_infiniband_settings() {
    local is_roce="$1"
    local network_type="$2"
    
    log_info "配置 InfiniBand 设置: $network_type"
    
    # ========== 基础 InfiniBand 配置 ==========
    export NCCL_IB_DISABLE=0
    log_info "设置 NCCL_IB_DISABLE=0 (启用 InfiniBand)"
    
    # ========== GPUDirect RDMA 配置 ==========
    export NCCL_NET_GDR_LEVEL=2
    log_info "设置 NCCL_NET_GDR_LEVEL=2 (启用 GPUDirect RDMA)"
    
    # ========== HCA 设备配置 ==========
    if command -v ibstat >/dev/null 2>&1; then
        local hca_name=$(ibstat | grep "CA '" | head -1 | awk '{print $2}' | tr -d "'")
        if [ -n "$hca_name" ]; then
            export NCCL_IB_HCA="$hca_name"
            log_info "设置 NCCL_IB_HCA=$hca_name"
        else
            log_warning "无法获取 HCA 设备名"
        fi
    fi
    
    # ========== 根据 IB 类型配置参数 ==========
    if [ "$is_roce" = true ]; then
        # RoCE 配置
        export NCCL_IB_GID_INDEX=3
        export NCCL_IB_TC=136
        export NCCL_IB_SL=0
        export NCCL_SOCKET_IFNAME=""
        log_info "RoCE 配置: GID_INDEX=3, TC=136, SL=0"
    else
        # 原生 IB 配置
        export NCCL_IB_GID_INDEX=0
        export NCCL_IB_TC=136
        export NCCL_IB_SL=0
        log_info "原生 IB 配置: GID_INDEX=0, TC=136, SL=0"
    fi
    
    # ========== IB 性能优化参数 ==========
    export NCCL_IB_TIMEOUT=22
    export NCCL_IB_RETRY_CNT=7
    log_info "IB 性能参数: TIMEOUT=22, RETRY_CNT=7"
    
    # 禁用其他传输方式
    export NCCL_P2P_DISABLE=0  # 启用 P2P
    log_info "启用 GPU P2P 通信"
}

# 配置以太网设置
configure_ethernet_settings() {
    log_info "配置以太网 (TCP/IP) 设置"
    
    # ========== 禁用 InfiniBand ==========
    export NCCL_IB_DISABLE=1
    log_info "设置 NCCL_IB_DISABLE=1 (禁用 InfiniBand)"
    
    # ========== 启用 Socket 传输 ==========
    # 让 NCCL 自动选择可用的以太网接口
    if [ "$MULTI_NODE_MODE" = true ]; then
        # 多节点模式：排除虚拟接口
        export NCCL_SOCKET_IFNAME=^docker0,lo,virbr
        log_info "多节点以太网: 排除虚拟接口"
    else
        # 单节点模式：可以使用 localhost
        unset NCCL_SOCKET_IFNAME
        log_info "单节点以太网: 使用默认接口选择"
    fi
    
    # ========== 禁用 GPUDirect RDMA ==========
    export NCCL_NET_GDR_LEVEL=0
    log_info "设置 NCCL_NET_GDR_LEVEL=0 (禁用 GPUDirect RDMA)"
    
    # ========== P2P 配置 ==========
    export NCCL_P2P_DISABLE=0  # 仍然启用 GPU P2P（通过 PCIe）
    log_info "启用 GPU P2P 通信 (PCIe)"
    
    log_success "以太网配置完成 - 将使用 TCP/IP 进行节点间通信"
}

# 配置 Socket 传输设置
configure_socket_settings() {
    log_info "配置 Socket 传输设置"
    
    # ========== 禁用所有硬件加速 ==========
    export NCCL_IB_DISABLE=1
    export NCCL_NET_GDR_LEVEL=0
    export NCCL_P2P_DISABLE=1  # 也禁用 P2P
    log_info "禁用所有硬件加速 (IB, GDR, P2P)"
    
    # ========== 强制使用 Socket ==========
    if [ "$MULTI_NODE_MODE" = true ]; then
        export NCCL_SOCKET_IFNAME=^docker0,lo,virbr
        log_info "多节点 Socket: 排除虚拟接口"
    else
        export NCCL_SOCKET_IFNAME=lo
        log_info "单节点 Socket: 使用 loopback 接口"
    fi
    
    log_warning "Socket 传输模式 - 性能可能较低，主要用于调试"
}

# 配置 NVLink 传输设置
configure_nvlink_settings() {
    log_info "配置 NVLink 传输设置（强制启用模式）"
    
    # ========== 检查 NVLink 可用性 ==========
    local nvlink_available=false
    local nvlink_count=0
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        # 检查是否有多个 GPU
        local gpu_count=$(nvidia-smi -L | wc -l)
        if [ "$gpu_count" -gt 1 ]; then
            # 检查 NVLink 连接
            if nvidia-smi nvlink --status &>/dev/null; then
                nvlink_count=$(nvidia-smi nvlink --status | grep -c "GB/s" 2>/dev/null || echo "0")
                # 清理可能的空格和换行符
                nvlink_count=$(echo "$nvlink_count" | tr -d ' \n\r\t')
                # 确保是数字
                if [[ "$nvlink_count" =~ ^[0-9]+$ ]] && [ "$nvlink_count" -gt 0 ]; then
                    nvlink_available=true
                    log_success "检测到 $nvlink_count 个活跃的 NVLink 连接"
                else
                    log_warning "未检测到活跃的 NVLink 连接"
                fi
            else
                log_warning "无法检查 NVLink 状态"
            fi
        else
            log_warning "检测到 $gpu_count 个 GPU，NVLink 需要多个 GPU"
        fi
    else
        log_warning "nvidia-smi 不可用，无法检查 NVLink 状态"
        # 检查是否有本地 nvlink 状态文件
        if [ -f "nvlink-smi_status.txt" ]; then
            nvlink_count=$(grep -c "GB/s" nvlink-smi_status.txt 2>/dev/null || echo "0")
            if [ "$nvlink_count" -gt 0 ]; then
                nvlink_available=true
                log_success "从本地文件检测到 $nvlink_count 个活跃的 NVLink 连接"
            fi
        fi
    fi
    
    # ========== 核心 NVLink 配置（强制启用）==========
    export NCCL_P2P_LEVEL=NVL          # 强制使用 NVLink
    export NCCL_NVLS_ENABLE=1          # 启用 NVLink SHARP
    export NCCL_P2P_DISABLE=0          # 确保 P2P 启用
    log_success "核心配置: NCCL_P2P_LEVEL=NVL, NCCL_NVLS_ENABLE=1, NCCL_P2P_DISABLE=0"
    
    # ========== 禁用其他网络后端 ==========
    export NCCL_IB_DISABLE=1           # 禁用 InfiniBand
    export NCCL_NET_DISABLE=1          # 禁用网络传输
    export NCCL_SOCKET_IFNAME=lo       # 仅使用本地回环
    log_info "网络禁用: NCCL_IB_DISABLE=1, NCCL_NET_DISABLE=1, NCCL_SOCKET_IFNAME=lo"
    
    # ========== 调试配置 ==========
    export NCCL_DEBUG=INFO             # 详细日志
    export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV  # 调试子系统
    log_info "调试配置: NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV"
    
    # ========== H100 优化配置 ==========
    export NCCL_ALGO=Ring,Tree         # 算法选择
    export NCCL_PROTO=Simple           # 协议选择
    export NCCL_BUFFSIZE=8388608       # 8MB 缓冲区
    export NCCL_NTHREADS=256           # 线程数
    log_info "H100 优化: NCCL_ALGO=Ring,Tree, NCCL_PROTO=Simple"
    log_info "性能优化: NCCL_BUFFSIZE=8388608, NCCL_NTHREADS=256"
    
    # ========== 额外的 NVLink 优化 ==========
    export NCCL_MIN_NCHANNELS=32       # 最小通道数
    export NCCL_MAX_NCHANNELS=32       # 最大通道数
    export NCCL_TREE_THRESHOLD=0       # 强制使用 Ring 算法
    export NCCL_CROSS_NIC=0            # 单节点不需要跨网卡
    export NCCL_IGNORE_CPU_AFFINITY=1  # 忽略 CPU 亲和性限制
    export NCCL_CUMEM_ENABLE=0         # 禁用 CUDA 统一内存，确保 NVLink 路径
    log_info "通道优化: NCCL_MIN_NCHANNELS=32, NCCL_MAX_NCHANNELS=32"
    log_info "高级优化: NCCL_TREE_THRESHOLD=0, NCCL_IGNORE_CPU_AFFINITY=1"
    
    # ========== GPUDirect 配置 ==========
    export NCCL_NET_GDR_LEVEL=0        # 禁用网络 GDR，使用 NVLink
    log_info "GPUDirect: NCCL_NET_GDR_LEVEL=0 (禁用网络 GPUDirect)"
    
    # ========== 拓扑检测和诊断 ==========
    export NCCL_TOPO_DUMP_FILE="/tmp/nccl_topo_nvlink.xml"
    export NCCL_GRAPH_DUMP_FILE="/tmp/nccl_graph_nvlink.xml"
    export NCCL_DEBUG_FILE="/tmp/nccl_debug_nvlink.%h.%p.log"
    log_info "拓扑文件: $NCCL_TOPO_DUMP_FILE"
    log_info "图文件: $NCCL_GRAPH_DUMP_FILE"
    log_info "调试日志: $NCCL_DEBUG_FILE"
    
    # ========== 结果总结 ==========
    if [ "$nvlink_available" = true ]; then
        log_success "✅ NVLink 强制启用配置完成 - 检测到 $nvlink_count 个活跃连接"
        log_success "🚀 预期性能: >300 GB/s 吞吐量, <0.02 ms 延迟"
        log_info "💡 查找日志关键词确认: 'Using network NVLink', 'P2P level: NVL', 'NVLS enabled'"
    else
        log_warning "⚠️  NVLink 强制启用配置完成 - 但硬件检测失败"
        log_warning "🔧 NCCL 将尝试使用 NVLink，如果硬件不支持将自动回退"
        log_info "📊 如果性能不佳，请检查硬件连接或使用 --network auto 重新测试"
    fi
    
    # ========== 环境变量验证 ==========
    log_info "🔍 关键环境变量验证:"
    log_info "  NCCL_P2P_LEVEL: ${NCCL_P2P_LEVEL}"
    log_info "  NCCL_NVLS_ENABLE: ${NCCL_NVLS_ENABLE}"
    log_info "  NCCL_IB_DISABLE: ${NCCL_IB_DISABLE}"
    log_info "  NCCL_NET_DISABLE: ${NCCL_NET_DISABLE}"
    log_info "  NCCL_BUFFSIZE: ${NCCL_BUFFSIZE}"
    log_info "  NCCL_NTHREADS: ${NCCL_NTHREADS}"
}

# 配置 PCIe P2P 传输设置（智能检测 NVLink 和 PCIe）
configure_pcie_p2p_settings() {
    log_info "配置 GPU 间高速传输设置（智能检测 NVLink/PCIe P2P）"
    
    # ========== 检查 GPU 和连接可用性 ==========
    local p2p_available=false
    local nvlink_available=false
    local gpu_count=0
    local nvlink_count=0
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count=$(nvidia-smi -L | wc -l)
        if [ "$gpu_count" -gt 1 ]; then
            p2p_available=true
            log_success "检测到 $gpu_count 个 GPU，GPU 间通信可用"
            
            # 检查 NVLink 连接
            if nvidia-smi nvlink -s >/dev/null 2>&1; then
                nvlink_count=$(nvidia-smi nvlink -s 2>/dev/null | grep -c "Active" || echo "0")
                if [ "$nvlink_count" -gt 0 ]; then
                    nvlink_available=true
                    log_success "检测到 $nvlink_count 个活跃的 NVLink 连接"
                else
                    log_info "未检测到活跃的 NVLink 连接，将使用 PCIe P2P"
                fi
            else
                log_info "NVLink 状态查询失败，假设使用 PCIe P2P"
            fi
        else
            log_warning "检测到 $gpu_count 个 GPU，P2P 通信需要多个 GPU"
        fi
    else
        log_warning "nvidia-smi 不可用，无法检查 GPU 状态"
    fi
    
    # ========== 强制禁用所有网络传输 ==========
    export NCCL_IB_DISABLE=1
    export NCCL_NET_DISABLE=1  # 强制禁用所有网络传输
    export NCCL_SOCKET_FAMILY=AF_UNSPEC  # 禁用 Socket 协议族
    log_info "设置 NCCL_IB_DISABLE=1 (禁用 InfiniBand)"
    log_info "设置 NCCL_NET_DISABLE=1 (强制禁用所有网络传输)"
    log_info "设置 NCCL_SOCKET_FAMILY=AF_UNSPEC (禁用 Socket 协议族)"
    
    # ========== 智能配置 P2P 级别 ==========
    export NCCL_P2P_DISABLE=0
    if [ "$nvlink_available" = true ]; then
        export NCCL_P2P_LEVEL=NVL  # 优先使用 NVLink
        export NCCL_NVLS_ENABLE=1  # 启用 NVLink Switch（H100 支持）
        export NCCL_NVLS_CHUNKSIZE=524288  # NVLink Switch 块大小优化
        log_success "设置 NCCL_P2P_LEVEL=NVL (优先 NVLink，检测到 $nvlink_count 个连接)"
        log_info "设置 NCCL_NVLS_ENABLE=1 (启用 NVLink Switch for H100)"
        log_info "设置 NCCL_NVLS_CHUNKSIZE=524288 (NVLink Switch 优化)"
    else
        export NCCL_P2P_LEVEL=PIX  # 回退到 PCIe
        export NCCL_NVLS_ENABLE=0  # 禁用 NVLink Switch
        log_info "设置 NCCL_P2P_LEVEL=PIX (使用 PCIe P2P)"
        log_info "设置 NCCL_NVLS_ENABLE=0 (禁用 NVLink Switch)"
    fi
    
    export NCCL_SHM_DISABLE=1  # 禁用共享内存，强制使用 P2P
    log_info "设置 NCCL_P2P_DISABLE=0, NCCL_SHM_DISABLE=1 (强制 P2P 路径)"
    
    # ========== 强制禁用网络 Socket 传输 ==========
    export NCCL_SOCKET_IFNAME=""  # 明确禁用 Socket 网络
    export NCCL_SOCKET_NTHREADS=0  # 禁用 Socket 线程
    log_info "设置 NCCL_SOCKET_IFNAME=\"\" (明确禁用网络 Socket)"
    log_info "设置 NCCL_SOCKET_NTHREADS=0 (禁用 Socket 线程)"
    
    # ========== GPUDirect 配置 ==========
    export NCCL_NET_GDR_LEVEL=0  # 禁用网络 GDR，使用 GPU 间直连
    log_info "设置 NCCL_NET_GDR_LEVEL=0 (禁用网络 GPUDirect)"
    
    # ========== 性能优化配置 ==========
    export NCCL_CROSS_NIC=0    # 单节点不需要跨网卡
    if [ "$nvlink_available" = true ]; then
        export NCCL_ALGO=Auto  # NVLink 使用自动算法选择
        export NCCL_MIN_NCHANNELS=1
        export NCCL_MAX_NCHANNELS=32  # NVLink 支持更多通道
        log_info "NVLink 优化: ALGO=Auto, MAX_NCHANNELS=32"
    else
        export NCCL_ALGO=Ring  # PCIe 使用 Ring 算法
        export NCCL_MIN_NCHANNELS=1
        export NCCL_MAX_NCHANNELS=16  # PCIe 通道限制
        log_info "PCIe 优化: ALGO=Ring, MAX_NCHANNELS=16"
    fi
    
    # ========== 通用 P2P 优化 ==========
    export NCCL_IGNORE_CPU_AFFINITY=1  # 忽略 CPU 亲和性限制
    export NCCL_BUFFSIZE=8388608  # 增大缓冲区大小
    export NCCL_NTHREADS=16  # 设置线程数
    export NCCL_MAX_NRINGS=8  # 最大环数
    log_info "通用优化: IGNORE_CPU_AFFINITY=1, BUFFSIZE=8388608"
    
    # ========== 调试和诊断配置 ==========
    if [ "$nvlink_available" = true ]; then
        export NCCL_DEBUG_SUBSYS="INIT,P2P,GRAPH,NVLS,ENV"  # 包含 NVLink 调试
        export NCCL_DEBUG_FILE="/tmp/nccl_debug_nvlink.%h.%p.log"
        log_info "NVLink 调试: NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
    else
        export NCCL_DEBUG_SUBSYS="INIT,P2P,GRAPH,ENV,TUNING"  # PCIe 调试
        export NCCL_DEBUG_FILE="/tmp/nccl_debug_pcie.%h.%p.log"
        log_info "PCIe 调试: NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
    fi
    log_info "调试日志: $NCCL_DEBUG_FILE"
    
    # ========== 拓扑检测和验证 ==========
    if [ "$nvlink_available" = true ]; then
        export NCCL_TOPO_DUMP_FILE="/tmp/nccl_topo_nvlink.xml"
        export NCCL_GRAPH_DUMP_FILE="/tmp/nccl_graph_nvlink.xml"
    else
        export NCCL_TOPO_DUMP_FILE="/tmp/nccl_topo_pcie.xml"
        export NCCL_GRAPH_DUMP_FILE="/tmp/nccl_graph_pcie.xml"
    fi
    log_info "拓扑文件: $NCCL_TOPO_DUMP_FILE"
    log_info "图文件: $NCCL_GRAPH_DUMP_FILE"
    
    # ========== 高级优化配置 ==========
    export NCCL_P2P_NET_CHUNKSIZE=131072  # P2P网络块大小优化
    export NCCL_CUMEM_ENABLE=0  # 禁用CUDA统一内存，确保P2P路径
    export NCCL_DMABUF_ENABLE=1  # 启用DMA-BUF支持（如果可用）
    export NCCL_REG_CACHE_ENABLE=1  # 启用注册缓存
    log_info "高级优化: NET_CHUNKSIZE=131072, CUMEM_ENABLE=0"
    log_info "内存优化: DMABUF_ENABLE=1, REG_CACHE_ENABLE=1"
    
    # ========== ACS 检测和警告 ==========
    if command -v lspci >/dev/null 2>&1; then
        local acs_enabled=$(sudo lspci -vvv 2>/dev/null | grep "ACSCtl.*SrcValid+" | wc -l || echo "0")
        if [ "$acs_enabled" -gt 0 ]; then
            log_warning "检测到 ACS (Access Control Services) 可能已启用"
            log_warning "这可能严重影响 P2P 性能，建议："
            log_info "  1. 在 BIOS 中禁用 VT-d/IOMMU"
            log_info "  2. 或使用: sudo setpci -s <GPU_BUS> CAP_EXP+28.w=0000"
        fi
    fi
    
    # ========== 配置完成总结 ==========
    if [ "$p2p_available" = true ]; then
        if [ "$nvlink_available" = true ]; then
            log_success "NVLink 配置完成 - 使用 NVLink 进行 GPU 间高速通信"
            log_info "预期带宽: ~900 GB/s (18 链路 × 26.562 GB/s × 双向)"
            log_info "预期延迟: < 1 μs"
        else
            log_success "PCIe P2P 配置完成 - 使用 PCIe 进行 GPU 间通信"
            log_info "预期带宽: ~64 GB/s (PCIe 5.0 x16)"
            log_info "预期延迟: 2-5 μs"
        fi
        
        # 验证建议
        log_info ""
        log_info "建议运行以下命令验证配置："
        log_info "  ./diagnose_topology.sh  # 深度拓扑诊断"
        log_info "  nvidia-smi topo -mp     # 查看GPU拓扑"
        log_info "  nvidia-smi nvlink -s    # 检查NVLink状态"
        if command -v nvbandwidth >/dev/null 2>&1; then
            log_info "  nvbandwidth             # 测试实际带宽"
        else
            log_info "  建议安装 nvbandwidth 工具进行带宽测试"
        fi
    else
        log_warning "GPU 间通信配置完成 - 但可能回退到共享内存通信"
        log_info "建议检查 GPU 数量和系统配置"
        log_warning "注意：消费级GPU（RTX 30/40系列）可能不支持P2P"
    fi
}

# 配置共享内存传输设置
configure_shm_settings() {
    log_info "配置共享内存传输设置"
    
    # ========== 禁用所有网络传输 ==========
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1  # 也禁用 P2P，强制使用共享内存
    log_info "禁用网络传输和 P2P (IB, P2P)"
    
    # ========== 启用共享内存 ==========
    # NCCL 会自动回退到共享内存传输
    export NCCL_SOCKET_IFNAME=""  # 明确禁用 Socket 网络
    log_info "设置 NCCL_SOCKET_IFNAME=\"\" (禁用网络，使用共享内存)"
    
    # ========== GPUDirect 配置 ==========
    export NCCL_NET_GDR_LEVEL=0  # 禁用网络 GDR
    log_info "设置 NCCL_NET_GDR_LEVEL=0 (禁用网络 GPUDirect)"
    
    # ========== 共享内存优化 ==========
    export NCCL_CROSS_NIC=0    # 单节点不需要跨网卡
    log_info "共享内存优化: CROSS_NIC=0"
    
    # ========== 拓扑检测 ==========
    export NCCL_TOPO_DUMP_FILE="/tmp/nccl_topo_shm.xml"
    log_info "拓扑文件: $NCCL_TOPO_DUMP_FILE"
    
    log_success "共享内存配置完成 - 将使用共享内存进行 GPU 间通信"
    log_warning "共享内存传输性能较低，主要用于兼容性测试"
}

# 显示 NCCL 配置摘要
display_nccl_config_summary() {
    log_info ""
    log_success "NCCL 环境变量配置完成"
    log_info "配置摘要:"
    log_info "  网络后端: $NETWORK_BACKEND"
    
    # 基础网络配置
    log_info "  NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-未设置}"
    log_info "  NCCL_NET_DISABLE: ${NCCL_NET_DISABLE:-未设置}"
    log_info "  NCCL_NET_GDR_LEVEL: ${NCCL_NET_GDR_LEVEL:-未设置}"
    log_info "  NCCL_IB_HCA: ${NCCL_IB_HCA:-未设置}"
    log_info "  NCCL_IB_GID_INDEX: ${NCCL_IB_GID_INDEX:-未设置}"
    
    # P2P 和 NVLink 配置
    log_info "  NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE:-未设置}"
    log_info "  NCCL_P2P_LEVEL: ${NCCL_P2P_LEVEL:-未设置}"
    log_info "  NCCL_SHM_DISABLE: ${NCCL_SHM_DISABLE:-未设置}"
    log_info "  NCCL_NVLS_ENABLE: ${NCCL_NVLS_ENABLE:-未设置}"
    log_info "  NCCL_NVLS_CHUNKSIZE: ${NCCL_NVLS_CHUNKSIZE:-未设置}"
    
    # Socket 配置
    log_info "  NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-未设置}$([ "${NCCL_SOCKET_IFNAME+x}" = "x" ] && [ -z "$NCCL_SOCKET_IFNAME" ] && echo " (空字符串-禁用)" || echo "")"
    log_info "  NCCL_SOCKET_NTHREADS: ${NCCL_SOCKET_NTHREADS:-未设置}"
    log_info "  NCCL_SOCKET_FAMILY: ${NCCL_SOCKET_FAMILY:-未设置}"
    
    # 性能优化配置
    log_info "  NCCL_ALGO: ${NCCL_ALGO:-未设置}"
    log_info "  NCCL_MIN_NCHANNELS: ${NCCL_MIN_NCHANNELS:-未设置}"
    log_info "  NCCL_MAX_NCHANNELS: ${NCCL_MAX_NCHANNELS:-未设置}"
    log_info "  NCCL_IGNORE_CPU_AFFINITY: ${NCCL_IGNORE_CPU_AFFINITY:-未设置}"
    log_info "  NCCL_BUFFSIZE: ${NCCL_BUFFSIZE:-未设置}"
    log_info "  NCCL_NTHREADS: ${NCCL_NTHREADS:-未设置}"
    log_info "  NCCL_MAX_NRINGS: ${NCCL_MAX_NRINGS:-未设置}"
    log_info "  NCCL_CROSS_NIC: ${NCCL_CROSS_NIC:-未设置}"
    
    # 调试和诊断配置
    log_info "  NCCL_DEBUG: $NCCL_DEBUG"
    log_info "  NCCL_DEBUG_SUBSYS: ${NCCL_DEBUG_SUBSYS:-未设置}"
    log_info "  NCCL_DEBUG_FILE: ${NCCL_DEBUG_FILE:-未设置}"
    log_info "  NCCL_TOPO_DUMP_FILE: ${NCCL_TOPO_DUMP_FILE:-未设置}"
    log_info "  NCCL_GRAPH_DUMP_FILE: ${NCCL_GRAPH_DUMP_FILE:-未设置}"
    
    # 高级优化配置
    log_info "  NCCL_P2P_NET_CHUNKSIZE: ${NCCL_P2P_NET_CHUNKSIZE:-未设置}"
    log_info "  NCCL_CUMEM_ENABLE: ${NCCL_CUMEM_ENABLE:-未设置}"
    log_info "  NCCL_DMABUF_ENABLE: ${NCCL_DMABUF_ENABLE:-未设置}"
    log_info "  NCCL_REG_CACHE_ENABLE: ${NCCL_REG_CACHE_ENABLE:-未设置}"
    
    # 多节点配置
    log_info "  多节点模式: $([ "$MULTI_NODE_MODE" = true ] && echo "是" || echo "否")"
    
    # 配置状态检查
    log_info ""
    log_info "配置状态检查:"
    
    # P2P 状态检查
    if [ "${NCCL_P2P_DISABLE:-1}" = "0" ]; then
        if [ "${NCCL_P2P_LEVEL:-}" = "NVL" ]; then
            log_success "✓ 已启用 NVLink P2P 通信"
            if [ "${NCCL_NVLS_ENABLE:-0}" = "1" ]; then
                log_success "✓ 已启用 NVLink Switch（H100 优化）"
            fi
        elif [ "${NCCL_P2P_LEVEL:-}" = "PIX" ]; then
            log_success "✓ 已启用 PCIe P2P 通信"
        else
            log_info "✓ 已启用 P2P 通信（级别：${NCCL_P2P_LEVEL:-自动}）"
        fi
    else
        log_warning "⚠ P2P 通信已禁用"
    fi
    
    # 网络传输状态检查
    if [ "${NCCL_IB_DISABLE:-0}" = "1" ] && [ "${NCCL_NET_DISABLE:-0}" = "1" ]; then
        log_success "✓ 已禁用网络传输，强制使用 GPU 间直连"
    elif [ "${NCCL_IB_DISABLE:-0}" = "1" ]; then
        log_info "✓ 已禁用 InfiniBand，使用其他传输方式"
    elif [ "${NCCL_NET_DISABLE:-0}" = "1" ]; then
        log_info "✓ 已禁用网络传输，使用本地传输"
    else
        log_warning "⚠ 网络传输未完全禁用，可能影响 P2P 性能"
    fi
    
    # 算法优化检查
    if [ "${NCCL_ALGO:-}" = "Auto" ]; then
        log_success "✓ 使用自动算法选择（适合 NVLink）"
    elif [ "${NCCL_ALGO:-}" = "Ring" ]; then
        log_info "✓ 使用 Ring 算法（适合 PCIe）"
    fi
    
    # 调试配置检查
    if [ -n "${NCCL_DEBUG_FILE:-}" ]; then
        log_success "✓ 已配置调试日志文件: ${NCCL_DEBUG_FILE}"
    fi
    
    if [ -n "${NCCL_TOPO_DUMP_FILE:-}" ]; then
        log_success "✓ 已配置拓扑转储文件: ${NCCL_TOPO_DUMP_FILE}"
    fi
}

# 创建动态 NCCL 测试脚本
# 功能说明：
# 1. 优先使用独立的模板文件 nccl_python_template.py
# 2. 如果模板文件不存在，则使用内嵌代码创建
# 3. 通过环境变量传递测试参数
create_nccl_test() {
    log_header "创建 NCCL 测试脚本"
    
    local test_script="/tmp/nccl_test.py"
    # 使用脚本所在目录的绝对路径
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local template_script="$script_dir/nccl_python_template.py"
    
    # 计算张量元素数量
    local tensor_elements
    case "$TEST_SIZE" in
        "1M"|"1m")
            tensor_elements=262144  # 1MB / 4 bytes = 262144 elements
            ;;
        "10M"|"10m")
            tensor_elements=2621440  # 10MB / 4 bytes = 2621440 elements
            ;;
        "100M"|"100m")
            tensor_elements=26214400  # 100MB / 4 bytes = 26214400 elements
            ;;
        "1G"|"1g")
            tensor_elements=268435456  # 1GB / 4 bytes = 268435456 elements
            ;;
        "10G"|"10g")
            tensor_elements=2684354560  # 10GB / 4 bytes = 2684354560 elements
            ;;
        *)
            log_warning "未知的测试大小: $TEST_SIZE，使用默认值 1M"
            log_info "支持的大小格式: 1M, 10M, 100M, 1G, 10G"
            tensor_elements=262144
            ;;
    esac
    
    log_info "配置测试数据大小: $TEST_SIZE (约 $tensor_elements 个元素)"
    
    # 检查模板文件是否存在
    if [ ! -f "$template_script" ]; then
        log_error "模板文件不存在: $template_script"
        log_error "请确保 nccl_python_template.py 文件在当前目录中"
        return 1
    fi
    
    # 使用模板文件创建测试脚本
    log_info "使用模板文件: $template_script"
    cp "$template_script" "$test_script"
    chmod +x "$test_script"
    
    # 设置环境变量传递参数给 Python 脚本
    export TENSOR_ELEMENTS="$tensor_elements"
    export TEST_DURATION="$TEST_DURATION"
    export NCCL_BACKEND="nccl"
    
    log_success "测试脚本创建成功: $test_script"
    log_info "测试参数通过环境变量传递:"
    log_info "  TENSOR_ELEMENTS: $tensor_elements"
    log_info "  TEST_DURATION: $TEST_DURATION"
}

# 运行单节点 NCCL 测试
run_single_node_test() {
    log_header "运行单节点 NCCL 测试"
    
    # 检查 GPU 数量
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "nvidia-smi 命令不可用"
        return 1
    fi
    
    local gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    log_info "检测到 $gpu_count 个 GPU"
    
    if [ "$gpu_count" -lt 1 ]; then
        log_error "未检测到可用的 GPU"
        return 1
    elif [ "$gpu_count" -eq 1 ]; then
        log_warning "只有 1 个 GPU，NCCL 测试意义有限"
        log_info "建议使用多 GPU 环境进行测试"
    fi
    
    # 检查测试脚本是否存在
    local test_script="/tmp/nccl_test.py"
    if [ ! -f "$test_script" ]; then
        log_error "测试脚本不存在: $test_script"
        return 1
    fi
    
    # 设置输出文件
    local output_file="/tmp/nccl_test_output.log"
    
    # 运行测试
    log_info "启动 NCCL 测试 (使用 $gpu_count 个 GPU)..."
    log_info "测试输出将保存到: $output_file"
    
    # 配置分布式参数
    local master_addr="localhost"
    local master_port="29500"
    local nnodes="1"
    local node_rank="0"
    
    # 多节点模式配置
    if [ "$MULTI_NODE_MODE" = true ]; then
        if [ -n "$MASTER_ADDR" ]; then
            master_addr="$MASTER_ADDR"
        else
            log_error "多节点模式需要指定 --master-addr 参数"
            return 1
        fi
        
        if [ -n "$MASTER_PORT" ]; then
            master_port="$MASTER_PORT"
        fi
        
        # 从环境变量获取多节点配置
        if [ -n "$WORLD_SIZE" ] && [ -n "$NODE_RANK" ] && [ -n "$NPROC_PER_NODE" ]; then
            # 使用环境变量配置
            nnodes=$((WORLD_SIZE / NPROC_PER_NODE))
            node_rank="$NODE_RANK"
            gpu_count="$NPROC_PER_NODE"
            log_info "使用环境变量配置多节点参数:"
            log_info "  WORLD_SIZE: $WORLD_SIZE"
            log_info "  NODE_RANK: $NODE_RANK"
            log_info "  NPROC_PER_NODE: $NPROC_PER_NODE"
            log_info "  计算得出 NNODES: $nnodes"
        else
            log_warning "多节点模式建议设置环境变量: WORLD_SIZE, NODE_RANK, NPROC_PER_NODE"
            log_info "当前使用默认配置 (单节点模式)"
        fi
    fi
    
    log_info "分布式配置参数:"
    log_info "  Master 地址: $master_addr"
    log_info "  Master 端口: $master_port"
    log_info "  节点数量: $nnodes"
    log_info "  节点编号: $node_rank"
    log_info "  每节点GPU数: $gpu_count"
    
    # 检查 torch.distributed.launch 是否可用
    if python3 -c "import torch.distributed.launch" 2>/dev/null; then
        # 使用新的 torchrun 命令（如果可用）
        if command -v torchrun >/dev/null 2>&1; then
            log_info "使用 torchrun 启动分布式测试"
            if torchrun \
                --nproc_per_node="$gpu_count" \
                --nnodes="$nnodes" \
                --node_rank="$node_rank" \
                --master_addr="$master_addr" \
                --master_port="$master_port" \
                "$test_script" 2>&1 | tee "$output_file"; then
                log_success "NCCL 测试执行完成"
            else
                log_error "NCCL 测试执行失败"
                return 1
            fi
        else
            # 使用传统的 torch.distributed.launch
            log_info "使用 torch.distributed.launch 启动分布式测试"
            if python3 -m torch.distributed.launch \
                --nproc_per_node="$gpu_count" \
                --nnodes="$nnodes" \
                --node_rank="$node_rank" \
                --master_addr="$master_addr" \
                --master_port="$master_port" \
                "$test_script" 2>&1 | tee "$output_file"; then
                log_success "NCCL 测试执行完成"
            else
                log_error "NCCL 测试执行失败"
                return 1
            fi
        fi
    else
        log_error "torch.distributed.launch 不可用"
        return 1
    fi
    
    # 分析输出
    analyze_nccl_output "/tmp/nccl_test_output.log" "$NETWORK_BACKEND"
}

# 分析 NCCL 输出
analyze_nccl_output() {
    log_header "分析 NCCL 测试输出"
    
    # 支持传入文件路径和期望网络后端参数
    local output_file="${1:-/tmp/nccl_test_output.log}"
    local expected_network="${2:-$NETWORK_BACKEND}"
    
    if [ ! -f "$output_file" ]; then
        log_error "测试输出文件不存在: $output_file"
        return 1
    fi
    
    local file_size=$(wc -l < "$output_file")
    log_info "输出文件大小: $file_size 行"
    log_info "期望网络后端: $expected_network"
    
    # 检查测试是否成功完成
    if grep -q "NCCL AllReduce 测试成功" "$output_file"; then
        log_success "✅ NCCL 测试成功完成"
    elif grep -q "NCCL AllReduce 测试失败" "$output_file"; then
        log_error "❌ NCCL 测试失败"
    else
        log_warning "⚠️  无法确定测试结果"
    fi
    
    log_info ""
    log_info "NCCL 网络选择分析与验证:"
    
    # 检测实际使用的网络类型
    local actual_network="unknown"
    local network_detected=false
    
    # 智能网络检测逻辑 - 结合环境变量、性能数据和日志关键词
    local nvlink_env_configured=false
    local nvlink_performance_indicators=false
    local explicit_network_logs=""
    
    # 1. 检查 NCCL 环境变量配置
    if grep -q "NCCL_P2P_LEVEL.*NVL" "$output_file" && grep -q "NCCL_NVLS_ENABLE.*1" "$output_file"; then
        nvlink_env_configured=true
        log_info "🔧 检测到 NVLink 环境变量配置: NCCL_P2P_LEVEL=NVL, NCCL_NVLS_ENABLE=1"
    fi
    
    # 2. 分析性能数据判断网络类型
    local avg_throughput=$(grep -E "平均吞吐量.*GB/s" "$output_file" | head -1 | grep -o '[0-9]\+\.[0-9]\+')
    local min_latency=$(grep -E "最小延迟.*ms" "$output_file" | head -1 | grep -o '[0-9]\+\.[0-9]\+')
    
    if [ -n "$avg_throughput" ] && [ -n "$min_latency" ]; then
        # NVLink 性能特征：高吞吐量 (>100 GB/s) + 低延迟 (<0.1 ms)
        # PCIe P2P 性能特征：中等吞吐量 (30-80 GB/s) + 中等延迟 (0.03-0.1 ms)
        # Socket/Ethernet 性能特征：低吞吐量 (<30 GB/s) + 高延迟 (>0.1 ms)
        
        local throughput_int=$(echo "$avg_throughput" | cut -d. -f1)
        local latency_float=$(echo "$min_latency")
        
        if [ "$throughput_int" -gt 100 ] && [ "$(echo "$latency_float < 0.05" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
            nvlink_performance_indicators=true
            log_info "📊 性能指标显示 NVLink 特征: 吞吐量 ${avg_throughput} GB/s, 延迟 ${min_latency} ms"
        elif [ "$throughput_int" -ge 30 ] && [ "$throughput_int" -le 100 ]; then
            log_info "📊 性能指标显示 PCIe P2P 特征: 吞吐量 ${avg_throughput} GB/s, 延迟 ${min_latency} ms"
        else
            log_info "📊 性能指标显示网络传输特征: 吞吐量 ${avg_throughput} GB/s, 延迟 ${min_latency} ms"
        fi
    fi
    
    # 3. 检测明确的网络日志信息
    if grep -q "Using network InfiniBand" "$output_file"; then
        actual_network="ib"
        network_detected=true
        explicit_network_logs="Using network InfiniBand"
        log_info "🔍 明确检测到: InfiniBand 网络"
    elif grep -q "NET/IB" "$output_file"; then
        actual_network="ib"
        network_detected=true
        explicit_network_logs="NET/IB"
        log_info "🔍 明确检测到: InfiniBand 网络 (NET/IB)"
    elif grep -q "RoCE" "$output_file"; then
        actual_network="ib"
        network_detected=true
        explicit_network_logs="RoCE"
        log_info "🔍 明确检测到: RoCE (Ethernet over IB) 网络"
    elif grep -q "NET/Socket" "$output_file"; then
        actual_network="socket"
        network_detected=true
        explicit_network_logs="NET/Socket"
        log_info "🔍 明确检测到: Socket 网络传输"
    elif grep -q "NET/SHM" "$output_file"; then
        actual_network="shm"
        network_detected=true
        explicit_network_logs="NET/SHM"
        log_info "🔍 明确检测到: 共享内存通信"
    elif grep -q "NET/NVL" "$output_file" || grep -q "NVLink.*enabled" "$output_file"; then
        actual_network="nvlink"
        network_detected=true
        explicit_network_logs="NET/NVL or NVLink enabled"
        log_info "🔍 明确检测到: NVLink 通信"
    elif grep -q "Ethernet" "$output_file" && ! grep -q "RoCE" "$output_file"; then
        actual_network="ethernet"
        network_detected=true
        explicit_network_logs="Ethernet"
        log_info "🔍 明确检测到: 以太网传输"
    fi
    
    # 4. 智能推断网络类型（当没有明确日志时）
    if [ "$network_detected" = false ]; then
        if [ "$nvlink_env_configured" = true ] && [ "$nvlink_performance_indicators" = true ]; then
            actual_network="nvlink"
            network_detected=true
            log_success "🧠 智能推断: NVLink 通信 (基于环境配置 + 性能特征)"
        elif [ "$nvlink_env_configured" = true ]; then
            # 环境变量配置了 NVLink，但性能不符合，可能是 PCIe P2P
            actual_network="pcie"
            network_detected=true
            log_warning "🧠 智能推断: PCIe P2P 通信 (NVLink 配置但性能不符)"
            log_warning "   💡 可能原因: NVLink 硬件不可用，NCCL 回退到 PCIe P2P"
        elif [ -n "$avg_throughput" ]; then
            local throughput_int=$(echo "$avg_throughput" | cut -d. -f1)
            if [ "$throughput_int" -ge 30 ]; then
                actual_network="pcie"
                network_detected=true
                log_info "🧠 智能推断: PCIe P2P 通信 (基于性能特征)"
            else
                actual_network="socket"
                network_detected=true
                log_info "🧠 智能推断: Socket 网络传输 (基于性能特征)"
            fi
        else
            # 最后的关键词检测
            if grep -q -i "pcie\|p2p" "$output_file"; then
                actual_network="pcie"
                network_detected=true
                log_info "🔍 关键词检测到: PCIe P2P 通信"
            fi
        fi
    fi
    
    # 比较期望值与实际值
    log_info ""
    log_info "网络配置验证结果:"
    
    if [ "$network_detected" = false ]; then
        log_warning "⚠️  无法明确检测到网络类型"
        actual_network="unknown"
    fi
    
    # 验证网络选择是否符合期望
    case "$expected_network" in
        "auto")
            log_success "✅ 自动模式: NCCL 选择了 $actual_network 网络"
            log_info "ℹ️  自动模式下，NCCL 按优先级选择最佳网络路径"
            ;;
        "ib")
            if [ "$actual_network" = "ib" ]; then
                log_success "✅ 网络配置匹配: 期望 InfiniBand，实际使用 InfiniBand"
            else
                log_error "❌ 网络配置不匹配: 期望 InfiniBand，实际使用 $actual_network"
                log_warning "💡 可能原因: IB 设备不可用、配置错误或被其他网络覆盖"
            fi
            ;;
        "nvlink")
            if [ "$actual_network" = "nvlink" ]; then
                log_success "✅ 网络配置匹配: 期望 NVLink，实际使用 NVLink"
            else
                log_error "❌ 网络配置不匹配: 期望 NVLink，实际使用 $actual_network"
                log_warning "💡 NVLink 配置诊断:"
                
                # 详细的 NVLink 诊断
                if [ "$nvlink_env_configured" = true ]; then
                    log_success "   ✅ NCCL 环境变量配置正确 (NCCL_P2P_LEVEL=NVL, NCCL_NVLS_ENABLE=1)"
                else
                    log_error "   ❌ NCCL 环境变量配置缺失或错误"
                fi
                
                if [ -n "$avg_throughput" ]; then
                    local throughput_int=$(echo "$avg_throughput" | cut -d. -f1)
                    if [ "$throughput_int" -gt 100 ]; then
                        log_success "   ✅ 性能指标符合 NVLink 特征 (${avg_throughput} GB/s)"
                    elif [ "$throughput_int" -ge 30 ]; then
                        log_warning "   ⚠️  性能指标显示 PCIe P2P 特征 (${avg_throughput} GB/s)"
                        log_warning "       这表明 NCCL 回退到了 PCIe P2P 通信"
                    else
                        log_error "   ❌ 性能指标过低 (${avg_throughput} GB/s)，可能使用网络传输"
                    fi
                fi
                
                # 提供具体的解决建议
                log_warning "   🔧 可能的解决方案:"
                if [ "$actual_network" = "pcie" ]; then
                    log_warning "      1. 检查 GPU 拓扑: 确认 GPU 之间有 NVLink 连接"
                    log_warning "      2. 检查 NCCL 版本: 确保支持当前 GPU 的 NVLink"
                    log_warning "      3. 检查系统配置: 确认 NVLink 驱动和固件正常"
                    log_warning "      4. 尝试设置 NCCL_DEBUG=INFO 获取更多调试信息"
                elif [ "$actual_network" = "socket" ]; then
                    log_warning "      1. 检查 NCCL_P2P_DISABLE 是否被意外设置为 1"
                    log_warning "      2. 检查 GPU 可见性: 确认所有 GPU 对 NCCL 可见"
                    log_warning "      3. 检查 CUDA 版本兼容性"
                else
                    log_warning "      1. 检查硬件支持: 确认 GPU 型号支持 NVLink"
                    log_warning "      2. 检查物理连接: 确认 NVLink 线缆连接正常"
                    log_warning "      3. 检查系统状态: 重启后重试"
                fi
            fi
            ;;
        "pcie")
            if [ "$actual_network" = "pcie" ]; then
                log_success "✅ 网络配置匹配: 期望 PCIe P2P，实际使用 PCIe P2P"
            else
                log_error "❌ 网络配置不匹配: 期望 PCIe P2P，实际使用 $actual_network"
                log_warning "💡 可能原因: PCIe P2P 不可用或被其他网络覆盖"
            fi
            ;;
        "ethernet")
            if [ "$actual_network" = "ethernet" ]; then
                log_success "✅ 网络配置匹配: 期望以太网，实际使用以太网"
            else
                log_warning "⚠️  网络配置不完全匹配: 期望以太网，实际使用 $actual_network"
                if [ "$actual_network" = "ib" ]; then
                    log_info "ℹ️  检测到 InfiniBand/RoCE，这通常比以太网性能更好"
                fi
            fi
            ;;
        "socket")
            if [ "$actual_network" = "socket" ]; then
                log_success "✅ 网络配置匹配: 期望 Socket，实际使用 Socket"
            else
                log_warning "⚠️  网络配置不匹配: 期望 Socket，实际使用 $actual_network"
                log_info "ℹ️  NCCL 选择了更高性能的网络路径"
            fi
            ;;
        *)
            log_info "ℹ️  未知期望网络类型: $expected_network，实际使用: $actual_network"
            ;;
    esac
    
    # 额外的网络状态检查
    if [ "$actual_network" = "socket" ] && [ "$expected_network" != "socket" ]; then
        log_warning "⚠️  NCCL 回退到 Socket 网络，可能存在以下问题:"
        log_warning "   • InfiniBand 设备配置错误"
        log_warning "   • GPU P2P 通信被禁用"
        log_warning "   • 网络环境变量配置不当"
    fi
    
    if [ "$actual_network" = "shm" ] && [ "$MULTI_NODE_MODE" = true ]; then
        log_warning "⚠️  多节点模式下使用共享内存，这可能不是期望的行为"
    fi
    
    # 检查 GPUDirect RDMA
    if grep -q "GDR" "$output_file" || grep -q "GPUDirect" "$output_file"; then
        log_success "✅ GPUDirect RDMA 已启用"
    else
        log_warning "⚠️  未检测到 GPUDirect RDMA"
    fi
    
    # 检查 NCCL 初始化信息
    log_info ""
    log_info "NCCL 初始化信息:"
    
    # 1. NCCL 版本信息 - 修复正则表达式以匹配实际格式
    local nccl_version=$(grep -E "(NCCL 版本:|NCCL version)" "$output_file" | head -1)
    if [ -n "$nccl_version" ]; then
        log_success "  版本: $nccl_version"
    else
        log_warning "  未找到 NCCL 版本信息"
    fi
    
    # 2. 通信器初始化信息 - 查找分布式环境初始化
    local comm_info=$(grep -E "(分布式环境初始化成功|后端: nccl|世界大小:|本地排名:)" "$output_file" | head -5)
    if [ -n "$comm_info" ]; then
        log_info "  通信器初始化:"
        echo "$comm_info" | while read line; do
            log_info "    $line"
        done
    fi
    
    # 3. GPU 设备信息
    local gpu_info=$(grep -E "(使用设备: cuda:|GPU: NVIDIA)" "$output_file" | head -3)
    if [ -n "$gpu_info" ]; then
        log_info "  GPU 设备:"
        echo "$gpu_info" | while read line; do
            log_info "    $line"
        done
    fi
    
    # 4. NCCL 环境变量配置 - 从环境信息部分提取
    local nccl_env_section=$(grep -A 20 "=== NCCL 环境信息 ===" "$output_file" | grep -E "NCCL_.*:" | head -5)
    if [ -n "$nccl_env_section" ]; then
        log_info "  NCCL 环境配置:"
        echo "$nccl_env_section" | while read line; do
            # 高亮重要的 NVLink 相关配置
            if echo "$line" | grep -q -E "(P2P_LEVEL.*NVL|NVLS_ENABLE.*1)"; then
                log_success "    ✅ $line"
            else
                log_info "    $line"
            fi
        done
    fi
    
    # 5. 拓扑检测信息 - 查找实际的 NCCL 拓扑信息
    local topo_info=$(grep -E "(topology|TOPO|Graph|Ring|Tree)" "$output_file" | grep -v "NCCL_" | head -2)
    if [ -n "$topo_info" ]; then
        log_info "  拓扑检测:"
        echo "$topo_info" | while read line; do
            log_info "    $line"
        done
    fi
    
    # 6. 性能测试启动信息
    local perf_start=$(grep -E "(开始预热|开始性能测试|测试时长)" "$output_file" | head -3)
    if [ -n "$perf_start" ]; then
        log_info "  性能测试状态:"
        echo "$perf_start" | while read line; do
            log_info "    $line"
        done
    fi
    
    # 7. 检查初始化完整性
    if [ -z "$nccl_version" ] && [ -z "$comm_info" ]; then
        log_warning "未找到完整的 NCCL 初始化信息"
        log_info "尝试查找其他初始化相关信息..."
        local other_init=$(grep -E "(NCCL|nccl|分布式)" "$output_file" | head -5)
        if [ -n "$other_init" ]; then
            echo "$other_init" | while read line; do
                log_info "    $line"
            done
        fi
    else
        log_success "  ✅ NCCL 初始化信息检查完成"
    fi
    
    # 检查错误和警告
    log_info ""
    log_info "错误和警告检查:"
    
    # 安全地获取错误计数，确保返回有效整数
    local error_count=0
    if [ -f "$output_file" ]; then
        error_count=$(grep -c -i "error\|Error\|ERROR" "$output_file" 2>/dev/null | head -1)
        # 确保是有效整数
        if ! [[ "$error_count" =~ ^[0-9]+$ ]]; then
            error_count=0
        fi
    fi
    
    # 安全地获取警告计数，确保返回有效整数
    local warning_count=0
    if [ -f "$output_file" ]; then
        warning_count=$(grep -c -i "warning\|Warning\|WARNING" "$output_file" 2>/dev/null | head -1)
        # 确保是有效整数
        if ! [[ "$warning_count" =~ ^[0-9]+$ ]]; then
            warning_count=0
        fi
    fi
    
    if [ "$error_count" -gt 0 ]; then
        log_warning "发现 $error_count 个错误信息"
        grep -i "error\|Error\|ERROR" "$output_file" 2>/dev/null | head -3 | while read line; do
            log_warning "  $line"
        done
    else
        log_success "未发现错误信息"
    fi
    
    if [ "$warning_count" -gt 0 ]; then
        log_info "发现 $warning_count 个警告信息"
    fi
    
    # 性能信息分析
    log_info ""
    log_info "性能信息摘要:"
    
    # 1. 测试数据规模
    local data_size_info=$(grep -E "(张量大小.*个元素.*MB)" "$output_file" | head -1)
    if [ -n "$data_size_info" ]; then
        log_info "  📊 测试数据规模: $data_size_info"
    fi
    
    # 2. 迭代性能数据分析
    local iteration_data=$(grep -E "\[Rank [0-9]+\] 迭代.*ms.*GB/s" "$output_file")
    if [ -n "$iteration_data" ]; then
        log_success "  ✅ 性能测试数据已收集"
        
        # 分析最新的性能数据（最后几次迭代）
        local latest_iterations=$(echo "$iteration_data" | tail -10)
        local max_throughput=$(echo "$latest_iterations" | grep -oE "[0-9]+\.[0-9]+ GB/s" | sort -nr | head -1)
        local min_latency=$(echo "$latest_iterations" | grep -oE "[0-9]+\.[0-9]+ ms" | sort -n | head -1)
        
        if [ -n "$max_throughput" ]; then
            log_success "    🚀 峰值吞吐量: $max_throughput"
        fi
        
        if [ -n "$min_latency" ]; then
            log_success "    ⚡ 最低延迟: $min_latency"
        fi
        
        # 统计迭代完成情况
        local completed_iterations=$(grep -c "已完成.*次迭代" "$output_file" 2>/dev/null || echo "0")
        if [ "$completed_iterations" -gt 0 ]; then
            log_info "    📈 完成迭代统计: $completed_iterations 个里程碑"
        fi
        
        # 显示最后几次迭代的性能
        log_info "    📋 最近性能样本:"
        echo "$latest_iterations" | tail -3 | while read line; do
            log_info "      $line"
        done
    else
        log_warning "  未找到迭代性能数据"
        
        # 尝试查找其他性能指标
        local other_perf=$(grep -E "(ms|GB/s|Gbps|延迟|吞吐量)" "$output_file" | head -3)
        if [ -n "$other_perf" ]; then
            log_info "  找到其他性能信息:"
            echo "$other_perf" | while read line; do
                log_info "    $line"
            done
        else
            log_info "  建议检查测试是否正常完成"
        fi
    fi
    
    # 3. 测试完成状态
    local test_completion=$(grep -E "(测试完成|测试结束|All tests completed)" "$output_file")
    if [ -n "$test_completion" ]; then
        log_success "  ✅ 测试执行完成"
    else
        local test_duration=$(grep -E "测试时长.*秒" "$output_file" | head -1)
        if [ -n "$test_duration" ]; then
            log_info "  ⏱️  $test_duration"
        fi
    fi
    
    # 环境变量检查
    log_info ""
    log_info "NCCL 环境变量验证:"
    local env_vars=$(grep -E "NCCL_.*:" "$output_file")
    if [ -n "$env_vars" ]; then
        echo "$env_vars" | head -5 | while read line; do
            log_info "  $line"
        done
    else
        log_warning "未找到环境变量信息"
    fi
    
    log_info ""
    log_info "详细日志位置: $output_file"
    log_info "查看完整日志: cat $output_file"
}



# 生成测试报告
generate_report() {
    log_header "生成测试报告"
    
    local report_file="/tmp/nccl_test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
=== NCCL InfiniBand 测试报告 ===
生成时间: $(date)
脚本版本: $VERSION

=== 系统信息 ===
操作系统: $(uname -s) $(uname -r)
Python 版本: $(python3 --version 2>&1 | awk '{print $2}')
PyTorch 版本: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装")
CUDA 版本: $(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "不可用")

=== GPU 信息 ===
$(nvidia-smi -L 2>/dev/null || echo "无法获取 GPU 信息")

=== InfiniBand 信息 ===
$(ibstat 2>/dev/null | head -20 || echo "无法获取 IB 信息")

=== NCCL 环境变量 ===
NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-未设置}
NCCL_NET_GDR_LEVEL: ${NCCL_NET_GDR_LEVEL:-未设置}
NCCL_IB_HCA: ${NCCL_IB_HCA:-未设置}
NCCL_IB_GID_INDEX: ${NCCL_IB_GID_INDEX:-未设置}
NCCL_DEBUG: ${NCCL_DEBUG:-未设置}

=== 测试日志 ===
$(tail -50 "$LOG_FILE" 2>/dev/null || echo "无法读取测试日志")

EOF
    
    log_success "测试报告已生成: $report_file"
    echo "$report_file"
}

# 生成测试总结
generate_summary() {
    log_header "测试总结"
    
    # 安全地获取错误计数，确保返回有效整数
    local error_count=0
    if [ -f "$LOG_FILE" ]; then
        error_count=$(grep -c "ERROR" "$LOG_FILE" 2>/dev/null | head -1)
        # 确保是有效整数
        if ! [[ "$error_count" =~ ^[0-9]+$ ]]; then
            error_count=0
        fi
    fi
    
    # 安全地获取警告计数，确保返回有效整数
    local warning_count=0
    if [ -f "$LOG_FILE" ]; then
        warning_count=$(grep -c "WARNING" "$LOG_FILE" 2>/dev/null | head -1)
        # 确保是有效整数
        if ! [[ "$warning_count" =~ ^[0-9]+$ ]]; then
            warning_count=0
        fi
    fi
    
    log_info "测试统计:"
    log_info "  错误数量: $error_count"
    log_info "  警告数量: $warning_count"
    
    if [ "$error_count" -eq 0 ]; then
        log_success "✅ NCCL 测试成功完成"
        log_info "InfiniBand 网络可以正常用于 NCCL 通信"
    else
        log_error "❌ NCCL 测试发现问题"
        log_info "请检查错误日志并解决相关问题"
    fi
    
    log_info ""
    log_info "详细日志: $LOG_FILE"
}

# 主函数
main() {
    # 解析命令行参数
    parse_arguments "$@"
    
    # 初始化日志
    echo "NCCL InfiniBand 测试开始 - $(date)" > "$LOG_FILE"
    
    # 显示脚本信息
    log_header "$SCRIPT_NAME v$VERSION"
    log_info "专注于 NCCL 分布式通信测试"
    log_info "日志文件: $LOG_FILE"
    log ""
    
    # 仅显示环境配置
    if [ "$ENV_ONLY" = true ]; then
        log_info "仅显示网络配置 (跳过依赖检查)"
        setup_nccl_env
        log_success "网络配置显示完成"
        exit 0
    fi
    
    # 仅检查环境
    if [ "$CHECK_ONLY" = true ]; then
        if check_nccl_dependencies; then
            log_success "环境检查通过"
        else
            log_error "环境检查失败"
            exit 1
        fi
        exit 0
    fi
    
    # 完整测试流程
    local step_failed=false
    
    # 1. 检查 NCCL 依赖
    if ! check_nccl_dependencies; then
        log_error "NCCL 依赖检查失败"
        step_failed=true
    fi
    
    # 2. 设置 NCCL 环境
    if ! $step_failed; then
        setup_nccl_env
    fi
    
    # 3. 创建并运行测试
    if ! $step_failed; then
        if create_nccl_test; then
            run_single_node_test
        else
            log_error "测试脚本创建失败"
            step_failed=true
        fi
    fi
    
    # 4. 生成报告和总结
    generate_report >/dev/null
    generate_summary
    
    if $step_failed; then
        exit 1
    fi
}

# 信号处理
cleanup() {
    log_warning "收到中断信号，正在清理..."
    
    # 清理临时文件
    rm -f /tmp/nccl_test_*.py
    rm -f /tmp/nccl_test_output.log
    
    log_info "清理完成"
    exit 130
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 脚本入口
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi