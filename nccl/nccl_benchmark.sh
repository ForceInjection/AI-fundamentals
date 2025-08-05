#!/bin/bash
# =============================================================================
# NCCL 测试验证脚本
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
SCRIPT_NAME="NCCL Benchmark"

# 全局变量
LOG_FILE="/tmp/nccl_test_$(date +%Y%m%d_%H%M%S).log"
ERROR_COUNT=0
WARNING_COUNT=0
QUIET_MODE=false
TEST_SIZE="1M"  # 测试数据大小: 1M, 10M, 100M, 1G
TEST_DURATION=30  # 测试持续时间(秒)
MULTI_NODE_MODE=false
MASTER_ADDR=""  # 多节点模式下必须明确指定
MASTER_PORT="29500"
NETWORK_BACKEND="auto"  # 网络后端: auto, ib, ethernet, socket
OPTIMIZATION_LEVEL="balanced"  # 优化级别: conservative, balanced, aggressive
DRY_RUN=false

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

# =============================================================================
# 统一配置管理器 - 消除重复代码和提升维护性
# =============================================================================

# NCCL 配置管理器 - 统一管理所有配置项
declare -A NCCL_CONFIG_CACHE
declare -A SYSTEM_INFO_CACHE

# 缓存系统信息，避免重复调用
cache_system_info() {
    if [ -z "${SYSTEM_INFO_CACHE[gpu_count]:-}" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            SYSTEM_INFO_CACHE[gpu_count]=$(nvidia-smi -L 2>/dev/null | wc -l)
        else
            SYSTEM_INFO_CACHE[gpu_count]=0
        fi
    fi
    
    if [ -z "${SYSTEM_INFO_CACHE[nvlink_available]:-}" ]; then
        SYSTEM_INFO_CACHE[nvlink_available]=false
        SYSTEM_INFO_CACHE[nvlink_count]=0
        
        if [ "${SYSTEM_INFO_CACHE[gpu_count]}" -gt 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
            # 统一使用 nvidia-smi nvlink --status 命令，与主检测逻辑保持一致
            if nvidia-smi nvlink --status &>/dev/null; then
                # 检测显示带宽的NVLink（如 "26.562 GB/s"）
                local nvlink_count=$(nvidia-smi nvlink --status 2>/dev/null | grep -c "GB/s" 2>/dev/null || echo "0")
                nvlink_count=$(echo "$nvlink_count" | tr -d ' \n\r\t')
                if [[ "$nvlink_count" =~ ^[0-9]+$ ]] && [ "$nvlink_count" -gt 0 ]; then
                    SYSTEM_INFO_CACHE[nvlink_available]=true
                    SYSTEM_INFO_CACHE[nvlink_count]=$nvlink_count
                fi
            fi
        fi
    fi
    
    if [ -z "${SYSTEM_INFO_CACHE[ib_available]:-}" ]; then
        SYSTEM_INFO_CACHE[ib_available]=false
        if command -v ibv_devinfo >/dev/null 2>&1; then
            local ib_output
            if ib_output=$(ibv_devinfo 2>/dev/null) && echo "$ib_output" | grep -q "hca_id:"; then
                SYSTEM_INFO_CACHE[ib_available]=true
            fi
        fi
    fi
}

# 统一的 NCCL 配置设置器
set_nccl_config() {
    local key="$1"
    local value="$2"
    local description="${3:-}"
    
    export "NCCL_$key"="$value"
    NCCL_CONFIG_CACHE["$key"]="$value"
    
    if [ -n "$description" ]; then
        log_info "设置 NCCL_$key=$value ($description)"
    fi
}

# 批量设置 NCCL 配置
set_nccl_configs() {
    local -n config_array=$1
    local description="${2:-}"
    
    if [ -n "$description" ]; then
        log_info "$description"
    fi
    
    for key in "${!config_array[@]}"; do
        set_nccl_config "$key" "${config_array[$key]}"
    done
}

# 设置通用的 NCCL 基础配置
setup_common_nccl_config() {
    log_info "设置通用 NCCL 基础配置"
    
    # 缓存系统信息
    cache_system_info
    
    # 基础配置组
    declare -A base_config=(
        ["DEBUG"]="INFO"
        ["DEBUG_SUBSYS"]="INIT,NET"
        ["IGNORE_CPU_AFFINITY"]="1"
        ["BUFFSIZE"]="8388608"
        ["CROSS_NIC"]="0"
        ["NET_GDR_LEVEL"]="0"
    )
    
    set_nccl_configs base_config "基础配置: 调试、性能优化、GPUDirect"
}

# 网络配置预设
setup_network_config() {
    local network_type="$1"
    
    case "$network_type" in
        "ib_enable")
            declare -A ib_config=(
                ["IB_DISABLE"]="0"
                ["NET_GDR_LEVEL"]="2"
                ["P2P_DISABLE"]="0"
                ["P2P_LEVEL"]="PIX"
            )
            set_nccl_configs ib_config "启用 InfiniBand 配置"
            ;;
        "ib_disable")
            set_nccl_config "IB_DISABLE" "1" "禁用 InfiniBand"
            ;;
        "p2p_nvlink")
            declare -A nvlink_config=(
                ["P2P_LEVEL"]="NVL"
                ["NVLS_ENABLE"]="1"
                ["P2P_DISABLE"]="0"
                ["IB_DISABLE"]="1"
                ["NET_DISABLE"]="1"
            )
            set_nccl_configs nvlink_config "NVLink P2P 配置"
            ;;
        "p2p_pcie")
            declare -A pcie_config=(
                ["P2P_LEVEL"]="PIX"
                ["NVLS_ENABLE"]="0"
                ["P2P_DISABLE"]="0"
                ["IB_DISABLE"]="1"
            )
            set_nccl_configs pcie_config "PCIe P2P 配置"
            ;;
        "p2p_disable")
            declare -A no_p2p_config=(
                ["P2P_DISABLE"]="1"
                ["P2P_LEVEL"]="0"
                ["NVLS_ENABLE"]="0"
            )
            set_nccl_configs no_p2p_config "禁用 P2P 配置"
            ;;
        "socket_only")
            declare -A socket_config=(
                ["IB_DISABLE"]="1"
                ["P2P_DISABLE"]="1"
                ["SHM_DISABLE"]="1"
                ["NET_DISABLE"]="0"
            )
            set_nccl_configs socket_config "Socket 传输配置"
            ;;
        "pxn_enable")
            # 智能选择 P2P_LEVEL：优先使用 NVLink，回退到 PCIe
            local p2p_level="PIX"  # 默认 PCIe P2P
            if [ "$DETECTED_NVLINK_AVAILABLE" = true ]; then
                p2p_level="NVL"  # 使用 NVLink
                log_success "PXN 模式: 检测到 NVLink，设置 P2P_LEVEL=NVL (节点内 $DETECTED_NVLINK_COUNT 个连接)"
            else
                log_info "PXN 模式: 未检测到 NVLink，设置 P2P_LEVEL=PIX (节点内 PCIe P2P)"
            fi
            
            declare -A pxn_config=(
                ["ALGO"]="Ring,Tree,CollNet"
                ["PROTO"]="Simple,LL,LL128"
                ["NET_GDR_LEVEL"]="2"
                ["P2P_DISABLE"]="0"
                ["P2P_LEVEL"]="$p2p_level"
                ["IB_DISABLE"]="0"
                ["CROSS_NIC"]="1"
                ["BUFFSIZE"]="8388608"
                ["MIN_NCHANNELS"]="4"
                ["MAX_NCHANNELS"]="16"
            )
            set_nccl_configs pxn_config "PXN 智能 P2P 配置 (P2P_LEVEL=$p2p_level)"
            ;;
    esac
}

# 性能优化配置预设
setup_performance_config() {
    local perf_type="$1"
    local opt_level="${2:-balanced}"  # 默认平衡模式
    
    case "$perf_type" in
        "nvlink_optimized")
            case "$opt_level" in
                "conservative")
                    # 保守配置（原有配置，保持兼容性）
                    declare -A nvlink_perf=(
                        ["ALGO"]="Ring,Tree"
                        ["PROTO"]="Simple"
                        ["NTHREADS"]="256"
                        ["MIN_NCHANNELS"]="16"
                        ["MAX_NCHANNELS"]="32"
                        ["TREE_THRESHOLD"]="0"
                        ["CUMEM_ENABLE"]="0"
                        ["NVLS_ENABLE"]="1"
                        ["NET_GDR_LEVEL"]="1"
                    )
                    set_nccl_configs nvlink_perf "NVLink 保守配置"
                    ;;
                "balanced")
                    # 平衡优化配置（推荐）
                    declare -A nvlink_perf=(
                        ["NTHREADS"]="384"
                        ["BUFFSIZE"]="12582912"  # 12MB
                        ["MIN_NCHANNELS"]="16"
                        ["MAX_NCHANNELS"]="32"
                        ["NVLS_ENABLE"]="1"
                        ["NET_GDR_LEVEL"]="2"
                        ["TREE_THRESHOLD"]="0"
                    )
                    set_nccl_configs nvlink_perf "NVLink 平衡配置"
                    # 移除部分限制性配置，启用自动选择
                    unset NCCL_ALGO NCCL_PROTO
                    log_info "✓ 启用算法和协议自动选择"
                    ;;
                "aggressive")
                    # 激进优化配置（最大性能）
                    declare -A nvlink_perf=(
                        ["NTHREADS"]="512"
                        ["BUFFSIZE"]="16777216"  # 16MB
                        ["MIN_NCHANNELS"]="16"
                        ["MAX_NCHANNELS"]="32"
                        ["NVLS_ENABLE"]="1"
                        ["NVLS_CHUNKSIZE"]="1048576"  # 1MB
                        ["NET_GDR_LEVEL"]="2"
                        ["CROSS_NIC"]="1"
                        ["CHECK_POINTERS"]="1"
                        ["TREE_THRESHOLD"]="0"
                    )
                    set_nccl_configs nvlink_perf "NVLink 激进配置"
                    # 完全移除限制性配置，启用 NCCL 完全自动优化
                    unset NCCL_ALGO NCCL_PROTO NCCL_CUMEM_ENABLE 
                    unset NCCL_TREE_THRESHOLD NCCL_NET_DISABLE
                    log_success "✓ 移除所有算法限制，启用 NCCL 完全自动优化"
                    ;;
            esac
            ;;
        "pcie_optimized")
            declare -A pcie_perf=(
                ["ALGO"]="Ring"
                ["MAX_NCHANNELS"]="16"
                ["MIN_NCHANNELS"]="1"
                ["NTHREADS"]="128"
                ["P2P_NET_CHUNKSIZE"]="131072"
                ["CUMEM_ENABLE"]="0"
                ["DMABUF_ENABLE"]="1"
                ["REG_CACHE_ENABLE"]="1"
                ["BUFFSIZE"]="8388608"  # 8MB
                ["NET_GDR_LEVEL"]="1"
            )
            set_nccl_configs pcie_perf "PCIe 性能优化"
            ;;
        "ib_optimized")
            declare -A ib_perf=(
                ["IB_TC"]="136"
                ["IB_SL"]="0"
                ["IB_TIMEOUT"]="22"
                ["IB_RETRY_CNT"]="7"
                ["IB_GID_INDEX"]="0"
                ["IB_PKEY"]="0"
                ["NET_GDR_LEVEL"]="2"
                ["BUFFSIZE"]="8388608"  # 8MB
            )
            set_nccl_configs ib_perf "InfiniBand 性能优化"
            ;;
        "ethernet_optimized")
            declare -A eth_perf=(
                ["NTHREADS"]="64"
                ["BUFFSIZE"]="4194304"  # 4MB
                ["MIN_NCHANNELS"]="1"
                ["MAX_NCHANNELS"]="8"
                ["NET_GDR_LEVEL"]="0"
                ["SOCKET_NTHREADS"]="8"
                ["NSOCKS_PERTHREAD"]="1"
            )
            set_nccl_configs eth_perf "以太网性能优化"
            ;;
        "shm_optimized")
            declare -A shm_perf=(
                ["NTHREADS"]="32"
                ["BUFFSIZE"]="2097152"  # 2MB
                ["MIN_NCHANNELS"]="1"
                ["MAX_NCHANNELS"]="4"
                ["NET_GDR_LEVEL"]="0"
                ["SHM_DISABLE"]="0"
                ["CUMEM_ENABLE"]="0"
            )
            set_nccl_configs shm_perf "共享内存性能优化"
            ;;
        "pxn_optimized")
            case "$opt_level" in
                "conservative")
                    declare -A pxn_perf=(
                        ["ALGO"]="Ring,Tree"
                        ["PROTO"]="Simple"
                        ["NTHREADS"]="256"
                        ["BUFFSIZE"]="8388608"  # 8MB
                        ["MIN_NCHANNELS"]="4"
                        ["MAX_NCHANNELS"]="12"
                        ["NET_GDR_LEVEL"]="1"
                        ["CROSS_NIC"]="0"
                    )
                    set_nccl_configs pxn_perf "PXN 保守配置"
                    ;;
                "balanced")
                    declare -A pxn_perf=(
                        ["ALGO"]="Ring,Tree,CollNet"
                        ["PROTO"]="Simple,LL"
                        ["NTHREADS"]="384"
                        ["BUFFSIZE"]="12582912"  # 12MB
                        ["MIN_NCHANNELS"]="6"
                        ["MAX_NCHANNELS"]="16"
                        ["NET_GDR_LEVEL"]="2"
                        ["CROSS_NIC"]="1"
                        ["P2P_NET_CHUNKSIZE"]="262144"
                    )
                    set_nccl_configs pxn_perf "PXN 平衡配置"
                    ;;
                "aggressive")
                    declare -A pxn_perf=(
                        ["NTHREADS"]="512"
                        ["BUFFSIZE"]="16777216"  # 16MB
                        ["MIN_NCHANNELS"]="8"
                        ["MAX_NCHANNELS"]="20"
                        ["NET_GDR_LEVEL"]="2"
                        ["CROSS_NIC"]="1"
                        ["P2P_NET_CHUNKSIZE"]="524288"
                        ["CHECK_POINTERS"]="1"
                        ["SOCKET_NTHREADS"]="16"
                        ["NSOCKS_PERTHREAD"]="2"
                    )
                    set_nccl_configs pxn_perf "PXN 激进配置"
                    # 启用完全自动优化
                    unset NCCL_ALGO NCCL_PROTO
                    log_success "✓ PXN 激进模式: 启用算法和协议自动选择"
                    ;;
            esac
            ;;
    esac
}

# 智能网络接口配置
setup_network_interface() {
    local interface_type="$1"
    
    case "$interface_type" in
        "auto_ethernet")
            if [ "$MULTI_NODE_MODE" = true ]; then
                # 智能检测物理网络接口
                local available_interfaces=""
                if command -v ip >/dev/null 2>&1; then
                    # 优先选择物理接口：eno*, eth*, enp*, ib*
                    available_interfaces=$(ip link show up | grep -E "^[0-9]+: (eno|eth|enp|ib)[0-9]+" | head -3 | cut -d: -f2 | cut -d@ -f1 | tr -d ' ' | tr '\n' ',' | sed 's/,$//')
                fi
                
                if [ -n "$available_interfaces" ]; then
                    set_nccl_config "SOCKET_IFNAME" "$available_interfaces" "多节点物理接口: $available_interfaces"
                    log_info "检测到物理网络接口: $available_interfaces"
                else
                    # 如果没有检测到物理接口，使用排除模式
                    set_nccl_config "SOCKET_IFNAME" "^docker0,lo,virbr0,veth,br-" "排除所有虚拟接口"
                    log_warning "未检测到物理接口，使用虚拟接口排除模式"
                fi
            else
                unset NCCL_SOCKET_IFNAME
                log_info "单节点模式: 使用 NCCL 自动接口选择"
            fi
            ;;
        "loopback_only")
            set_nccl_config "SOCKET_IFNAME" "lo" "仅使用回环接口"
            ;;
        "exclude_virtual")
            # 更全面的虚拟接口排除列表
            set_nccl_config "SOCKET_IFNAME" "^docker0,lo,virbr0,veth,br-,antrea-,kube-,vxlan" "排除所有虚拟接口"
            ;;
        "clear_interface")
            set_nccl_config "SOCKET_IFNAME" "" "清除接口限制"
            ;;
    esac
}

# 统一检测 GPU 拓扑 (使用缓存系统)
detect_gpu_topology() {
    # 确保系统信息已缓存
    cache_system_info
    
    # 从缓存获取信息
    DETECTED_GPU_COUNT=${SYSTEM_INFO_CACHE[gpu_count]}
    DETECTED_NVLINK_COUNT=${SYSTEM_INFO_CACHE[nvlink_count]}
    DETECTED_NVLINK_AVAILABLE=${SYSTEM_INFO_CACHE[nvlink_available]}
    
    log_info "GPU 拓扑检测: $DETECTED_GPU_COUNT 个GPU, $DETECTED_NVLINK_COUNT 个NVLink连接"
}

# 设置调试文件路径
setup_debug_files() {
    local network_type="$1"
    
    export NCCL_TOPO_DUMP_FILE="/tmp/nccl_topo_${network_type}.xml"
    export NCCL_GRAPH_DUMP_FILE="/tmp/nccl_graph_${network_type}.xml"
    export NCCL_DEBUG_FILE="/tmp/nccl_debug_${network_type}.%h.%p.log"
    
    log_info "调试文件: topo=${NCCL_TOPO_DUMP_FILE}, debug=${NCCL_DEBUG_FILE}"
}

# 显示帮助信息
show_help() {
    cat << EOF
NCCL 测试验证脚本 v${VERSION}

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
                                   多节点: InfiniBand > PXN > 以太网
                          ib       - 强制使用 InfiniBand/RoCE
                          pxn      - 强制使用 PXN (Process Exchange Network) 多节点模式
                          nvlink   - 强制使用 NVLink (单节点多GPU)
                          pcie     - 强制使用 PCIe P2P (单节点多GPU)
                          shm      - 强制使用共享内存 (单节点多GPU)
                          ethernet - 强制使用以太网 (TCP/IP)
                          socket   - 强制使用 Socket 传输
  --optimization-level LEVEL 优化级别 [默认: balanced]
                          conservative - 保守配置，稳定性优先
                          balanced     - 平衡配置，性能与稳定性兼顾 (推荐)
                          aggressive   - 激进配置，最大性能优化
  --dry-run               Dry-run 模式：检查环境、配置变量但不执行测试

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
    • 自动检测优先级: InfiniBand > PXN (Process Exchange Network) > 以太网
    • 推荐网络后端: auto (自动选择) > ib > pxn > ethernet > socket
    • 主要测试网络带宽和延迟
    • 适用于分布式训练场景
    • 必须设置环境变量: WORLD_SIZE, NODE_RANK, NPROC_PER_NODE
      - WORLD_SIZE: 总进程数 (节点数 × 每节点GPU数)
      - NODE_RANK: 当前节点编号 (0, 1, 2, ...)
      - NPROC_PER_NODE: 每节点GPU数

前置条件:
  • 建议先运行 ib_health_check.sh 确保 IB 网络正常
  • 可配合 ib_bandwidth_monitor.sh 监控测试期间的网络性能

示例:
  # 基础测试 (推荐使用 auto 模式)
  $0                                    # 自动检测最佳通信路径 (推荐)
  $0 --dry-run                         # Dry-run 模式：检查环境和配置但不执行测试
  
  # 单节点测试 (auto 模式会按优先级自动选择)
  $0 --network auto -s 1G -t 60        # 自动选择最佳路径，1GB 数据，60秒 (推荐)
  $0 --network nvlink -s 1G            # 强制使用 NVLink (如果可用)
  $0 --network pcie -s 1G              # 强制使用 PCIe P2P 通信
  $0 --network ib -s 10M               # 强制使用 InfiniBand
  $0 --network ethernet -s 1M          # 以太网兼容性测试
  $0 --network socket -s 1M            # Socket 调试模式
  
  # 多节点测试 (必须设置环境变量)
  # 节点0 (主节点):
  export WORLD_SIZE=8 NODE_RANK=0 NPROC_PER_NODE=4
  $0 -m --master-addr 192.168.1.100    # 自动选择网络 (推荐)
  
  # 节点1:
  export WORLD_SIZE=8 NODE_RANK=1 NPROC_PER_NODE=4
  $0 -m --master-addr 192.168.1.100 --network ib     # 强制使用 InfiniBand
  
  # 其他示例:
  export WORLD_SIZE=4 NODE_RANK=0 NPROC_PER_NODE=2
  $0 -m --master-addr 192.168.1.100 --network pxn    # 强制使用 PXN 模式
  $0 -m --master-addr 192.168.1.100 --network ethernet # 强制使用以太网
  $0 -m --master-addr 192.168.1.100 -s 100M -t 120   # 大数据长时间测试
  
  # 或者使用 nccl_multinode_launcher.sh (自动设置环境变量):
  ./nccl_multinode_launcher.sh --nodes 2 --gpus-per-node 4 --master-addr 192.168.1.100

EOF
}

# 显示版本信息
show_version() {
    echo "$SCRIPT_NAME v$VERSION"
    echo "专注于 NCCL 通信测试"
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
        auto|ib|pxn|nvlink|pcie|shm|ethernet|socket)
            log_success "网络后端: $NETWORK_BACKEND"
            ;;
        *)
            log_error "无效的网络后端: $NETWORK_BACKEND"
            log_info "支持的网络后端: auto, ib, pxn, nvlink, pcie, shm, ethernet, socket"
            validation_failed=true
            ;;
    esac
    
    # 验证优化级别
    case "$OPTIMIZATION_LEVEL" in
        conservative|balanced|aggressive)
            log_success "优化级别: $OPTIMIZATION_LEVEL"
            ;;
        *)
            log_error "无效的优化级别: $OPTIMIZATION_LEVEL"
            log_info "支持的优化级别: conservative, balanced, aggressive"
            validation_failed=true
            ;;
    esac
    
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
            --optimization-level)
                if [ -z "$2" ]; then
                    log_error "--optimization-level 选项需要参数"
                    exit 1
                fi
                OPTIMIZATION_LEVEL="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
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
        log_error "请安装 Python3: sudo apt-get install python3 (Ubuntu) 或 brew install python3 (macOS)"
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
            log_error "请安装支持 CUDA 的 PyTorch 版本"
            deps_ok=false
        fi
        
        # 检查 NCCL
        if python3 -c "import torch; torch.cuda.nccl.version()" 2>/dev/null; then
            local nccl_version=$(python3 -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null)
            log_success "NCCL 版本: $nccl_version"
        else
            log_error "无法获取 NCCL 版本信息，NCCL 可能未正确安装"
            deps_ok=false
        fi
    else
        log_error "PyTorch 未安装或不可用"
        log_error "请安装 PyTorch: pip3 install torch"
        deps_ok=false
    fi
    
    # 检查 NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi &>/dev/null; then
            local gpu_count=$(nvidia-smi -L | wc -l)
            if [ "$gpu_count" -eq 0 ]; then
                log_error "未检测到 NVIDIA GPU"
                deps_ok=false
            else
                log_success "检测到 $gpu_count 个 NVIDIA GPU"
                if [ "$QUIET_MODE" = false ]; then
                    nvidia-smi -L | while read line; do
                        log_info "  $line"
                    done
                fi
            fi
        else
            log_error "nvidia-smi 执行失败，可能是驱动问题"
            log_error "请检查 NVIDIA 驱动安装: nvidia-smi"
            deps_ok=false
        fi
    else
        log_error "nvidia-smi 命令不可用"
        log_error "请安装 NVIDIA 驱动和 CUDA 工具包"
        deps_ok=false
    fi
    
    # 检查 InfiniBand 硬件可用性（仅警告，不强制退出）
    if command -v ibv_devinfo >/dev/null 2>&1; then
        local ib_output
        if ib_output=$(ibv_devinfo 2>/dev/null) && [ -n "$ib_output" ]; then
            # 检查是否有实际的IB设备
            if echo "$ib_output" | grep -q "hca_id:"; then
                log_success "InfiniBand 设备可用"
            else
                log_warning "InfiniBand 工具可用但未检测到硬件设备"
            fi
        else
            log_warning "InfiniBand 设备不可用 (可能影响性能)"
        fi
    else
        log_warning "InfiniBand 工具未安装 (建议先运行 ib_health_check.sh)"
    fi
    
    if [ "$deps_ok" = true ]; then
        log_success "NCCL 环境依赖检查通过"
        return 0
    else
        log_error "NCCL 环境依赖检查失败，无法继续执行"
        log_error "请解决上述环境问题后重新运行脚本"
        exit 1
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
        "pxn")
            setup_pxn_network
            ;;
        "nvlink")
            setup_nvlink_network
            ;;
        "pcie")
            setup_pcie_network
            ;;
        "shm")
            setup_shm_network
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
        # 检查是否已经设置了特定的物理接口
        if [ -n "${NCCL_SOCKET_IFNAME:-}" ] && [[ ! "${NCCL_SOCKET_IFNAME}" =~ ^\^ ]]; then
            # 已经设置了物理接口，保持不变
            log_info "多节点模式: 保持已配置的物理接口 ($NCCL_SOCKET_IFNAME)"
        else
            # 未设置物理接口或使用排除模式，应用默认排除配置
            export NCCL_SOCKET_IFNAME=^docker0,lo,virbr0,veth,br-,antrea-,kube-,vxlan
            log_info "多节点模式: 排除虚拟接口 (docker0,lo,virbr0,veth,br-,antrea-,kube-,vxlan)"
        fi
    fi
    
    # 显示最终配置摘要
    display_nccl_config_summary
}

# 自动检测网络配置 - 按照 NCCL 优先级
setup_auto_network() {
    log_info "自动检测网络环境 (按 NCCL 优先级: NVLink > PCIe P2P > 共享内存 > 网络传输 > PXN)..."
    
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
                    local nvlink_count=$(nvidia-smi nvlink --status | grep -c "GB/s" 2>/dev/null)
    nvlink_count=${nvlink_count:-0}
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
                        configure_nvlink_settings "$OPTIMIZATION_LEVEL"
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
        local p2p_verified=false
        if command -v nvidia-smi >/dev/null 2>&1; then
            local gpu_count=$(nvidia-smi -L | wc -l)
            if [ "$gpu_count" -gt 1 ]; then
                log_info "检测到 $gpu_count 个 GPU，检查 PCIe P2P 支持..."
                
                # 尝试检查 P2P 拓扑
                if nvidia-smi topo -p2p r >/dev/null 2>&1; then
                    local p2p_matrix=$(nvidia-smi topo -p2p r 2>/dev/null)
                    if echo "$p2p_matrix" | grep -q "OK"; then
                        p2p_available=true
                        p2p_verified=true
                        log_success "PCIe P2P 拓扑验证成功"
                    else
                        log_warning "PCIe P2P 拓扑检查显示不支持 P2P"
                    fi
                else
                    # 如果无法检查拓扑，基于 GPU 型号进行推测
                    local gpu_model=$(nvidia-smi -L | head -1 | grep -oE "Tesla|Quadro|RTX|GTX|A[0-9]+|H[0-9]+|V[0-9]+")
                    if echo "$gpu_model" | grep -qE "Tesla|Quadro|A[0-9]+|H[0-9]+|V[0-9]+"; then
                        p2p_available=true
                        log_info "检测到企业级 GPU ($gpu_model)，假设支持 PCIe P2P"
                    else
                        log_warning "检测到消费级 GPU ($gpu_model)，P2P 支持不确定"
                        p2p_available=true  # 仍然尝试，让 NCCL 自己决定
                    fi
                fi
                
                if [ "$p2p_available" = true ]; then
                    log_success "自动选择 PCIe P2P 通信 (第二优先级)"
                    configure_pcie_p2p_settings
                    return 0
                fi
            fi
        fi
        log_info "PCIe P2P 检测: $([ "$p2p_available" = true ] && echo "可用" || echo "不可用")$([ "$p2p_verified" = true ] && echo " (已验证)" || echo "")"
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
    
    # ========== 第四优先级：网络传输 (InfiniBand > PXN > 以太网) ==========
    log_info "检测网络传输选项..."
    
    # 检测 InfiniBand 设备
    local has_ib=false
    local network_type="未知"
    local is_roce=false
    local ib_error=""
    
    # 检查 InfiniBand 工具是否可用
    if ! command -v ibv_devinfo >/dev/null 2>&1; then
        ib_error="ibv_devinfo 命令不可用"
        log_info "InfiniBand 检测: $ib_error"
    else
        # 尝试获取 IB 设备信息
        local ib_output
        if ib_output=$(ibv_devinfo 2>/dev/null); then
            if [ -n "$ib_output" ]; then
                local link_layer=$(echo "$ib_output" | grep "link_layer:" | head -1 | awk '{print $2}')
                if [ "$link_layer" = "Ethernet" ]; then
                    has_ib=true
                    is_roce=true
                    network_type="RoCE (Ethernet over IB)"
                    log_info "检测到 RoCE 环境"
                elif [ "$link_layer" = "InfiniBand" ]; then
                    has_ib=true
                    network_type="原生 InfiniBand"
                    log_info "检测到原生 IB 环境"
                else
                    ib_error="未识别的链路层类型: $link_layer"
                fi
            else
                ib_error="未找到 InfiniBand 设备"
            fi
        else
            ib_error="ibv_devinfo 执行失败"
        fi
    fi
    
    if [ "$has_ib" = true ]; then
        log_success "自动选择 InfiniBand 网络 (网络传输最高优先级)"
        configure_infiniband_settings "$is_roce" "$network_type"
    else
        log_warning "InfiniBand 不可用: $ib_error，检测 PXN 支持..."
        
        # ========== 第五优先级：PXN (Process Exchange Network) 多节点模式 ==========
        if [ "$MULTI_NODE_MODE" = true ]; then
            local pxn_available=false
            local pxn_reason=""
            
            # 检查是否有高速网络接口支持 PXN
            if command -v ip >/dev/null 2>&1; then
                local high_speed_interfaces=$(ip link show up 2>/dev/null | grep -E "^[0-9]+: (eth|en|ib)[0-9]+" | head -3)
                if [ -n "$high_speed_interfaces" ]; then
                    # 检查是否配置了主节点地址
                    if [ -n "$MASTER_ADDR" ]; then
                        pxn_available=true
                        pxn_reason="检测到高速网络接口且配置了主节点地址"
                        log_success "自动选择 PXN 模式 (多节点高性能通信)"
                        log_info "PXN 检测: $pxn_reason"
                        configure_pxn_settings
                        return 0
                    else
                        pxn_reason="缺少主节点地址配置 (--master-addr)"
                    fi
                else
                    pxn_reason="未检测到高速网络接口"
                fi
            else
                pxn_reason="无法检查网络接口状态"
            fi
            
            log_info "PXN 检测: 不可用 ($pxn_reason)，回退到以太网"
        else
            log_info "单节点模式: 跳过 PXN 检测"
        fi
        
        log_success "自动选择以太网传输 (网络传输备选)"
        configure_ethernet_settings
    fi
}

# 强制使用 InfiniBand 网络
setup_infiniband_network() {
    log_info "强制使用 InfiniBand 网络..."
    
    # ========== 硬件检查 ==========
    local has_ib=false
    local is_roce=false
    local network_type="InfiniBand (强制)"
    local ib_error=""
    
    # 检查 InfiniBand 工具是否可用
    if ! command -v ibv_devinfo >/dev/null 2>&1; then
        ib_error="ibv_devinfo 命令不可用，请安装 InfiniBand 驱动和工具"
        log_error "硬件检查失败: $ib_error"
        log_error "无法强制使用 InfiniBand 网络"
        log_info "解决方案:"
        log_info "  1. 安装 InfiniBand 驱动: apt-get install infiniband-diags"
        log_info "  2. 或使用其他网络后端: --network ethernet 或 --network auto"
        exit 1
    fi
    
    # 尝试获取 IB 设备信息
    local ib_output
    if ib_output=$(ibv_devinfo 2>/dev/null); then
        if [ -n "$ib_output" ]; then
            local link_layer=$(echo "$ib_output" | grep "link_layer:" | head -1 | awk '{print $2}')
            if [ "$link_layer" = "Ethernet" ]; then
                has_ib=true
                is_roce=true
                network_type="RoCE (强制)"
                log_success "检测到 RoCE 设备"
            elif [ "$link_layer" = "InfiniBand" ]; then
                has_ib=true
                network_type="原生 InfiniBand (强制)"
                log_success "检测到原生 InfiniBand 设备"
            else
                ib_error="未识别的链路层类型: $link_layer"
            fi
        else
            ib_error="未找到 InfiniBand 设备"
        fi
    else
        ib_error="ibv_devinfo 执行失败，可能没有 InfiniBand 硬件"
    fi
    
    # 如果没有检测到 IB 设备，直接退出
    if [ "$has_ib" = false ]; then

        log_error "硬件检查失败: $ib_error"
        log_error "无法强制使用 InfiniBand 网络"
        log_info "解决方案:"
        log_info "  1. 检查 InfiniBand 硬件是否正确安装"
        log_info "  2. 检查 InfiniBand 驱动是否正确加载"
        log_info "  3. 或使用其他网络后端: --network ethernet 或 --network auto"
        exit 1
    fi
    
    # 额外检查：验证 HCA 设备状态
    if command -v ibstat >/dev/null 2>&1; then
        local ibstat_output
        if ibstat_output=$(ibstat 2>/dev/null); then
            if ! echo "$ibstat_output" | grep -q "State: Active"; then
                log_error "InfiniBand 设备未处于活跃状态"
                log_error "当前设备状态:"
                echo "$ibstat_output" | grep "State:" | head -3 | while read line; do
                    log_info "  $line"
                done
                log_info "解决方案:"
                log_info "  1. 检查 InfiniBand 网络连接"
                log_info "  2. 重启 InfiniBand 服务: systemctl restart openibd"
                log_info "  3. 或使用其他网络后端: --network ethernet"
                exit 1
            else
                log_success "InfiniBand 设备状态正常"
            fi
        fi
    fi
    
    configure_infiniband_settings "$is_roce" "$network_type"
}

# 强制使用 NVLink 传输
setup_nvlink_network() {
    log_info "强制使用 NVLink 传输..."
    
    # ========== 基础条件检查 ==========
    # NVLink 仅适用于单节点多GPU场景
    if [ "$MULTI_NODE_MODE" = true ]; then
        log_error "NVLink 仅支持单节点多GPU模式，不支持多节点"
        log_error "无法强制使用 NVLink 网络"
        log_info "解决方案:"
        log_info "  1. 使用单节点模式 (移除 -m 或 --multi-node 参数)"
        log_info "  2. 或使用多节点兼容的网络后端: --network ib 或 --network ethernet"
        exit 1
    fi
    
    # ========== 硬件检查 ==========
    local nvlink_available=false
    local nvlink_active=false
    local gpu_count=0
    local nvlink_error=""
    
    # 检查 nvidia-smi 是否可用
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        nvlink_error="nvidia-smi 命令不可用，请安装 NVIDIA 驱动"
        log_error "硬件检查失败: $nvlink_error"
        log_error "无法强制使用 NVLink 网络"
        log_info "解决方案:"
        log_info "  1. 安装 NVIDIA 驱动"
        log_info "  2. 或使用其他网络后端: --network auto 或 --network ethernet"
        exit 1
    fi
    
    # 检查 GPU 数量
    if ! gpu_count=$(nvidia-smi -L | wc -l 2>/dev/null); then
        nvlink_error="无法获取 GPU 信息"
        log_error "硬件检查失败: $nvlink_error"
        log_error "无法强制使用 NVLink 网络"
        log_info "解决方案:"
        log_info "  1. 安装 NVIDIA 驱动"
        log_info "  2. 或使用其他网络后端: --network auto 或 --network pcie"
        exit 1
    fi
    
    if [ "$gpu_count" -lt 2 ]; then
        nvlink_error="检测到 $gpu_count 个 GPU，NVLink 需要至少 2 个 GPU"
        log_error "硬件检查失败: $nvlink_error"
        log_error "无法强制使用 NVLink 网络"
        log_info "解决方案:"
        log_info "  1. 使用多 GPU 系统"
        log_info "  2. 或使用单 GPU 兼容的网络后端: --network ethernet"
        exit 1
    fi
    
    log_success "检测到 $gpu_count 个 GPU"
    
    # 检查 NVLink 硬件可用性
    if nvidia-smi nvlink --status &>/dev/null; then
        # 检测活跃的 NVLink 连接
        local nvlink_count=$(nvidia-smi nvlink --status 2>/dev/null | grep -c "GB/s" 2>/dev/null)
        nvlink_count=${nvlink_count:-0}
        nvlink_count=$(echo "$nvlink_count" | tr -d ' \n\r\t')
        
        if [[ "$nvlink_count" =~ ^[0-9]+$ ]] && [ "$nvlink_count" -gt 0 ]; then
            nvlink_available=true
            nvlink_active=true
            local avg_bandwidth=$(nvidia-smi nvlink --status 2>/dev/null | grep "GB/s" | head -1 | grep -oE "[0-9]+\.[0-9]+ GB/s" | head -1)
            log_success "检测到 $nvlink_count 个活跃的 NVLink 连接 (带宽: $avg_bandwidth)"
        else
            # 检查 NVLink 硬件拓扑
            local topo_output=$(nvidia-smi topo -m 2>/dev/null || echo "")
            if [ -n "$topo_output" ] && echo "$topo_output" | grep -qE "NV[0-9]+"; then
                nvlink_available=true
                nvlink_error="NVLink 硬件可用但当前未激活，可能被其他进程占用或需要GPU负载触发"
                log_warning "$nvlink_error"
                nvlink_error="未检测到 NVLink 硬件拓扑"
            fi
        fi
    else
        nvlink_error="nvidia-smi nvlink 命令执行失败，可能不支持 NVLink"
    fi
    
    # 如果 NVLink 不可用，直接退出
    if [ "$nvlink_available" = false ] || [ "$nvlink_active" = false ]; then
        log_error "硬件检查失败: $nvlink_error"
        log_error "无法强制使用 NVLink 网络"
        log_info "解决方案:"
        log_info "  1. 安装 NVIDIA 驱动"
        log_info "  2. 检查 NVLink 硬件连接"
        log_info "  3. 或使用其他网络后端: --network auto 或 --network pcie"
        exit 1
    fi
    
    configure_nvlink_settings "$OPTIMIZATION_LEVEL"
}

# 强制使用 PCIe P2P 传输
setup_pcie_network() {
    log_info "强制使用 PCIe P2P 传输..."
    
    # ========== 基础条件检查 ==========
    # PCIe P2P 仅适用于单节点多GPU场景
    if [ "$MULTI_NODE_MODE" = true ]; then
        log_error "PCIe P2P 仅支持单节点多GPU模式，不支持多节点"
        log_error "无法强制使用 PCIe P2P 网络"
        log_info "解决方案:"
        log_info "  1. 使用单节点模式 (移除 -m 或 --multi-node 参数)"
        log_info "  2. 或使用多节点兼容的网络后端: --network ib 或 --network ethernet"
        exit 1
    fi
    
    # ========== 硬件检查 ==========
    local gpu_count=0
    local pcie_error=""
    
    # 检查 nvidia-smi 是否可用
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        pcie_error="nvidia-smi 命令不可用，请安装 NVIDIA 驱动"
        log_error "硬件检查失败: $pcie_error"
        log_error "无法强制使用 PCIe P2P 网络"
        log_info "解决方案:"
        log_info "  1. 安装 NVIDIA 驱动"
        log_info "  2. 或使用其他网络后端: --network auto 或 --network ethernet"
        exit 1
    fi
    
    # 检查 GPU 数量
    if ! gpu_count=$(nvidia-smi -L | wc -l 2>/dev/null); then
        pcie_error="无法获取 GPU 信息"
        log_error "硬件检查失败: $pcie_error"
        log_error "无法强制使用 PCIe P2P 网络"
        exit 1
    fi
    
    if [ "$gpu_count" -lt 2 ]; then
        pcie_error="检测到 $gpu_count 个 GPU，PCIe P2P 需要至少 2 个 GPU"
        log_error "硬件检查失败: $pcie_error"
        log_error "无法强制使用 PCIe P2P 网络"
        log_info "解决方案:"
        log_info "  1. 使用多 GPU 系统"
        log_info "  2. 或使用单 GPU 兼容的网络后端: --network ethernet"
        exit 1
    fi
    
    log_success "检测到 $gpu_count 个 GPU，PCIe P2P 通信可用"
    
    # 检查是否有 NVLink 连接（如果有 NVLink，建议使用 NVLink 而不是 PCIe）
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi nvlink --status &>/dev/null; then
        local nvlink_count=$(nvidia-smi nvlink --status 2>/dev/null | grep -c "GB/s" 2>/dev/null)
        nvlink_count=${nvlink_count:-0}
        if [[ "$nvlink_count" =~ ^[0-9]+$ ]] && [ "$nvlink_count" -gt 0 ]; then
            log_warning "检测到 $nvlink_count 个活跃的 NVLink 连接"
            log_warning "建议使用 --network nvlink 以获得更好的性能"
            log_info "继续使用 PCIe P2P (性能可能不如 NVLink)..."
        fi
    fi
    
    configure_pcie_p2p_settings
}

# 强制使用以太网
setup_ethernet_network() {
    log_info "强制使用以太网 (TCP/IP)..."
    
    # ========== 硬件检查 ==========
    local ethernet_error=""
    local available_interfaces=""
    
    # 检查网络接口是否可用 (支持 Linux 和 macOS)
    if ! command -v ip >/dev/null 2>&1; then
        # 如果 ip 命令不可用，尝试使用 ifconfig (macOS/BSD)
        if ! command -v ifconfig >/dev/null 2>&1; then
            ethernet_error="ip 和 ifconfig 命令都不可用，无法检查网络接口"
            log_error "硬件检查失败: $ethernet_error"
            log_error "无法强制使用以太网"
            exit 1
        fi
        
        # 使用 ifconfig 检查网络接口 (macOS/BSD)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS: 查找 en* 接口
            available_interfaces=$(ifconfig -l | tr ' ' '\n' | grep -E "^en[0-9]+" | head -5)
        else
            # 其他 BSD 系统: 查找 eth*, en*, em* 接口
            available_interfaces=$(ifconfig -l | tr ' ' '\n' | grep -E "^(eth|en|em)[0-9]+" | head -5)
        fi
    else
        # 使用 ip 命令检查网络接口 (Linux)
        if ! available_interfaces=$(ip link show up 2>/dev/null | grep -E "^[0-9]+: (eth|en|ib)" | cut -d: -f2 | cut -d@ -f1 | tr -d ' ' 2>/dev/null); then
            ethernet_error="无法获取网络接口信息"
            log_error "硬件检查失败: $ethernet_error"
            log_error "无法强制使用以太网"
            exit 1
        fi
    fi
    
    # 检查是否有物理网络接口
    if [ -z "$available_interfaces" ]; then
        ethernet_error="未检测到可用的物理网络接口 (eth*, en*, ib*)"
        log_error "硬件检查失败: $ethernet_error"
        log_error "无法强制使用以太网"
        log_info "解决方案:"
        log_info "  1. 检查网络接口是否启用"
        log_info "  2. 或使用其他网络后端: --network socket"
        exit 1
    fi
    
    log_success "检测到可用的网络接口: $(echo $available_interfaces | tr '\n' ' ')"
    
    configure_ethernet_settings
}

# 强制使用 Socket 传输
setup_socket_network() {
    log_info "强制使用 Socket 传输..."
    
    # Socket 传输基本上在所有系统上都可用，只需检查 loopback 接口
    local socket_error=""
    
    # 检查 loopback 接口
    if ! command -v ip >/dev/null 2>&1; then
        # 如果 ip 命令不可用，尝试使用 ifconfig
        if ! command -v ifconfig >/dev/null 2>&1; then
            log_warning "ip 和 ifconfig 命令都不可用，无法检查 loopback 接口"
            log_warning "假设 loopback 接口可用并继续..."
        else
            # 使用 ifconfig 检查 loopback 接口 (支持 Linux 和 macOS)
            local lo_interface="lo"
            if [[ "$OSTYPE" == "darwin"* ]]; then
                lo_interface="lo0"
            fi
            
            if ! ifconfig "$lo_interface" >/dev/null 2>&1; then
                socket_error="loopback 接口不可用"
                log_error "硬件检查失败: $socket_error"
                log_error "无法强制使用 Socket 传输"
                exit 1
            fi
        fi
    else
        # 使用 ip 命令检查 loopback 接口
        if ! ip link show lo up >/dev/null 2>&1; then
            socket_error="loopback 接口不可用"
            log_error "硬件检查失败: $socket_error"
            log_error "无法强制使用 Socket 传输"
            exit 1
        fi
    fi
    
    log_success "Socket 传输可用"
    
    configure_socket_settings
}

# 强制使用 PXN (Process Exchange Network) 模式
setup_pxn_network() {
    log_info "强制使用 PXN (Process Exchange Network) 模式..."
    
    # ========== 基础条件检查 ==========
    # PXN 仅适用于多节点场景
    if [ "$MULTI_NODE_MODE" = false ]; then
        log_error "PXN 模式仅支持多节点分布式训练，不支持单节点"
        log_error "无法强制使用 PXN 网络"
        log_info "解决方案:"
        log_info "  1. 使用多节点模式 (添加 -m 或 --multi-node 参数)"
        log_info "  2. 或使用单节点兼容的网络后端: --network nvlink 或 --network pcie"
        exit 1
    fi
    
    # ========== 硬件检查 ==========
    local pxn_available=false
    local pxn_error=""
    local network_interfaces=""
    
    # 检查高速网络接口可用性 (InfiniBand 或高速以太网)
    if command -v ibv_devinfo >/dev/null 2>&1; then
        local ib_output
        if ib_output=$(ibv_devinfo 2>/dev/null) && echo "$ib_output" | grep -q "hca_id:"; then
            pxn_available=true
            log_success "检测到 InfiniBand 设备，PXN 可使用 IB 作为底层传输"
        fi
    fi
    
    # 如果没有 InfiniBand，检查高速以太网接口
    if [ "$pxn_available" = false ]; then
        if command -v ip >/dev/null 2>&1; then
            # 检查 10GbE 或更高速度的网络接口
            network_interfaces=$(ip link show up 2>/dev/null | grep -E "^[0-9]+: (eth|en|ib)[0-9]+" | head -5)
            if [ -n "$network_interfaces" ]; then
                pxn_available=true
                log_success "检测到高速网络接口，PXN 可使用以太网作为底层传输"
            else
                pxn_error="未检测到可用的高速网络接口"
            fi
        else
            pxn_error="无法检查网络接口状态"
        fi
    fi
    
    # 检查 MASTER_ADDR 配置
    if [ -z "$MASTER_ADDR" ]; then
        pxn_error="PXN 模式需要指定主节点地址 (--master-addr)"
        log_error "配置检查失败: $pxn_error"
        log_error "无法启用 PXN 模式"
        log_info "解决方案:"
        log_info "  1. 指定主节点地址: --master-addr <IP地址>"
        log_info "  2. 确保所有节点都能访问该地址"
        exit 1
    fi
    
    # 如果 PXN 不可用，直接退出
    if [ "$pxn_available" = false ]; then
        log_error "硬件检查失败: $pxn_error"
        log_error "无法强制使用 PXN 网络"
        log_info "解决方案:"
        log_info "  1. 安装 InfiniBand 硬件和驱动"
        log_info "  2. 配置高速以太网接口 (10GbE+)"
        log_info "  3. 或使用其他网络后端: --network ib 或 --network ethernet"
        exit 1
    fi
    
    configure_pxn_settings
}

# 配置 InfiniBand 相关设置
configure_infiniband_settings() {
    local is_roce="$1"
    local network_type="$2"
    
    log_info "配置 InfiniBand 设置: $network_type"
    
    # 使用统一配置管理器
    setup_common_nccl_config
    setup_debug_files "ib"
    setup_network_config "ib_enable"
    
    # HCA 设备自动检测
    local hca_name=""
    if command -v ibstat >/dev/null 2>&1; then
        hca_name=$(ibstat 2>/dev/null | grep "CA '" | head -1 | awk '{print $2}' | tr -d "'")
    fi
    
    if [ -n "$hca_name" ]; then
        set_nccl_config "IB_HCA" "$hca_name" "HCA 设备"
    else
        log_warning "无法获取 HCA 设备名，使用 NCCL 自动检测"
    fi
    
    # 根据 IB 类型配置参数
    if [ "$is_roce" = true ]; then
        set_nccl_config "IB_GID_INDEX" "3" "RoCE 配置"
        setup_network_interface "clear_interface"
    else
        set_nccl_config "IB_GID_INDEX" "0" "原生 IB 配置"
    fi
    
    # IB 性能优化
    setup_performance_config "ib_optimized"
}

# 配置以太网设置
configure_ethernet_settings() {
    log_info "配置以太网 (TCP/IP) 设置"
    
    # 使用统一配置管理器
    setup_common_nccl_config
    setup_debug_files "ethernet"
    setup_network_config "ib_disable"
    
    # 智能网络接口配置
    setup_network_interface "auto_ethernet"
    
    # P2P 配置
    setup_network_config "p2p_pcie"
    
    # 以太网性能优化
    setup_performance_config "ethernet_optimized"
    
    log_success "以太网配置完成 - 将使用 TCP/IP 进行节点间通信"
}

# 配置 Socket 传输设置
configure_socket_settings() {
    log_info "配置 Socket 传输设置"
    
    # 使用统一配置管理器
    setup_common_nccl_config
    setup_debug_files "socket"
    setup_network_config "socket_only"
    
    # 配置 Socket 接口
    if [ "$MULTI_NODE_MODE" = true ]; then
        setup_network_interface "exclude_virtual"
    else
        setup_network_interface "loopback_only"
    fi
    
    # 容器环境特殊配置
    if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
        log_warning "检测到容器环境，应用额外的 Socket 强制配置"
        declare -A container_config=(
            ["NET_DISABLE"]="0"
            ["SOCKET_FORCE"]="1"
            ["IGNORE_DISABLED_P2P"]="1"
            ["CUMEM_ENABLE"]="0"
        )
        set_nccl_configs container_config "容器环境 Socket 配置"
    fi
    
    # 使用标准输出进行调试
    set_nccl_config "DEBUG_FILE" "" "使用标准输出调试"
    set_nccl_config "CHECK_DISABLE" "0" "启用检查"
    
    log_warning "Socket 传输模式 - 性能可能较低，主要用于调试"
    log_info "预期性能: <1 GB/s 吞吐量（受网络栈限制）"
}

# 配置 PXN (Process Exchange Network) 设置
configure_pxn_settings() {
    log_info "配置 PXN (Process Exchange Network) 设置"
    
    # 使用统一配置管理器
    setup_common_nccl_config
    setup_debug_files "pxn"
    
    # 检测 GPU 拓扑以智能选择 P2P_LEVEL
    detect_gpu_topology
    
    # 应用 PXN 网络配置预设（智能 P2P 配置）
    setup_network_config "pxn_enable"
    
    # 应用 PXN 性能优化配置
    setup_performance_config "pxn_optimized" "$OPTIMIZATION_LEVEL"
    
    # 智能网络接口配置 - PXN 需要高性能网络
    setup_network_interface "auto_ethernet"
    
    # PXN 环境变量设置
    export NCCL_PXN_DISABLE=0  # 确保 PXN 启用
    export NCCL_COLLNET_NODE_THRESHOLD=2  # 2个或更多节点时启用集合通信
    export NCCL_COLLNET_CHAIN_THRESHOLD=2  # 链式通信阈值
    
    # 多节点特定配置
    if [ -n "$MASTER_ADDR" ]; then
        log_info "PXN 主节点地址: $MASTER_ADDR:$MASTER_PORT"
        # 设置分布式训练环境变量
        export MASTER_ADDR="$MASTER_ADDR"
        export MASTER_PORT="$MASTER_PORT"
        export WORLD_SIZE="${WORLD_SIZE:-2}"  # 默认2个节点
        export RANK="${RANK:-0}"  # 默认为主节点
    fi
    
    # 结果总结
    log_success "✅ PXN 配置完成 - 多节点高性能通信模式"
    
    # 显示智能 P2P 选择结果
    if [ "$DETECTED_NVLINK_AVAILABLE" = true ]; then
        log_success "节点内通信: NVLink (P2P_LEVEL=NVL, $DETECTED_NVLINK_COUNT 个连接)"
        log_info "节点内预期带宽: ~900 GB/s, 延迟: < 1 μs"
    else
        log_info "节点内通信: PCIe P2P (P2P_LEVEL=PIX)"
        log_info "节点内预期带宽: ~64 GB/s, 延迟: 2-5 μs"
    fi
    log_info "节点间通信: PXN 集合通信 + 高速网络"
    
    case "$OPTIMIZATION_LEVEL" in
        "conservative")
            log_success "保守模式: 稳定性优先, 预期性能: ~50 GB/s"
            ;;
        "balanced")
            log_success "平衡模式: 性能与稳定性兼顾, 预期性能: ~100 GB/s"
            ;;
        "aggressive")
            log_success "激进模式: 最大性能优化, 预期性能: >200 GB/s"
            ;;
    esac
    log_info "优化级别: $OPTIMIZATION_LEVEL"
    log_info "主节点: $MASTER_ADDR:$MASTER_PORT"
    log_info "支持算法: Ring, Tree, CollNet"
    log_info "缓冲区大小: $(echo $NCCL_BUFFSIZE | numfmt --to=iec 2>/dev/null || echo $NCCL_BUFFSIZE)"
    
    # 特殊提示
    log_warning "⚠️  PXN 模式需要在所有参与节点上运行此脚本"
    log_warning "⚠️  确保所有节点的网络配置一致"
    log_info "💡 建议: 配合 ib_bandwidth_monitor.sh 监控网络性能"
}

# 强制使用共享内存网络
setup_shm_network() {
    log_info "强制使用共享内存传输..."
    
    # 检查是否为单节点模式
    if [ "$MULTI_NODE_MODE" = true ]; then
        log_error "共享内存传输仅支持单节点模式"
        log_error "多节点环境无法使用共享内存进行跨节点通信"
        log_info "解决方案:"
        log_info "  1. 使用 --network ethernet 进行多节点通信"
        log_info "  2. 使用 --network ib 进行 InfiniBand 通信"
        exit 1
    fi
    
    # 检查 GPU 数量
    local gpu_count=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count=$(nvidia-smi -L | wc -l)
        if [ "$gpu_count" -lt 2 ]; then
            log_warning "检测到 $gpu_count 个 GPU，共享内存通信需要至少 2 个 GPU"
            log_warning "单 GPU 环境下共享内存测试意义有限"
        else
            log_success "检测到 $gpu_count 个 GPU，适合共享内存通信测试"
        fi
    else
        log_warning "nvidia-smi 不可用，无法检查 GPU 数量"
    fi
    
    # 检查共享内存可用性
    local shm_available=false
    local shm_error=""
    
    # 检查 /dev/shm 挂载点
    if [ -d "/dev/shm" ] && [ -w "/dev/shm" ]; then
        # 检查 /dev/shm 的大小
        local shm_size
        if command -v df >/dev/null 2>&1; then
            shm_size=$(df -h /dev/shm 2>/dev/null | tail -1 | awk '{print $2}')
            if [ -n "$shm_size" ]; then
                log_success "共享内存可用: /dev/shm ($shm_size)"
                shm_available=true
            else
                shm_error="/dev/shm 大小检查失败"
            fi
        else
            log_warning "df 命令不可用，无法检查 /dev/shm 大小"
            shm_available=true  # 假设可用
        fi
    else
        shm_error="/dev/shm 不可用或不可写"
    fi
    
    if [ "$shm_available" = false ]; then
        log_error "硬件检查失败: $shm_error"
        log_error "无法强制使用共享内存传输"
        exit 1
    fi
    
    log_success "共享内存传输可用"
    
    configure_shm_settings
}

# 配置 NVLink 传输设置
configure_nvlink_settings() {
    local opt_level="${1:-balanced}"  # 默认平衡模式
    log_info "配置 NVLink 传输设置（优化级别: $opt_level）"
    
    # 使用统一配置管理器
    setup_common_nccl_config
    detect_gpu_topology
    setup_debug_files "nvlink"
    setup_network_config "p2p_nvlink"
    setup_network_interface "loopback_only"
    setup_performance_config "nvlink_optimized" "$opt_level"
    
    # 结果总结
    if [ "$DETECTED_NVLINK_AVAILABLE" = true ]; then
        log_success "✅ NVLink 配置完成 - 检测到 $DETECTED_NVLINK_COUNT 个活跃连接"
        case "$opt_level" in
            "conservative")
                log_success "保守模式: 稳定性优先, 预期性能: ~200 GB/s"
                ;;
            "balanced")
                log_success "平衡模式: 性能与稳定性兼顾, 预期性能: ~400 GB/s"
                ;;
            "aggressive")
                log_success "激进模式: 最大性能优化, 预期性能: >600 GB/s"
                ;;
        esac
    else
        log_warning "⚠️  NVLink 强制配置完成 - 硬件检测失败，NCCL 将尝试回退"
    fi
}

# 配置 PCIe P2P 传输设置（智能检测 NVLink 和 PCIe）
configure_pcie_p2p_settings() {
    log_info "配置 GPU 间高速传输设置（智能检测 NVLink/PCIe P2P）"
    
    # 使用统一配置管理器
    setup_common_nccl_config
    detect_gpu_topology
    setup_network_config "ib_disable"
    
    # 智能配置 P2P 级别
    if [ "$DETECTED_NVLINK_AVAILABLE" = true ]; then
        setup_network_config "p2p_nvlink"
        set_nccl_config "NVLS_CHUNKSIZE" "524288" "NVLink 块大小"
        set_nccl_config "ALGO" "Auto" "自动算法选择"
        set_nccl_config "MAX_NCHANNELS" "32" "最大通道数"
        setup_debug_files "nvlink"
        log_success "设置 NCCL_P2P_LEVEL=NVL (优先 NVLink，检测到 $DETECTED_NVLINK_COUNT 个连接)"
    else
        setup_network_config "p2p_pcie"
        set_nccl_config "ALGO" "Ring" "Ring 算法"
        set_nccl_config "MAX_NCHANNELS" "16" "最大通道数"
        setup_debug_files "pcie"
        log_info "设置 NCCL_P2P_LEVEL=PIX (使用 PCIe P2P)"
    fi
    
    # P2P 优化配置
    setup_performance_config "pcie_optimized"
    
    # ACS 检测和警告
    if command -v lspci >/dev/null 2>&1; then
        local acs_enabled=$(sudo lspci -vvv 2>/dev/null | grep "ACSCtl.*SrcValid+" | wc -l || echo "0")
        if [ "$acs_enabled" -gt 0 ]; then
            log_warning "检测到 ACS 可能已启用，可能影响 P2P 性能"
        fi
    fi
    
    # 配置完成总结
    if [ "$DETECTED_GPU_COUNT" -gt 1 ]; then
        if [ "$DETECTED_NVLINK_AVAILABLE" = true ]; then
            log_success "NVLink 配置完成 - 预期带宽: ~900 GB/s, 延迟: < 1 μs"
        else
            log_success "PCIe P2P 配置完成 - 预期带宽: ~64 GB/s, 延迟: 2-5 μs"
        fi
    else
        log_warning "GPU 间通信配置完成 - 但可能回退到共享内存通信"
    fi
}

# 配置共享内存传输设置
configure_shm_settings() {
    log_info "配置共享内存传输设置"
    
    # 使用统一配置管理器
    setup_common_nccl_config
    setup_debug_files "shm"
    setup_network_config "ib_disable"
    setup_network_config "p2p_disable"
    setup_network_interface "clear_interface"
    
    # 共享内存性能优化
    setup_performance_config "shm_optimized"
    
    log_success "共享内存配置完成 - 将使用共享内存进行 GPU 间通信"
    log_warning "共享内存传输性能较低，主要用于兼容性测试"
}

# 显示 NCCL 配置摘要
display_nccl_config_summary() {
    log_info ""
    log_success "NCCL 环境变量配置完成"
    log_info "配置摘要: 网络后端: $NETWORK_BACKEND"
    
    # 关键配置状态检查
    if [ "${NCCL_P2P_DISABLE:-1}" = "0" ]; then
        if [ "${NCCL_P2P_LEVEL:-}" = "NVL" ]; then
            log_success "✓ 已启用 NVLink P2P 通信"
        elif [ "${NCCL_P2P_LEVEL:-}" = "PIX" ]; then
            log_success "✓ 已启用 PCIe P2P 通信"
        else
            log_info "✓ 已启用 P2P 通信"
        fi
    else
        log_warning "⚠ P2P 通信已禁用"
    fi
    
    # 网络传输状态
    if [ "${NCCL_IB_DISABLE:-0}" = "1" ]; then
        log_info "✓ 已禁用 InfiniBand"
    fi
    
    # 调试配置
    if [ -n "${NCCL_DEBUG_FILE:-}" ]; then
        log_success "✓ 调试日志: ${NCCL_DEBUG_FILE}"
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
        master_addr="$MASTER_ADDR"
        
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
            log_error "多节点模式必须设置以下环境变量: WORLD_SIZE, NODE_RANK, NPROC_PER_NODE"
            log_error "当前环境变量状态:"
            log_error "  WORLD_SIZE: ${WORLD_SIZE:-未设置}"
            log_error "  NODE_RANK: ${NODE_RANK:-未设置}"
            log_error "  NPROC_PER_NODE: ${NPROC_PER_NODE:-未设置}"
            log_error ""
            log_error "请设置环境变量后重新运行，例如:"
            log_error "  export WORLD_SIZE=2        # 总进程数 (节点数 × 每节点GPU数)"
            log_error "  export NODE_RANK=0         # 当前节点编号 (0, 1, 2, ...)"
            log_error "  export NPROC_PER_NODE=4    # 每节点GPU数"
            log_error ""
            log_error "或者使用 nccl_multinode_launcher.sh 脚本自动设置这些变量"
            exit 1
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
            # 强制刷新输出缓冲区，确保NCCL调试信息被捕获
            export PYTHONUNBUFFERED=1
            export NCCL_DEBUG_FILE=""  # 清除文件输出，使用标准输出
            if stdbuf -oL -eL torchrun \
                --nproc_per_node="$gpu_count" \
                --nnodes="$nnodes" \
                --node_rank="$node_rank" \
                --master_addr="$master_addr" \
                --master_port="$master_port" \
                "$test_script" > >(tee "$output_file") 2>&1; then
                log_success "NCCL 测试执行完成"
            else
                log_error "NCCL 测试执行失败"
                return 1
            fi
        else
            # 使用传统的 torch.distributed.launch
            log_info "使用 torch.distributed.launch 启动分布式测试"
            # 强制刷新输出缓冲区，确保NCCL调试信息被捕获
            export PYTHONUNBUFFERED=1
            export NCCL_DEBUG_FILE=""  # 清除文件输出，使用标准输出
            if stdbuf -oL -eL python3 -m torch.distributed.launch \
                --nproc_per_node="$gpu_count" \
                --nnodes="$nnodes" \
                --node_rank="$node_rank" \
                --master_addr="$master_addr" \
                --master_port="$master_port" \
                "$test_script" > >(tee "$output_file") 2>&1; then
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
    log_info "NCCL 关键信息摘要:"
    
    # 1. 提取通信路径信息（仅展示，不做判断）
    log_info "📋 通信路径信息:"
    if grep -E "NCCL INFO Channel.*via" "$output_file" >/dev/null 2>&1; then
        local channel_paths=$(grep -E "NCCL INFO Channel.*via" "$output_file")
        echo "$channel_paths" | while read -r line; do
            if [ -n "$line" ]; then
                # 提取关键信息: Channel X : A[X] -> B[X] via PATH
                local path_info=$(echo "$line" | sed -n 's/.*NCCL INFO \(Channel [^:]*\) : \([^:]*\) via \(.*\)/\1: \2 via \3/p')
                if [ -n "$path_info" ]; then
                    log_info "   $path_info"
                fi
            fi
        done
    else
        log_info "   未找到 Channel 通信路径信息"
    fi
    
    # 2. 显示 GPUDirect RDMA 状态（仅展示）
    log_info "🔗 GPUDirect RDMA:"
    if grep -q "GDR" "$output_file" || grep -q "GPUDirect" "$output_file"; then
        log_info "   已启用"
    else
        log_info "   未检测到"
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
        local completed_iterations=$(grep -c "已完成.*次迭代" "$output_file" 2>/dev/null)
    completed_iterations=${completed_iterations:-0}
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

# 展示当前 NCCL 环境变量的值
display_nccl_environment_variables() {
    log_header "当前 NCCL 环境变量"
    
    # 定义需要展示的 NCCL 环境变量列表
    local nccl_vars=(
        "NCCL_DEBUG"
        "NCCL_DEBUG_SUBSYS"
        "NCCL_IB_DISABLE"
        "NCCL_NET_DISABLE"
        "NCCL_P2P_DISABLE"
        "NCCL_SHM_DISABLE"
        "NCCL_NET_GDR_LEVEL"
        "NCCL_IB_HCA"
        "NCCL_IB_GID_INDEX"
        "NCCL_IB_TIMEOUT"
        "NCCL_IB_RETRY_CNT"
        "NCCL_SOCKET_IFNAME"
        "NCCL_P2P_LEVEL"
        "NCCL_NVLS_ENABLE"
        "NCCL_ALGO"
        "NCCL_MAX_NCHANNELS"
        "NCCL_MIN_NCHANNELS"
        "NCCL_NVLS_CHUNKSIZE"
        "NCCL_TREE_THRESHOLD"
        "NCCL_RING_THRESHOLD"
        "NCCL_BUFFSIZE"
        "NCCL_NTHREADS"
        "NCCL_CHECKS_DISABLE"
        "NCCL_CHECK_POINTERS"
        "NCCL_LAUNCH_MODE"
    )
    
    log_info "核心配置变量:"
    local core_vars=("NCCL_DEBUG" "NCCL_DEBUG_SUBSYS" "NCCL_IB_DISABLE" "NCCL_NET_DISABLE" "NCCL_P2P_DISABLE" "NCCL_SHM_DISABLE")
    for var in "${core_vars[@]}"; do
        local value="${!var:-未设置}"
        if [ "$value" != "未设置" ]; then
            log_success "  $var = $value"
        else
            log_info "  $var = $value"
        fi
    done
    
    log_info ""
    log_info "网络配置变量:"
    local network_vars=("NCCL_NET_GDR_LEVEL" "NCCL_IB_HCA" "NCCL_IB_GID_INDEX" "NCCL_IB_TIMEOUT" "NCCL_IB_RETRY_CNT" "NCCL_SOCKET_IFNAME")
    for var in "${network_vars[@]}"; do
        local value="${!var:-未设置}"
        if [ "$value" != "未设置" ]; then
            log_success "  $var = $value"
        else
            log_info "  $var = $value"
        fi
    done
    
    log_info ""
    log_info "性能优化变量:"
    local perf_vars=("NCCL_P2P_LEVEL" "NCCL_NVLS_ENABLE" "NCCL_ALGO" "NCCL_MAX_NCHANNELS" "NCCL_MIN_NCHANNELS" "NCCL_NVLS_CHUNKSIZE")
    for var in "${perf_vars[@]}"; do
        local value="${!var:-未设置}"
        if [ "$value" != "未设置" ]; then
            log_success "  $var = $value"
        else
            log_info "  $var = $value"
        fi
    done
    
    log_info ""
    log_info "其他配置变量:"
    local other_vars=("NCCL_TREE_THRESHOLD" "NCCL_RING_THRESHOLD" "NCCL_BUFFSIZE" "NCCL_NTHREADS" "NCCL_CHECKS_DISABLE" "NCCL_CHECK_POINTERS" "NCCL_LAUNCH_MODE")
    for var in "${other_vars[@]}"; do
        local value="${!var:-未设置}"
        if [ "$value" != "未设置" ]; then
            log_success "  $var = $value"
        else
            log_info "  $var = $value"
        fi
    done
    
    # 统计已设置的变量数量
    local set_count=0
    local total_count=${#nccl_vars[@]}
    for var in "${nccl_vars[@]}"; do
        if [ -n "${!var:-}" ]; then
            ((set_count++))
        fi
    done
    
    log_info ""
    log_info "环境变量统计: $set_count/$total_count 个变量已设置"
    
    # 如果有自定义的 NCCL 环境变量（不在预定义列表中）
    log_info ""
    log_info "检查其他 NCCL 环境变量:"
    local custom_vars=$(env | grep "^NCCL_" | grep -v -E "^($(IFS='|'; echo "${nccl_vars[*]}"))" | cut -d= -f1 | sort)
    if [ -n "$custom_vars" ]; then
        log_warning "发现额外的 NCCL 环境变量:"
        echo "$custom_vars" | while read var; do
            local value="${!var:-}"
            log_warning "  $var = $value"
        done
    else
        log_info "  未发现额外的 NCCL 环境变量"
    fi
}



# 生成测试报告
generate_report() {
    log_header "生成测试报告"
    
    local report_file="/tmp/nccl_test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
=== NCCL 测试报告 ===
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
    echo "NCCL 测试开始 - $(date)" > "$LOG_FILE"
    
    # 显示脚本信息
    log_header "$SCRIPT_NAME v$VERSION"
    log_info "专注于 NCCL 分布式通信测试"
    log_info "日志文件: $LOG_FILE"
    log ""
    
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
        
        # 展示 NCCL 环境变量的值
        display_nccl_environment_variables
    fi

    # 3. 判断是否是 dry run，如果是则跳过测试创建和运行
    if [ "$DRY_RUN" = true ]; then
        log_success "Dry-run 完成：环境检查和配置均正常"
        log_info "如需执行实际测试，请移除 --dry-run 选项"
        exit 0
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