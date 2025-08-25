#!/bin/bash

# =============================================================================
# InfiniBand 网卡健康检查脚本
# 基于 Ubuntu 服务器 IB 网络分析报告
# 作者: Grissom
# 版本：1.1
# =============================================================================

# 版本信息
VERSION="1.1"
SCRIPT_NAME="IB Health Check"

# 默认选项
QUIET_MODE=false
SUMMARY_ONLY=false
USE_COLOR=true

# 显示帮助信息
show_help() {
    cat << EOF
InfiniBand 网卡相关健康检查脚本 v${VERSION}

用法: $0 [选项]

选项:
  -h, --help      显示此帮助信息
  -v, --version   显示版本信息
  -q, --quiet     静默模式 (仅输出错误和警告)
  -s, --summary   仅显示摘要信息
  --no-color      禁用彩色输出

运行模式说明:
  【完整模式】(默认)
    • 执行所有 10 项检查 (硬件、驱动、网络、优化等)
    • 显示详细的检查过程和结果
    • 提供完整的优化建议和故障排查指导
    • 适用于: 全面诊断、初次部署、故障排查

  【静默模式】(-q)
    • 执行所有 10 项检查
    • 仅显示警告和错误信息，隐藏正常状态
    • 适用于: 自动化监控、脚本集成、快速问题识别
    • 输出特点: 无输出表示一切正常，有输出表示需要关注

  【摘要模式】(-s)
    • 仅执行 4 项关键检查 (依赖、硬件、端口、性能计数器)
    • 显示简洁的总结报告
    • 适用于: 快速健康检查、定期巡检、状态概览
    • 输出特点: 重点关注网络基本功能，不包含优化建议

功能:
  • 检查 InfiniBand 硬件状态
  • 验证 OFED 驱动和内核模块
  • 分析端口状态和网络拓扑
  • 监控性能计数器
  • 评估系统优化配置
  • 提供详细的优化建议

使用场景示例:
  sudo $0                    # 新环境部署后的全面检查
  sudo $0 -q                 # 监控脚本中的异常检测
  sudo $0 -s                 # 日常巡检的快速状态确认
  sudo $0 --no-color         # 在不支持彩色的终端中运行

检查项目对比:
  完整模式: 系统信息 + IB硬件 + OFED驱动 + 端口状态 + 网络拓扑 + 
           性能计数器 + 网络接口 + 性能工具 + 系统优化 + 优化建议
  摘要模式: 依赖检查 + IB硬件 + 端口状态 + 性能计数器

注意:
  • 此脚本需要 root 权限运行
  • 脚本仅进行检查和提供建议，不会修改系统配置
  • 所有优化操作需要用户手工执行
  • 静默模式下无输出表示系统状态良好
  • 摘要模式专注于核心功能，不检查优化配置

EOF
}

# 显示版本信息
show_version() {
    echo "$SCRIPT_NAME v$VERSION"
    echo "基于 Ubuntu 服务器 IB 设备和网络分析报告"
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
            -s|--summary)
                SUMMARY_ONLY=true
                shift
                ;;
            --no-color)
                USE_COLOR=false
                shift
                ;;
            *)
                echo "错误: 未知选项 '$1'"
                echo "使用 '$0 --help' 查看帮助信息"
                exit 1
                ;;
        esac
    done
}

# 颜色定义 (根据USE_COLOR变量动态设置)
set_colors() {
    if [ "$USE_COLOR" = true ]; then
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        YELLOW='\033[1;33m'
        BLUE='\033[0;34m'
        PURPLE='\033[0;35m'
        CYAN='\033[0;36m'
        NC='\033[0m' # No Color
    else
        RED=''
        GREEN=''
        YELLOW=''
        BLUE=''
        PURPLE=''
        CYAN=''
        NC=''
    fi
}

# 全局变量
LOG_FILE="/tmp/ib_health_check_$(date +%Y%m%d_%H%M%S).log"
ERROR_COUNT=0
WARNING_COUNT=0
TOTAL_CHECKS=0

# 日志函数
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_info() {
    if [ "$QUIET_MODE" = false ] && [ "$SUMMARY_ONLY" = false ]; then
        log "${BLUE}[INFO]${NC} $1"
    else
        echo -e "${BLUE}[INFO]${NC} $1" >> "$LOG_FILE"
    fi
}

log_success() {
    if [ "$QUIET_MODE" = false ] && [ "$SUMMARY_ONLY" = false ]; then
        log "${GREEN}[PASS]${NC} $1"
    else
        echo -e "${GREEN}[PASS]${NC} $1" >> "$LOG_FILE"
    fi
}

log_warning() {
    WARNING_COUNT=$((WARNING_COUNT + 1))
    if [ "$SUMMARY_ONLY" = false ]; then
        log "${YELLOW}[WARN]${NC} $1"
    elif [ "$QUIET_MODE" = false ]; then
        # 摘要模式下仍显示警告到终端
        echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
    else
        echo -e "${YELLOW}[WARN]${NC} $1" >> "$LOG_FILE"
    fi
}

log_recommendation() {
    # 用于建议部分的警告，不增加WARNING_COUNT
    if [ "$SUMMARY_ONLY" = false ]; then
        log "${YELLOW}[建议]${NC} $1"
    else
        echo -e "${YELLOW}[建议]${NC} $1" >> "$LOG_FILE"
    fi
}

log_error() {
    ERROR_COUNT=$((ERROR_COUNT + 1))
    if [ "$SUMMARY_ONLY" = false ]; then
        log "${RED}[FAIL]${NC} $1"
    elif [ "$QUIET_MODE" = false ]; then
        # 摘要模式下仍显示错误到终端
        echo -e "${RED}[FAIL]${NC} $1" | tee -a "$LOG_FILE"
    else
        echo -e "${RED}[FAIL]${NC} $1" >> "$LOG_FILE"
    fi
}

log_header() {
    if [ "$QUIET_MODE" = false ] && [ "$SUMMARY_ONLY" = false ]; then
        log ""
        log "${PURPLE}=== $1 ===${NC}"
        log ""
    else
        echo "" >> "$LOG_FILE"
        echo -e "${PURPLE}=== $1 ===${NC}" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
    fi
}

# 摘要模式专用日志函数
log_summary() {
    if [ "$QUIET_MODE" = false ]; then
        echo -e "$1" | tee -a "$LOG_FILE"
    else
        echo -e "$1" >> "$LOG_FILE"
    fi
}

# 检查是否为root用户
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "此脚本需要root权限运行"
        log_info "请使用: sudo $0"
        exit 1
    fi
}

# 检查必要命令是否存在
check_dependencies() {
    log_header "检查依赖命令"
    
    local commands=("lspci" "ibstat" "ibv_devinfo" "perfquery" "ibnetdiscover" "ofed_info")
    local missing_commands=()
    
    for cmd in "${commands[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            log_success "命令 $cmd 可用"
        else
            log_error "命令 $cmd 未找到"
            missing_commands+=("$cmd")
        fi
    done
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))  # 依赖检查作为一个检查项目
    
    if [ ${#missing_commands[@]} -gt 0 ]; then
        log_error "缺少必要命令，请安装 MLNX_OFED 驱动包"
        log_info "缺少的命令: ${missing_commands[*]}"
        return 1
    fi
    
    return 0
}

# 系统基本信息检查
check_system_info() {
    log_header "系统基本信息"
    
    log_info "主机名: $(hostname)"
    log_info "操作系统: $(lsb_release -d | cut -f2)"
    log_info "内核版本: $(uname -r)"
    log_info "架构: $(uname -m)"
    log_info "检查时间: $(date)"
    
    # CPU信息
    local cpu_count=$(nproc)
    local cpu_model=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
    log_info "CPU核心数: $cpu_count"
    log_info "CPU型号: $cpu_model"
    
    # 内存信息
    local mem_total=$(free -h | awk '/^Mem:/ {print $2}')
    local mem_used=$(free -h | awk '/^Mem:/ {print $3}')
    local mem_available=$(free -h | awk '/^Mem:/ {print $7}')
    log_info "内存总量: $mem_total"
    log_info "内存使用: $mem_used"
    log_info "内存可用: $mem_available"
    
    # 系统负载
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | xargs)
    log_info "系统负载: $load_avg"
    
    # 检查swap状态
    local swap_total=$(free -h | awk '/^Swap:/ {print $2}')
    if [ "$swap_total" = "0B" ]; then
        log_success "Swap已禁用 (符合HPC最佳实践)"
    else
        log_warning "Swap已启用: $swap_total (建议在HPC环境中禁用)"
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

# InfiniBand 硬件检查
check_ib_hardware() {
    log_header "InfiniBand 硬件检查"
    
    # 检查PCI设备
    local ib_devices=$(lspci | grep -i infiniband | wc -l)
    if [ "$ib_devices" -gt 0 ]; then
        log_success "发现 $ib_devices 个 InfiniBand 设备"
        log_info "InfiniBand 设备列表:"
        lspci | grep -i infiniband | while read line; do
            log_info "  $line"
        done
    else
        log_error "未发现 InfiniBand 设备"
        return 1
    fi
    
    # 检查设备详细信息
    log_info ""
    log_info "设备详细信息:"
    if ibv_devinfo >/dev/null 2>&1; then
        local device_count=$(ibv_devinfo | grep "hca_id:" | wc -l)
        log_success "发现 $device_count 个 IB 设备"
        
        # 检查每个设备的状态
        ibv_devinfo | grep -E "(hca_id|fw_ver|node_guid|port.*state)" | while read line; do
            if [[ $line == *"hca_id"* ]]; then
                log_info "  设备: $(echo $line | cut -d: -f2 | xargs)"
            elif [[ $line == *"fw_ver"* ]]; then
                log_info "    固件版本: $(echo $line | cut -d: -f2 | xargs)"
            elif [[ $line == *"node_guid"* ]]; then
                log_info "    Node GUID: $(echo $line | cut -d: -f2- | xargs)"
            elif [[ $line == *"state"* ]]; then
                local state=$(echo $line | awk '{print $NF}')
                if [ "$state" = "PORT_ACTIVE" ]; then
                    log_success "    端口状态: $state"
                else
                    log_warning "    端口状态: $state"
                fi
            fi
        done
    else
        log_error "无法获取设备信息"
    fi
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))  # IB硬件检查作为一个检查项目
}

# OFED 驱动检查
check_ofed_driver() {
    log_header "OFED 驱动检查"
    
    # 检查OFED版本
    if command -v ofed_info >/dev/null 2>&1; then
        local ofed_version=$(ofed_info -s 2>/dev/null)
        if [ -n "$ofed_version" ]; then
            log_success "OFED 版本: $ofed_version"
            
            # 检查是否为推荐版本
            if [[ $ofed_version == *"23.10"* ]] || [[ $ofed_version == *"24."* ]]; then
                log_success "OFED 版本较新，支持 ConnectX-7"
            else
                log_warning "OFED 版本较旧，建议升级到最新版本"
            fi
        else
            log_error "无法获取 OFED 版本信息"
        fi
    else
        log_error "OFED 未安装或配置不正确"
    fi
    
    # 检查内核模块
    log_info ""
    log_info "关键内核模块状态:"
    local modules=("mlx5_core" "mlx5_ib" "ib_core" "ib_ipoib" "ib_uverbs" "rdma_cm")
    
    for module in "${modules[@]}"; do
        if lsmod | grep -q "^$module"; then
            log_success "  $module: 已加载"
        else
            log_error "  $module: 未加载"
        fi
    done
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))  # OFED驱动检查作为一个检查项目
}

# 端口状态检查
check_port_status() {
    log_header "端口状态检查"
    
    if ! command -v ibstat >/dev/null 2>&1; then
        log_error "ibstat 命令不可用"
        return 1
    fi
    
    # 获取所有端口状态
    local active_ports=0
    local total_ports=0
    
    log_info "端口状态详情:"
    ibstat 2>/dev/null | grep -E "(CA|Port|State|Rate|Physical state|Port GUID)" | while read line; do
        if [[ $line == *"CA '"* ]]; then
            local ca_name=$(echo "$line" | sed "s/CA '\(.*\)'/\1/")
            log_info "  设备: $ca_name"
        elif [[ $line == *"Port "* ]] && [[ $line != *"Port GUID"* ]]; then
            local port_num=$(echo "$line" | awk '{print $2}' | sed 's/:$//')
            log_info "    端口 $port_num:"
        elif [[ $line == *"State:"* ]]; then
            local state=$(echo "$line" | awk '{print $2}')
            if [ "$state" = "Active" ]; then
                log_success "      状态: $state"
            else
                log_warning "      状态: $state"
            fi
        elif [[ $line == *"Physical state:"* ]]; then
            local phy_state=$(echo "$line" | awk '{print $3}')
            if [ "$phy_state" = "LinkUp" ]; then
                log_success "      物理状态: $phy_state"
            else
                log_warning "      物理状态: $phy_state"
            fi
        elif [[ $line == *"Rate:"* ]]; then
            local rate=$(echo "$line" | awk '{print $2}')
            log_info "      速率: $rate Gbps"
            if [ "$rate" -ge 200 ]; then
                log_success "      速率优秀 (HDR)"
            elif [ "$rate" -ge 100 ]; then
                log_info "      速率良好 (EDR)"
            else
                log_warning "      速率较低"
            fi
        elif [[ $line == *"Port GUID:"* ]]; then
            # 清理行首空白字符并提取GUID
            local cleaned_line=$(echo "$line" | sed 's/^[[:space:]]*//')
            local port_guid=$(echo "$cleaned_line" | awk '{print $3}')
            if [[ $port_guid =~ ^0x[0-9a-fA-F]{16}$ ]]; then
                log_info "      端口 GUID: $port_guid"
            else
                log_warning "      端口 GUID: 格式异常"
            fi
        fi
    done
    
    # 统计活跃端口
    active_ports=$(ibstat 2>/dev/null | grep -c "State: Active")
    total_ports=$(ibstat 2>/dev/null | grep -c "Port [0-9]")
    
    log_info ""
    if [ "$active_ports" -eq "$total_ports" ] && [ "$total_ports" -gt 0 ]; then
        log_success "所有端口 ($active_ports/$total_ports) 均为 Active 状态"
    else
        log_warning "活跃端口: $active_ports/$total_ports"
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

# 网络拓扑检查
check_network_topology() {
    log_header "网络拓扑检查"
    
    if ! command -v ibnetdiscover >/dev/null 2>&1; then
        log_error "ibnetdiscover 命令不可用"
        return 1
    fi
    
    log_info "正在获取网络拓扑信息..."
    
    # 使用 ibnodes 获取准确的节点信息
    local nodes_output=$(timeout 30 ibnodes 2>/dev/null)
    local ports_output=$(timeout 30 ibnetdiscover -p 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$nodes_output" ]; then
        # 从 ibnodes 输出中统计节点数量
        local node_count=$(echo "$nodes_output" | wc -l)
        local switch_count=$(echo "$nodes_output" | grep -c "Switch")
        local ca_count=$(echo "$nodes_output" | grep -c "Ca")
        
        # 从端口连接报告中获取连接数
        local port_connections=0
        if [ -n "$ports_output" ]; then
            port_connections=$(echo "$ports_output" | wc -l)
        fi
        
        log_success "网络拓扑发现成功"
        log_info "  总节点数: $node_count"
        log_info "  交换机数: $switch_count"
        log_info "  计算节点数: $ca_count"
        log_info "  端口连接数: $port_connections"
        
        if [ "$switch_count" -ge 2 ]; then
            log_success "检测到多交换机冗余设计"
        elif [ "$switch_count" -eq 1 ]; then
            log_warning "仅检测到单个交换机"
        else
            log_error "未检测到交换机"
        fi
        
        # 检查子网管理器
        local sm_output=$(timeout 10 sminfo 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$sm_output" ]; then
            local sm_count=$(echo "$sm_output" | wc -l)
            log_success "检测到 $sm_count 个子网管理器"
        else
            log_warning "未检测到活跃的子网管理器"
        fi
    else
        log_error "网络拓扑发现失败或超时"
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

# 性能计数器检查
check_performance_counters() {
    log_header "性能计数器检查"
    
    if ! command -v perfquery >/dev/null 2>&1; then
        log_error "perfquery 命令不可用"
        return 1
    fi
    
    log_info "检查性能计数器..."
    
    # 获取所有LID
    local lids=$(ibstat 2>/dev/null | grep "Base lid:" | awk '{print $3}')
    
    if [ -z "$lids" ]; then
        log_error "无法获取端口LID信息"
        return 1
    fi
    
    local error_found=false
    local total_errors=0
    
    for lid in $lids; do
        log_info "  检查 LID $lid:"
        
        # 检查错误计数器
        local perf_output=$(timeout 10 perfquery -a "$lid" 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$perf_output" ]; then
            # 检查各种错误计数器
            local symbol_errors=$(echo "$perf_output" | grep "SymbolErrorCounter" | awk '{print $2}')
            local link_errors=$(echo "$perf_output" | grep "LinkErrorRecoveryCounter" | awk '{print $2}')
            local link_downed=$(echo "$perf_output" | grep "LinkDownedCounter" | awk '{print $2}')
            local rcv_errors=$(echo "$perf_output" | grep "PortRcvErrors" | awk '{print $2}')
            local xmit_discards=$(echo "$perf_output" | grep "PortXmitDiscards" | awk '{print $2}')
            local xmit_wait=$(echo "$perf_output" | grep "PortXmitWait" | awk '{print $2}')
            
            # 检查是否有错误
            local current_errors=0
            [ -n "$symbol_errors" ] && [ "$symbol_errors" -gt 0 ] && current_errors=$((current_errors + symbol_errors))
            [ -n "$link_errors" ] && [ "$link_errors" -gt 0 ] && current_errors=$((current_errors + link_errors))
            [ -n "$link_downed" ] && [ "$link_downed" -gt 0 ] && current_errors=$((current_errors + link_downed))
            [ -n "$rcv_errors" ] && [ "$rcv_errors" -gt 0 ] && current_errors=$((current_errors + rcv_errors))
            [ -n "$xmit_discards" ] && [ "$xmit_discards" -gt 0 ] && current_errors=$((current_errors + xmit_discards))
            
            if [ "$current_errors" -eq 0 ]; then
                log_success "    无错误计数器"
            else
                log_warning "    发现 $current_errors 个错误"
                error_found=true
                total_errors=$((total_errors + current_errors))
            fi
            
            # 检查拥塞
            if [ -n "$xmit_wait" ] && [ "$xmit_wait" -gt 0 ]; then
                log_warning "    检测到网络拥塞 (XmitWait: $xmit_wait)"
            else
                log_success "    无网络拥塞"
            fi
            
            # 显示数据传输统计
            local xmit_data=$(echo "$perf_output" | grep "PortXmitData" | awk '{print $2}')
            local rcv_data=$(echo "$perf_output" | grep "PortRcvData" | awk '{print $2}')
            
            if [ -n "$xmit_data" ] && [ -n "$rcv_data" ]; then
                log_info "    发送数据: $xmit_data 包"
                log_info "    接收数据: $rcv_data 包"
            fi
        else
            log_error "    无法获取性能计数器"
        fi
    done
    
    log_info ""
    if [ "$error_found" = false ]; then
        log_success "所有端口性能计数器正常"
    else
        log_warning "发现总计 $total_errors 个错误"
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

# 网络接口检查
check_network_interfaces() {
    log_header "网络接口检查"
    
    # 检查IPoIB接口
    local ipoib_interfaces=$(ip link show | grep -E "ib[0-9]|ibp[0-9]" | awk -F: '{print $2}' | xargs)
    
    if [ -n "$ipoib_interfaces" ]; then
        log_success "发现 IPoIB 接口: $ipoib_interfaces"
        
        for interface in $ipoib_interfaces; do
            log_info "  接口 $interface:"
            
            # 检查接口状态
            local state=$(ip link show "$interface" | grep -o "state [A-Z]*" | awk '{print $2}')
            if [ "$state" = "UP" ]; then
                log_success "    状态: $state"
            else
                log_warning "    状态: $state"
            fi
            
            # 检查MTU
            local mtu=$(ip link show "$interface" | grep -o "mtu [0-9]*" | awk '{print $2}')
            log_info "    MTU: $mtu"
            
            if [ "$mtu" -eq 2044 ]; then
            log_warning "    MTU为数据报模式标准值，建议优化为连接模式65520"
            elif [ "$mtu" -eq 65520 ]; then
            log_success "    MTU已优化为连接模式Jumbo Frame"
            else
                log_info "    MTU为自定义值"
            fi
            
            # 检查IP地址
            local ip_addr=$(ip addr show "$interface" | grep "inet " | awk '{print $2}')
            if [ -n "$ip_addr" ]; then
                log_info "    IP地址: $ip_addr"
            else
                log_warning "    未配置IP地址"
            fi
        done
    else
        log_warning "未发现 IPoIB 接口"
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

# 性能测试工具检查
check_performance_tools() {
    log_header "性能测试工具检查"
    
    local tools=("ib_send_bw" "ib_send_lat" "ib_write_bw" "ib_read_bw" "ib_write_lat" "ib_read_lat")
    local available_tools=()
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            available_tools+=("$tool")
            log_success "  $tool: 可用"
        else
            missing_tools+=("$tool")
            log_info "  $tool: 不可用"
        fi
    done
    
    log_info ""
    if [ ${#available_tools[@]} -gt 0 ]; then
        log_success "可用的性能测试工具: ${available_tools[*]}"
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_warning "缺少的性能测试工具: ${missing_tools[*]}"
        log_info "建议安装 perftest 包: apt install perftest"
    fi
    
    # 只计算一个检查项目
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

# 系统优化建议
check_system_optimization() {
    log_header "系统优化检查"
    
    # 检查内核参数 (仅在IPoIB场景下有意义)
    local ipoib_interfaces=$(ip link show | grep -E "ib[0-9]|ibp[0-9]" | awk -F: '{print $2}' | xargs)
    
    if [ -n "$ipoib_interfaces" ]; then
        log_info "检查网络内核参数 (适用于IPoIB场景):"
        
        local rmem_max=$(sysctl -n net.core.rmem_max 2>/dev/null)
        local wmem_max=$(sysctl -n net.core.wmem_max 2>/dev/null)
        
        if [ "$rmem_max" -ge 268435456 ]; then
            log_success "  net.core.rmem_max: $rmem_max (已优化)"
        else
            log_warning "  net.core.rmem_max: $rmem_max (建议设置为268435456)"
        fi
        
        if [ "$wmem_max" -ge 268435456 ]; then
            log_success "  net.core.wmem_max: $wmem_max (已优化)"
        else
            log_warning "  net.core.wmem_max: $wmem_max (建议设置为268435456)"
        fi
    else
        log_info "网络内核参数检查:"
        log_info "  未检测到IPoIB接口，跳过网络内核参数检查"
        log_info "  注意: 如果GPU直接使用IB网络(RDMA)，网络内核参数无关紧要"
    fi
    
    # 检查CPU频率调节器
    log_info ""
    log_info "检查CPU频率调节器:"
    local governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)
    if [ "$governor" = "performance" ]; then
        log_success "  CPU调节器: $governor (推荐用于HPC)"
    elif [ -n "$governor" ]; then
        log_warning "  CPU调节器: $governor (建议设置为performance)"
    else
        log_info "  无法检测CPU调节器"
    fi
    
    # 检查NUMA配置
    log_info ""
    log_info "检查NUMA配置:"
    if command -v numactl >/dev/null 2>&1; then
        local numa_nodes=$(numactl --hardware | grep "available:" | awk '{print $2}')
        log_info "  NUMA节点数: $numa_nodes"
        
        if [ "$numa_nodes" -gt 1 ]; then
            log_info "  建议检查IB设备与CPU的NUMA亲和性"
        fi
    else
        log_warning "  numactl未安装，无法检查NUMA配置"
    fi
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))  # 系统优化检查作为一个检查项目
}

# 生成优化建议
generate_recommendations() {
    log_header "优化建议"
    
    log_info "基于检查结果的优化建议 (请手工执行):"
    log_info ""
    
    local has_recommendations=false
    
    # MTU优化建议
    local ipoib_interfaces=$(ip link show | grep -E "ib[0-9]|ibp[0-9]" | awk -F: '{print $2}' | xargs)
    if [ -n "$ipoib_interfaces" ]; then
        for interface in $ipoib_interfaces; do
            local mtu=$(ip link show "$interface" | grep -o "mtu [0-9]*" | awk '{print $2}')
            if [ "$mtu" -eq 2044 ]; then
            has_recommendations=true
            log_recommendation "发现 $interface MTU 为数据报模式标准值 (当前: $mtu)"
            log_info "建议优化命令:"
            log_info "  # 临时设置 (重启后失效)"
            log_info "  sudo ip link set $interface mtu 65520"
            log_info "  # 永久设置 (编辑网络配置文件)"
            log_info "  sudo vim /etc/netplan/01-netcfg.yaml"
            log_info "  # 或者编辑 /etc/network/interfaces"
            log_info "  # 注意: 连接模式 MTU 65520 需要特殊配置支持"
            fi
        done
    fi
    
    # 内核参数优化 (仅适用于IPoIB场景)
    if [ -n "$ipoib_interfaces" ]; then
        local rmem_max=$(sysctl -n net.core.rmem_max 2>/dev/null)
        local wmem_max=$(sysctl -n net.core.wmem_max 2>/dev/null)
        
        if [ "$rmem_max" -lt 268435456 ] || [ "$wmem_max" -lt 268435456 ]; then
            has_recommendations=true
            log_recommendation "网络内核参数未优化 (适用于IPoIB场景)"
            log_info "适用场景: 仅当使用IPoIB网络接口时需要优化"
            log_info "注意: 如果GPU直接使用IB网络(RDMA)，则无需修改这些参数"
            log_info ""
            log_info "当前设置:"
            log_info "  net.core.rmem_max = $rmem_max (建议: 268435456)"
            log_info "  net.core.wmem_max = $wmem_max (建议: 268435456)"
            log_info ""
            log_info "优化命令 (仅在使用IPoIB时执行):"
            log_info "  # 创建优化配置文件"
            log_info "  sudo tee /etc/sysctl.d/99-ipoib-network.conf << EOF"
            log_info "# IPoIB 网络优化参数 (仅适用于IP over InfiniBand)"
            log_info "net.core.rmem_max = 268435456"
            log_info "net.core.wmem_max = 268435456"
            log_info "net.core.rmem_default = 67108864"
            log_info "net.core.wmem_default = 67108864"
            log_info "net.core.netdev_max_backlog = 5000"
            log_info "EOF"
            log_info "  # 应用配置"
            log_info "  sudo sysctl -p /etc/sysctl.d/99-ipoib-network.conf"
            log_info ""
        fi
    else
        log_info "内核参数优化建议:"
        log_info "  未检测到IPoIB接口，无需修改网络内核参数"
        log_info "  注意: 如果GPU直接使用IB网络(RDMA)，内核网络参数优化无效"
        log_info ""
    fi
    
    # CPU调节器优化 (通用HPC优化)
    local governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)
    if [ "$governor" != "performance" ] && [ -n "$governor" ]; then
        has_recommendations=true
        log_recommendation "CPU调节器未设置为性能模式 (当前: $governor)"
        log_info "适用场景: 所有高性能计算场景 (包括GPU计算、IPoIB网络等)"
        log_info "优化命令:"
        log_info "  # 临时设置 (重启后失效)"
        log_info "  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
        log_info "  # 永久设置 (安装cpufrequtils)"
        log_info "  sudo apt install cpufrequtils"
        log_info "  echo 'GOVERNOR=\"performance\"' | sudo tee /etc/default/cpufrequtils"
        log_info "  sudo systemctl restart cpufrequtils"
        log_info ""
    fi
    
    # NUMA亲和性优化建议 (通用HPC优化)
    if command -v numactl >/dev/null 2>&1; then
        local numa_nodes=$(numactl --hardware | grep "available:" | awk '{print $2}')
        if [ "$numa_nodes" -gt 1 ]; then
            has_recommendations=true
            log_info "NUMA亲和性优化建议:"
            log_info "适用场景: 所有高性能计算场景 (GPU计算、IB网络、高性能应用)"
            log_info "  # 检查IB设备的NUMA节点"
            log_info "  cat /sys/class/infiniband/*/device/numa_node"
            log_info "  # 检查GPU设备的NUMA节点 (如果有GPU)"
            log_info "  nvidia-smi topo -m"
            log_info "  # 检查CPU NUMA拓扑"
            log_info "  numactl --hardware"
            log_info "  # 绑定应用到特定NUMA节点 (示例)"
            log_info "  numactl --cpunodebind=0 --membind=0 your_application"
            log_info "  # GPU应用NUMA绑定示例"
            log_info "  numactl --cpunodebind=0 --membind=0 python gpu_training.py"
            log_info ""
        fi
    fi
    
    # 监控脚本建议
    log_info "监控和维护建议:"
    log_info "  # 设置定期健康检查 (添加到crontab)"
    log_info "  sudo crontab -e"
    log_info "  # 添加以下行:"
    log_info "  0 8 * * * $(realpath "$0") > /var/log/ib_daily_check.log 2>&1"
    log_info ""
    log_info "  # 实时性能监控命令"
    log_info "  watch -n 30 'ibstat | grep -E \"(State|Rate)\"'"
    log_info "  watch -n 10 'perfquery -a | grep -E \"(Error|Wait)\"'"
    log_info ""
    
    # 性能测试建议
    log_info "性能基线测试建议:"
    log_info "  # 安装性能测试工具 (如果未安装)"
    log_info "  sudo apt update && sudo apt install perftest"
    log_info ""
    log_info "  # 带宽测试 (需要两台机器)"
    log_info "  # 服务端: ib_send_bw -d mlx5_0 --report_gbits"
    log_info "  # 客户端: ib_send_bw -d mlx5_0 --report_gbits <server_ip>"
    log_info ""
    log_info "  # 延迟测试"
    log_info "  # 服务端: ib_send_lat -d mlx5_0"
    log_info "  # 客户端: ib_send_lat -d mlx5_0 <server_ip>"
    log_info ""
    
    # 故障排查建议
    log_info "故障排查工具:"
    log_info "  # 重置性能计数器"
    log_info "  sudo perfquery -R"
    log_info "  # 检查链路质量"
    log_info "  ibdiagnet"
    log_info "  # 检查网络连通性"
    log_info "  ibping -S <source_lid> <dest_lid>"
    log_info ""
    
    if [ "$has_recommendations" = false ]; then
        log_success "当前配置已优化，无需额外调整"
    else
        log_info "发现可优化项目，请参考上述建议手工执行"
    fi
}

# 生成总结报告
generate_summary() {
    if [ "$SUMMARY_ONLY" = true ]; then
        # 摘要模式：使用专用日志函数确保输出到终端
        log_summary ""
        log_summary "${PURPLE}=== 检查总结 ===${NC}"
        log_summary ""
        
        log_summary "${BLUE}[INFO]${NC} 检查完成时间: $(date)"
        log_summary "${BLUE}[INFO]${NC} 总检查项目: $TOTAL_CHECKS"
        
        # 计算通过的检查模块数量
        local passed_modules=$((TOTAL_CHECKS - ERROR_COUNT))
        if [ "$passed_modules" -lt 0 ]; then
            passed_modules=0
        fi
        
        log_summary "${GREEN}[PASS]${NC} 通过模块: $passed_modules/$TOTAL_CHECKS"
        log_summary "${BLUE}[INFO]${NC} 警告事件: $WARNING_COUNT"
        log_summary "${BLUE}[INFO]${NC} 失败事件: $ERROR_COUNT"
        log_summary "${BLUE}[INFO]${NC} 详细日志: $LOG_FILE"
        
        log_summary ""
        if [ "$ERROR_COUNT" -eq 0 ] && [ "$WARNING_COUNT" -eq 0 ]; then
            log_summary "${GREEN}[PASS]${NC} 🎉 所有检查项目均通过！InfiniBand网络状态优秀。"
        elif [ "$ERROR_COUNT" -eq 0 ]; then
            log_summary "${BLUE}[INFO]${NC} ⚠️  检查完成，有 $WARNING_COUNT 个警告事件需要关注。"
        else
            log_summary "${RED}[FAIL]${NC} ❌ 检查发现 $ERROR_COUNT 个严重问题，需要立即处理。"
        fi
        
        log_summary ""
        log_summary "${BLUE}[INFO]${NC} 建议:"
        if [ "$ERROR_COUNT" -gt 0 ]; then
            log_summary "${BLUE}[INFO]${NC} 1. 优先解决失败项目"
            log_summary "${BLUE}[INFO]${NC} 2. 检查OFED驱动安装"
            log_summary "${BLUE}[INFO]${NC} 3. 验证硬件连接"
        fi
        if [ "$WARNING_COUNT" -gt 0 ]; then
            log_summary "${BLUE}[INFO]${NC} 1. 查看优化建议章节"
            log_summary "${BLUE}[INFO]${NC} 2. 考虑性能调优"
            log_summary "${BLUE}[INFO]${NC} 3. 定期监控网络状态"
        fi
        
        log_summary ""
        log_summary "${BLUE}[INFO]${NC} 如需技术支持，请提供日志文件: $LOG_FILE"
    else
        # 完整模式：使用原有日志函数
        log_header "检查总结"
        
        log_info "检查完成时间: $(date)"
        log_info "总检查项目: $TOTAL_CHECKS"
        
        # 计算通过的检查模块数量
        local passed_modules=$((TOTAL_CHECKS - ERROR_COUNT))
        if [ "$passed_modules" -lt 0 ]; then
            passed_modules=0
        fi
        
        log_success "通过模块: $passed_modules/$TOTAL_CHECKS"
        log_info "警告事件: $WARNING_COUNT"
        log_info "失败事件: $ERROR_COUNT"
        log_info "详细日志: $LOG_FILE"
        
        log_info ""
        if [ "$ERROR_COUNT" -eq 0 ] && [ "$WARNING_COUNT" -eq 0 ]; then
            log_success "🎉 所有检查项目均通过！InfiniBand网络状态优秀。"
        elif [ "$ERROR_COUNT" -eq 0 ]; then
            log_info "⚠️  检查完成，有 $WARNING_COUNT 个警告事件需要关注。"
        else
            log_error "❌ 检查发现 $ERROR_COUNT 个严重问题，需要立即处理。"
        fi
        
        log_info ""
        log_info "建议:"
        if [ "$ERROR_COUNT" -gt 0 ]; then
            log_info "1. 优先解决失败项目"
            log_info "2. 检查OFED驱动安装"
            log_info "3. 验证硬件连接"
        fi
        if [ "$WARNING_COUNT" -gt 0 ]; then
            log_info "1. 查看优化建议章节"
            log_info "2. 考虑性能调优"
            log_info "3. 定期监控网络状态"
        fi
        
        log_info ""
        log_info "如需技术支持，请提供日志文件: $LOG_FILE"
    fi
}

# 主函数
main() {
    # 解析命令行参数
    parse_arguments "$@"
    
    # 设置颜色
    set_colors
    
    # 脚本开始
    if [ "$SUMMARY_ONLY" = false ]; then
        clear
        log_header "$SCRIPT_NAME v$VERSION"
        log_info "开始 InfiniBand 网络健康检查..."
        log_info "日志文件: $LOG_FILE"
        log_info ""
    fi
    
    # 检查root权限
    check_root
    
    # 执行各项检查
    if ! check_dependencies; then
        log_error "依赖检查失败，无法继续执行"
        exit 1
    fi
    
    # 根据模式执行不同的检查
    if [ "$SUMMARY_ONLY" = false ]; then
        check_system_info
        check_ib_hardware
        check_ofed_driver
        check_port_status
        check_network_topology
        check_performance_counters
        check_network_interfaces
        check_performance_tools
        check_system_optimization
        
        # 生成建议
        generate_recommendations
    else
        # 摘要模式：只执行关键检查
        check_ib_hardware
        check_port_status
        check_performance_counters
    fi
    
    # 生成总结
    generate_summary
    
    # 根据结果设置退出码
    if [ "$ERROR_COUNT" -gt 0 ]; then
        exit 1
    elif [ "$WARNING_COUNT" -gt 0 ]; then
        exit 2
    else
        exit 0
    fi
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi