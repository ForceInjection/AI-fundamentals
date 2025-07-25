#!/bin/bash

# =============================================================================
# InfiniBand 网络带宽监控脚本
# 功能: 实时监控 InfiniBand 网络带宽，SAR 格式输出
# 作者：Grissom
# 版本: 2.0
# =============================================================================

# 全局变量
SCRIPT_NAME="$(basename "$0")"
VERSION="2.0"
MONITOR_INTERVAL=5
MONITOR_DURATION=0  # 0 表示无限监控
DEVICE=""
PORT=""
PORTS_LIST=""
DEVICES_PORTS_CONFIG=""  # 多设备端口配置，格式: device1:port1,port2;device2:port3,port4
QUIET_MODE=false

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 日志函数
log_info() {
    [ "$QUIET_MODE" = false ] && echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warning() {
    [ "$QUIET_MODE" = false ] && echo -e "${YELLOW}[WARN]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
InfiniBand 网络带宽监控脚本 (简化版) v${VERSION}

用法: $SCRIPT_NAME [选项]

选项:
  -h, --help              显示此帮助信息
  -v, --version           显示版本信息
  -d, --device DEVICE     指定设备名称 (如: mlx5_0)
  -p, --port PORT         指定端口号 (默认: 1)
  --ports PORTS           指定多个端口 (如: 1,2,3 或 1-4)
  --multi-devices CONFIG  指定多设备多端口配置 (如: mlx5_0:1,2;mlx5_1:1,3)
  -i, --interval SECONDS  监控间隔秒数 (默认: 5)
  -t, --time SECONDS      监控总时长秒数 (默认: 0，表示无限监控)
  -q, --quiet             静默模式 (仅输出数据)
  --list-ports            列出所有可监控的端口

输出格式: SAR 风格 (固定)
  时间戳     接口      rxpck/s   txpck/s    rxMB/s    txMB/s   %ifutil

示例:
  $SCRIPT_NAME --list-ports           # 列出所有可监控端口
  $SCRIPT_NAME                        # 自动检测设备并监控
  $SCRIPT_NAME -d mlx5_0 -p 1 -i 1   # 监控指定设备，1秒间隔
  $SCRIPT_NAME -d mlx5_0 --ports 1,2 # 同时监控多个端口
  $SCRIPT_NAME --multi-devices "mlx5_0:1,2;mlx5_1:1"  # 监控多设备多端口
  $SCRIPT_NAME --multi-devices "mlx5_0:1-3;mlx5_1:2,4;mlx5_2:1"  # 复杂配置
  $SCRIPT_NAME -t 60                  # 监控60秒后自动停止
  $SCRIPT_NAME -t 300 -i 1            # 监控5分钟，1秒间隔
  $SCRIPT_NAME -q                     # 静默模式，仅输出数据

多设备配置格式说明:
  - 设备间用分号(;)分隔
  - 设备名和端口用冒号(:)分隔
  - 端口间用逗号(,)分隔
  - 支持端口范围，如 1-4 表示端口 1,2,3,4
  - 示例: "mlx5_0:1,2,3;mlx5_1:1-4;mlx5_2:2"

EOF
}

# 显示版本信息
show_version() {
    echo "$SCRIPT_NAME version $VERSION"
}

# 检查依赖
check_dependencies() {
    local missing_deps=()
    local required_commands=("perfquery" "ibstat" "bc")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "缺少必要的命令: ${missing_deps[*]}"
        log_error "请安装: sudo apt install infiniband-diags bc"
        exit 1
    fi
}

# 自动检测IB设备
detect_ib_devices() {
    local devices
    devices=$(ibstat -l 2>/dev/null | head -5)
    
    if [ -z "$devices" ]; then
        log_error "未检测到 InfiniBand 设备"
        exit 1
    fi
    
    echo "$devices"
}

# 获取设备端口信息
get_device_ports() {
    local device="$1"
    local ports
    
    ports=$(ibstat "$device" 2>/dev/null | grep "Number of ports" | awk '{print $4}')
    
    if [ -z "$ports" ] || [ "$ports" -eq 0 ]; then
        log_error "设备 $device 没有可用端口"
        exit 1
    fi
    
    echo "$ports"
}

# 列出所有可监控的端口
list_available_ports() {
    [ "$QUIET_MODE" = false ] && echo "=== 可监控的 InfiniBand 端口 ==="
    
    # 检测设备
    local devices
    devices=$(ibstat -l 2>/dev/null | head -5)
    
    if [ -z "$devices" ]; then
        log_warning "未检测到 InfiniBand 设备"
        echo "请检查: lspci | grep -i infiniband"
        echo "加载驱动: sudo modprobe mlx5_ib"
        return 1
    fi
    
    local active_ports=0
    
    printf "%-15s %-8s %-12s %-15s\n" "设备" "端口" "状态" "速率"
    printf "%-15s %-8s %-12s %-15s\n" "----" "----" "----" "----"
    
    for device in $devices; do
        local device_info
        device_info=$(ibstat "$device" 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            local num_ports
            num_ports=$(echo "$device_info" | grep "Number of ports" | awk '{print $4}')
            
            if [ -n "$num_ports" ] && [ "$num_ports" -gt 0 ]; then
                for ((port=1; port<=num_ports; port++)); do
                    local port_info
                    port_info=$(ibstat "$device" "$port" 2>/dev/null)
                    
                    if [ $? -eq 0 ]; then
                        local state
                        local rate
                        state=$(echo "$port_info" | grep "State:" | awk '{print $2}')
                        rate=$(echo "$port_info" | grep "Rate:" | awk '{print $2}')
                        
                        printf "%-15s %-8s %-12s %-15s\n" "$device" "$port" "$state" "$rate"
                        
                        if [ "$state" = "Active" ]; then
                            active_ports=$((active_ports + 1))
                        fi
                    fi
                done
            fi
        fi
    done
    
    [ "$QUIET_MODE" = false ] && echo "活跃端口: $active_ports 个"
}

# 解析多设备配置
parse_multi_devices_config() {
    local config="$1"
    local -A devices_ports_map
    
    # 分割设备配置 (用分号分隔)
    IFS=';' read -ra DEVICE_CONFIGS <<< "$config"
    
    for device_config in "${DEVICE_CONFIGS[@]}"; do
        # 分割设备名和端口列表 (用冒号分隔)
        if [[ "$device_config" == *":"* ]]; then
            local device_name
            local ports_str
            device_name=$(echo "$device_config" | cut -d':' -f1)
            ports_str=$(echo "$device_config" | cut -d':' -f2)
            
            # 验证设备是否存在
            if ! ibstat "$device_name" >/dev/null 2>&1; then
                log_error "设备 $device_name 不存在或无法访问"
                return 1
            fi
            
            # 解析端口列表
            local ports_array
            ports_array=($(parse_ports_list "$ports_str" "$device_name"))
            
            if [ ${#ports_array[@]} -eq 0 ]; then
                log_warning "设备 $device_name 没有有效的端口配置"
                continue
            fi
            
            devices_ports_map["$device_name"]="${ports_array[*]}"
        else
            log_error "无效的设备配置格式: $device_config (应为 device:ports 格式)"
            return 1
        fi
    done
    
    # 输出解析结果 (格式: device1:port1,port2;device2:port3,port4)
    local result=""
    for device in "${!devices_ports_map[@]}"; do
        local ports_str="${devices_ports_map[$device]}"
        ports_str=$(echo "$ports_str" | tr ' ' ',')
        if [ -n "$result" ]; then
            result="${result};${device}:${ports_str}"
        else
            result="${device}:${ports_str}"
        fi
    done
    
    echo "$result"
}

# 解析端口列表
parse_ports_list() {
    local ports_str="$1"
    local device="$2"
    local ports_array=()
    
    local max_ports
    max_ports=$(get_device_ports "$device")
    
    IFS=',' read -ra PORT_RANGES <<< "$ports_str"
    
    for range in "${PORT_RANGES[@]}"; do
        if [[ "$range" == *"-"* ]]; then
            local start_port
            local end_port
            start_port=$(echo "$range" | cut -d'-' -f1)
            end_port=$(echo "$range" | cut -d'-' -f2)
            
            for ((port=start_port; port<=end_port; port++)); do
                if [ "$port" -le "$max_ports" ] && [ "$port" -ge 1 ]; then
                    ports_array+=("$port")
                fi
            done
        else
            if [[ "$range" =~ ^[0-9]+$ ]] && [ "$range" -le "$max_ports" ] && [ "$range" -ge 1 ]; then
                ports_array+=("$range")
            fi
        fi
    done
    
    local unique_ports
    unique_ports=($(printf '%s\n' "${ports_array[@]}" | sort -nu))
    
    echo "${unique_ports[@]}"
}

# 获取端口状态
get_port_status() {
    local device="$1"
    local port="$2"
    
    local status
    status=$(ibstat "$device" "$port" 2>/dev/null | grep "State:" | awk '{print $2}')
    
    echo "$status"
}

# 获取链路速率
get_link_rate() {
    local device="$1"
    local port="$2"
    
    local rate_str
    rate_str=$(ibstat "$device" "$port" 2>/dev/null | grep "Rate:" | awk '{print $2}')
    
    # 转换为 Gbps
    case "$rate_str" in
        "10")
            echo "10"
            ;;
        "20")
            echo "20"
            ;;
        "40")
            echo "40"
            ;;
        "56")
            echo "56"
            ;;
        "100")
            echo "100"
            ;;
        "200")
            echo "200"
            ;;
        *)
            echo "0"
            ;;
    esac
}

# 获取性能计数器
get_performance_counters() {
    local device="$1"
    local port="$2"
    
    local perfquery_output
    local perfquery_exit
    
    # 尝试不同的 perfquery 调用方式
    perfquery_output=$(timeout 10 perfquery -C "$device" -P "$port" 2>/dev/null)
    perfquery_exit=$?
    
    if [ $perfquery_exit -ne 0 ]; then
        local device_lid
        device_lid=$(ibstat "$device" "$port" 2>/dev/null | grep "Base lid:" | awk '{print $3}')
        
        if [ -n "$device_lid" ] && [ "$device_lid" != "0" ]; then
            perfquery_output=$(timeout 10 perfquery "$device_lid" "$port" 2>/dev/null)
            perfquery_exit=$?
        fi
    fi
    
    if [ $perfquery_exit -ne 0 ]; then
        echo "0 0 0 0"
        return 1
    fi
    
    # 解析输出
    local xmit_data
    local rcv_data
    local xmit_pkts
    local rcv_pkts
    
    xmit_data=$(echo "$perfquery_output" | grep "PortXmitData:" | awk '{print $2}')
    rcv_data=$(echo "$perfquery_output" | grep "PortRcvData:" | awk '{print $2}')
    xmit_pkts=$(echo "$perfquery_output" | grep "PortXmitPkts:" | awk '{print $2}')
    rcv_pkts=$(echo "$perfquery_output" | grep "PortRcvPkts:" | awk '{print $2}')
    
    # 验证数据
    xmit_data=${xmit_data:-0}
    rcv_data=${rcv_data:-0}
    xmit_pkts=${xmit_pkts:-0}
    rcv_pkts=${rcv_pkts:-0}
    
    echo "$xmit_data $rcv_data $xmit_pkts $rcv_pkts"
}

# 计算带宽
calculate_bandwidth() {
    local prev_xmit="$1"
    local curr_xmit="$2"
    local prev_rcv="$3"
    local curr_rcv="$4"
    local interval="$5"
    
    # 检查参数有效性
    if [ "$interval" -le 0 ]; then
        echo "0.00 0.00"
        return
    fi
    
    if [ "$prev_xmit" -eq 0 ] || [ "$prev_rcv" -eq 0 ]; then
        echo "0.00 0.00"
        return
    fi
    
    local xmit_diff
    local rcv_diff
    xmit_diff=$((curr_xmit - prev_xmit))
    rcv_diff=$((curr_rcv - prev_rcv))
    
    # 处理计数器回绕 (支持32位和64位计数器)
    if [ $xmit_diff -lt 0 ]; then
        # 尝试64位回绕，如果仍然为负则使用32位回绕
        local xmit_diff_64=$((18446744073709551616 + xmit_diff))
        if [ $xmit_diff_64 -gt 0 ] && [ $xmit_diff_64 -lt 18446744073709551616 ]; then
            xmit_diff=$xmit_diff_64
        else
            xmit_diff=$((4294967296 + xmit_diff))
        fi
    fi
    
    if [ $rcv_diff -lt 0 ]; then
        # 尝试64位回绕，如果仍然为负则使用32位回绕
        local rcv_diff_64=$((18446744073709551616 + rcv_diff))
        if [ $rcv_diff_64 -gt 0 ] && [ $rcv_diff_64 -lt 18446744073709551616 ]; then
            rcv_diff=$rcv_diff_64
        else
            rcv_diff=$((4294967296 + rcv_diff))
        fi
    fi
    
    # 计算带宽 (4字节单位转换为 MB/s)
    local xmit_mbps
    local rcv_mbps
    xmit_mbps=$(echo "scale=2; $xmit_diff * 4 / $interval / 1024 / 1024" | bc)
    rcv_mbps=$(echo "scale=2; $rcv_diff * 4 / $interval / 1024 / 1024" | bc)
    
    echo "$xmit_mbps $rcv_mbps"
}

# 计算包速率
calculate_packet_rate() {
    local prev_xmit_pkts="$1"
    local curr_xmit_pkts="$2"
    local prev_rcv_pkts="$3"
    local curr_rcv_pkts="$4"
    local interval="$5"
    
    # 检查参数有效性
    if [ "$interval" -le 0 ]; then
        echo "0 0"
        return
    fi
    
    if [ "$prev_xmit_pkts" -eq 0 ] || [ "$prev_rcv_pkts" -eq 0 ]; then
        echo "0 0"
        return
    fi
    
    local xmit_pkt_diff
    local rcv_pkt_diff
    xmit_pkt_diff=$((curr_xmit_pkts - prev_xmit_pkts))
    rcv_pkt_diff=$((curr_rcv_pkts - prev_rcv_pkts))
    
    # 处理计数器回绕 (支持32位和64位计数器)
    if [ $xmit_pkt_diff -lt 0 ]; then
        # 尝试64位回绕，如果仍然为负则使用32位回绕
        local xmit_pkt_diff_64=$((18446744073709551616 + xmit_pkt_diff))
        if [ $xmit_pkt_diff_64 -gt 0 ] && [ $xmit_pkt_diff_64 -lt 18446744073709551616 ]; then
            xmit_pkt_diff=$xmit_pkt_diff_64
        else
            xmit_pkt_diff=$((4294967296 + xmit_pkt_diff))
        fi
    fi
    
    if [ $rcv_pkt_diff -lt 0 ]; then
        # 尝试64位回绕，如果仍然为负则使用32位回绕
        local rcv_pkt_diff_64=$((18446744073709551616 + rcv_pkt_diff))
        if [ $rcv_pkt_diff_64 -gt 0 ] && [ $rcv_pkt_diff_64 -lt 18446744073709551616 ]; then
            rcv_pkt_diff=$rcv_pkt_diff_64
        else
            rcv_pkt_diff=$((4294967296 + rcv_pkt_diff))
        fi
    fi
    
    # 计算包速率
    local xmit_pps
    local rcv_pps
    xmit_pps=$((xmit_pkt_diff / interval))
    rcv_pps=$((rcv_pkt_diff / interval))
    
    echo "$xmit_pps $rcv_pps"
}

# 计算利用率
calculate_utilization() {
    local xmit_mbps="$1"
    local rcv_mbps="$2"
    local link_rate_gbps="$3"
    
    # 检查参数有效性
    if [ "$link_rate_gbps" -eq 0 ] || [ -z "$link_rate_gbps" ]; then
        echo "0.00"
        return
    fi
    
    # 检查输入参数是否为有效数字
    if ! [[ "$xmit_mbps" =~ ^[0-9]+\.?[0-9]*$ ]] || ! [[ "$rcv_mbps" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "0.00"
        return
    fi
    
    local link_rate_mbps
    # 转换为 MB/s (Gbps -> MB/s: Gbps × 1000 ÷ 8 = Gbps × 125)
    link_rate_mbps=$(echo "$link_rate_gbps * 125" | bc)
    
    # 检查计算结果是否有效
    if [ -z "$link_rate_mbps" ] || [ "$(echo "$link_rate_mbps <= 0" | bc)" -eq 1 ]; then
        echo "0.00"
        return
    fi
    
    local max_rate
    max_rate=$(echo "scale=2; if ($xmit_mbps > $rcv_mbps) $xmit_mbps else $rcv_mbps" | bc)
    
    local utilization
    utilization=$(echo "scale=2; $max_rate * 100 / $link_rate_mbps" | bc)
    
    echo "$utilization"
}

# SAR 格式输出头部
output_sar_header() {
    printf "%-15s %-18s %10s %10s %10s %10s %10s\n" \
        "时间" "接口" "rxpck/s" "txpck/s" "rxMB/s" "txMB/s" "%ifutil"
}

# SAR 格式输出
format_sar_output() {
    local timestamp="$1"
    local device="$2"
    local port="$3"
    local rxpps="$4"
    local txpps="$5"
    local rxmbps="$6"
    local txmbps="$7"
    local utilization="$8"

    printf "%-15s %-18s %-10d %-10d %-10.2f %-10.2f %-10.2f\n" \
        "$timestamp" "${device}:${port}" "$rxpps" "$txpps" "$rxmbps" "$txmbps" "$utilization"
}

# 监控单个端口
monitor_single_port() {
    local device="$1"
    local port="$2"
    
    # 检查端口状态
    local port_status
    port_status=$(get_port_status "$device" "$port")
    
    if [ "$port_status" != "Active" ]; then
        log_error "端口 ${device}:${port} 状态不是 Active (当前: $port_status)"
        return 1
    fi
    
    # 获取链路速率
    local link_rate
    link_rate=$(get_link_rate "$device" "$port")
    
    if [ "$MONITOR_DURATION" -gt 0 ]; then
        [ "$QUIET_MODE" = false ] && log_info "监控 ${device}:${port} (${link_rate} Gbps) - 持续时间: ${MONITOR_DURATION}秒"
    else
        [ "$QUIET_MODE" = false ] && log_info "监控 ${device}:${port} (${link_rate} Gbps)"
    fi
    
    # 获取初始计数器
    local prev_counters
    prev_counters=$(get_performance_counters "$device" "$port")
    
    if [ $? -ne 0 ]; then
        log_error "无法获取性能计数器"
        return 1
    fi
    
    local prev_xmit_data
    local prev_rcv_data
    local prev_xmit_pkts
    local prev_rcv_pkts
    read -r prev_xmit_data prev_rcv_data prev_xmit_pkts prev_rcv_pkts <<< "$prev_counters"
    
    # 输出头部
    [ "$QUIET_MODE" = false ] && output_sar_header
    
    # 记录开始时间
    local start_time
    start_time=$(date +%s)
    
    while true; do
        sleep "$MONITOR_INTERVAL"
        
        # 检查是否超过监控时间
        if [ "$MONITOR_DURATION" -gt 0 ]; then
            local current_time
            current_time=$(date +%s)
            local elapsed_time
            elapsed_time=$((current_time - start_time))
            
            if [ "$elapsed_time" -ge "$MONITOR_DURATION" ]; then
                [ "$QUIET_MODE" = false ] && log_info "监控时间已到达 ${MONITOR_DURATION} 秒，停止监控"
                break
            fi
        fi
        
        # 获取当前计数器
        local curr_counters
        curr_counters=$(get_performance_counters "$device" "$port")
        
        if [ $? -ne 0 ]; then
            log_warning "获取计数器失败，跳过此次采样"
            continue
        fi
        
        local curr_xmit_data
        local curr_rcv_data
        local curr_xmit_pkts
        local curr_rcv_pkts
        read -r curr_xmit_data curr_rcv_data curr_xmit_pkts curr_rcv_pkts <<< "$curr_counters"
        
        # 计算带宽
        local bandwidth
        bandwidth=$(calculate_bandwidth "$prev_xmit_data" "$curr_xmit_data" "$prev_rcv_data" "$curr_rcv_data" "$MONITOR_INTERVAL")
        local txmbps
        local rxmbps
        read -r txmbps rxmbps <<< "$bandwidth"
        
        # 计算包速率
        local packet_rate
        packet_rate=$(calculate_packet_rate "$prev_xmit_pkts" "$curr_xmit_pkts" "$prev_rcv_pkts" "$curr_rcv_pkts" "$MONITOR_INTERVAL")
        local txpps
        local rxpps
        read -r txpps rxpps <<< "$packet_rate"
        
        # 计算利用率
        local utilization
        utilization=$(calculate_utilization "$txmbps" "$rxmbps" "$link_rate")
        
        # 输出 SAR 格式
        local timestamp
        timestamp=$(date +%H:%M:%S)
        format_sar_output "$timestamp" "$device" "$port" "$rxpps" "$txpps" "$rxmbps" "$txmbps" "$utilization"
        
        # 更新前一次的值
        prev_xmit_data=$curr_xmit_data
        prev_rcv_data=$curr_rcv_data
        prev_xmit_pkts=$curr_xmit_pkts
        prev_rcv_pkts=$curr_rcv_pkts
    done
}

# 监控多个端口
monitor_multiple_ports() {
    local device="$1"
    shift
    local ports=("$@")
    
    if [ "$MONITOR_DURATION" -gt 0 ]; then
        [ "$QUIET_MODE" = false ] && log_info "监控多个端口: ${device}:${ports[*]} - 持续时间: ${MONITOR_DURATION}秒"
    else
        [ "$QUIET_MODE" = false ] && log_info "监控多个端口: ${device}:${ports[*]}"
    fi
    
    # 检查所有端口状态
    for port in "${ports[@]}"; do
        local port_status
        port_status=$(get_port_status "$device" "$port")
        
        if [ "$port_status" != "Active" ]; then
            log_error "端口 ${device}:${port} 状态不是 Active (当前: $port_status)"
            return 1
        fi
    done
    
    # 初始化数据结构
    declare -A prev_xmit_data
    declare -A prev_rcv_data
    declare -A prev_xmit_pkts
    declare -A prev_rcv_pkts
    declare -A link_rates
    
    # 获取初始计数器和链路速率
    for port in "${ports[@]}"; do
        local counters
        counters=$(get_performance_counters "$device" "$port")
        
        if [ $? -eq 0 ]; then
            read -r prev_xmit_data[$port] prev_rcv_data[$port] prev_xmit_pkts[$port] prev_rcv_pkts[$port] <<< "$counters"
            link_rates[$port]=$(get_link_rate "$device" "$port")
        fi
    done
    
    # 输出头部
    [ "$QUIET_MODE" = false ] && output_sar_header
    
    # 记录开始时间
    local start_time
    start_time=$(date +%s)
    
    while true; do
        sleep "$MONITOR_INTERVAL"
        
        # 检查是否超过监控时间
        if [ "$MONITOR_DURATION" -gt 0 ]; then
            local current_time
            current_time=$(date +%s)
            local elapsed_time
            elapsed_time=$((current_time - start_time))
            
            if [ "$elapsed_time" -ge "$MONITOR_DURATION" ]; then
                [ "$QUIET_MODE" = false ] && log_info "监控时间已到达 ${MONITOR_DURATION} 秒，停止监控"
                break
            fi
        fi
        
        local timestamp
        timestamp=$(date +%H:%M:%S)
        
        for port in "${ports[@]}"; do
            # 获取当前计数器
            local curr_counters
            curr_counters=$(get_performance_counters "$device" "$port")
            
            if [ $? -ne 0 ]; then
                continue
            fi
            
            local curr_xmit_data
            local curr_rcv_data
            local curr_xmit_pkts
            local curr_rcv_pkts
            read -r curr_xmit_data curr_rcv_data curr_xmit_pkts curr_rcv_pkts <<< "$curr_counters"
            
            # 计算带宽
            local bandwidth
            bandwidth=$(calculate_bandwidth "${prev_xmit_data[$port]}" "$curr_xmit_data" "${prev_rcv_data[$port]}" "$curr_rcv_data" "$MONITOR_INTERVAL")
            local txmbps
            local rxmbps
            read -r txmbps rxmbps <<< "$bandwidth"
            
            # 计算包速率
            local packet_rate
            packet_rate=$(calculate_packet_rate "${prev_xmit_pkts[$port]}" "$curr_xmit_pkts" "${prev_rcv_pkts[$port]}" "$curr_rcv_pkts" "$MONITOR_INTERVAL")
            local txpps
            local rxpps
            read -r txpps rxpps <<< "$packet_rate"
            
            # 计算利用率
            local utilization
            utilization=$(calculate_utilization "$txmbps" "$rxmbps" "${link_rates[$port]}")
            
            # 输出 SAR 格式
            format_sar_output "$timestamp" "$device" "$port" "$rxpps" "$txpps" "$rxmbps" "$txmbps" "$utilization"
            
            # 更新前一次的值
            prev_xmit_data[$port]=$curr_xmit_data
            prev_rcv_data[$port]=$curr_rcv_data
            prev_xmit_pkts[$port]=$curr_xmit_pkts
            prev_rcv_pkts[$port]=$curr_rcv_pkts
        done
    done
}

# 监控多设备多端口
monitor_multi_devices() {
    local config="$1"
    
    [ "$QUIET_MODE" = false ] && log_info "开始多设备多端口监控"
    
    # 解析配置
    IFS=';' read -ra DEVICE_CONFIGS <<< "$config"
    
    # 存储所有设备端口的数据
    declare -A prev_xmit_data
    declare -A prev_rcv_data
    declare -A prev_xmit_pkts
    declare -A prev_rcv_pkts
    declare -A link_rates
    
    # 初始化所有设备端口的计数器
    for device_config in "${DEVICE_CONFIGS[@]}"; do
        local device_name
        local ports_str
        device_name=$(echo "$device_config" | cut -d':' -f1)
        ports_str=$(echo "$device_config" | cut -d':' -f2)
        
        IFS=',' read -ra ports <<< "$ports_str"
        
        for port in "${ports[@]}"; do
            local key="${device_name}:${port}"
            
            # 检查端口状态
            local port_status
            port_status=$(get_port_status "$device_name" "$port")
            
            if [ "$port_status" != "Active" ]; then
                log_warning "端口 ${key} 状态不是 Active (当前: $port_status)，跳过监控"
                continue
            fi
            
            # 获取初始计数器
            local counters
            counters=$(get_performance_counters "$device_name" "$port")
            
            if [ $? -eq 0 ]; then
                read -r prev_xmit_data[$key] prev_rcv_data[$key] prev_xmit_pkts[$key] prev_rcv_pkts[$key] <<< "$counters"
                link_rates[$key]=$(get_link_rate "$device_name" "$port")
                [ "$QUIET_MODE" = false ] && log_info "初始化端口 ${key}，链路速率: ${link_rates[$key]} Gbps"
            else
                log_warning "无法获取端口 ${key} 的初始计数器"
            fi
        done
    done
    
    # 检查是否有有效的端口
    if [ ${#prev_xmit_data[@]} -eq 0 ]; then
        log_error "没有有效的端口可监控"
        return 1
    fi
    
    # 输出头部
    [ "$QUIET_MODE" = false ] && output_sar_header
    
    # 记录开始时间
    local start_time
    start_time=$(date +%s)
    
    if [ "$MONITOR_DURATION" -gt 0 ]; then
        [ "$QUIET_MODE" = false ] && log_info "开始监控，持续时间: ${MONITOR_DURATION} 秒"
    else
        [ "$QUIET_MODE" = false ] && log_info "开始无限监控 (Ctrl+C 停止)"
    fi
    
    while true; do
        sleep "$MONITOR_INTERVAL"
        
        # 检查是否超过监控时间
        if [ "$MONITOR_DURATION" -gt 0 ]; then
            local current_time
            current_time=$(date +%s)
            local elapsed_time
            elapsed_time=$((current_time - start_time))
            
            if [ "$elapsed_time" -ge "$MONITOR_DURATION" ]; then
                [ "$QUIET_MODE" = false ] && log_info "监控时间已到达 ${MONITOR_DURATION} 秒，停止监控"
                break
            fi
        fi
        
        local timestamp
        timestamp=$(date +%H:%M:%S)
        
        # 遍历所有设备端口
        for device_config in "${DEVICE_CONFIGS[@]}"; do
            local device_name
            local ports_str
            device_name=$(echo "$device_config" | cut -d':' -f1)
            ports_str=$(echo "$device_config" | cut -d':' -f2)
            
            IFS=',' read -ra ports <<< "$ports_str"
            
            for port in "${ports[@]}"; do
                local key="${device_name}:${port}"
                
                # 跳过未初始化的端口
                if [ -z "${prev_xmit_data[$key]}" ]; then
                    continue
                fi
                
                # 获取当前计数器
                local curr_counters
                curr_counters=$(get_performance_counters "$device_name" "$port")
                
                if [ $? -ne 0 ]; then
                    continue
                fi
                
                local curr_xmit_data
                local curr_rcv_data
                local curr_xmit_pkts
                local curr_rcv_pkts
                read -r curr_xmit_data curr_rcv_data curr_xmit_pkts curr_rcv_pkts <<< "$curr_counters"
                
                # 计算带宽
            local bandwidth
            bandwidth=$(calculate_bandwidth "${prev_xmit_data[$key]}" "$curr_xmit_data" "${prev_rcv_data[$key]}" "$curr_rcv_data" "$MONITOR_INTERVAL")
            local txmbps
            local rxmbps
            read -r txmbps rxmbps <<< "$bandwidth"
            
            # 计算包速率
            local packet_rate
            packet_rate=$(calculate_packet_rate "${prev_xmit_pkts[$key]}" "$curr_xmit_pkts" "${prev_rcv_pkts[$key]}" "$curr_rcv_pkts" "$MONITOR_INTERVAL")
            local txpps
            local rxpps
            read -r txpps rxpps <<< "$packet_rate"
            
            # 计算利用率
            local utilization
            utilization=$(calculate_utilization "$txmbps" "$rxmbps" "${link_rates[$key]}")
            
            # 输出 SAR 格式
            format_sar_output "$timestamp" "$device_name" "$port" "$rxpps" "$txpps" "$rxmbps" "$txmbps" "$utilization"
                
                # 更新前一次的值
                prev_xmit_data[$key]=$curr_xmit_data
                prev_rcv_data[$key]=$curr_rcv_data
                prev_xmit_pkts[$key]=$curr_xmit_pkts
                prev_rcv_pkts[$key]=$curr_rcv_pkts
            done
        done
    done
}

# 信号处理
cleanup() {
    [ "$QUIET_MODE" = false ] && echo -e "\n监控已停止"
    exit 0
}

trap cleanup SIGINT SIGTERM

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
            -d|--device)
                DEVICE="$2"
                shift 2
                ;;
            -p|--port)
                PORT="$2"
                shift 2
                ;;
            --ports)
                PORTS_LIST="$2"
                shift 2
                ;;
            --multi-devices)
                DEVICES_PORTS_CONFIG="$2"
                shift 2
                ;;
            -i|--interval)
                MONITOR_INTERVAL="$2"
                shift 2
                ;;
            -t|--time)
                MONITOR_DURATION="$2"
                shift 2
                ;;
            -q|--quiet)
                QUIET_MODE=true
                shift
                ;;
            --list-ports)
                list_available_ports
                exit 0
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
    parse_arguments "$@"
    
    # 验证参数
    if ! [[ "$MONITOR_INTERVAL" =~ ^[0-9]+$ ]] || [ "$MONITOR_INTERVAL" -le 0 ]; then
        log_error "监控间隔必须是正整数"
        exit 1
    fi
    
    if ! [[ "$MONITOR_DURATION" =~ ^[0-9]+$ ]] || [ "$MONITOR_DURATION" -lt 0 ]; then
        log_error "监控时长必须是非负整数"
        exit 1
    fi
    
    # 检查依赖
    check_dependencies
    
    # 处理多设备多端口监控
    if [ -n "$DEVICES_PORTS_CONFIG" ]; then
        # 解析并验证多设备配置
        local parsed_config
        parsed_config=$(parse_multi_devices_config "$DEVICES_PORTS_CONFIG")
        
        if [ $? -ne 0 ]; then
            log_error "多设备配置解析失败"
            exit 1
        fi
        
        monitor_multi_devices "$DEVICES_PORTS_CONFIG"
        exit 0
    fi
    
    # 自动检测设备
    if [ -z "$DEVICE" ]; then
        local devices
        devices=$(detect_ib_devices)
        DEVICE=$(echo "$devices" | head -1)
        
        [ "$QUIET_MODE" = false ] && log_info "自动选择设备: $DEVICE"
    fi
    
    # 处理多端口监控
    if [ -n "$PORTS_LIST" ]; then
        local ports_array
        ports_array=($(parse_ports_list "$PORTS_LIST" "$DEVICE"))
        
        if [ ${#ports_array[@]} -eq 0 ]; then
            log_error "没有有效的端口可监控"
            exit 1
        fi
        
        monitor_multiple_ports "$DEVICE" "${ports_array[@]}"
    else
        # 单端口监控
        if [ -z "$PORT" ]; then
            PORT="1"
        fi
        
        # 验证端口
        local max_ports
        max_ports=$(get_device_ports "$DEVICE")
        
        if [ "$PORT" -gt "$max_ports" ]; then
            log_error "端口 $PORT 超出设备 $DEVICE 的最大端口数 ($max_ports)"
            exit 1
        fi
        
        monitor_single_port "$DEVICE" "$PORT"
    fi
}

# 执行主函数
main "$@"