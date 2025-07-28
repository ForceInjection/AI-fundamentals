#!/bin/bash

# =============================================================================
# InfiniBand 网络带宽监控脚本 (单文件版)
# 功能: 实时监控 InfiniBand 网络带宽，SAR 格式输出
# 作者：Grissom
# 版本: 2.2 (单文件版 - 内置通用工具库)
# =============================================================================

# ==================== 通用工具函数库 (内置) ====================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    [ "${QUIET_MODE:-false}" = false ] && echo -e "${GREEN}[INFO]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warning() {
    [ "${QUIET_MODE:-false}" = false ] && echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_debug() {
    if [ "${DEBUG_MODE:-false}" = true ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1" >&2
    fi
}

log_success() {
    [ "${QUIET_MODE:-false}" = false ] && echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

# 验证正整数参数
# 参数: $1=值, $2=参数名称, $3=最小值(可选,默认1)
validate_positive_integer() {
    local value="$1"
    local param_name="$2"
    local min_value="${3:-1}"
    
    # 检查是否为空
    if [ -z "$value" ]; then
        log_error "参数 $param_name 不能为空"
        return 1
    fi
    
    # 检查是否为正整数
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        log_error "参数 $param_name 必须是正整数: $value"
        return 1
    fi
    
    # 检查最小值
    if [ "$value" -lt "$min_value" ]; then
        log_error "参数 $param_name 必须大于等于 $min_value: $value"
        return 1
    fi
    
    return 0
}

# 验证非负整数参数
# 参数: $1=值, $2=参数名称
validate_non_negative_integer() {
    local value="$1"
    local param_name="$2"
    
    # 检查是否为空
    if [ -z "$value" ]; then
        log_error "参数 $param_name 不能为空"
        return 1
    fi
    
    # 检查是否为非负整数
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        log_error "参数 $param_name 必须是非负整数: $value"
        return 1
    fi
    
    return 0
}

# 验证浮点数参数
# 参数: $1=值, $2=参数名称
validate_float() {
    local value="$1"
    local param_name="$2"
    
    # 检查是否为空
    if [ -z "$value" ]; then
        log_error "参数 $param_name 不能为空"
        return 1
    fi
    
    # 检查是否为有效的浮点数
    if ! [[ "$value" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        log_error "参数 $param_name 必须是有效的数字: $value"
        return 1
    fi
    
    return 0
}

# 验证端口范围
# 参数: $1=起始端口, $2=结束端口, $3=最小值, $4=最大值
validate_port_range() {
    local start_port="$1"
    local end_port="$2"
    local min_port="$3"
    local max_port="$4"
    
    # 验证起始端口
    if ! validate_positive_integer "$start_port" "起始端口"; then
        return 1
    fi
    
    # 验证结束端口
    if ! validate_positive_integer "$end_port" "结束端口"; then
        return 1
    fi
    
    # 检查端口范围
    if [ "$start_port" -lt "$min_port" ] || [ "$start_port" -gt "$max_port" ]; then
        log_error "起始端口超出范围 ($min_port-$max_port): $start_port"
        return 1
    fi
    
    if [ "$end_port" -lt "$min_port" ] || [ "$end_port" -gt "$max_port" ]; then
        log_error "结束端口超出范围 ($min_port-$max_port): $end_port"
        return 1
    fi
    
    if [ "$start_port" -gt "$end_port" ]; then
        log_error "起始端口不能大于结束端口: $start_port > $end_port"
        return 1
    fi
    
    return 0
}

# 处理计数器回绕
# 参数: $1=差值
# 返回: 处理后的差值
handle_counter_wraparound() {
    local diff="$1"
    
    # 如果差值为正，直接返回
    if [ "$diff" -ge 0 ]; then
        echo "$diff"
        return 0
    fi
    
    # 尝试64位计数器回绕
    local diff_64=$((18446744073709551616 + diff))
    if [ "$diff_64" -gt 0 ] && [ "$diff_64" -lt 18446744073709551616 ]; then
        echo "$diff_64"
        return 0
    fi
    
    # 使用32位计数器回绕
    local diff_32=$((4294967296 + diff))
    echo "$diff_32"
    return 0
}

# 安全的数学计算（避免除零错误）
# 参数: $1=表达式
safe_calc() {
    local expression="$1"
    
    # 检查除零
    if [[ "$expression" =~ /[[:space:]]*0[[:space:]]*$ ]]; then
        echo "0"
        return 1
    fi
    
    # 使用bc进行计算
    local result
    result=$(echo "scale=2; $expression" | bc 2>/dev/null)
    
    # 检查计算结果
    if [ -z "$result" ] || [ "$result" = "0" ]; then
        echo "0"
        return 1
    fi
    
    echo "$result"
    return 0
}

# 计算百分比
# 参数: $1=部分值, $2=总值
calculate_percentage() {
    local part="$1"
    local total="$2"
    
    # 验证输入参数是否为有效数字（包括小数）
    if ! [[ "$part" =~ ^[0-9]+\.?[0-9]*$ ]] || ! [[ "$total" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "0.00"
        return 1
    fi
    
    # 检查总值是否大于0（使用bc进行浮点数比较）
    local is_zero
    is_zero=$(echo "scale=2; if ($total <= 0) 1 else 0" | bc 2>/dev/null)
    if [ "$is_zero" -eq 1 ]; then
        echo "0.00"
        return 1
    fi
    
    safe_calc "$part * 100 / $total"
}

# 退出并显示错误信息
die() {
    log_error "$1"
    exit "${2:-1}"
}

# 初始化通用工具库
init_common_utils() {
    # 设置默认值
    QUIET_MODE="${QUIET_MODE:-false}"
    DEBUG_MODE="${DEBUG_MODE:-false}"
    
    # 检查必要的命令
    local required_commands=("bc" "date" "grep" "awk" "sed")
    local missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [ ${#missing_commands[@]} -gt 0 ]; then
        die "缺少必要的命令: ${missing_commands[*]}"
    fi
    
    log_debug "通用工具库初始化完成"
    return 0
}

# ==================== 主脚本开始 ====================

# 全局变量
SCRIPT_NAME="$(basename "$0")"
VERSION="2.2"
MONITOR_INTERVAL=5
MONITOR_DURATION=0  # 0 表示无限监控
DEVICE=""
PORT=""
PORTS_LIST=""
DEVICES_PORTS_CONFIG=""  # 多设备端口配置，格式: device1:port1,port2;device2:port3,port4
QUIET_MODE=false
DEBUG_MODE=false

# 初始化通用工具库
init_common_utils

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

验证功能:
  脚本会在开始监控前自动验证:
  - 设备是否存在且可访问
  - 端口是否在有效范围内
  - 端口状态是否为 Active (活跃)
  - 对于非活跃端口，会提示用户确认是否继续

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

故障排除:
  如果遇到设备或端口验证失败，请:
  1. 使用 --list-ports 查看可用的活跃端口
  2. 检查 InfiniBand 驱动是否正确加载
  3. 确认网络连接状态
  4. 检查设备权限

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

# 获取设备端口信息 - 确保返回数字
get_device_ports() {
    local device="$1"
    local ports
    
    # 使用临时文件避免命令替换中的混合输出
    local temp_file="/tmp/ibstat_$$"
    if ibstat "$device" >"$temp_file" 2>/dev/null; then
        ports=$(grep "Number of ports" "$temp_file" | awk '{print $4}' | head -1)
        rm -f "$temp_file"
        
        # 验证并清理输出，确保只返回数字
        if [[ "$ports" =~ ^[0-9]+$ ]] && [ "$ports" -gt 0 ]; then
            echo "$ports"
            return 0
        fi
    else
        rm -f "$temp_file"
    fi
    
    # 如果获取失败，记录错误但不退出（让调用者处理）
    log_error "设备 $device 没有可用端口或无法访问" >&2
    echo "0"
    return 1
}

# 列出所有可监控的端口
list_available_ports() {
    [ "$QUIET_MODE" = false ] && echo "=== 可监控的 InfiniBand 端口 ==="
    
    # 检测设备
    local devices
    devices=$(ibstat -l 2>/dev/null)
    
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
    local skipped_ranges=()
    
    local max_ports
    max_ports=$(get_device_ports "$device")
    
    IFS=',' read -ra PORT_RANGES <<< "$ports_str"
    
    for range in "${PORT_RANGES[@]}"; do
        if [[ "$range" == *"-"* ]]; then
            local start_port
            local end_port
            start_port=$(echo "$range" | cut -d'-' -f1)
            end_port=$(echo "$range" | cut -d'-' -f2)
            
            # 验证端口范围
            if [[ "$start_port" =~ ^[0-9]+$ ]] && [[ "$end_port" =~ ^[0-9]+$ ]] && \
               [ "$start_port" -ge 1 ] && [ "$start_port" -le "$max_ports" ] && \
               [ "$end_port" -ge 1 ] && [ "$end_port" -le "$max_ports" ] && \
               [ "$start_port" -le "$end_port" ]; then
                for ((port=start_port; port<=end_port; port++)); do
                    ports_array+=("$port")
                done
            else
                # 记录跳过的端口范围
                skipped_ranges+=("$range")
            fi
        else
            # 验证单个端口
            if [[ "$range" =~ ^[0-9]+$ ]] && \
               [ "$range" -ge 1 ] && [ "$range" -le "$max_ports" ]; then
                ports_array+=("$range")
            else
                # 记录跳过的端口
                skipped_ranges+=("$range")
            fi
        fi
    done
    
    # 输出跳过的端口信息（仅在非静默模式下）
    if [ ${#skipped_ranges[@]} -gt 0 ] && [ "$QUIET_MODE" = false ]; then
        log_warning "设备 $device 跳过无效端口/范围: ${skipped_ranges[*]} (有效范围: 1-$max_ports)" >&2
    fi
    
    local unique_ports
    unique_ports=($(printf '%s\n' "${ports_array[@]}" | sort -nu))
    
    echo "${unique_ports[@]}"
}

# 获取端口状态 - 确保返回状态字符串
get_port_status() {
    local device="$1"
    local port="$2"
    
    # 使用临时文件避免命令替换中的混合输出
    local temp_file="/tmp/ibstat_port_$$"
    if ibstat "$device" "$port" >"$temp_file" 2>/dev/null; then
        local status
        status=$(grep "State:" "$temp_file" | awk '{print $2}' | head -1)
        rm -f "$temp_file"
        
        # 验证状态字符串
        if [ -n "$status" ]; then
            echo "$status"
            return 0
        fi
    else
        rm -f "$temp_file"
    fi
    
    # 返回未知状态
    echo "Unknown"
    return 1
}

# 验证设备是否存在且可访问
validate_device() {
    local device="$1"
    
    if [ -z "$device" ]; then
        log_error "设备名称不能为空"
        return 1
    fi
    
    # 检查设备是否存在
    if ! ibstat "$device" >/dev/null 2>&1; then
        log_error "设备 '$device' 不存在或无法访问"
        log_info "可用设备列表:"
        local available_devices
        available_devices=$(ibstat -l 2>/dev/null)
        if [ -n "$available_devices" ]; then
            echo "$available_devices" | while read -r dev; do
                echo "  - $dev"
            done
        else
            echo "  (未检测到任何 InfiniBand 设备)"
        fi
        return 1
    fi
    
    return 0
}

# 验证端口是否存在且活跃
validate_port() {
    local device="$1"
    local port="$2"
    local check_active="${3:-true}"  # 默认检查端口是否活跃
    local silent="${4:-false}"       # 是否静默模式，避免重复错误信息
    
    if [ -z "$port" ]; then
        [ "$silent" = false ] && log_error "端口号不能为空"
        return 1
    fi
    
    # 静默验证端口号格式
    if [[ ! "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ]; then
        [ "$silent" = false ] && log_error "端口号必须是正整数: $port"
        return 1
    fi
    
    # 检查端口是否在有效范围内
    local max_ports
    max_ports=$(get_device_ports "$device")
    
    if [ "$port" -lt 1 ] || [ "$port" -gt "$max_ports" ]; then
        [ "$silent" = false ] && log_error "端口 $port 超出设备 $device 的有效范围 (1-$max_ports)"
        return 1
    fi
    
    # 检查端口是否存在
    local port_info
    port_info=$(ibstat "$device" "$port" 2>/dev/null)
    
    if [ $? -ne 0 ] || [ -z "$port_info" ]; then
        [ "$silent" = false ] && log_error "设备 $device 的端口 $port 不存在或无法访问"
        return 1
    fi
    
    # 如果需要检查端口是否活跃
    if [ "$check_active" = "true" ]; then
        local port_status
        port_status=$(get_port_status "$device" "$port")
        
        if [ "$port_status" != "Active" ]; then
            [ "$silent" = false ] && log_warning "设备 $device 的端口 $port 状态为 '$port_status'，不是 'Active'"
            [ "$silent" = false ] && log_info "建议使用以下命令查看活跃端口:"
            [ "$silent" = false ] && log_info "  $SCRIPT_NAME --list-ports"
            
            # 询问用户是否继续
            if [ "$QUIET_MODE" = false ] && [ "$silent" = false ]; then
                echo -n "是否继续监控非活跃端口? [y/N]: "
                read -r response
                case "$response" in
                    [yY]|[yY][eE][sS])
                        log_info "继续监控非活跃端口 $device:$port"
                        ;;
                    *)
                        log_info "取消监控"
                        return 1
                        ;;
                esac
            else
                # 静默模式下，对非活跃端口返回警告但继续执行
                [ "$silent" = false ] && log_warning "静默模式：继续监控非活跃端口 $device:$port"
            fi
        fi
    fi
    
    return 0
}

# 验证设备和端口配置（统一处理单端口和多端口）
validate_device_ports_config() {
    local device="$1"
    shift
    local ports=("$@")
    
    # 验证设备
    if ! validate_device "$device"; then
        return 1
    fi
    
    # 如果没有提供端口，返回错误
    if [ ${#ports[@]} -eq 0 ]; then
        log_error "必须提供至少一个端口"
        return 1
    fi
    
    local valid_ports=()
    local invalid_ports=()
    
    for port in "${ports[@]}"; do
        # 跳过空端口
        if [ -z "$port" ]; then
            continue
        fi
        
        # 使用静默模式验证端口，避免重复错误信息
        if validate_port "$device" "$port" "false" "true"; then
            # 检查端口状态
            local port_status
            port_status=$(get_port_status "$device" "$port")
            if [ "$port_status" = "Active" ]; then
                valid_ports+=("$port")
            else
                log_warning "端口 $device:$port 状态为 '$port_status' (非活跃)"
                invalid_ports+=("$port")
            fi
        else
            invalid_ports+=("$port")
        fi
    done
    
    if [ ${#valid_ports[@]} -eq 0 ]; then
        log_error "设备 $device 没有有效的活跃端口可监控"
        if [ ${#invalid_ports[@]} -gt 0 ]; then
            log_info "无效或非活跃端口: ${invalid_ports[*]}"
        fi
        return 1
    fi
    
    if [ ${#invalid_ports[@]} -gt 0 ]; then
        log_warning "以下端口将被跳过: ${invalid_ports[*]}"
    fi
    
    # 根据端口数量输出不同的信息
    if [ ${#ports[@]} -eq 1 ]; then
        log_info "设备 $device 端口 ${valid_ports[0]} 验证通过"
    else
        log_info "设备 $device 有效活跃端口: ${valid_ports[*]}"
    fi
    
    return 0
}

# 为了向后兼容，保留原函数名作为别名
validate_device_port_config() {
    validate_device_ports_config "$@"
}

validate_multiple_ports_config() {
    validate_device_ports_config "$@"
}

# 验证多设备多端口配置并返回有效配置
validate_multi_devices_config() {
    local config="$1"
    
    if [ -z "$config" ]; then
        log_error "多设备配置不能为空"
        return 1
    fi
    
    local valid_configs=()
    local invalid_configs=()
    
    # 分割设备配置 (用分号分隔)
    IFS=';' read -ra DEVICE_CONFIGS <<< "$config"
    
    for device_config in "${DEVICE_CONFIGS[@]}"; do
        # 跳过空配置
        if [ -z "$device_config" ]; then
            continue
        fi
        
        # 分割设备名和端口列表 (用冒号分隔)
        if [[ "$device_config" == *":"* ]]; then
            local device_name
            local ports_str
            device_name=$(echo "$device_config" | cut -d':' -f1)
            ports_str=$(echo "$device_config" | cut -d':' -f2)
            
            # 跳过空设备名或空端口
            if [ -z "$device_name" ] || [ -z "$ports_str" ]; then
                continue
            fi
            
            # 验证设备 - 使用静默检查避免重复错误信息
            if ! ibstat "$device_name" >/dev/null 2>&1; then
                log_warning "设备 '$device_name' 不存在或无法访问，跳过配置: $device_config"
                invalid_configs+=("$device_config")
                continue
            fi
            
            # 解析端口列表 - 先检查端口范围有效性
            local ports_array
            ports_array=($(parse_ports_list "$ports_str" "$device_name"))
            
            if [ ${#ports_array[@]} -eq 0 ]; then
                log_warning "设备 $device_name 没有有效的端口配置: $ports_str"
                invalid_configs+=("$device_config")
                continue
            fi
            
            # 验证端口
            local valid_ports=()
            local skipped_ports=()
            for port in "${ports_array[@]}"; do
                # 跳过空端口
                if [ -z "$port" ]; then
                    continue
                fi
                
                # 使用静默模式验证端口，避免重复错误信息
                if validate_port "$device_name" "$port" "false" "true"; then
                    local port_status
                    port_status=$(get_port_status "$device_name" "$port")
                    if [ "$port_status" = "Active" ]; then
                        valid_ports+=("$port")
                    else
                        skipped_ports+=("$port:$port_status")
                        log_warning "端口 $device_name:$port 状态为 '$port_status' (非活跃)，已跳过"
                    fi
                else
                    skipped_ports+=("$port:不存在")
                    log_warning "端口 $device_name:$port 不存在或无法访问，已跳过"
                fi
            done
            
            # 输出端口验证结果
            if [ ${#valid_ports[@]} -gt 0 ]; then
                valid_configs+=("$device_name:$(IFS=','; echo "${valid_ports[*]}")")
                log_info "设备 $device_name 有效活跃端口: ${valid_ports[*]}" >&2
            else
                log_warning "设备 $device_name 没有活跃端口"
                invalid_configs+=("$device_config")
            fi
            
            # 显示跳过的端口详情
            if [ ${#skipped_ports[@]} -gt 0 ]; then
                log_info "设备 $device_name 跳过的端口: ${skipped_ports[*]}" >&2
            fi
        else
            log_error "无效的设备配置格式: $device_config (应为 device:ports 格式)"
            invalid_configs+=("$device_config")
        fi
    done
    
    if [ ${#valid_configs[@]} -eq 0 ]; then
        log_error "没有有效的设备端口配置可监控"
        if [ ${#invalid_configs[@]} -gt 0 ]; then
            log_info "无效配置: ${invalid_configs[*]}"
        fi
        return 1
    fi
    
    if [ ${#invalid_configs[@]} -gt 0 ]; then
        log_warning "以下配置将被跳过: ${invalid_configs[*]}"
    fi
    
    log_info "有效的设备端口配置: ${valid_configs[*]}" >&2
    
    # 输出有效配置供后续使用
    echo "$(IFS=';'; echo "${valid_configs[*]}")"
    return 0
}

# 获取链路速率 - 确保返回数字
get_link_rate() {
    local device="$1"
    local port="$2"
    
    # 使用临时文件避免命令替换中的混合输出
    local temp_file="/tmp/ibstat_rate_$$"
    if ibstat "$device" "$port" >"$temp_file" 2>/dev/null; then
        local rate_str
        rate_str=$(grep "Rate:" "$temp_file" | awk '{print $2}' | head -1)
        rm -f "$temp_file"
        
        # 验证速率字符串并返回
        if [ -n "$rate_str" ]; then
            # 检查是否为有效的 InfiniBand 速率值
            case "$rate_str" in
                10|20|40|56|100|200)
                    echo "$rate_str"
                    ;;
                *)
                    echo "0"
                    ;;
            esac
        else
            echo "0"
        fi
    else
        rm -f "$temp_file"
        echo "0"
    fi
}

# 获取性能计数器 - 确保返回数字数据
get_performance_counters() {
    local device="$1"
    local port="$2"
    
    local perfquery_output
    local perfquery_exit
    local debug_info=""
    local temp_file="/tmp/perfquery_$$"
    
    # 方法1: 尝试使用 CA 名称和端口号
    if timeout 10 perfquery -C "$device" -P "$port" >"$temp_file" 2>&1; then
        perfquery_output=$(cat "$temp_file")
        perfquery_exit=0
        debug_info="Method1(-C $device -P $port): success"
    else
        perfquery_exit=$?
        debug_info="Method1(-C $device -P $port): exit=$perfquery_exit"
    fi
    
    # 如果方法1失败，尝试方法2: 使用设备名和端口号（不带 -C -P 参数）
    if [ $perfquery_exit -ne 0 ]; then
        if timeout 10 perfquery "$device" "$port" >"$temp_file" 2>&1; then
            perfquery_output=$(cat "$temp_file")
            perfquery_exit=0
            debug_info="$debug_info, Method2($device $port): success"
        else
            perfquery_exit=$?
            debug_info="$debug_info, Method2($device $port): exit=$perfquery_exit"
        fi
    fi
    
    # 如果方法2失败，尝试方法3: 使用 LID
    if [ $perfquery_exit -ne 0 ]; then
        local device_lid
        local temp_ibstat="/tmp/ibstat_lid_$$"
        if ibstat "$device" "$port" >"$temp_ibstat" 2>/dev/null; then
            device_lid=$(grep "Base lid:" "$temp_ibstat" | awk '{print $3}' | head -1)
            rm -f "$temp_ibstat"
            
            # 验证LID
            if [ -n "$device_lid" ] && [ "$device_lid" != "0" ] && [ "$device_lid" != "0x0" ]; then
                if timeout 10 perfquery "$device_lid" "$port" >"$temp_file" 2>&1; then
                    perfquery_output=$(cat "$temp_file")
                    perfquery_exit=0
                    debug_info="$debug_info, Method3(LID=$device_lid $port): success"
                else
                    perfquery_exit=$?
                    debug_info="$debug_info, Method3(LID=$device_lid $port): exit=$perfquery_exit"
                fi
            else
                debug_info="$debug_info, Method3: No valid LID found"
            fi
        else
            rm -f "$temp_ibstat"
            debug_info="$debug_info, Method3: Cannot get device info"
        fi
    fi
    
    # 如果方法3失败，尝试方法4: 使用 GUID
    if [ $perfquery_exit -ne 0 ]; then
        local device_guid
        local temp_ibstat="/tmp/ibstat_guid_$$"
        if ibstat "$device" "$port" >"$temp_ibstat" 2>/dev/null; then
            device_guid=$(grep "Port GUID:" "$temp_ibstat" | awk '{print $3}' | head -1)
            rm -f "$temp_ibstat"
            
            # 验证GUID
            if [ -n "$device_guid" ] && [ "$device_guid" != "0x0000000000000000" ]; then
                if timeout 10 perfquery -G "$device_guid" "$port" >"$temp_file" 2>&1; then
                    perfquery_output=$(cat "$temp_file")
                    perfquery_exit=0
                    debug_info="$debug_info, Method4(GUID=$device_guid $port): success"
                else
                    perfquery_exit=$?
                    debug_info="$debug_info, Method4(GUID=$device_guid $port): exit=$perfquery_exit"
                fi
            else
                debug_info="$debug_info, Method4: No valid GUID found"
            fi
        else
            rm -f "$temp_ibstat"
            debug_info="$debug_info, Method4: Cannot get device info"
        fi
    fi
    
    # 清理临时文件
    rm -f "$temp_file"
    
    # 如果所有方法都失败，输出调试信息并返回错误
    if [ $perfquery_exit -ne 0 ]; then
        [ "$QUIET_MODE" = false ] && log_warning "perfquery 失败: $debug_info" >&2
        echo "0 0 0 0"
        return 1
    fi
    
    # 解析输出
    local xmit_data
    local rcv_data
    local xmit_pkts
    local rcv_pkts
    
    # 尝试不同的输出格式 - 修复正则表达式以匹配实际的 perfquery 输出格式
    # 格式1: PortXmitData:....................51811956
    xmit_data=$(echo "$perfquery_output" | grep -E "PortXmitData:" | sed 's/.*PortXmitData:[.]*\([0-9]*\).*/\1/' | head -1)
    rcv_data=$(echo "$perfquery_output" | grep -E "PortRcvData:" | sed 's/.*PortRcvData:[.]*\([0-9]*\).*/\1/' | head -1)
    xmit_pkts=$(echo "$perfquery_output" | grep -E "PortXmitPkts:" | sed 's/.*PortXmitPkts:[.]*\([0-9]*\).*/\1/' | head -1)
    rcv_pkts=$(echo "$perfquery_output" | grep -E "PortRcvPkts:" | sed 's/.*PortRcvPkts:[.]*\([0-9]*\).*/\1/' | head -1)
    
    # 如果上面的格式不匹配，尝试其他格式
    # 格式2: PortXmitData: 51811956 (空格分隔)
    if [ -z "$xmit_data" ]; then
        xmit_data=$(echo "$perfquery_output" | grep -E "PortXmitData[[:space:]]*:" | awk '{print $2}' | head -1)
    fi
    if [ -z "$rcv_data" ]; then
        rcv_data=$(echo "$perfquery_output" | grep -E "PortRcvData[[:space:]]*:" | awk '{print $2}' | head -1)
    fi
    if [ -z "$xmit_pkts" ]; then
        xmit_pkts=$(echo "$perfquery_output" | grep -E "PortXmitPkts[[:space:]]*:" | awk '{print $2}' | head -1)
    fi
    if [ -z "$rcv_pkts" ]; then
        rcv_pkts=$(echo "$perfquery_output" | grep -E "PortRcvPkts[[:space:]]*:" | awk '{print $2}' | head -1)
    fi
    
    # 格式3: 通用匹配 (最后的数字)
    if [ -z "$xmit_data" ]; then
        xmit_data=$(echo "$perfquery_output" | grep -i "xmitdata" | awk '{print $NF}' | head -1)
    fi
    if [ -z "$rcv_data" ]; then
        rcv_data=$(echo "$perfquery_output" | grep -i "rcvdata" | awk '{print $NF}' | head -1)
    fi
    if [ -z "$xmit_pkts" ]; then
        xmit_pkts=$(echo "$perfquery_output" | grep -i "xmitpkts" | awk '{print $NF}' | head -1)
    fi
    if [ -z "$rcv_pkts" ]; then
        rcv_pkts=$(echo "$perfquery_output" | grep -i "rcvpkts" | awk '{print $NF}' | head -1)
    fi
    
    # 清理并验证数据，确保只返回数字
    xmit_data=$(echo "$xmit_data" | sed 's/\x1b\[[0-9;]*m//g' | tr -d ' \t\n\r')
    rcv_data=$(echo "$rcv_data" | sed 's/\x1b\[[0-9;]*m//g' | tr -d ' \t\n\r')
    xmit_pkts=$(echo "$xmit_pkts" | sed 's/\x1b\[[0-9;]*m//g' | tr -d ' \t\n\r')
    rcv_pkts=$(echo "$rcv_pkts" | sed 's/\x1b\[[0-9;]*m//g' | tr -d ' \t\n\r')
    
    # 验证数据并转换为十进制（处理十六进制输入）
    xmit_data=$(printf "%d" "${xmit_data:-0}" 2>/dev/null || echo "0")
    rcv_data=$(printf "%d" "${rcv_data:-0}" 2>/dev/null || echo "0")
    xmit_pkts=$(printf "%d" "${xmit_pkts:-0}" 2>/dev/null || echo "0")
    rcv_pkts=$(printf "%d" "${rcv_pkts:-0}" 2>/dev/null || echo "0")
    
    # 调试输出（仅在非静默模式下）
    if [ "$QUIET_MODE" = false ] && [ "$xmit_data" -eq 0 ] && [ "$rcv_data" -eq 0 ]; then
        log_warning "获取到的计数器全为0，perfquery 输出样本:" >&2
        if [ -n "$perfquery_output" ]; then
            echo "$perfquery_output" | head -5 | while IFS= read -r line; do
                [ -n "$line" ] && log_warning "  $line" >&2
            done
        fi
    fi
    
    echo "$xmit_data $rcv_data $xmit_pkts $rcv_pkts"
}

# 计算带宽
calculate_bandwidth() {
    local prev_xmit="$1"
    local curr_xmit="$2"
    local prev_rcv="$3"
    local curr_rcv="$4"
    local interval="$5"
    
    # 设置默认值
    prev_xmit="${prev_xmit:-0}"
    curr_xmit="${curr_xmit:-0}"
    prev_rcv="${prev_rcv:-0}"
    curr_rcv="${curr_rcv:-0}"
    interval="${interval:-1}"
    
    # 使用通用验证函数验证参数
    if ! validate_non_negative_integer "$prev_xmit" "prev_xmit" || \
       ! validate_non_negative_integer "$curr_xmit" "curr_xmit" || \
       ! validate_non_negative_integer "$prev_rcv" "prev_rcv" || \
       ! validate_non_negative_integer "$curr_rcv" "curr_rcv" || \
       ! validate_positive_integer "$interval" "interval"; then
        echo "0.00 0.00"
        return 1
    fi
    
    # 检查初始值
    if [ "$prev_xmit" -eq 0 ] || [ "$prev_rcv" -eq 0 ]; then
        echo "0.00 0.00"
        return 0
    fi
    
    # 计算差值
    local xmit_diff=$((curr_xmit - prev_xmit))
    local rcv_diff=$((curr_rcv - prev_rcv))
    
    # 使用通用函数处理计数器回绕
    xmit_diff=$(handle_counter_wraparound "$xmit_diff")
    rcv_diff=$(handle_counter_wraparound "$rcv_diff")
    
    # 计算带宽 (4字节单位转换为 MB/s)
    local xmit_mbps
    local rcv_mbps
    xmit_mbps=$(safe_calc "$xmit_diff * 4 / $interval / 1024 / 1024")
    rcv_mbps=$(safe_calc "$rcv_diff * 4 / $interval / 1024 / 1024")
    
    echo "$xmit_mbps $rcv_mbps"
}

# 计算包速率
calculate_packet_rate() {
    local prev_xmit_pkts="$1"
    local curr_xmit_pkts="$2"
    local prev_rcv_pkts="$3"
    local curr_rcv_pkts="$4"
    local interval="$5"
    
    # 设置默认值
    prev_xmit_pkts="${prev_xmit_pkts:-0}"
    curr_xmit_pkts="${curr_xmit_pkts:-0}"
    prev_rcv_pkts="${prev_rcv_pkts:-0}"
    curr_rcv_pkts="${curr_rcv_pkts:-0}"
    interval="${interval:-1}"
    
    # 使用通用验证函数验证参数
    if ! validate_non_negative_integer "$prev_xmit_pkts" "prev_xmit_pkts" || \
       ! validate_non_negative_integer "$curr_xmit_pkts" "curr_xmit_pkts" || \
       ! validate_non_negative_integer "$prev_rcv_pkts" "prev_rcv_pkts" || \
       ! validate_non_negative_integer "$curr_rcv_pkts" "curr_rcv_pkts" || \
       ! validate_positive_integer "$interval" "interval"; then
        echo "0 0"
        return 1
    fi
    
    # 检查初始值
    if [ "$prev_xmit_pkts" -eq 0 ] || [ "$prev_rcv_pkts" -eq 0 ]; then
        echo "0 0"
        return 0
    fi
    
    # 计算差值
    local xmit_pkt_diff=$((curr_xmit_pkts - prev_xmit_pkts))
    local rcv_pkt_diff=$((curr_rcv_pkts - prev_rcv_pkts))
    
    # 使用通用函数处理计数器回绕
    xmit_pkt_diff=$(handle_counter_wraparound "$xmit_pkt_diff")
    rcv_pkt_diff=$(handle_counter_wraparound "$rcv_pkt_diff")
    
    # 计算包速率
    local xmit_pps=$((xmit_pkt_diff / interval))
    local rcv_pps=$((rcv_pkt_diff / interval))
    
    echo "$xmit_pps $rcv_pps"
}

# 计算利用率
calculate_utilization() {
    local xmit_mbps="$1"
    local rcv_mbps="$2"
    local link_rate_gbps="$3"
    
    # 设置默认值
    xmit_mbps="${xmit_mbps:-0}"
    rcv_mbps="${rcv_mbps:-0}"
    link_rate_gbps="${link_rate_gbps:-0}"
    
    # 使用通用验证函数验证参数
    if ! validate_positive_integer "$link_rate_gbps" "link_rate_gbps"; then
        echo "0.00"
        return 1
    fi
    
    if ! validate_float "$xmit_mbps" "xmit_mbps" || ! validate_float "$rcv_mbps" "rcv_mbps"; then
        echo "0.00"
        return 1
    fi
    
    # 转换为 MB/s (Gbps -> MB/s: Gbps × 1000 ÷ 8 = Gbps × 125)
    local link_rate_mbps
    link_rate_mbps=$(safe_calc "$link_rate_gbps * 125")
    
    if [ "$?" -ne 0 ]; then
        echo "0.00"
        return 1
    fi
    
    # 计算最大速率
    local max_rate
    max_rate=$(safe_calc "if ($xmit_mbps > $rcv_mbps) $xmit_mbps else $rcv_mbps")
    
    # 计算利用率百分比
    local utilization
    utilization=$(calculate_percentage "$max_rate" "$link_rate_mbps")
    
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

# 关联数组辅助函数 - 兼容不支持declare -A的shell版本
# 设置关联数组值
set_assoc_value() {
    local array_name="$1"
    local key="$2"
    local value="$3"
    local safe_key
    # 更安全的键转换，移除所有特殊字符
    safe_key=$(echo "$key" | sed 's/[^a-zA-Z0-9]/_/g')
    eval "${array_name}_${safe_key}='$value'"
}

# 获取关联数组值
get_assoc_value() {
    local array_name="$1"
    local key="$2"
    local safe_key
    # 更安全的键转换，移除所有特殊字符
    safe_key=$(echo "$key" | sed 's/[^a-zA-Z0-9]/_/g')
    eval "echo \$${array_name}_${safe_key}"
}

# 检查关联数组键是否存在
has_assoc_key() {
    local array_name="$1"
    local key="$2"
    local safe_key
    # 更安全的键转换，移除所有特殊字符
    safe_key=$(echo "$key" | sed 's/[^a-zA-Z0-9]/_/g')
    local value
    value=$(eval "echo \$${array_name}_${safe_key}")
    [ -n "$value" ]
}

# 统一的监控函数 - 支持单设备单端口、单设备多端口、多设备多端口
monitor_unified() {
    local monitor_type="$1"
    shift
    
    # 存储所有设备端口的数据 - 使用兼容的数组声明方式
    # 关联数组在某些shell版本中不支持declare -A，使用eval方式
    local prev_xmit_data_keys=()
    local prev_rcv_data_keys=()
    local prev_xmit_pkts_keys=()
    local prev_rcv_pkts_keys=()
    local link_rates_keys=()
    local port_keys=()
    
    # 根据监控类型初始化端口列表
    case "$monitor_type" in
        "single")
            local device="$1"
            local port="$2"
            local key="${device}:${port}"
            
            # 检查端口状态
            local port_status
            port_status=$(get_port_status "$device" "$port")
            
            if [ "$port_status" != "Active" ]; then
                log_error "端口 ${device}:${port} 状态不是 Active (当前: $port_status)"
                return 1
            fi
            
            # 获取初始计数器
            local counters
            counters=$(get_performance_counters "$device" "$port")
            
            if [ $? -ne 0 ]; then
                log_error "无法获取性能计数器"
                return 1
            fi

            # 解析计数器并存储
            local xmit_data rcv_data xmit_pkts rcv_pkts
            read -r xmit_data rcv_data xmit_pkts rcv_pkts <<< "$counters"
            
            set_assoc_value "prev_xmit_data" "$key" "$xmit_data"
            set_assoc_value "prev_rcv_data" "$key" "$rcv_data"
            set_assoc_value "prev_xmit_pkts" "$key" "$xmit_pkts"
            set_assoc_value "prev_rcv_pkts" "$key" "$rcv_pkts"
            
            local link_rate
            link_rate=$(get_link_rate "$device" "$port")
            set_assoc_value "link_rates" "$key" "$link_rate"
            port_keys=("$key")
            
            local link_rate
            link_rate=$(get_assoc_value "link_rates" "$key")
            if [ "$MONITOR_DURATION" -gt 0 ]; then
                [ "$QUIET_MODE" = false ] && log_info "监控 ${device}:${port} (${link_rate} Gbps) - 持续时间: ${MONITOR_DURATION}秒"
            else
                [ "$QUIET_MODE" = false ] && log_info "监控 ${device}:${port} (${link_rate} Gbps)"
            fi
            ;;
            
        "multiple")
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
            
            # 初始化端口数据
            for port in "${ports[@]}"; do
                local key="${device}:${port}"
                local counters
                counters=$(get_performance_counters "$device" "$port")
                
                if [ $? -eq 0 ]; then
                    # 解析计数器并存储
                    local xmit_data rcv_data xmit_pkts rcv_pkts
                    read -r xmit_data rcv_data xmit_pkts rcv_pkts <<< "$counters"
                    
                    set_assoc_value "prev_xmit_data" "$key" "$xmit_data"
                    set_assoc_value "prev_rcv_data" "$key" "$rcv_data"
                    set_assoc_value "prev_xmit_pkts" "$key" "$xmit_pkts"
                    set_assoc_value "prev_rcv_pkts" "$key" "$rcv_pkts"
                    
                    local link_rate
                    link_rate=$(get_link_rate "$device" "$port")
                    set_assoc_value "link_rates" "$key" "$link_rate"
                    port_keys+=("$key")
                fi
            done
            ;;
            
        "multi_devices")
            local config="$1"
            
            [ "$QUIET_MODE" = false ] && log_info "开始多设备多端口监控"
            
            # 使用已验证的配置（从validate_multi_devices_config获得）
            IFS=';' read -ra DEVICE_CONFIGS <<< "$config"
            
            # 初始化所有设备端口的计数器
            for device_config in "${DEVICE_CONFIGS[@]}"; do
                # 跳过空配置
                [ -z "$device_config" ] && continue
                
                local device_name
                local ports_str
                device_name=$(echo "$device_config" | cut -d':' -f1)
                ports_str=$(echo "$device_config" | cut -d':' -f2)
                
                IFS=',' read -ra ports <<< "$ports_str"
                
                for port in "${ports[@]}"; do
                    # 跳过空端口
                    [ -z "$port" ] && continue
                    
                    local key="${device_name}:${port}"
                    
                    # 由于配置已经验证过，这里只需要获取初始计数器
                    local counters
                    counters=$(get_performance_counters "$device_name" "$port" 2>/dev/null)
                    
                    if [ $? -eq 0 ] && [ -n "$counters" ]; then
                        # 使用临时变量避免颜色代码混入数组索引
                        local temp_counters="$counters"
                        local xmit_data rcv_data xmit_pkts rcv_pkts
                        read -r xmit_data rcv_data xmit_pkts rcv_pkts <<< "$temp_counters"
                        
                        set_assoc_value "prev_xmit_data" "$key" "$xmit_data"
                        set_assoc_value "prev_rcv_data" "$key" "$rcv_data"
                        set_assoc_value "prev_xmit_pkts" "$key" "$xmit_pkts"
                        set_assoc_value "prev_rcv_pkts" "$key" "$rcv_pkts"
                        
                        local link_rate
                        link_rate=$(get_link_rate "$device_name" "$port" 2>/dev/null)
                        
                        # 验证链路速率
                        if [ -z "$link_rate" ] || [ "$link_rate" -eq 0 ]; then
                            log_warning "端口 ${key} 链路速率为 0，使用默认值 100 Gbps"
                            link_rate=100
                        fi
                        
                        set_assoc_value "link_rates" "$key" "$link_rate"
                        port_keys+=("$key")
                        [ "$QUIET_MODE" = false ] && log_info "设备 $device_name 端口 $port 初始化完成，链路速率: ${link_rate} Gbps"
                    else
                        # 清理设备端口键，避免颜色代码混入
                        local clean_key="${device_name}:${port}"
                        log_warning "无法获取端口 ${clean_key} 的初始计数器，跳过监控"
                    fi
                done
            done
            
            # 检查是否有有效的端口
            if [ ${#port_keys[@]} -eq 0 ]; then
                log_error "没有有效的端口可监控"
                return 1
            fi
            ;;
            
        *)
            log_error "未知的监控类型: $monitor_type"
            return 1
            ;;
    esac
    
    if [ "$MONITOR_DURATION" -gt 0 ]; then
        [ "$QUIET_MODE" = false ] && log_info "开始监控，持续时间: ${MONITOR_DURATION} 秒"
    else
        [ "$QUIET_MODE" = false ] && log_info "开始无限监控 (Ctrl+C 停止)"
    fi
    
    # 输出头部
    [ "$QUIET_MODE" = false ] && output_sar_header

    # 记录开始时间（在验证和初始化完成后）
    local start_time
    start_time=$(date +%s)
    
    # 监控循环
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
        
        # 遍历所有端口
        for key in "${port_keys[@]}"; do
            # 跳过空键或未初始化的端口
            if [ -z "$key" ] || ! has_assoc_key "prev_xmit_data" "$key"; then
                continue
            fi
            
            # 解析设备名和端口号
            local device_name
            local port
            device_name=$(echo "$key" | cut -d':' -f1)
            port=$(echo "$key" | cut -d':' -f2)
            
            # 跳过空的设备名或端口号
            if [ -z "$device_name" ] || [ -z "$port" ]; then
                continue
            fi
            
            # 获取当前计数器
            local curr_counters
            curr_counters=$(get_performance_counters "$device_name" "$port" 2>/dev/null)
            
            if [ $? -ne 0 ] || [ -z "$curr_counters" ]; then
                if [ "$monitor_type" = "single" ]; then
                    log_warning "获取计数器失败，跳过此次采样"
                fi
                continue
            fi
            
            local curr_xmit_data
            local curr_rcv_data
            local curr_xmit_pkts
            local curr_rcv_pkts
            read -r curr_xmit_data curr_rcv_data curr_xmit_pkts curr_rcv_pkts <<< "$curr_counters"
            
            # 获取前一次的值
            local prev_xmit
            local prev_rcv
            local prev_xmit_pkt
            local prev_rcv_pkt
            prev_xmit=$(get_assoc_value "prev_xmit_data" "$key")
            prev_rcv=$(get_assoc_value "prev_rcv_data" "$key")
            prev_xmit_pkt=$(get_assoc_value "prev_xmit_pkts" "$key")
            prev_rcv_pkt=$(get_assoc_value "prev_rcv_pkts" "$key")
            
            # 计算带宽
            local bandwidth
            bandwidth=$(calculate_bandwidth "$prev_xmit" "$curr_xmit_data" "$prev_rcv" "$curr_rcv_data" "$MONITOR_INTERVAL")
            local txmbps
            local rxmbps
            read -r txmbps rxmbps <<< "$bandwidth"
            
            # 计算包速率
            local packet_rate
            packet_rate=$(calculate_packet_rate "$prev_xmit_pkt" "$curr_xmit_pkts" "$prev_rcv_pkt" "$curr_rcv_pkts" "$MONITOR_INTERVAL")
            local txpps
            local rxpps
            read -r txpps rxpps <<< "$packet_rate"
            
            # 计算利用率
            local link_rate
            link_rate=$(get_assoc_value "link_rates" "$key")
            
            # 验证链路速率
            if [ -z "$link_rate" ] || [ "$link_rate" -eq 0 ]; then
                link_rate=100  # 使用默认值
            fi
            
            local utilization
            utilization=$(calculate_utilization "$txmbps" "$rxmbps" "$link_rate")
            
            # 输出 SAR 格式
            format_sar_output "$timestamp" "$device_name" "$port" "$rxpps" "$txpps" "$rxmbps" "$txmbps" "$utilization"
            
            # 更新前一次的值
            set_assoc_value "prev_xmit_data" "$key" "$curr_xmit_data"
            set_assoc_value "prev_rcv_data" "$key" "$curr_rcv_data"
            set_assoc_value "prev_xmit_pkts" "$key" "$curr_xmit_pkts"
            set_assoc_value "prev_rcv_pkts" "$key" "$curr_rcv_pkts"
        done
    done
}

# 兼容性函数 - 保持向后兼容
monitor_single_port() {
    monitor_unified "single" "$@"
}

monitor_multiple_ports() {
    monitor_unified "multiple" "$@"
}

monitor_multi_devices() {
    monitor_unified "multi_devices" "$@"
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
            --debug)
                DEBUG_MODE=true
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
    if ! validate_positive_integer "$MONITOR_INTERVAL" "监控间隔"; then
        exit 1
    fi
    
    if ! validate_non_negative_integer "$MONITOR_DURATION" "监控时长"; then
        exit 1
    fi
    
    # 检查依赖
    check_dependencies
    
    # 处理多设备多端口监控
    if [ -n "$DEVICES_PORTS_CONFIG" ]; then
        # 预先验证多设备配置
        [ "$QUIET_MODE" = false ] && log_info "验证多设备多端口配置..."
        
        # 验证多设备配置并获取有效配置
        local valid_config
        valid_config=$(validate_multi_devices_config "$DEVICES_PORTS_CONFIG")
        
        if [ $? -ne 0 ] || [ -z "$valid_config" ]; then
            log_error "多设备配置验证失败"
            log_info "建议使用 --list-ports 查看可用的活跃端口"
            exit 1
        fi
        
        # 使用有效配置进行监控
        monitor_unified "multi_devices" "$valid_config"
        exit 0
    fi
    
    # 自动检测设备
    if [ -z "$DEVICE" ]; then
        local devices
        devices=$(detect_ib_devices)
        DEVICE=$(echo "$devices" | head -1)
        
        [ "$QUIET_MODE" = false ] && log_info "自动选择设备: $DEVICE"
    fi
    
    # 验证设备
    [ "$QUIET_MODE" = false ] && log_info "验证设备 $DEVICE..."
    if ! validate_device "$DEVICE"; then
        log_error "设备验证失败"
        log_info "建议使用 --list-ports 查看可用设备"
        exit 1
    fi
    
    # 处理多端口监控
    if [ -n "$PORTS_LIST" ]; then
        local ports_array
        ports_array=($(parse_ports_list "$PORTS_LIST" "$DEVICE"))
        
        if [ ${#ports_array[@]} -eq 0 ]; then
            log_error "没有有效的端口可监控"
            exit 1
        fi
        
        # 验证多端口配置
        [ "$QUIET_MODE" = false ] && log_info "验证设备 $DEVICE 的多端口配置..."
        if ! validate_multiple_ports_config "$DEVICE" "${ports_array[@]}"; then
            log_error "多端口配置验证失败"
            log_info "建议使用 --list-ports 查看可用的活跃端口"
            exit 1
        fi
        
        monitor_unified "multiple" "$DEVICE" "${ports_array[@]}"
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
        
        # 验证设备和端口配置
        [ "$QUIET_MODE" = false ] && log_info "验证设备 $DEVICE 端口 $PORT..."
        if ! validate_device_port_config "$DEVICE" "$PORT"; then
            log_error "设备端口配置验证失败"
            log_info "建议使用 --list-ports 查看可用的活跃端口"
            exit 1
        fi
        
        monitor_unified "single" "$DEVICE" "$PORT"
    fi
}

# 执行主函数
main "$@"