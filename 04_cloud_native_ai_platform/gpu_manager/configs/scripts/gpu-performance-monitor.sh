#!/bin/bash
# GPU性能监控脚本
# 提供全面的GPU性能监控和报告功能

set -e

# 配置变量
LOG_DIR="/var/log/gpu-monitor"
REPORT_DIR="/var/log/gpu-reports"
ALERT_THRESHOLD_UTIL=90
ALERT_THRESHOLD_TEMP=80
ALERT_THRESHOLD_MEM=85
MONITOR_INTERVAL=30
REPORT_INTERVAL=3600
EMAIL_ALERTS="admin@company.com"

# 创建必要的目录
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/gpu-monitor.log"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_DIR/gpu-monitor.log" >&2
}

# 检查依赖
check_dependencies() {
    local deps=("nvidia-smi" "nvidia-ml-py" "jq" "curl")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "依赖 $dep 未安装"
            return 1
        fi
    done
    log "依赖检查通过"
}

# 获取GPU基本信息
get_gpu_info() {
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.max_limit \
        --format=csv,noheader,nounits | while IFS=',' read -r index name driver memory power; do
        echo "GPU $index: $name (Driver: $driver, Memory: ${memory}MB, Max Power: ${power}W)"
    done
}

# 获取GPU实时状态
get_gpu_status() {
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,fan.speed \
        --format=csv,noheader,nounits
}

# 检查GPU健康状态
check_gpu_health() {
    local alerts=()
    
    while IFS=',' read -r index util_gpu util_mem mem_used mem_total temp power fan; do
        # 移除空格
        index=$(echo "$index" | tr -d ' ')
        util_gpu=$(echo "$util_gpu" | tr -d ' ')
        util_mem=$(echo "$util_mem" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' ')
        mem_total=$(echo "$mem_total" | tr -d ' ')
        temp=$(echo "$temp" | tr -d ' ')
        power=$(echo "$power" | tr -d ' ')
        fan=$(echo "$fan" | tr -d ' ')
        
        # 计算内存使用率
        if [[ "$mem_total" != "0" && "$mem_total" != "[Not Supported]" ]]; then
            mem_usage_percent=$((mem_used * 100 / mem_total))
        else
            mem_usage_percent=0
        fi
        
        # 检查GPU利用率
        if [[ "$util_gpu" != "[Not Supported]" && "$util_gpu" -gt "$ALERT_THRESHOLD_UTIL" ]]; then
            alerts+=("GPU $index: 高利用率 ${util_gpu}%")
        fi
        
        # 检查温度
        if [[ "$temp" != "[Not Supported]" && "$temp" -gt "$ALERT_THRESHOLD_TEMP" ]]; then
            alerts+=("GPU $index: 高温度 ${temp}°C")
        fi
        
        # 检查内存使用率
        if [[ "$mem_usage_percent" -gt "$ALERT_THRESHOLD_MEM" ]]; then
            alerts+=("GPU $index: 高内存使用率 ${mem_usage_percent}%")
        fi
        
        # 记录状态
        log "GPU $index: Util=${util_gpu}%, Mem=${mem_usage_percent}%, Temp=${temp}°C, Power=${power}W"
        
    done < <(get_gpu_status)
    
    # 处理告警
    if [[ ${#alerts[@]} -gt 0 ]]; then
        for alert in "${alerts[@]}"; do
            log_error "ALERT: $alert"
            send_alert "$alert"
        done
    fi
}

# 发送告警
send_alert() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 记录到告警日志
    echo "[$timestamp] $message" >> "$LOG_DIR/gpu-alerts.log"
    
    # 发送邮件告警（如果配置了）
    if [[ -n "$EMAIL_ALERTS" ]] && command -v mail &> /dev/null; then
        echo "GPU Alert: $message" | mail -s "GPU监控告警 - $(hostname)" "$EMAIL_ALERTS"
    fi
    
    # 发送到Webhook（如果配置了）
    if [[ -n "$WEBHOOK_URL" ]]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"GPU Alert on $(hostname): $message\"}" \
            2>/dev/null || true
    fi
}

# 生成性能报告
generate_performance_report() {
    local report_file="$REPORT_DIR/gpu-report-$(date '+%Y%m%d-%H%M%S').json"
    
    log "生成性能报告: $report_file"
    
    # 获取系统信息
    local hostname=$(hostname)
    local timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    local uptime=$(uptime -p)
    
    # 获取GPU信息
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    
    # 构建JSON报告
    cat > "$report_file" << EOF
{
  "timestamp": "$timestamp",
  "hostname": "$hostname",
  "uptime": "$uptime",
  "gpu_count": $gpu_count,
  "driver_version": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)",
  "gpus": [
EOF
    
    # 添加每个GPU的详细信息
    local first=true
    nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.max_limit,fan.speed,clocks.gr,clocks.mem \
        --format=csv,noheader,nounits | while IFS=',' read -r index name uuid util_gpu util_mem mem_used mem_total temp power power_max fan clock_gpu clock_mem; do
        
        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo "," >> "$report_file"
        fi
        
        # 清理数据
        name=$(echo "$name" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        uuid=$(echo "$uuid" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        cat >> "$report_file" << EOF
    {
      "index": $index,
      "name": "$name",
      "uuid": "$uuid",
      "utilization": {
        "gpu": $(echo "$util_gpu" | tr -d ' '),
        "memory": $(echo "$util_mem" | tr -d ' ')
      },
      "memory": {
        "used": $(echo "$mem_used" | tr -d ' '),
        "total": $(echo "$mem_total" | tr -d ' '),
        "usage_percent": $(($(echo "$mem_used" | tr -d ' ') * 100 / $(echo "$mem_total" | tr -d ' ')))
      },
      "temperature": $(echo "$temp" | tr -d ' '),
      "power": {
        "draw": $(echo "$power" | tr -d ' '),
        "max_limit": $(echo "$power_max" | tr -d ' ')
      },
      "fan_speed": $(echo "$fan" | tr -d ' '),
      "clocks": {
        "graphics": $(echo "$clock_gpu" | tr -d ' '),
        "memory": $(echo "$clock_mem" | tr -d ' ')
      }
    }EOF
    done
    
    echo "" >> "$report_file"
    echo "  ]" >> "$report_file"
    echo "}" >> "$report_file"
    
    log "性能报告已生成: $report_file"
}

# 清理旧日志
cleanup_logs() {
    local days_to_keep=7
    
    log "清理 $days_to_keep 天前的日志文件"
    
    find "$LOG_DIR" -name "*.log" -mtime +$days_to_keep -delete 2>/dev/null || true
    find "$REPORT_DIR" -name "*.json" -mtime +$days_to_keep -delete 2>/dev/null || true
    
    log "日志清理完成"
}

# 监控循环
monitor_loop() {
    log "开始GPU监控循环，间隔: ${MONITOR_INTERVAL}秒"
    
    local last_report_time=0
    
    while true; do
        local current_time=$(date +%s)
        
        # 检查GPU健康状态
        check_gpu_health
        
        # 定期生成报告
        if [[ $((current_time - last_report_time)) -ge $REPORT_INTERVAL ]]; then
            generate_performance_report
            cleanup_logs
            last_report_time=$current_time
        fi
        
        sleep "$MONITOR_INTERVAL"
    done
}

# 显示帮助信息
show_help() {
    cat << EOF
GPU性能监控脚本

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -i, --info              显示GPU基本信息
  -s, --status            显示GPU当前状态
  -c, --check             执行一次健康检查
  -r, --report            生成性能报告
  -m, --monitor           启动监控循环
  -d, --daemon            以守护进程模式运行
  --interval SECONDS      设置监控间隔（默认: 30秒）
  --alert-util PERCENT    设置利用率告警阈值（默认: 90%）
  --alert-temp CELSIUS    设置温度告警阈值（默认: 80°C）
  --alert-mem PERCENT     设置内存告警阈值（默认: 85%）
  --email EMAIL           设置告警邮箱
  --webhook URL           设置Webhook URL

示例:
  $0 --info                    # 显示GPU信息
  $0 --monitor                 # 启动监控
  $0 --daemon --interval 60    # 以守护进程模式运行，60秒间隔
EOF
}

# 守护进程模式
run_daemon() {
    log "启动GPU监控守护进程"
    
    # 创建PID文件
    local pid_file="/var/run/gpu-monitor.pid"
    echo $$ > "$pid_file"
    
    # 设置信号处理
    trap 'log "收到终止信号，停止监控"; rm -f "$pid_file"; exit 0' TERM INT
    
    # 启动监控
    monitor_loop
}

# 主函数
main() {
    # 检查是否为root用户
    if [[ $EUID -ne 0 ]]; then
        log_error "此脚本需要root权限运行"
        exit 1
    fi
    
    # 检查依赖
    if ! check_dependencies; then
        exit 1
    fi
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -i|--info)
                get_gpu_info
                exit 0
                ;;
            -s|--status)
                get_gpu_status
                exit 0
                ;;
            -c|--check)
                check_gpu_health
                exit 0
                ;;
            -r|--report)
                generate_performance_report
                exit 0
                ;;
            -m|--monitor)
                monitor_loop
                exit 0
                ;;
            -d|--daemon)
                run_daemon
                exit 0
                ;;
            --interval)
                MONITOR_INTERVAL="$2"
                shift 2
                ;;
            --alert-util)
                ALERT_THRESHOLD_UTIL="$2"
                shift 2
                ;;
            --alert-temp)
                ALERT_THRESHOLD_TEMP="$2"
                shift 2
                ;;
            --alert-mem)
                ALERT_THRESHOLD_MEM="$2"
                shift 2
                ;;
            --email)
                EMAIL_ALERTS="$2"
                shift 2
                ;;
            --webhook)
                WEBHOOK_URL="$2"
                shift 2
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 默认显示帮助信息
    show_help
}

# 执行主函数
main "$@"