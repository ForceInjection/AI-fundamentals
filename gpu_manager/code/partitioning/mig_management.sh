#!/bin/bash

# MIG (Multi-Instance GPU) 管理脚本
# 来源：GPU 管理相关技术深度解析 - 4.1.1.2 GPU 实例的创建和管理
# 功能：提供MIG实例的创建、监控和管理功能

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查权限
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        log_error "此脚本需要root权限运行"
        exit 1
    fi
}

# 检查MIG支持
check_mig_support() {
    log_info "检查MIG支持状态..."
    nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader,nounits
}

# 启用MIG模式
enable_mig() {
    local gpu_id=${1:-0}
    log_info "在GPU $gpu_id 上启用MIG模式..."
    
    # 启用 MIG 模式
    nvidia-smi -i $gpu_id -mig 1
    
    log_warn "MIG模式已启用，需要重启GPU或系统以生效"
}

# 创建GPU实例
create_gpu_instances() {
    local gpu_id=${1:-0}
    local config=${2:-"1g.5gb,2g.10gb"}
    
    log_info "在GPU $gpu_id 上创建GPU实例: $config"
    
    # 创建 GPU 实例
    nvidia-smi mig -i $gpu_id -cgi $config
    
    # 创建计算实例
    nvidia-smi mig -i $gpu_id -cci
    
    log_info "GPU实例创建完成"
}

# 监控MIG实例
monitor_instances() {
    local gpu_id=${1:-0}
    
    log_info "监控GPU $gpu_id 的MIG实例状态..."
    
    echo "=== GPU实例列表 ==="
    nvidia-smi -L
    
    echo "\n=== MIG实例详情 ==="
    nvidia-smi mig -i $gpu_id -lgi
    
    echo "\n=== 实例配置信息 ==="
    nvidia-smi mig -i $gpu_id -lgip
}

# 重置MIG配置
reset_mig() {
    local gpu_id=${1:-0}
    
    log_warn "重置GPU $gpu_id 的MIG配置..."
    
    # 删除计算实例
    nvidia-smi mig -i $gpu_id -dci || true
    
    # 删除GPU实例
    nvidia-smi mig -i $gpu_id -dgi || true
    
    log_info "MIG配置已重置"
}

# 验证实例创建
verify_instances() {
    local gpu_id=${1:-0}
    
    log_info "验证GPU $gpu_id 的实例创建状态..."
    
    local instance_count=$(nvidia-smi mig -i $gpu_id -lgi | grep "GPU instance" | wc -l)
    
    if [[ $instance_count -gt 0 ]]; then
        log_info "发现 $instance_count 个GPU实例"
        return 0
    else
        log_error "未发现任何GPU实例"
        return 1
    fi
}

# 显示帮助信息
show_help() {
    echo "MIG管理脚本使用说明:"
    echo "  $0 check                    - 检查MIG支持状态"
    echo "  $0 enable [gpu_id]          - 启用MIG模式"
    echo "  $0 create [gpu_id] [config] - 创建GPU实例"
    echo "  $0 monitor [gpu_id]         - 监控实例状态"
    echo "  $0 reset [gpu_id]           - 重置MIG配置"
    echo "  $0 verify [gpu_id]          - 验证实例创建"
    echo "  $0 help                     - 显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 enable 0"
    echo "  $0 create 0 \"1g.5gb,2g.10gb\""
    echo "  $0 monitor 0"
}

# 主函数
main() {
    check_permissions
    
    case "${1:-help}" in
        "check")
            check_mig_support
            ;;
        "enable")
            enable_mig "${2:-0}"
            ;;
        "create")
            create_gpu_instances "${2:-0}" "${3:-1g.5gb,2g.10gb}"
            ;;
        "monitor")
            monitor_instances "${2:-0}"
            ;;
        "reset")
            reset_mig "${2:-0}"
            ;;
        "verify")
            verify_instances "${2:-0}"
            ;;
        "help")
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"