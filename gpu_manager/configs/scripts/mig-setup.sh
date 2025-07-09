#!/bin/bash
# MIG (Multi-Instance GPU) 配置管理脚本
# 支持A100, H100, L40S等支持MIG的GPU

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# MIG实例配置映射
declare -A MIG_PROFILES=(
    ["1g.5gb"]="1g.5gb"
    ["1g.10gb"]="1g.10gb"
    ["2g.10gb"]="2g.10gb"
    ["3g.20gb"]="3g.20gb"
    ["4g.20gb"]="4g.20gb"
    ["7g.40gb"]="7g.40gb"
    ["1g.5gb+me"]="1g.5gb+me"
    ["2g.10gb+me"]="2g.10gb+me"
    ["3g.20gb+me"]="3g.20gb+me"
    ["4g.20gb+me"]="4g.20gb+me"
)

# 检查GPU是否支持MIG
check_mig_support() {
    local gpu_id=${1:-0}
    
    log_info "检查GPU $gpu_id 的MIG支持..."
    
    # 检查GPU型号
    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits -i $gpu_id)
    log_debug "GPU型号: $gpu_name"
    
    # 检查MIG支持
    local mig_support=$(nvidia-smi --query-gpu=mig.mode.supported --format=csv,noheader,nounits -i $gpu_id)
    
    if [[ "$mig_support" == "Supported" ]]; then
        log_info "✅ GPU $gpu_id 支持MIG"
        return 0
    else
        log_error "❌ GPU $gpu_id 不支持MIG"
        return 1
    fi
}

# 启用MIG模式
enable_mig() {
    local gpu_id=${1:-0}
    
    log_info "在GPU $gpu_id 上启用MIG模式..."
    
    # 检查当前MIG状态
    local current_mode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader,nounits -i $gpu_id)
    
    if [[ "$current_mode" == "Enabled" ]]; then
        log_warn "MIG模式已经启用"
        return 0
    fi
    
    # 启用MIG模式
    sudo nvidia-smi -i $gpu_id -mig 1
    
    # 重置GPU以应用MIG设置
    log_info "重置GPU以应用MIG设置..."
    sudo nvidia-smi -i $gpu_id -r
    
    # 等待GPU重置完成
    sleep 5
    
    # 验证MIG模式
    local new_mode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader,nounits -i $gpu_id)
    if [[ "$new_mode" == "Enabled" ]]; then
        log_info "✅ MIG模式启用成功"
    else
        log_error "❌ MIG模式启用失败"
        return 1
    fi
}

# 禁用MIG模式
disable_mig() {
    local gpu_id=${1:-0}
    
    log_info "在GPU $gpu_id 上禁用MIG模式..."
    
    # 检查当前MIG状态
    local current_mode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader,nounits -i $gpu_id)
    
    if [[ "$current_mode" == "Disabled" ]]; then
        log_warn "MIG模式已经禁用"
        return 0
    fi
    
    # 删除所有MIG实例
    log_info "删除所有MIG实例..."
    sudo nvidia-smi mig -dci -i $gpu_id || true
    sudo nvidia-smi mig -dgi -i $gpu_id || true
    
    # 禁用MIG模式
    sudo nvidia-smi -i $gpu_id -mig 0
    
    # 重置GPU
    log_info "重置GPU..."
    sudo nvidia-smi -i $gpu_id -r
    
    # 等待GPU重置完成
    sleep 5
    
    # 验证MIG模式
    local new_mode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader,nounits -i $gpu_id)
    if [[ "$new_mode" == "Disabled" ]]; then
        log_info "✅ MIG模式禁用成功"
    else
        log_error "❌ MIG模式禁用失败"
        return 1
    fi
}

# 创建GPU实例
create_gpu_instance() {
    local gpu_id=$1
    local profile=$2
    local count=${3:-1}
    
    log_info "在GPU $gpu_id 上创建 $count 个 $profile GPU实例..."
    
    # 验证配置文件
    if [[ -z "${MIG_PROFILES[$profile]}" ]]; then
        log_error "不支持的MIG配置: $profile"
        log_info "支持的配置: ${!MIG_PROFILES[@]}"
        return 1
    fi
    
    # 创建GPU实例
    for ((i=1; i<=count; i++)); do
        log_debug "创建第 $i 个GPU实例..."
        if sudo nvidia-smi mig -cgi $profile -i $gpu_id; then
            log_info "✅ GPU实例 $i 创建成功"
        else
            log_error "❌ GPU实例 $i 创建失败"
            return 1
        fi
    done
}

# 创建计算实例
create_compute_instance() {
    local gpu_id=$1
    local gi_id=${2:-"all"}
    
    log_info "在GPU $gpu_id 的GPU实例 $gi_id 上创建计算实例..."
    
    if [[ "$gi_id" == "all" ]]; then
        # 为所有GPU实例创建计算实例
        if sudo nvidia-smi mig -cci -i $gpu_id; then
            log_info "✅ 所有计算实例创建成功"
        else
            log_error "❌ 计算实例创建失败"
            return 1
        fi
    else
        # 为指定GPU实例创建计算实例
        if sudo nvidia-smi mig -cci -gi $gi_id -i $gpu_id; then
            log_info "✅ 计算实例创建成功"
        else
            log_error "❌ 计算实例创建失败"
            return 1
        fi
    fi
}

# 删除所有MIG实例
delete_all_instances() {
    local gpu_id=${1:-0}
    
    log_info "删除GPU $gpu_id 上的所有MIG实例..."
    
    # 删除计算实例
    log_debug "删除计算实例..."
    sudo nvidia-smi mig -dci -i $gpu_id || true
    
    # 删除GPU实例
    log_debug "删除GPU实例..."
    sudo nvidia-smi mig -dgi -i $gpu_id || true
    
    log_info "✅ 所有MIG实例删除完成"
}

# 列出MIG实例
list_mig_instances() {
    local gpu_id=${1:-"all"}
    
    log_info "列出MIG实例信息..."
    
    if [[ "$gpu_id" == "all" ]]; then
        echo "=== GPU实例列表 ==="
        nvidia-smi mig -lgi
        echo ""
        echo "=== 计算实例列表 ==="
        nvidia-smi mig -lci
    else
        echo "=== GPU $gpu_id 实例列表 ==="
        nvidia-smi mig -lgi -i $gpu_id
        echo ""
        echo "=== GPU $gpu_id 计算实例列表 ==="
        nvidia-smi mig -lci -i $gpu_id
    fi
}

# 显示MIG状态
show_mig_status() {
    local gpu_id=${1:-"all"}
    
    log_info "显示MIG状态..."
    
    if [[ "$gpu_id" == "all" ]]; then
        nvidia-smi --query-gpu=index,name,mig.mode.current,mig.mode.pending --format=table
    else
        nvidia-smi --query-gpu=index,name,mig.mode.current,mig.mode.pending --format=table -i $gpu_id
    fi
}

# 预设配置模板
apply_preset() {
    local gpu_id=$1
    local preset=$2
    
    log_info "应用预设配置: $preset"
    
    case $preset in
        "dev")
            # 开发环境：2个小实例用于测试
            create_gpu_instance $gpu_id "1g.5gb" 2
            create_compute_instance $gpu_id
            ;;
        "training")
            # 训练环境：1个大实例用于训练
            create_gpu_instance $gpu_id "7g.40gb" 1
            create_compute_instance $gpu_id
            ;;
        "inference")
            # 推理环境：多个中等实例
            create_gpu_instance $gpu_id "2g.10gb" 3
            create_compute_instance $gpu_id
            ;;
        "mixed")
            # 混合环境：1个大实例 + 2个小实例
            create_gpu_instance $gpu_id "3g.20gb" 1
            create_gpu_instance $gpu_id "1g.5gb" 2
            create_compute_instance $gpu_id
            ;;
        *)
            log_error "不支持的预设配置: $preset"
            log_info "支持的预设: dev, training, inference, mixed"
            return 1
            ;;
    esac
}

# 验证MIG配置
verify_mig_config() {
    local gpu_id=${1:-0}
    
    log_info "验证MIG配置..."
    
    # 检查MIG模式
    local mig_mode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader,nounits -i $gpu_id)
    if [[ "$mig_mode" == "Enabled" ]]; then
        log_info "✅ MIG模式已启用"
    else
        log_warn "⚠️ MIG模式未启用"
        return 1
    fi
    
    # 检查GPU实例数量
    local gi_count=$(nvidia-smi mig -lgi -i $gpu_id 2>/dev/null | grep -c "GPU instance" || echo "0")
    log_info "GPU实例数量: $gi_count"
    
    # 检查计算实例数量
    local ci_count=$(nvidia-smi mig -lci -i $gpu_id 2>/dev/null | grep -c "Compute instance" || echo "0")
    log_info "计算实例数量: $ci_count"
    
    if [[ $gi_count -gt 0 && $ci_count -gt 0 ]]; then
        log_info "✅ MIG配置验证成功"
        return 0
    else
        log_warn "⚠️ 未找到有效的MIG实例"
        return 1
    fi
}

# 显示帮助信息
show_help() {
    echo "MIG配置管理脚本"
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  enable [GPU_ID]                启用MIG模式"
    echo "  disable [GPU_ID]               禁用MIG模式"
    echo "  create-gi GPU_ID PROFILE [COUNT] 创建GPU实例"
    echo "  create-ci GPU_ID [GI_ID]       创建计算实例"
    echo "  delete GPU_ID                  删除所有实例"
    echo "  list [GPU_ID]                  列出MIG实例"
    echo "  status [GPU_ID]                显示MIG状态"
    echo "  preset GPU_ID PRESET           应用预设配置"
    echo "  verify [GPU_ID]                验证MIG配置"
    echo "  check [GPU_ID]                 检查MIG支持"
    echo ""
    echo "MIG配置文件:"
    echo "  1g.5gb, 1g.10gb, 2g.10gb, 3g.20gb, 4g.20gb, 7g.40gb"
    echo ""
    echo "预设配置:"
    echo "  dev       - 开发环境 (2x 1g.5gb)"
    echo "  training  - 训练环境 (1x 7g.40gb)"
    echo "  inference - 推理环境 (3x 2g.10gb)"
    echo "  mixed     - 混合环境 (1x 3g.20gb + 2x 1g.5gb)"
    echo ""
    echo "示例:"
    echo "  $0 enable 0                    # 在GPU 0上启用MIG"
    echo "  $0 create-gi 0 1g.5gb 2       # 创建2个1g.5gb实例"
    echo "  $0 preset 0 dev               # 应用开发环境预设"
    echo "  $0 list                        # 列出所有MIG实例"
}

# 主函数
main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    local command=$1
    shift
    
    case $command in
        "enable")
            local gpu_id=${1:-0}
            check_mig_support $gpu_id && enable_mig $gpu_id
            ;;
        "disable")
            local gpu_id=${1:-0}
            disable_mig $gpu_id
            ;;
        "create-gi")
            if [[ $# -lt 2 ]]; then
                log_error "用法: $0 create-gi GPU_ID PROFILE [COUNT]"
                exit 1
            fi
            create_gpu_instance $1 $2 ${3:-1}
            ;;
        "create-ci")
            if [[ $# -lt 1 ]]; then
                log_error "用法: $0 create-ci GPU_ID [GI_ID]"
                exit 1
            fi
            create_compute_instance $1 ${2:-"all"}
            ;;
        "delete")
            if [[ $# -lt 1 ]]; then
                log_error "用法: $0 delete GPU_ID"
                exit 1
            fi
            delete_all_instances $1
            ;;
        "list")
            list_mig_instances ${1:-"all"}
            ;;
        "status")
            show_mig_status ${1:-"all"}
            ;;
        "preset")
            if [[ $# -lt 2 ]]; then
                log_error "用法: $0 preset GPU_ID PRESET"
                exit 1
            fi
            check_mig_support $1 && enable_mig $1 && delete_all_instances $1 && apply_preset $1 $2
            ;;
        "verify")
            verify_mig_config ${1:-0}
            ;;
        "check")
            check_mig_support ${1:-0}
            ;;
        "help"|"--help"|"h")
            show_help
            ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"