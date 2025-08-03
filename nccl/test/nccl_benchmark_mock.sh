#!/bin/bash
# =============================================================================
# NCCL Benchmark Mock Script
# 功能: 为测试环境提供兼容性包装，解决bash版本和环境依赖问题
# =============================================================================

# 获取原始脚本路径
ORIGINAL_SCRIPT="$(dirname "$0")/../nccl_benchmark.sh"

# 检查bash版本并设置兼容性
check_bash_compatibility() {
    local bash_version=$(bash --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
    local major_version=$(echo "$bash_version" | cut -d. -f1)
    
    if [ "$major_version" -lt 4 ]; then
        echo "警告: 检测到 bash $bash_version，某些功能可能不兼容"
        export BASH_COMPAT_MODE=true
    else
        export BASH_COMPAT_MODE=false
    fi
}

# Mock 环境变量设置
setup_mock_environment() {
    # 设置基本的 mock 环境
    export MOCK_MODE=true
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET
    
    # Mock GPU 环境
    if [ ! -f "/tmp/mock_nvidia_smi" ]; then
        cat > /tmp/mock_nvidia_smi << 'EOF'
#!/bin/bash
case "$1" in
    "-L")
        echo "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-12345678-1234-1234-1234-123456789abc)"
        echo "GPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-87654321-4321-4321-4321-cba987654321)"
        ;;
    "nvlink")
        if [ "$2" = "--status" ]; then
            echo "GPU 0: 26.562 GB/s"
            echo "GPU 1: 26.562 GB/s"
        fi
        ;;
    *)
        echo "Mock nvidia-smi output"
        ;;
esac
EOF
        chmod +x /tmp/mock_nvidia_smi
    fi
    
    # Mock InfiniBand 环境
    if [ ! -f "/tmp/mock_ibv_devinfo" ]; then
        cat > /tmp/mock_ibv_devinfo << 'EOF'
#!/bin/bash
echo "hca_id: mlx5_0"
echo "    transport:                  InfiniBand (0)"
echo "    fw_ver:                     16.35.2000"
echo "    node_guid:                  248a:0703:00b4:7a96"
echo "    sys_image_guid:             248a:0703:00b4:7a96"
EOF
        chmod +x /tmp/mock_ibv_devinfo
    fi
    
    # 将 mock 工具添加到 PATH
    export PATH="/tmp:$PATH"
}

# 兼容性包装函数
declare_associative_array() {
    local array_name="$1"
    
    if [ "$BASH_COMPAT_MODE" = "true" ]; then
        # 对于旧版本 bash，使用普通变量模拟
        eval "${array_name}_keys=''"
        eval "${array_name}_values=''"
    else
        # 新版本 bash 使用关联数组
        declare -gA "$array_name"
    fi
}

# 兼容性数组设置函数
set_array_value() {
    local array_name="$1"
    local key="$2"
    local value="$3"
    
    if [ "$BASH_COMPAT_MODE" = "true" ]; then
        # 简化的键值存储
        eval "${array_name}_${key}='$value'"
    else
        eval "${array_name}['$key']='$value'"
    fi
}

# 兼容性数组获取函数
get_array_value() {
    local array_name="$1"
    local key="$2"
    
    if [ "$BASH_COMPAT_MODE" = "true" ]; then
        eval "echo \$${array_name}_${key}"
    else
        eval "echo \${${array_name}['$key']}"
    fi
}

# 预处理原始脚本以解决兼容性问题
preprocess_script() {
    local temp_script="/tmp/nccl_benchmark_processed.sh"
    local is_dry_run=false
    
    # 检查是否为 dry-run 模式
    for arg in "$@"; do
        if [ "$arg" = "--dry-run" ]; then
            is_dry_run=true
            break
        fi
    done
    
    # 复制原始脚本并进行兼容性修改
    cp "$ORIGINAL_SCRIPT" "$temp_script"
    
    # 如果是兼容模式，替换关联数组声明
    if [ "$BASH_COMPAT_MODE" = "true" ]; then
        sed -i.bak 's/declare -A /# declare -A /g' "$temp_script"
    fi
    
    # 如果是 dry-run 模式，跳过环境依赖检查
    if [ "$is_dry_run" = "true" ]; then
        # 只替换函数调用，不替换函数定义
        sed -i.bak 's/if ! check_nccl_dependencies; then/if ! true; then # Mock: 跳过环境检查/g' "$temp_script"
        
        # 添加 mock 环境检查函数
        cat > /tmp/mock_env_functions.sh << 'EOF'
# Mock 环境检查函数
check_nccl_dependencies() {
    log_info "Mock: 跳过 NCCL 环境依赖检查 (dry-run 模式)"
    log_success "Mock: NCCL 环境依赖检查通过"
    return 0
}
EOF
        
        # 将 mock 函数插入到脚本开头（在日志函数定义之后）
        sed -i.bak '/^# 日志函数/r /tmp/mock_env_functions.sh' "$temp_script"
    fi
    
    # 添加兼容性函数
    if [ "$BASH_COMPAT_MODE" = "true" ]; then
        cat > /tmp/compat_functions.sh << 'EOF'
# 兼容性函数
NCCL_CONFIG_CACHE_keys=""
SYSTEM_INFO_CACHE_keys=""

set_nccl_config() {
    local key="$1"
    local value="$2"
    local description="${3:-}"
    
    export "NCCL_$key"="$value"
    eval "NCCL_CONFIG_CACHE_$key='$value'"
    
    if [ -n "$description" ]; then
        log_info "设置 NCCL_$key=$value ($description)"
    fi
}
EOF
        
        # 将兼容性函数插入到脚本中
        sed -i.bak '/^# NCCL 配置管理器/r /tmp/compat_functions.sh' "$temp_script"
    fi
    
    echo "$temp_script"
}

# 环境依赖检查的宽松模式
setup_lenient_environment_check() {
    # 在 dry-run 模式下，创建宽松的环境检查
    if [[ "$*" == *"--dry-run"* ]]; then
        export LENIENT_MODE=true
        
        # Mock Python 和 PyTorch
        if [ ! -f "/tmp/mock_python3" ]; then
            cat > /tmp/mock_python3 << 'EOF'
#!/bin/bash
case "$1" in
    "-c")
        case "$2" in
            *"import torch"*)
                echo "Mock PyTorch 2.0.0"
                exit 0
                ;;
            *)
                echo "Mock Python 3.8.0"
                exit 0
                ;;
        esac
        ;;
    *)
        echo "Mock Python 3.8.0"
        exit 0
        ;;
esac
EOF
            chmod +x /tmp/mock_python3
        fi
        
        # 创建 mock pip3
        if [ ! -f "/tmp/mock_pip3" ]; then
            cat > /tmp/mock_pip3 << 'EOF'
#!/bin/bash
echo "Mock pip 21.0.0"
exit 0
EOF
            chmod +x /tmp/mock_pip3
        fi
    fi
}

# 主函数
main() {
    echo "🔧 NCCL Benchmark Mock Environment"
    echo "原始脚本: $ORIGINAL_SCRIPT"
    
    # 检查原始脚本是否存在
    if [ ! -f "$ORIGINAL_SCRIPT" ]; then
        echo "错误: 找不到原始脚本 $ORIGINAL_SCRIPT"
        exit 1
    fi
    
    # 设置兼容性环境
    check_bash_compatibility
    setup_mock_environment
    setup_lenient_environment_check "$@"
    
    # 预处理脚本
    local processed_script=$(preprocess_script "$@")
    
    echo "✓ Mock 环境设置完成"
    echo "✓ Bash 兼容性: $([ "$BASH_COMPAT_MODE" = "true" ] && echo "兼容模式" || echo "原生模式")"
    echo "✓ 环境检查: $([ "$LENIENT_MODE" = "true" ] && echo "宽松模式" || echo "标准模式")"
    echo ""
    
    # 执行处理后的脚本
    bash "$processed_script" "$@"
    local exit_code=$?
    
    # 清理临时文件
    rm -f "$processed_script" "$processed_script.bak" /tmp/compat_functions.sh
    
    exit $exit_code
}

# 清理函数
cleanup() {
    # 清理 mock 文件
    rm -f /tmp/mock_nvidia_smi /tmp/mock_ibv_devinfo /tmp/mock_python3 /tmp/mock_pip3
    rm -f /tmp/nccl_benchmark_processed.sh /tmp/nccl_benchmark_processed.sh.bak
    rm -f /tmp/compat_functions.sh
}

# 设置退出时清理
trap cleanup EXIT

# 运行主函数
main "$@"