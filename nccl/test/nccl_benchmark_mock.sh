#!/bin/bash
# =============================================================================
# NCCL Benchmark 增强版 Mock 包装器
# 功能: 使用独立的 mock 模块提供完整的测试环境
# =============================================================================

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_SCRIPT="$(dirname "$SCRIPT_DIR")/nccl_benchmark.sh"
MOCK_DIR="$SCRIPT_DIR/mock"

# 加载 Mock 模块
source "$MOCK_DIR/mock_system_info.sh"

# 创建临时的 mock 命令
setup_mock_commands() {
    local mock_bin_dir="/tmp/nccl_mock_bin"
    mkdir -p "$mock_bin_dir"
    
    # Mock nvidia-smi
    cat > "$mock_bin_dir/nvidia-smi" << 'EOF'
#!/bin/bash
case "$1" in
    "-L")
        for i in $(seq 0 $((${MOCK_GPU_COUNT:-4} - 1))); do
            echo "GPU $i: NVIDIA A100-SXM4-40GB (UUID: GPU-$(printf "%08d" $i)-1234-5678-9abc-def012345678)"
        done
        ;;
    "nvlink")
        if [ "$2" = "--status" ] && [ "${MOCK_NVLINK_COUNT:-0}" -gt 0 ]; then
            for i in $(seq 1 ${MOCK_NVLINK_COUNT}); do
                echo "GPU $((i-1)): 26.562 GB/s"
            done
        fi
        ;;
    *)
        echo "Mock nvidia-smi: 未知参数 $*"
        ;;
esac
EOF
    chmod +x "$mock_bin_dir/nvidia-smi"
    
    # Mock ibv_devinfo
    cat > "$mock_bin_dir/ibv_devinfo" << 'EOF'
#!/bin/bash
if [ "${MOCK_IB_AVAILABLE:-false}" = "true" ]; then
    cat << 'IB_INFO'
hca_id: mlx5_0
    transport:                  InfiniBand (0)
    fw_ver:                     16.35.2000
    node_guid:                  248a:0703:00b4:7a96
    sys_image_guid:             248a:0703:00b4:7a96
    vendor_id:                  0x02c9
    vendor_part_id:             4123
    hw_ver:                     0x0
    board_id:                   MT_0000000013
    phys_port_cnt:              1
        port:   1
            state:              PORT_ACTIVE (4)
            max_mtu:            4096 (5)
            active_mtu:         4096 (5)
            sm_lid:             1
            port_lid:           1
            port_lmc:           0x00
            link_layer:         InfiniBand
IB_INFO
else
    exit 1
fi
EOF
    chmod +x "$mock_bin_dir/ibv_devinfo"
    
    # 将 mock 命令添加到 PATH
    export PATH="$mock_bin_dir:$PATH"
    echo "✓ Mock 命令已设置"
}

# 创建增强的脚本包装器
create_enhanced_wrapper() {
    local wrapper_script="/tmp/nccl_benchmark_enhanced.sh"
    
    # 复制原始脚本
    cp "$ORIGINAL_SCRIPT" "$wrapper_script"
    
    # 在脚本开头插入 mock 系统信息函数的重写
    cat > /tmp/mock_override.sh << 'EOF'

# Mock 系统信息函数重写
cache_system_info() {
    if [ -n "${MOCK_GPU_COUNT:-}" ]; then
        SYSTEM_INFO_CACHE[gpu_count]=$MOCK_GPU_COUNT
    elif command -v nvidia-smi >/dev/null 2>&1; then
        SYSTEM_INFO_CACHE[gpu_count]=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        SYSTEM_INFO_CACHE[gpu_count]=0
    fi
    
    SYSTEM_INFO_CACHE[nvlink_available]=false
    SYSTEM_INFO_CACHE[nvlink_count]=0
    
    if [ -n "${MOCK_NVLINK_COUNT:-}" ]; then
        if [ "$MOCK_NVLINK_COUNT" -gt 0 ]; then
            SYSTEM_INFO_CACHE[nvlink_available]=true
            SYSTEM_INFO_CACHE[nvlink_count]=$MOCK_NVLINK_COUNT
        fi
    elif [ "${SYSTEM_INFO_CACHE[gpu_count]}" -gt 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi nvlink --status &>/dev/null; then
            local nvlink_count=$(nvidia-smi nvlink --status 2>/dev/null | grep -c "GB/s" 2>/dev/null || echo "0")
            nvlink_count=$(echo "$nvlink_count" | tr -d ' \n\r\t')
            if [[ "$nvlink_count" =~ ^[0-9]+$ ]] && [ "$nvlink_count" -gt 0 ]; then
                SYSTEM_INFO_CACHE[nvlink_available]=true
                SYSTEM_INFO_CACHE[nvlink_count]=$nvlink_count
            fi
        fi
    fi
    
    SYSTEM_INFO_CACHE[ib_available]=false
    if [ -n "${MOCK_IB_AVAILABLE:-}" ] && [ "$MOCK_IB_AVAILABLE" = "true" ]; then
        SYSTEM_INFO_CACHE[ib_available]=true
    elif command -v ibv_devinfo >/dev/null 2>&1; then
        local ib_output
        if ib_output=$(ibv_devinfo 2>/dev/null) && echo "$ib_output" | grep -q "hca_id:"; then
            SYSTEM_INFO_CACHE[ib_available]=true
        fi
    fi
}

EOF
    
    # 在原始脚本的 cache_system_info 函数定义之后插入重写版本
    sed -i.bak '/^cache_system_info() {/,/^}$/c\
# Mock 增强的 cache_system_info 函数\
source /tmp/mock_override.sh' "$wrapper_script"
    
    echo "$wrapper_script"
}

# 清理函数
cleanup_enhanced_mock() {
    rm -rf /tmp/nccl_mock_bin
    rm -f /tmp/nccl_benchmark_enhanced.sh /tmp/nccl_benchmark_enhanced.sh.bak
    rm -f /tmp/mock_override.sh
    cleanup_mock_system_info
}

# 主函数
main() {
    echo "🚀 NCCL Benchmark 增强版 Mock 环境"
    echo "原始脚本: $ORIGINAL_SCRIPT"
    
    # 检查原始脚本
    if [ ! -f "$ORIGINAL_SCRIPT" ]; then
        echo "❌ 错误: 找不到原始脚本 $ORIGINAL_SCRIPT"
        exit 1
    fi
    
    # 检查是否指定了 mock 场景
    local mock_scenario=""
    local script_args=()
    
    for arg in "$@"; do
        case "$arg" in
            --mock-scenario=*)
                mock_scenario="${arg#--mock-scenario=}"
                ;;
            *)
                script_args+=("$arg")
                ;;
        esac
    done
    
    # 初始化 Mock 环境
    init_mock_system_info
    
    # 如果指定了场景，则设置对应的 mock 数据
    if [ -n "$mock_scenario" ]; then
        echo "🎭 设置 Mock 场景: $mock_scenario"
        set_mock_scenario "$mock_scenario"
    fi
    
    # 导出环境变量
    export_mock_environment
    
    # 设置 mock 命令
    setup_mock_commands
    
    # 创建增强的包装器
    local enhanced_script=$(create_enhanced_wrapper)
    
    echo "✅ Mock 环境设置完成"
    echo ""
    
    # 执行增强的脚本
    bash "$enhanced_script" "${script_args[@]}"
    local exit_code=$?
    
    # 清理
    cleanup_enhanced_mock
    
    exit $exit_code
}

# 设置退出时清理
trap cleanup_enhanced_mock EXIT

# 显示使用帮助
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "NCCL Benchmark 增强版 Mock 包装器"
    echo ""
    echo "用法: $0 [--mock-scenario=SCENARIO] [NCCL_BENCHMARK_ARGS...]"
    echo ""
    echo "Mock 场景:"
    echo "  single_gpu        - 单 GPU 环境"
    echo "  multi_gpu_nvlink  - 多 GPU + NVLink 环境"
    echo "  multi_gpu_pcie    - 多 GPU + PCIe 环境"
    echo "  cluster_ib        - 集群 + InfiniBand 环境"
    echo ""
    echo "示例:"
    echo "  $0 --mock-scenario=multi_gpu_nvlink --dry-run"
    echo "  $0 --mock-scenario=cluster_ib --pxn-enable"
    echo ""
    exit 0
fi

# 运行主函数
main "$@"