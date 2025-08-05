#!/bin/bash
# =============================================================================
# Mock 系统信息模块
# 功能: 为测试环境提供模拟的系统信息，包括 GPU、NVLink、InfiniBand 等
# =============================================================================

# Mock 系统信息缓存
declare -A MOCK_SYSTEM_INFO_CACHE

# 初始化 Mock 环境
init_mock_system_info() {
    # 设置默认的 mock 值
    MOCK_SYSTEM_INFO_CACHE[gpu_count]=${MOCK_GPU_COUNT:-4}
    MOCK_SYSTEM_INFO_CACHE[nvlink_available]=${MOCK_NVLINK_AVAILABLE:-false}
    MOCK_SYSTEM_INFO_CACHE[nvlink_count]=${MOCK_NVLINK_COUNT:-0}
    MOCK_SYSTEM_INFO_CACHE[ib_available]=${MOCK_IB_AVAILABLE:-false}
    
    # 根据 NVLink 数量设置可用性
    if [ "${MOCK_SYSTEM_INFO_CACHE[nvlink_count]}" -gt 0 ]; then
        MOCK_SYSTEM_INFO_CACHE[nvlink_available]=true
    fi
    
    echo "✓ Mock 系统信息初始化完成"
    echo "  - GPU 数量: ${MOCK_SYSTEM_INFO_CACHE[gpu_count]}"
    echo "  - NVLink 可用: ${MOCK_SYSTEM_INFO_CACHE[nvlink_available]}"
    echo "  - NVLink 数量: ${MOCK_SYSTEM_INFO_CACHE[nvlink_count]}"
    echo "  - InfiniBand 可用: ${MOCK_SYSTEM_INFO_CACHE[ib_available]}"
}

# Mock 版本的 cache_system_info 函数
mock_cache_system_info() {
    # 使用 mock 数据填充系统信息缓存
    SYSTEM_INFO_CACHE[gpu_count]=${MOCK_SYSTEM_INFO_CACHE[gpu_count]}
    SYSTEM_INFO_CACHE[nvlink_available]=${MOCK_SYSTEM_INFO_CACHE[nvlink_available]}
    SYSTEM_INFO_CACHE[nvlink_count]=${MOCK_SYSTEM_INFO_CACHE[nvlink_count]}
    SYSTEM_INFO_CACHE[ib_available]=${MOCK_SYSTEM_INFO_CACHE[ib_available]}
}

# Mock GPU 拓扑检测
mock_detect_gpu_topology() {
    echo "检测到 ${MOCK_SYSTEM_INFO_CACHE[gpu_count]} 个 NVIDIA GPU"
    
    if [ "${MOCK_SYSTEM_INFO_CACHE[nvlink_available]}" = "true" ]; then
        echo "检测到 ${MOCK_SYSTEM_INFO_CACHE[nvlink_count]} 个 NVLink 连接"
        echo "GPU 拓扑: NVLink 高速互连"
    else
        echo "未检测到 NVLink 连接"
        echo "GPU 拓扑: PCIe 连接"
    fi
    
    if [ "${MOCK_SYSTEM_INFO_CACHE[ib_available]}" = "true" ]; then
        echo "检测到 InfiniBand 网络"
    else
        echo "未检测到 InfiniBand 网络"
    fi
}

# 设置 Mock 场景
set_mock_scenario() {
    local scenario="$1"
    
    case "$scenario" in
        "single_gpu")
            MOCK_SYSTEM_INFO_CACHE[gpu_count]=1
            MOCK_SYSTEM_INFO_CACHE[nvlink_available]=false
            MOCK_SYSTEM_INFO_CACHE[nvlink_count]=0
            MOCK_SYSTEM_INFO_CACHE[ib_available]=false
            ;;
        "multi_gpu_nvlink")
            MOCK_SYSTEM_INFO_CACHE[gpu_count]=8
            MOCK_SYSTEM_INFO_CACHE[nvlink_available]=true
            MOCK_SYSTEM_INFO_CACHE[nvlink_count]=8
            MOCK_SYSTEM_INFO_CACHE[ib_available]=false
            ;;
        "multi_gpu_pcie")
            MOCK_SYSTEM_INFO_CACHE[gpu_count]=4
            MOCK_SYSTEM_INFO_CACHE[nvlink_available]=false
            MOCK_SYSTEM_INFO_CACHE[nvlink_count]=0
            MOCK_SYSTEM_INFO_CACHE[ib_available]=false
            ;;
        "cluster_ib")
            MOCK_SYSTEM_INFO_CACHE[gpu_count]=8
            MOCK_SYSTEM_INFO_CACHE[nvlink_available]=true
            MOCK_SYSTEM_INFO_CACHE[nvlink_count]=8
            MOCK_SYSTEM_INFO_CACHE[ib_available]=true
            ;;
        *)
            echo "未知的 Mock 场景: $scenario"
            echo "可用场景: single_gpu, multi_gpu_nvlink, multi_gpu_pcie, cluster_ib"
            return 1
            ;;
    esac
    
    echo "✓ 设置 Mock 场景: $scenario"
    mock_detect_gpu_topology
}

# 导出 Mock 环境变量
export_mock_environment() {
    export MOCK_GPU_COUNT=${MOCK_SYSTEM_INFO_CACHE[gpu_count]}
    export MOCK_NVLINK_AVAILABLE=${MOCK_SYSTEM_INFO_CACHE[nvlink_available]}
    export MOCK_NVLINK_COUNT=${MOCK_SYSTEM_INFO_CACHE[nvlink_count]}
    export MOCK_IB_AVAILABLE=${MOCK_SYSTEM_INFO_CACHE[ib_available]}
    
    echo "✓ Mock 环境变量已导出"
}

# 清理 Mock 环境
cleanup_mock_system_info() {
    unset MOCK_GPU_COUNT MOCK_NVLINK_AVAILABLE MOCK_NVLINK_COUNT MOCK_IB_AVAILABLE
    unset MOCK_SYSTEM_INFO_CACHE
    echo "✓ Mock 环境已清理"
}

# 如果直接运行此脚本，则进行测试
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "🧪 Mock 系统信息模块测试"
    
    # 测试不同场景
    for scenario in "single_gpu" "multi_gpu_nvlink" "multi_gpu_pcie" "cluster_ib"; do
        echo ""
        echo "--- 测试场景: $scenario ---"
        set_mock_scenario "$scenario"
        export_mock_environment
        echo ""
    done
    
    cleanup_mock_system_info
fi