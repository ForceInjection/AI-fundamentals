#!/bin/bash
# =============================================================================
# NCCL InfiniBand 测试验证脚本
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
SCRIPT_NAME="NCCL IB Test"

# 全局变量
LOG_FILE="/tmp/nccl_ib_test_$(date +%Y%m%d_%H%M%S).log"
ERROR_COUNT=0
WARNING_COUNT=0
QUIET_MODE=false
TEST_SIZE="1M"  # 测试数据大小: 1M, 10M, 100M, 1G
TEST_DURATION=30  # 测试持续时间(秒)
MULTI_NODE_MODE=false
MASTER_ADDR="localhost"
MASTER_PORT="29500"

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

# 显示帮助信息
show_help() {
    cat << EOF
NCCL InfiniBand 测试验证脚本 v${VERSION}

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -v, --version           显示版本信息
  -q, --quiet             静默模式 (仅输出关键信息)
  -s, --size SIZE         测试数据大小 (1M, 10M, 100M, 1G) [默认: 1M]
                          1M  = 约 1MB  (262K 元素)
                          10M = 约 10MB (2.6M 元素) 
                          100M= 约 100MB(26M 元素)
                          1G  = 约 1GB  (268M 元素)
  -t, --time SECONDS      测试持续时间 (秒) [默认: 30]
  -m, --multi-node        多节点模式
  --master-addr ADDR      主节点地址 [默认: localhost]
  --master-port PORT      主节点端口 [默认: 29500]
  --check-only            仅检查 NCCL 环境，不运行测试
  --env-only              仅设置环境变量并显示配置

功能:
  • 检查 NCCL 和 PyTorch 环境
  • 自动检测网络类型并配置 NCCL 参数
  • 运行 AllReduce 性能测试
  • 分析网络通信效率
  • 生成详细的测试报告

测试模式:
  单节点模式: 测试单机多 GPU 之间的 NCCL 通信
  多节点模式: 测试跨节点的 NCCL 通信 (需要在每个节点运行)

前置条件:
  • 建议先运行 ib_health_check.sh 确保 IB 网络正常
  • 可配合 ib_bandwidth_monitor.sh 监控测试期间的网络性能

示例:
  $0                                    # 基本测试 (1MB 数据)
  $0 -s 10M                            # 10MB 数据测试
  $0 -s 100M -t 60                     # 100MB 数据，测试 60 秒
  $0 -s 1G                             # 1GB 大数据测试 (需要足够 GPU 内存)
  $0 --check-only                      # 仅检查环境
  $0 --env-only                        # 仅显示环境配置
  $0 -m --master-addr 192.168.1.100    # 多节点模式

EOF
}

# 显示版本信息
show_version() {
    echo "$SCRIPT_NAME v$VERSION"
    echo "专注于 NCCL InfiniBand 通信测试"
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
                TEST_SIZE="$2"
                shift 2
                ;;
            -t|--time)
                TEST_DURATION="$2"
                shift 2
                ;;
            -m|--multi-node)
                MULTI_NODE_MODE=true
                shift
                ;;
            --master-addr)
                MASTER_ADDR="$2"
                shift 2
                ;;
            --master-port)
                MASTER_PORT="$2"
                shift 2
                ;;
            --check-only)
                CHECK_ONLY=true
                shift
                ;;
            --env-only)
                ENV_ONLY=true
                shift
                ;;
            *)
                log_error "未知选项: $1"
                echo "使用 '$0 --help' 查看帮助信息"
                exit 1
                ;;
        esac
    done
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
            deps_ok=false
        fi
        
        # 检查 NCCL
        if python3 -c "import torch; torch.cuda.nccl.version()" 2>/dev/null; then
            local nccl_version=$(python3 -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null)
            log_success "NCCL 版本: $nccl_version"
        else
            log_warning "无法获取 NCCL 版本信息"
        fi
    else
        log_error "PyTorch 未安装或不可用"
        deps_ok=false
    fi
    
    # 检查 NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi &>/dev/null; then
            local gpu_count=$(nvidia-smi -L | wc -l)
            log_success "检测到 $gpu_count 个 NVIDIA GPU"
            if [ "$QUIET_MODE" = false ]; then
                nvidia-smi -L | while read line; do
                    log_info "  $line"
                done
            fi
        else
            log_error "nvidia-smi 执行失败"
            deps_ok=false
        fi
    else
        log_error "nvidia-smi 命令不可用"
        deps_ok=false
    fi
    
    # 简单检查 InfiniBand 可用性
    if command -v ibstat >/dev/null 2>&1; then
        if ibstat &>/dev/null; then
            log_success "InfiniBand 设备可用"
        else
            log_warning "InfiniBand 设备不可用 (可能影响性能)"
        fi
    else
        log_warning "ibstat 命令不可用 (建议先运行 ib_health_check.sh)"
    fi
    
    if [ "$deps_ok" = true ]; then
        log_success "NCCL 环境依赖检查通过"
        return 0
    else
        log_error "NCCL 环境依赖检查失败"
        return 1
    fi
}

# 智能检测网络类型并设置 NCCL 环境变量
# 功能说明：
# 1. 自动检测 InfiniBand 网络类型（原生 IB vs RoCE）
# 2. 根据网络类型设置最优的 NCCL 环境变量
# 3. 配置 GPUDirect RDMA 和性能优化参数
setup_nccl_env() {
    log_header "配置 NCCL 环境变量"
    
    # 检测网络类型
    # 使用 ibv_devinfo 命令检查 link_layer 字段来区分网络类型
    local is_roce=false
    local network_type="未知"
    
    if command -v ibv_devinfo >/dev/null 2>&1; then
        # 获取第一个 IB 设备的链路层类型
        # link_layer: Ethernet 表示 RoCE，InfiniBand 表示原生 IB
        local link_layer=$(ibv_devinfo | grep "link_layer:" | head -1 | awk '{print $2}')
        if [ "$link_layer" = "Ethernet" ]; then
            is_roce=true
            network_type="RoCE (Ethernet over IB)"
            log_info "检测到 RoCE 环境，配置相应的 NCCL 参数"
        elif [ "$link_layer" = "InfiniBand" ]; then
            network_type="原生 InfiniBand"
            log_info "检测到原生 IB 环境，配置标准 NCCL 参数"
        fi
    else
        log_warning "无法检测网络类型，使用默认配置"
    fi
    
    # ========== 基础 NCCL InfiniBand 配置 ==========
    # NCCL_IB_DISABLE=0: 启用 InfiniBand 传输层
    # 这是使用 IB 网络进行 NCCL 通信的前提条件
    export NCCL_IB_DISABLE=0
    log_info "设置 NCCL_IB_DISABLE=0 (启用 InfiniBand)"
    
    # ========== GPUDirect RDMA 配置 ==========
    # NCCL_NET_GDR_LEVEL=2: 启用完整的 GPUDirect RDMA 支持
    # Level 0: 禁用 GDR
    # Level 1: 启用 GDR 但不使用 GPU 内存作为网络缓冲区
    # Level 2: 完全启用 GDR，包括 GPU 内存网络缓冲区（最佳性能）
    export NCCL_NET_GDR_LEVEL=2
    log_info "设置 NCCL_NET_GDR_LEVEL=2 (启用 GPUDirect RDMA)"
    
    # ========== HCA 设备配置 ==========
    # 自动检测并设置 Host Channel Adapter (HCA) 设备名
    # HCA 是 InfiniBand 网络适配器的标准术语
    if command -v ibstat >/dev/null 2>&1; then
        # 从 ibstat 输出中提取第一个 CA 设备名
        # 格式示例: CA 'mlx5_0'
        local hca_name=$(ibstat | grep "CA '" | head -1 | awk '{print $2}' | tr -d "'")
        if [ -n "$hca_name" ]; then
            export NCCL_IB_HCA="$hca_name"
            log_info "设置 NCCL_IB_HCA=$hca_name"
        else
            log_warning "无法获取 HCA 设备名"
        fi
    fi
    
    # ========== 网络类型特定配置 ==========
    # 根据检测到的网络类型（RoCE vs 原生 IB）设置不同的参数
    if [ "$is_roce" = true ]; then
        # ---------- RoCE (RDMA over Converged Ethernet) 配置 ----------
        # GID Index 3 通常对应 RoCE v2 (IPv4/IPv6 over Ethernet)
        # RoCE v1 使用 GID Index 1-2，RoCE v2 使用 GID Index 3+
        export NCCL_IB_GID_INDEX=3
        log_info "设置 NCCL_IB_GID_INDEX=3 (RoCE v2)"
        
        # Traffic Class (TC) 和 Service Level (SL) 配置
        # TC=136: 设置 DSCP 值，用于 QoS 流量分类
        # SL=0: 服务级别，影响 IB 交换机的优先级队列选择
        export NCCL_IB_TC=136
        export NCCL_IB_SL=0
        log_info "设置 RoCE 流控制参数 (TC=136, SL=0)"
        
        # 禁用 Socket 接口，强制 NCCL 使用 RoCE
        # 这确保 NCCL 不会回退到 TCP/IP 传输
        export NCCL_SOCKET_IFNAME=""
        log_info "禁用 Socket 接口，强制使用 RoCE"
    else
        # ---------- 原生 InfiniBand 配置 ----------
        # GID Index 0 是原生 IB 的默认全局标识符索引
        # 原生 IB 不需要 IP 路由，直接使用 IB 地址空间
        export NCCL_IB_GID_INDEX=0
        log_info "设置 NCCL_IB_GID_INDEX=0 (原生 IB)"
        
        # 原生 IB 的流控制参数
        # 这些参数与 RoCE 相同，但在原生 IB 中有不同的语义
        export NCCL_IB_TC=136
        export NCCL_IB_SL=0
        log_info "设置 IB 流控制参数"
    fi
    
    # ========== NCCL 调试配置 ==========
    # 根据脚本的静默模式设置不同的调试级别
    if [ "$QUIET_MODE" = false ]; then
        # 详细模式：启用信息级别的调试输出
        # DEBUG=INFO: 显示初始化、网络选择、拓扑发现等关键信息
        # SUBSYS=INIT,NET: 只显示初始化和网络子系统的调试信息
        export NCCL_DEBUG=INFO
        export NCCL_DEBUG_SUBSYS=INIT,NET
        log_info "启用 NCCL 调试信息 (DEBUG=INFO, SUBSYS=INIT,NET)"
    else
        # 静默模式：只显示警告和错误信息
        # DEBUG=WARN: 只显示警告级别及以上的消息
        # SUBSYS=ALL: 监控所有子系统的警告信息
        export NCCL_DEBUG=WARN
        export NCCL_DEBUG_SUBSYS=ALL
        log_info "设置 NCCL 调试级别为 WARN"
    fi
    
    # ========== InfiniBand 性能优化参数 ==========
    # IB_TIMEOUT: InfiniBand 操作超时时间（以 4.096μs 为单位）
    # 22 对应约 90ms 超时，适合大多数网络环境
    # 较大的值可以处理网络延迟较高的情况
    export NCCL_IB_TIMEOUT=22
    
    # IB_RETRY_CNT: InfiniBand 重传次数
    # 7 是推荐值，在网络不稳定时提供足够的重试机会
    # 过高的值会增加故障恢复时间，过低可能导致不必要的失败
    export NCCL_IB_RETRY_CNT=7
    log_info "设置 IB 超时和重试参数 (TIMEOUT=22, RETRY_CNT=7)"
    
    # 多节点配置
    if [ "$MULTI_NODE_MODE" = true ]; then
        export NCCL_SOCKET_IFNAME=^docker0,lo
        log_info "多节点模式: 排除 docker0 和 lo 接口"
    fi
    
    # 显示配置摘要
    log_info ""
    log_success "NCCL 环境变量配置完成"
    log_info "配置摘要:"
    log_info "  网络类型: $network_type"
    log_info "  NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
    log_info "  NCCL_NET_GDR_LEVEL: $NCCL_NET_GDR_LEVEL"
    log_info "  NCCL_IB_HCA: ${NCCL_IB_HCA:-未设置}"
    log_info "  NCCL_IB_GID_INDEX: $NCCL_IB_GID_INDEX"
    log_info "  NCCL_DEBUG: $NCCL_DEBUG"
    log_info "  多节点模式: $([ "$MULTI_NODE_MODE" = true ] && echo "是" || echo "否")"
}

# 创建动态 NCCL 测试脚本
# 功能说明：
# 1. 根据用户指定的测试数据大小动态生成 Python 测试脚本
# 2. 实现完整的 NCCL AllReduce 测试逻辑
# 3. 包含性能测量、内存检查和结果验证
create_nccl_test() {
    log_header "创建 NCCL 测试脚本"
    
    local test_script="/tmp/nccl_ib_test.py"
    
    # ========== 动态计算测试数据大小 ==========
    # 将用户友好的大小表示（如 1M, 100M, 1G）转换为张量元素数量
    # 假设使用 float32 数据类型（4 字节/元素）
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
        *)
            log_warning "未知的测试大小: $TEST_SIZE，使用默认值 1M"
            tensor_elements=262144
            ;;
    esac
    
    log_info "配置测试数据大小: $TEST_SIZE (约 $tensor_elements 个元素)"
    
    cat > "$test_script" << EOF
import torch
import torch.distributed as dist
import os
import time
import socket
import sys

def get_local_ip():
    """获取本机 IP 地址"""
    try:
        # 连接到一个不存在的地址来获取本机 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def print_nccl_info():
    """打印 NCCL 相关信息"""
    print("=== NCCL 环境信息 ===")
    
    # NCCL 版本
    if hasattr(torch.cuda.nccl, 'version'):
        print(f"NCCL 版本: {torch.cuda.nccl.version()}")
    else:
        print("NCCL 版本: 未知")
    
    # 环境变量
    nccl_vars = [
        'NCCL_IB_DISABLE', 'NCCL_NET_GDR_LEVEL', 'NCCL_IB_HCA',
        'NCCL_IB_GID_INDEX', 'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS'
    ]
    
    print("NCCL 环境变量:")
    for var in nccl_vars:
        value = os.environ.get(var, '未设置')
        print(f"  {var}: {value}")

def test_nccl_allreduce():
    """
    测试 NCCL AllReduce 操作
    
    功能说明：
    1. 创建测试张量并检查 GPU 内存容量
    2. 执行 AllReduce 操作并测量性能
    3. 验证计算结果的正确性
    4. 计算并报告性能指标
    
    返回值：
    bool: 测试是否成功（结果验证通过）
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}] 开始 NCCL AllReduce 测试")
    
    # ========== 动态张量创建和内存检查 ==========
    device = torch.device(f"cuda:{rank}")
    tensor_size = $tensor_elements  # 动态设置的元素数量
    
    # GPU 内存容量检查
    # 确保测试数据不会超出 GPU 内存限制
    gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    required_memory_gb = tensor_size * 4 / 1024**3  # float32 = 4 bytes per element
    
    print(f"[Rank {rank}] GPU 内存: {gpu_memory_gb:.1f} GB")
    print(f"[Rank {rank}] 所需内存: {required_memory_gb:.2f} GB")
    
    # 内存安全检查：保留 20% 的 GPU 内存作为缓冲
    if required_memory_gb > gpu_memory_gb * 0.8:
        print(f"[Rank {rank}] 警告: 测试数据可能超出 GPU 内存容量")
    
    # 尝试分配 GPU 内存
    # 每个 rank 使用不同的初始值 (rank + 1)，便于验证 AllReduce 结果
    try:
        tensor = torch.ones(tensor_size, device=device, dtype=torch.float32) * (rank + 1)
    except RuntimeError as e:
        print(f"[Rank {rank}] 错误: 无法分配 GPU 内存 - {e}")
        return False
    
    print(f"[Rank {rank}] 初始张量值: {tensor[0].item()}")
    print(f"[Rank {rank}] 张量大小: {tensor_size} elements ({tensor.numel() * tensor.element_size() / 1024 / 1024:.2f} MB)")
    
    # ========== 分布式同步 ==========
    # 确保所有进程都完成了张量初始化后再开始测试
    # 这对于准确的性能测量至关重要
    dist.barrier()
    
    # ========== NCCL AllReduce 执行和性能测量 ==========
    # 使用高精度计时器测量 AllReduce 操作的端到端延迟
    start_time = time.time()
    
    # 执行 AllReduce SUM 操作
    # 所有 rank 的张量将被求和，结果广播到所有 rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # 确保 GPU 操作完全完成
    # 这对于准确测量异步 GPU 操作的时间至关重要
    torch.cuda.synchronize()
    end_time = time.time()
    
    # ========== 结果验证 ==========
    # AllReduce SUM 的预期结果：sum(1, 2, ..., world_size)
    # 例如：3个rank时，预期结果 = 1 + 2 + 3 = 6
    expected_sum = sum(range(1, world_size + 1))
    actual_result = tensor[0].item()
    duration_ms = (end_time - start_time) * 1000
    
    print(f"[Rank {rank}] AllReduce 结果: {actual_result}")
    print(f"[Rank {rank}] 预期结果: {expected_sum}")
    print(f"[Rank {rank}] 耗时: {duration_ms:.2f} ms")
    
    # ========== 性能指标计算 ==========
    # 计算数据吞吐量（注意：这是应用层吞吐量，不等同于网络带宽）
    # 吞吐量 = 数据量 / 时间，单位转换为 Gbps
    data_size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
    throughput_gbps = (data_size_mb * 8) / (duration_ms / 1000) / 1000  # MB -> Gb -> Gbps
    print(f"[Rank {rank}] 数据吞吐量: {throughput_gbps:.2f} Gbps (注意：非网络带宽)")
    print(f"[Rank {rank}] 数据大小: {data_size_mb:.2f} MB")
    
    # ========== AllReduce 算法效率分析 ==========
    if world_size > 1:
        # Ring AllReduce 理论传输量计算
        # 公式：2 * (N-1) / N * data_size
        # 其中 N 是参与的 GPU 数量
        # 
        # 算法原理：
        # 1. Reduce-Scatter 阶段：每个 GPU 发送 (N-1)/N 的数据
        # 2. AllGather 阶段：每个 GPU 再次发送 (N-1)/N 的数据
        # 3. 总传输量：2 * (N-1)/N * data_size
        theoretical_transfer_mb = 2 * (world_size - 1) / world_size * data_size_mb
        print(f"[Rank {rank}] 理论传输量: {theoretical_transfer_mb:.2f} MB (Ring AllReduce)")
        
        # 网络效率分析
        # 实际传输效率 = 理论传输量 / 测量的数据吞吐量对应的数据量
        network_efficiency = theoretical_transfer_mb / data_size_mb
        print(f"[Rank {rank}] 网络传输倍数: {network_efficiency:.2f}x (相对于数据大小)")
    else:
        print(f"[Rank {rank}] 单GPU测试，无网络传输")
    
    # ========== 结果正确性验证 ==========
    # 使用浮点数比较的容差检查
    # 1e-6 的容差足以处理浮点运算的精度误差
    return abs(actual_result - expected_sum) < 1e-6

def main():
    try:
        # 初始化分布式环境
        dist.init_process_group(backend='nccl')
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_ip = get_local_ip()
        
        print(f"=== NCCL 测试信息 ===")
        print(f"Rank: {rank}/{world_size}")
        print(f"本机 IP: {local_ip}")
        print(f"CUDA 设备: {torch.cuda.current_device()}")
        print(f"GPU 名称: {torch.cuda.get_device_name()}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 只在 rank 0 打印 NCCL 信息
        if rank == 0:
            print_nccl_info()
        
        # 执行测试
        success = test_nccl_allreduce()
        
        # 同步所有进程
        dist.barrier()
        
        if rank == 0:
            if success:
                print("✅ NCCL AllReduce 测试成功")
            else:
                print("❌ NCCL AllReduce 测试失败")
                sys.exit(1)
        
        # 清理
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"[Rank {rank if 'rank' in locals() else 'Unknown'}] 测试过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF
    
    if [ -f "$test_script" ]; then
        log_success "NCCL 测试脚本创建完成: $test_script"
        log_info "脚本大小: $(wc -l < "$test_script") 行"
    else
        log_error "NCCL 测试脚本创建失败"
        return 1
    fi
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
    local test_script="/tmp/nccl_ib_test.py"
    if [ ! -f "$test_script" ]; then
        log_error "测试脚本不存在: $test_script"
        return 1
    fi
    
    # 设置输出文件
    local output_file="/tmp/nccl_test_output.log"
    
    # 运行测试
    log_info "启动 NCCL 测试 (使用 $gpu_count 个 GPU)..."
    log_info "测试输出将保存到: $output_file"
    
    # 检查 torch.distributed.launch 是否可用
    if python3 -c "import torch.distributed.launch" 2>/dev/null; then
        # 使用新的 torchrun 命令（如果可用）
        if command -v torchrun >/dev/null 2>&1; then
            log_info "使用 torchrun 启动分布式测试"
            if torchrun \
                --nproc_per_node="$gpu_count" \
                --nnodes=1 \
                --node_rank=0 \
                --master_addr=localhost \
                --master_port=29500 \
                "$test_script" 2>&1 | tee "$output_file"; then
                log_success "NCCL 测试执行完成"
            else
                log_error "NCCL 测试执行失败"
                return 1
            fi
        else
            # 使用传统的 torch.distributed.launch
            log_info "使用 torch.distributed.launch 启动分布式测试"
            if python3 -m torch.distributed.launch \
                --nproc_per_node="$gpu_count" \
                --nnodes=1 \
                --node_rank=0 \
                --master_addr=localhost \
                --master_port=29500 \
                "$test_script" 2>&1 | tee "$output_file"; then
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
    analyze_nccl_output
}

# 分析 NCCL 输出
analyze_nccl_output() {
    log_header "分析 NCCL 测试输出"
    
    local output_file="/tmp/nccl_test_output.log"
    
    if [ ! -f "$output_file" ]; then
        log_error "测试输出文件不存在: $output_file"
        return 1
    fi
    
    local file_size=$(wc -l < "$output_file")
    log_info "输出文件大小: $file_size 行"
    
    # 检查测试是否成功完成
    if grep -q "NCCL AllReduce 测试成功" "$output_file"; then
        log_success "✅ NCCL 测试成功完成"
    elif grep -q "NCCL AllReduce 测试失败" "$output_file"; then
        log_error "❌ NCCL 测试失败"
    else
        log_warning "⚠️  无法确定测试结果"
    fi
    
    log_info ""
    log_info "NCCL 网络选择分析:"
    
    # 检查是否使用了 InfiniBand/RoCE
    local network_detected=false
    
    if grep -q "Using network InfiniBand" "$output_file"; then
        log_success "✅ NCCL 正在使用 InfiniBand 网络"
        network_detected=true
    elif grep -q "NET/IB" "$output_file"; then
        log_success "✅ NCCL 正在使用 InfiniBand 网络 (NET/IB)"
        network_detected=true
    elif grep -q "RoCE" "$output_file" || grep -q "Ethernet" "$output_file"; then
        log_success "✅ NCCL 正在使用 RoCE (Ethernet over IB) 网络"
        network_detected=true
    fi
    
    if ! $network_detected; then
        # 检查是否回退到其他网络
        if grep -q "NET/Socket" "$output_file"; then
            log_warning "⚠️  NCCL 回退到 Socket 网络 (可能 IB 配置有问题)"
        elif grep -q "NET/SHM" "$output_file"; then
            log_info "ℹ️  NCCL 使用共享内存 (单节点内通信)"
        else
            log_warning "⚠️  未明确检测到网络类型"
        fi
    fi
    
    # 检查 GPUDirect RDMA
    if grep -q "GDR" "$output_file" || grep -q "GPUDirect" "$output_file"; then
        log_success "✅ GPUDirect RDMA 已启用"
    else
        log_warning "⚠️  未检测到 GPUDirect RDMA"
    fi
    
    # 检查 NCCL 初始化信息
    log_info ""
    log_info "NCCL 初始化信息:"
    local nccl_init_lines=$(grep -E "(NCCL version|Using network|comm 0x)" "$output_file" | head -5)
    if [ -n "$nccl_init_lines" ]; then
        echo "$nccl_init_lines" | while read line; do
            log_info "  $line"
        done
    else
        log_warning "未找到 NCCL 初始化信息"
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
    log_info "性能信息分析:"
    
    # 分别提取延迟、吞吐量和数据大小信息
    local latency_lines=$(grep -E "耗时.*ms" "$output_file" 2>/dev/null)
    local throughput_lines=$(grep -E "数据吞吐量.*Gbps" "$output_file" 2>/dev/null)
    local data_size_lines=$(grep -E "数据大小.*MB" "$output_file" 2>/dev/null)
    local transfer_lines=$(grep -E "理论传输量.*MB" "$output_file" 2>/dev/null)
    
    if [ -n "$latency_lines" ]; then
        log_info "  延迟信息:"
        while IFS= read -r line; do
            log_info "    $line"
        done <<< "$latency_lines"
    fi
    
    if [ -n "$throughput_lines" ]; then
        log_info "  数据吞吐量:"
        while IFS= read -r line; do
            log_info "    $line"
        done <<< "$throughput_lines"
    fi
    
    if [ -n "$data_size_lines" ]; then
        log_info "  测试数据大小:"
        echo "$data_size_lines" | head -1 | while IFS= read -r line; do
            log_info "    $line"
        done
    fi
    
    if [ -n "$transfer_lines" ]; then
        log_info "  网络传输分析:"
        echo "$transfer_lines" | head -1 | while IFS= read -r line; do
            log_info "    $line"
        done
    fi
    
    if [ -z "$latency_lines" ] && [ -z "$throughput_lines" ]; then
        log_warning "未找到性能信息"
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



# 生成测试报告
generate_report() {
    log_header "生成测试报告"
    
    local report_file="/tmp/nccl_test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
=== NCCL InfiniBand 测试报告 ===
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
    echo "NCCL InfiniBand 测试开始 - $(date)" > "$LOG_FILE"
    
    # 显示脚本信息
    log_header "$SCRIPT_NAME v$VERSION"
    log_info "专注于 NCCL 分布式通信测试"
    log_info "日志文件: $LOG_FILE"
    log ""
    
    # 仅显示环境配置
    if [ "$ENV_ONLY" = true ]; then
        if check_nccl_dependencies; then
            setup_nccl_env
            log_success "环境配置完成"
        else
            log_error "环境检查失败"
            exit 1
        fi
        exit 0
    fi
    
    # 仅检查环境
    if [ "$CHECK_ONLY" = true ]; then
        if check_nccl_dependencies; then
            log_success "环境检查通过"
        else
            log_error "环境检查失败"
            exit 1
        fi
        exit 0
    fi
    
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