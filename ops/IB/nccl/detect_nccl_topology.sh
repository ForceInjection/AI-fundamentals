#!/bin/bash

# NCCL 通信路径检测脚本 v2.0.0
# 按照NCCL实际优先级进行检测：NVLink > PCIe P2P > SHM > NET
# 修复版本：解决NVLink和PCIe P2P检测问题

SCRIPT_NAME="NCCL 通信路径检测"
VERSION="2.0.0"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "[INFO] $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

# 检查必要命令的可用性
check_dependencies() {
    local deps_ok=true
    
    # 检查 nvidia-smi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "nvidia-smi 命令不可用，无法检测 GPU 信息"
        deps_ok=false
    fi
    
    # 检查 lspci
    if ! command -v lspci >/dev/null 2>&1; then
        log_warning "lspci 命令不可用，PCIe 信息检测可能受限"
    fi
    
    # 检查 ibstat (可选)
    if ! command -v ibstat >/dev/null 2>&1; then
        log_warning "ibstat 命令不可用，InfiniBand 信息检测将跳过"
    fi
    
    if [ "$deps_ok" = false ]; then
        log_error "关键依赖检查失败，无法继续"
        exit 1
    fi
    
    log_success "依赖检查通过"
}

# 改进的NVLink检测函数
check_nvlink_status() {
    local nvlink_available=false
    local topo_output
    
    # 获取GPU拓扑信息
    topo_output=$(nvidia-smi topo -m 2>/dev/null || echo "")
    
    if [ -n "$topo_output" ]; then
        # 检查拓扑矩阵中的NVLink连接（NV开头的连接）
        if echo "$topo_output" | grep -qE "NV[0-9]+"; then
            nvlink_available=true
            local nvlink_connections
            nvlink_connections=$(echo "$topo_output" | grep -oE "NV[0-9]+" | sort -u)
            log_success "NVLink可用：从GPU拓扑检测到NVLink连接"
            log_info "NVLink连接类型: $(echo "$nvlink_connections" | tr '\n' ' ')"
            
            # 计算NVLink连接数量
            local total_links=0
            for conn in $nvlink_connections; do
                local link_num
                link_num=$(echo "$conn" | grep -oE "[0-9]+")
                if [ -n "$link_num" ]; then
                    total_links=$((total_links + link_num))
                fi
            done
            log_info "总NVLink连接数: $total_links"
            
            # 尝试获取详细的NVLink状态
            if command -v nvidia-smi >/dev/null 2>&1; then
                log_info "尝试获取NVLink详细状态..."
                if nvidia-smi nvlink -s 2>/dev/null; then
                    log_info "NVLink状态获取成功"
                else
                    log_warning "无法获取详细NVLink状态，但拓扑显示NVLink可用"
                fi
                
                # 尝试获取NVLink能力信息
                if nvidia-smi nvlink -c 2>/dev/null; then
                    log_info "NVLink能力信息获取成功"
                else
                    log_warning "无法获取NVLink能力信息"
                fi
            fi
        else
            log_info "GPU拓扑中未检测到NVLink连接"
        fi
    else
        log_warning "无法获取GPU拓扑信息"
    fi
    
    # 使用退出状态返回结果
    if [ "$nvlink_available" = true ]; then
        return 0  # 成功
    else
        return 1  # 失败
    fi
}

# 改进的PCIe P2P检测函数
check_pcie_p2p() {
    local pcie_p2p=false
    local topo_output
    
    # 获取GPU拓扑信息
    topo_output=$(nvidia-smi topo -m 2>/dev/null || echo "")
    
    if [ -n "$topo_output" ]; then
        log_info "分析GPU拓扑中的PCIe连接类型..."
        
        # 检查各种可能的P2P连接类型
        if echo "$topo_output" | grep -qE "(PXB|PIX|PHB)"; then
            pcie_p2p=true
            log_success "PCIe P2P可用：检测到GPU间PCIe点对点通信支持"
            
            # 分析连接类型
            local pxb_count pix_count phb_count
            pxb_count=$(echo "$topo_output" | grep -o "PXB" | wc -l | tr -d ' ' || echo "0")
            pix_count=$(echo "$topo_output" | grep -o "PIX" | wc -l | tr -d ' ' || echo "0")
            phb_count=$(echo "$topo_output" | grep -o "PHB" | wc -l | tr -d ' ' || echo "0")
            
            log_info "P2P连接统计: PXB($pxb_count) PIX($pix_count) PHB($phb_count)"
            
            # 详细分析连接质量
            if [ "$pix_count" -gt 0 ]; then
                log_info "检测到PIX连接：单PCIe桥接，性能良好"
            fi
            if [ "$pxb_count" -gt 0 ]; then
                log_info "检测到PXB连接：多PCIe桥接，性能中等"
            fi
            if [ "$phb_count" -gt 0 ]; then
                log_info "检测到PHB连接：通过PCIe主桥，性能一般"
            fi
            
        elif echo "$topo_output" | grep -qE "(NODE|SYS)"; then
            # NODE和SYS连接可能支持有限的P2P
            local node_count sys_count
            node_count=$(echo "$topo_output" | grep -o "NODE" | wc -l | tr -d ' ' || echo "0")
            sys_count=$(echo "$topo_output" | grep -o "SYS" | wc -l | tr -d ' ' || echo "0")
            
            log_warning "PCIe P2P受限：仅支持跨NUMA节点的有限P2P通信"
            log_info "连接统计: NODE($node_count) SYS($sys_count)"
            log_info "NODE: NUMA节点内PCIe连接，性能受限"
            log_info "SYS: 跨NUMA节点连接，性能较差"
            pcie_p2p=false  # 保守判断
        else
            log_warning "PCIe P2P不可用或受限"
        fi
        
        # 显示PCIe链路详细信息
        if command -v lspci >/dev/null 2>&1; then
            log_info "PCIe链路详细信息："
            for gpu in $(lspci | grep -i nvidia | grep -v Audio | cut -d' ' -f1 2>/dev/null || true); do
                if [ -n "$gpu" ]; then
                    echo "GPU $gpu:"
                    lspci -vvv -s "$gpu" 2>/dev/null | grep -E "(LnkCap|LnkSta)" | head -2 || true
                fi
            done
        else
            log_warning "lspci 不可用，跳过详细PCIe链路检查"
        fi
    else
        log_warning "无法获取GPU拓扑信息"
    fi
    
    # 使用退出状态返回结果
    if [ "$pcie_p2p" = true ]; then
        return 0  # 成功
    else
        return 1  # 失败
    fi
}

# 错误处理函数
cleanup() {
    log_warning "检测过程被中断，正在清理..."
    exit 130
}

# 显示帮助信息
show_help() {
    cat << EOF
$SCRIPT_NAME v$VERSION

用途：
    检测和分析NCCL通信路径，按照NCCL实际优先级进行检测和建议配置

语法：
    $0 [选项]

选项：
    --no-test           跳过NCCL实际测试，仅进行硬件检测
    -h, --help          显示此帮助信息
    -v, --version       显示版本信息

检测优先级：
    1. NVLink          GPU间专用高速互联（最优）
    2. PCIe P2P        通过PCIe总线点对点通信（良好）
    3. 共享内存        通过CPU内存中转（基础）
    4. 网络传输        主要用于跨节点通信（特殊用途）

示例：
    $0                  # 完整检测（包含60秒NCCL测试）
    $0 --no-test        # 仅硬件检测，跳过NCCL测试
    $0 --help           # 显示帮助信息

输出说明：
    脚本会按照NCCL的实际优先级顺序检测各种通信路径，
    并给出最优的配置建议和环境变量设置。

注意事项：
    - 需要NVIDIA GPU和相应驱动
    - 建议以root权限运行以获取完整信息
    - NCCL测试需要60秒（初始化10s + 预热15s + 稳定性测试35s）
    - 建议在GPU空闲时运行以获得准确结果

EOF
}

# 显示版本信息
show_version() {
    echo "$SCRIPT_NAME v$VERSION"
    echo "NCCL通信路径检测和优化建议工具"
}

# 参数处理
parse_arguments() {
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                show_version
                exit 0
                ;;
            --no-test)
                NO_TEST=true
                ;;
            *)
                log_error "未知参数: $1"
                echo "使用 '$0 --help' 查看帮助信息"
                exit 1
                ;;
        esac
        shift
    done
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 主执行逻辑
main() {
    log_header "$SCRIPT_NAME v$VERSION"
    log_info "按照 NCCL 实际优先级进行检测：NVLink > PCIe P2P > SHM > NET"
    
    # 检查依赖
    check_dependencies
    
    # 1. 检查GPU基本信息
    log_header "GPU基本信息"
    if nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null; then
        log_success "GPU 信息获取成功"
    else
        log_error "无法获取 GPU 信息"
        exit 1
    fi
    
    # 2. 检查GPU拓扑结构
    log_header "GPU拓扑结构"
    if nvidia-smi topo -m 2>/dev/null; then
        log_success "GPU 拓扑信息获取成功"
    else
        log_warning "无法获取 GPU 拓扑信息"
    fi
    
    # 3. 检查NVLink详细状态（NCCL最高优先级）
    log_header "NVLink连接详情（优先级1 - 最高）"
    check_nvlink_status
    nvlink_available=$?
    
    # 4. 检查PCIe P2P支持（NCCL第二优先级）
    log_header "PCIe P2P支持检查（优先级2）"
    check_pcie_p2p
    pcie_p2p=$?
    
    # 5. 共享内存检查（NCCL第三优先级）
    log_header "共享内存支持检查（优先级3）"
    shm_available=true  # 共享内存通常总是可用的
    log_success "共享内存可用：系统默认支持进程间共享内存通信"
    
    # 6. InfiniBand设备检查（NCCL第四优先级 - 网络传输）
    log_header "InfiniBand设备状态（优先级4 - 网络传输）"
    ib_available=false
    if command -v ibstat >/dev/null 2>&1; then
        if ibstat 2>/dev/null | grep -E "(CA|Port|State|Rate)"; then
            if ibstat 2>/dev/null | grep -q "Active"; then
                ib_available=true
                log_success "InfiniBand可用：检测到活跃的IB设备"
            else
                log_warning "InfiniBand设备存在但状态非活跃"
            fi
        else
            log_info "未检测到 InfiniBand 设备"
        fi
    else
        log_info "未安装InfiniBand工具或无IB设备"
    fi
    
    # 7. NCCL通信路径实际测试
    log_header "NCCL通信路径验证"
    if [ "$NO_TEST" = true ]; then
        log_info "跳过NCCL实际测试（--no-test 参数）"
    elif [ -f "./nccl_test.sh" ]; then
        log_info "运行NCCL调试测试（60秒）..."
        log_info "测试包含：初始化(10s) + 预热(15s) + 稳定性测试(35s)"
        export NCCL_DEBUG=INFO
        if timeout 90s ./nccl_test.sh --network auto -s 100M -t 60 2>&1 | \
           grep -E "(Channel|via|Ring|Tree|Using|bandwidth|latency)" | head -15; then
            log_success "NCCL 路径测试完成"
        else
            log_warning "NCCL 路径测试可能未完全成功"
        fi
        unset NCCL_DEBUG
    else
        log_warning "未找到nccl_test.sh脚本，跳过实际路径测试"
    fi
    
    # 8. 按NCCL优先级进行通信路径分析和建议
    log_header "NCCL通信路径优先级分析"
    log_info "NCCL自动选择优先级：NVLink > PCIe P2P > 共享内存 > 网络传输"
    
    # 按优先级给出建议
    log_header "推荐配置（按NCCL优先级）"
    
    local optimal_path="未知"
    local performance_level="未知"
    
    if [ "$nvlink_available" -eq 0 ]; then
        optimal_path="NVLink"
        performance_level="最高"
        log_success "🚀 优先级1 - NVLink直连（最优）："
        log_info "   配置：--network auto（NCCL将自动选择NVLink）"
        log_info "   预期性能：延迟 < 10μs，带宽 > 200 GB/s"
        log_info "   技术特点：GPU间直接高速互联，零拷贝传输"
        log_info "   环境变量：export NCCL_P2P_DISABLE=0; export NCCL_NVLS_ENABLE=1"
        primary_choice="NVLink"
    elif [ "$pcie_p2p" -eq 0 ]; then
        optimal_path="PCIe P2P"
        performance_level="高"
        log_success "⚡ 优先级2 - PCIe P2P（良好）："
        log_info "   配置：--network auto（NCCL将选择PCIe P2P）"
        log_info "   预期性能：延迟 30-100μs，带宽取决于PCIe版本"
        log_info "   技术特点：通过PCIe总线进行GPU间点对点通信"
        log_info "   环境变量：export NCCL_P2P_DISABLE=0; export NCCL_SHM_DISABLE=1"
        primary_choice="PCIe P2P"
    elif [ "$shm_available" = true ]; then
        optimal_path="共享内存"
        performance_level="中等"
        log_success "📝 优先级3 - 共享内存（基础）："
        log_info "   配置：--network auto（NCCL将使用共享内存）"
        log_info "   预期性能：延迟较高，带宽受限于CPU内存"
        log_info "   技术特点：通过系统内存进行数据交换"
        log_info "   环境变量：export NCCL_P2P_DISABLE=1; export NCCL_SHM_DISABLE=0"
        primary_choice="共享内存"
    else
        log_warning "⚠️  无可用的GPU间通信方式"
        log_warning "   建议检查硬件配置和驱动程序"
        primary_choice="无"
    fi
    
    # 网络传输作为补充说明
    if [ "$ib_available" = true ]; then
        log_header "网络传输补充说明"
        log_info "📡 优先级4 - 网络传输（特殊用途）："
        log_info "   配置：--network ib（强制使用IB网络）"
        log_info "   用途：测试IB网卡loopback性能，非GPU间直连"
        log_info "   注意：单节点中通常不会自动选择网络传输"
    fi
    
    # 技术说明和最佳实践
    log_header "技术说明"
    log_info "• NCCL优先级说明："
    log_info "  - NVLink: GPU间专用高速互联，性能最优"
    log_info "  - PCIe P2P: 通过PCIe总线直接通信，性能良好"
    log_info "  - 共享内存: 通过CPU内存中转，性能一般"
    log_info "  - 网络传输: 主要用于跨节点通信，单节点中为特殊测试"
    
    log_success "• 当前系统最优选择：$optimal_path ($performance_level 性能)"
    
    # 计算可用路径数量
    local available_count=0
    [ "$nvlink_available" -eq 0 ] && available_count=$((available_count + 1))
    [ "$pcie_p2p" -eq 0 ] && available_count=$((available_count + 1))
    [ "$shm_available" = true ] && available_count=$((available_count + 1))
    [ "$ib_available" = true ] && available_count=$((available_count + 1))
    
    log_info "• 可用通信路径数量：$available_count"
    
    log_header "环境变量配置建议"
    log_info "• 推荐配置（让NCCL自动选择）："
    log_info "  export NCCL_DEBUG=INFO  # 查看选择的通信路径"
    log_info "  ./nccl_test.sh --network auto"
    
    log_info "• 强制配置（仅用于调试）："
    if [ "$nvlink_available" -eq 0 ]; then
        log_info "  export NCCL_P2P_LEVEL=NVL  # 强制使用NVLink"
    fi
    if [ "$ib_available" = true ]; then
        log_info "  export NCCL_IB_DISABLE=0   # 强制启用IB（测试用）"
    fi
    log_info "  export NCCL_P2P_DISABLE=1     # 禁用P2P（调试用）"
    
    log_success "检测完成！建议使用 --network auto 让NCCL自动选择最优通信路径。"
}

# 解析命令行参数
parse_arguments "$@"

# 执行主函数
main