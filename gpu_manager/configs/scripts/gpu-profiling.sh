#!/bin/bash
# GPU性能分析脚本
# 提供全面的GPU性能分析和调优工具

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置变量
PROFILE_DURATION=60
OUTPUT_DIR="gpu_profiling_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_DIR/profiling.log"
REPORT_FILE="$OUTPUT_DIR/profiling_report.md"
SAMPLE_INTERVAL=1
PROFILE_TYPE="all"
TARGET_PID=""
TARGET_COMMAND=""

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo -e "${PURPLE}[SECTION]${NC} $1" | tee -a "$LOG_FILE"
}

# 创建输出目录
setup_output_directory() {
    mkdir -p "$OUTPUT_DIR"
    log_info "性能分析结果将保存到: $OUTPUT_DIR"
}

# 检查依赖工具
check_dependencies() {
    log_info "检查依赖工具..."
    
    local missing_tools=()
    
    # 必需工具
    if ! command -v nvidia-smi &> /dev/null; then
        missing_tools+=("nvidia-smi")
    fi
    
    # 可选工具
    local optional_tools=("nvtop" "gpustat" "nvprof" "nsys" "ncu" "nvidia-ml-py")
    local available_tools=()
    
    for tool in "${optional_tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            available_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "缺少必需工具: ${missing_tools[*]}"
        return 1
    fi
    
    log_info "可用的性能分析工具: ${available_tools[*]}"
    
    # 检查Python工具
    if command -v python3 &> /dev/null; then
        python3 -c "import pynvml" 2>/dev/null && log_info "pynvml可用" || log_warn "pynvml不可用"
        python3 -c "import GPUtil" 2>/dev/null && log_info "GPUtil可用" || log_warn "GPUtil不可用"
    fi
    
    log_info "✅ 依赖检查完成"
}

# GPU基础信息收集
collect_gpu_info() {
    log_section "收集GPU基础信息"
    
    # GPU设备信息
    log_info "GPU设备信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.gr,clocks.mem,compute_cap --format=csv > "$OUTPUT_DIR/gpu_info.csv"
    
    # 显示摘要
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader | while IFS=',' read -r index name memory compute_cap; do
        log_info "GPU $index: $name, 内存: $memory, 计算能力: $compute_cap"
    done
    
    # 驱动和CUDA信息
    local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    log_info "NVIDIA驱动版本: $driver_version"
    
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "CUDA版本: $cuda_version"
    fi
    
    # GPU拓扑信息
    if nvidia-smi topo -m &> /dev/null; then
        log_info "GPU拓扑信息:"
        nvidia-smi topo -m > "$OUTPUT_DIR/gpu_topology.txt" 2>/dev/null || log_warn "无法获取GPU拓扑信息"
    fi
    
    log_info "✅ GPU基础信息收集完成"
}

# 实时GPU监控
real_time_monitoring() {
    log_section "开始实时GPU监控 (${PROFILE_DURATION}秒)"
    
    # 创建监控脚本
    cat > "$OUTPUT_DIR/monitor_gpu.py" << 'EOF'
import time
import csv
import subprocess
import sys
from datetime import datetime

def get_gpu_stats():
    """获取GPU统计信息"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.gr,clocks.mem',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        return []

def get_process_stats():
    """获取GPU进程信息"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-compute-apps=pid,process_name,gpu_uuid,used_memory',
            '--format=csv,noheader'
        ], capture_output=True, text=True, check=True)
        
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []

def monitor_gpu(duration, interval, output_file):
    """监控GPU性能"""
    print(f"开始监控GPU性能，持续时间: {duration}秒，采样间隔: {interval}秒")
    
    # 准备CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'timestamp', 'gpu_index', 'gpu_name', 'gpu_util_percent', 
            'memory_util_percent', 'memory_total_mb', 'memory_used_mb', 
            'memory_free_mb', 'temperature_c', 'power_draw_w', 
            'graphics_clock_mhz', 'memory_clock_mhz'
        ])
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration:
            gpu_stats = get_gpu_stats()
            
            for stat_line in gpu_stats:
                if stat_line.strip():
                    # 解析统计信息
                    parts = [part.strip() for part in stat_line.split(',')]
                    if len(parts) >= 12:
                        writer.writerow(parts)
            
            sample_count += 1
            if sample_count % 10 == 0:
                print(f"已采样 {sample_count} 次...")
            
            time.sleep(interval)
    
    print(f"监控完成，共采样 {sample_count} 次")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python3 monitor_gpu.py <duration> <interval> <output_file>")
        sys.exit(1)
    
    duration = int(sys.argv[1])
    interval = float(sys.argv[2])
    output_file = sys.argv[3]
    
    monitor_gpu(duration, interval, output_file)
EOF
    
    # 运行监控
    if command -v python3 &> /dev/null; then
        log_info "启动Python监控脚本..."
        python3 "$OUTPUT_DIR/monitor_gpu.py" "$PROFILE_DURATION" "$SAMPLE_INTERVAL" "$OUTPUT_DIR/gpu_monitoring.csv" &
        MONITOR_PID=$!
    fi
    
    # 同时使用nvidia-smi进行监控
    log_info "启动nvidia-smi监控..."
    {
        echo "timestamp,gpu_util,memory_util,temperature,power_draw,graphics_clock,memory_clock"
        for ((i=0; i<PROFILE_DURATION; i++)); do
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            local stats=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.gr,clocks.mem --format=csv,noheader,nounits | head -1)
            echo "$timestamp,$stats"
            sleep 1
        done
    } > "$OUTPUT_DIR/nvidia_smi_monitoring.csv" &
    NVIDIA_SMI_PID=$!
    
    # 如果有nvtop，也启动它
    if command -v nvtop &> /dev/null; then
        log_info "启动nvtop监控..."
        timeout "${PROFILE_DURATION}s" nvtop -d 1 > "$OUTPUT_DIR/nvtop_output.txt" 2>&1 &
        NVTOP_PID=$!
    fi
    
    # 等待监控完成
    log_info "监控进行中，请等待 ${PROFILE_DURATION} 秒..."
    
    # 显示进度条
    for ((i=1; i<=PROFILE_DURATION; i++)); do
        printf "\r进度: [%-50s] %d%%" $(printf "#%.0s" $(seq 1 $((i*50/PROFILE_DURATION)))) $((i*100/PROFILE_DURATION))
        sleep 1
    done
    echo
    
    # 等待所有监控进程完成
    wait $NVIDIA_SMI_PID 2>/dev/null || true
    [[ -n "$MONITOR_PID" ]] && wait $MONITOR_PID 2>/dev/null || true
    [[ -n "$NVTOP_PID" ]] && kill $NVTOP_PID 2>/dev/null || true
    
    log_info "✅ 实时监控完成"
}

# GPU进程分析
analyze_gpu_processes() {
    log_section "分析GPU进程"
    
    # 当前GPU进程
    log_info "当前GPU进程:"
    nvidia-smi pmon -c 1 > "$OUTPUT_DIR/gpu_processes.txt" 2>/dev/null || {
        nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv > "$OUTPUT_DIR/gpu_processes.csv"
        log_info "GPU进程信息已保存到 gpu_processes.csv"
    }
    
    # 详细进程信息
    if nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null | grep -v "No running processes found"; then
        log_info "检测到运行中的GPU进程:"
        nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader | while IFS=',' read -r pid process_name gpu_uuid used_memory; do
            if [[ -n "$pid" && "$pid" != "No running processes found" ]]; then
                log_info "PID: $pid, 进程: $process_name, 内存使用: $used_memory"
                
                # 获取进程详细信息
                if ps -p "$pid" -o pid,ppid,user,cmd --no-headers 2>/dev/null; then
                    ps -p "$pid" -o pid,ppid,user,cmd --no-headers >> "$OUTPUT_DIR/process_details.txt"
                fi
            fi
        done
    else
        log_info "当前没有运行中的GPU进程"
    fi
    
    log_info "✅ GPU进程分析完成"
}

# 内存分析
analyze_gpu_memory() {
    log_section "GPU内存分析"
    
    # 内存使用情况
    log_info "GPU内存使用情况:"
    nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv > "$OUTPUT_DIR/memory_usage.csv"
    
    # 显示内存摘要
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader | while IFS=',' read -r index name total used free; do
        local used_percent=$(echo "scale=1; $used * 100 / $total" | bc 2>/dev/null || echo "N/A")
        log_info "GPU $index ($name): 总内存 $total, 已用 $used ($used_percent%), 空闲 $free"
    done
    
    # 内存碎片分析
    if command -v python3 &> /dev/null; then
        cat > "$OUTPUT_DIR/memory_analysis.py" << 'EOF'
import subprocess
import json

def analyze_memory_fragmentation():
    """分析GPU内存碎片"""
    try:
        # 获取详细内存信息
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        memory_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    index, total, used, free = parts
                    memory_info.append({
                        'gpu_index': int(index),
                        'total_mb': int(total),
                        'used_mb': int(used),
                        'free_mb': int(free),
                        'utilization_percent': round(int(used) * 100 / int(total), 2)
                    })
        
        # 保存分析结果
        with open('memory_analysis.json', 'w') as f:
            json.dump(memory_info, f, indent=2)
        
        print("GPU内存分析:")
        for info in memory_info:
            print(f"GPU {info['gpu_index']}: {info['utilization_percent']}% 使用率")
            if info['utilization_percent'] > 90:
                print(f"  警告: GPU {info['gpu_index']} 内存使用率过高")
            elif info['utilization_percent'] < 10:
                print(f"  提示: GPU {info['gpu_index']} 内存利用率较低")
    
    except Exception as e:
        print(f"内存分析失败: {e}")

if __name__ == "__main__":
    analyze_memory_fragmentation()
EOF
        
        cd "$OUTPUT_DIR"
        python3 memory_analysis.py
        cd - > /dev/null
    fi
    
    log_info "✅ GPU内存分析完成"
}

# 性能瓶颈分析
analyze_performance_bottlenecks() {
    log_section "性能瓶颈分析"
    
    # 分析监控数据
    if [[ -f "$OUTPUT_DIR/gpu_monitoring.csv" ]]; then
        log_info "分析GPU利用率数据..."
        
        # 创建分析脚本
        cat > "$OUTPUT_DIR/bottleneck_analysis.py" << 'EOF'
import csv
import statistics
import json

def analyze_gpu_utilization(csv_file):
    """分析GPU利用率数据"""
    gpu_utils = []
    memory_utils = []
    temperatures = []
    power_draws = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gpu_util = float(row['gpu_util_percent'])
                    memory_util = float(row['memory_util_percent'])
                    temp = float(row['temperature_c'])
                    power = float(row['power_draw_w'])
                    
                    gpu_utils.append(gpu_util)
                    memory_utils.append(memory_util)
                    temperatures.append(temp)
                    power_draws.append(power)
                except (ValueError, KeyError):
                    continue
        
        if not gpu_utils:
            print("没有有效的监控数据")
            return
        
        # 计算统计信息
        analysis = {
            'gpu_utilization': {
                'mean': round(statistics.mean(gpu_utils), 2),
                'median': round(statistics.median(gpu_utils), 2),
                'max': round(max(gpu_utils), 2),
                'min': round(min(gpu_utils), 2),
                'stdev': round(statistics.stdev(gpu_utils) if len(gpu_utils) > 1 else 0, 2)
            },
            'memory_utilization': {
                'mean': round(statistics.mean(memory_utils), 2),
                'median': round(statistics.median(memory_utils), 2),
                'max': round(max(memory_utils), 2),
                'min': round(min(memory_utils), 2),
                'stdev': round(statistics.stdev(memory_utils) if len(memory_utils) > 1 else 0, 2)
            },
            'temperature': {
                'mean': round(statistics.mean(temperatures), 2),
                'max': round(max(temperatures), 2),
                'min': round(min(temperatures), 2)
            },
            'power_draw': {
                'mean': round(statistics.mean(power_draws), 2),
                'max': round(max(power_draws), 2),
                'min': round(min(power_draws), 2)
            }
        }
        
        # 瓶颈分析
        bottlenecks = []
        recommendations = []
        
        # GPU利用率分析
        if analysis['gpu_utilization']['mean'] < 50:
            bottlenecks.append("GPU利用率低")
            recommendations.append("考虑增加批次大小或优化数据加载")
        elif analysis['gpu_utilization']['mean'] > 95:
            bottlenecks.append("GPU利用率过高")
            recommendations.append("可能需要更强的GPU或优化算法")
        
        # 内存利用率分析
        if analysis['memory_utilization']['mean'] > 90:
            bottlenecks.append("GPU内存使用率过高")
            recommendations.append("考虑减少批次大小或使用梯度累积")
        elif analysis['memory_utilization']['mean'] < 30:
            bottlenecks.append("GPU内存利用率低")
            recommendations.append("可以增加批次大小以提高效率")
        
        # 温度分析
        if analysis['temperature']['max'] > 80:
            bottlenecks.append("GPU温度过高")
            recommendations.append("检查散热系统，考虑降低功耗限制")
        
        # 功耗分析
        if analysis['power_draw']['mean'] > 250:  # 假设功耗限制
            bottlenecks.append("功耗较高")
            recommendations.append("考虑优化算法或调整功耗限制")
        
        analysis['bottlenecks'] = bottlenecks
        analysis['recommendations'] = recommendations
        
        # 保存分析结果
        with open('bottleneck_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # 打印结果
        print("=== GPU性能瓶颈分析 ===")
        print(f"GPU平均利用率: {analysis['gpu_utilization']['mean']}%")
        print(f"内存平均利用率: {analysis['memory_utilization']['mean']}%")
        print(f"平均温度: {analysis['temperature']['mean']}°C")
        print(f"平均功耗: {analysis['power_draw']['mean']}W")
        
        if bottlenecks:
            print("\n检测到的瓶颈:")
            for bottleneck in bottlenecks:
                print(f"- {bottleneck}")
        
        if recommendations:
            print("\n优化建议:")
            for rec in recommendations:
                print(f"- {rec}")
    
    except Exception as e:
        print(f"分析失败: {e}")

if __name__ == "__main__":
    analyze_gpu_utilization('gpu_monitoring.csv')
EOF
        
        cd "$OUTPUT_DIR"
        python3 bottleneck_analysis.py
        cd - > /dev/null
    fi
    
    log_info "✅ 性能瓶颈分析完成"
}

# CUDA分析（如果可用）
cuda_profiling() {
    if [[ -z "$TARGET_COMMAND" ]]; then
        log_warn "跳过CUDA分析（未指定目标命令）"
        return 0
    fi
    
    log_section "CUDA性能分析"
    
    # 使用nsys进行分析
    if command -v nsys &> /dev/null; then
        log_info "使用Nsight Systems进行分析..."
        nsys profile -o "$OUTPUT_DIR/nsys_profile" --stats=true $TARGET_COMMAND > "$OUTPUT_DIR/nsys_output.txt" 2>&1 || {
            log_warn "Nsight Systems分析失败"
        }
    fi
    
    # 使用ncu进行分析
    if command -v ncu &> /dev/null; then
        log_info "使用Nsight Compute进行分析..."
        ncu --set full -o "$OUTPUT_DIR/ncu_profile" $TARGET_COMMAND > "$OUTPUT_DIR/ncu_output.txt" 2>&1 || {
            log_warn "Nsight Compute分析失败"
        }
    fi
    
    # 使用nvprof进行分析（如果可用）
    if command -v nvprof &> /dev/null; then
        log_info "使用nvprof进行分析..."
        nvprof --log-file "$OUTPUT_DIR/nvprof_output.txt" --print-gpu-trace $TARGET_COMMAND || {
            log_warn "nvprof分析失败"
        }
    fi
    
    log_info "✅ CUDA性能分析完成"
}

# 生成性能分析报告
generate_profiling_report() {
    log_section "生成性能分析报告"
    
    cat > "$REPORT_FILE" << EOF
# GPU性能分析报告

## 分析概览
- 分析时间: $(date)
- 分析持续时间: ${PROFILE_DURATION}秒
- 采样间隔: ${SAMPLE_INTERVAL}秒
- 分析类型: $PROFILE_TYPE

## 系统环境
- 操作系统: $(uname -a)
- NVIDIA驱动版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
EOF
    
    if command -v nvcc &> /dev/null; then
        echo "- CUDA版本: $(nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-)" >> "$REPORT_FILE"
    fi
    
    cat >> "$REPORT_FILE" << EOF

## GPU设备信息
\`\`\`
$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv)
\`\`\`

## 性能监控结果

EOF
    
    # 添加瓶颈分析结果
    if [[ -f "$OUTPUT_DIR/bottleneck_analysis.json" ]]; then
        echo "### 性能瓶颈分析" >> "$REPORT_FILE"
        echo '```json' >> "$REPORT_FILE"
        cat "$OUTPUT_DIR/bottleneck_analysis.json" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    # 添加内存分析结果
    if [[ -f "$OUTPUT_DIR/memory_analysis.json" ]]; then
        echo "### 内存使用分析" >> "$REPORT_FILE"
        echo '```json' >> "$REPORT_FILE"
        cat "$OUTPUT_DIR/memory_analysis.json" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    # 添加文件列表
    echo "## 生成的文件" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    find "$OUTPUT_DIR" -type f -name "*.csv" -o -name "*.json" -o -name "*.txt" | while read -r file; do
        local filename=$(basename "$file")
        local filesize=$(du -h "$file" | cut -f1)
        echo "- \`$filename\` ($filesize)" >> "$REPORT_FILE"
    done
    
    echo "" >> "$REPORT_FILE"
    echo "## 优化建议" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "基于分析结果，建议关注以下方面：" >> "$REPORT_FILE"
    echo "1. GPU利用率优化" >> "$REPORT_FILE"
    echo "2. 内存使用优化" >> "$REPORT_FILE"
    echo "3. 温度和功耗管理" >> "$REPORT_FILE"
    echo "4. 数据传输优化" >> "$REPORT_FILE"
    
    log_info "✅ 性能分析报告生成完成: $REPORT_FILE"
}

# 清理临时文件
cleanup_temp_files() {
    log_info "清理临时文件..."
    
    # 保留重要结果文件，删除临时文件
    find "$OUTPUT_DIR" -name "*.py" -delete 2>/dev/null || true
    
    log_info "✅ 清理完成"
}

# 显示帮助信息
show_help() {
    echo "GPU性能分析脚本"
    echo "用法: $0 [选项] [分析类型]"
    echo ""
    echo "分析类型:"
    echo "  basic      基础性能分析"
    echo "  monitor    实时监控分析"
    echo "  process    进程分析"
    echo "  memory     内存分析"
    echo "  bottleneck 瓶颈分析"
    echo "  cuda       CUDA分析（需要目标命令）"
    echo "  all        所有分析（默认）"
    echo ""
    echo "选项:"
    echo "  --duration SECONDS     分析持续时间（默认: 60秒）"
    echo "  --interval SECONDS     采样间隔（默认: 1秒）"
    echo "  --output-dir DIR       输出目录（默认: gpu_profiling_TIMESTAMP）"
    echo "  --target-pid PID       目标进程PID"
    echo "  --target-command CMD   目标命令（用于CUDA分析）"
    echo "  --help                显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 运行所有分析"
    echo "  $0 monitor                            # 只运行监控分析"
    echo "  $0 --duration 120 bottleneck          # 运行120秒瓶颈分析"
    echo "  $0 --target-command 'python train.py' cuda  # CUDA分析"
    echo "  $0 --target-pid 1234 process          # 分析特定进程"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --duration)
                PROFILE_DURATION="$2"
                shift 2
                ;;
            --interval)
                SAMPLE_INTERVAL="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                LOG_FILE="$OUTPUT_DIR/profiling.log"
                REPORT_FILE="$OUTPUT_DIR/profiling_report.md"
                shift 2
                ;;
            --target-pid)
                TARGET_PID="$2"
                shift 2
                ;;
            --target-command)
                TARGET_COMMAND="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            basic|monitor|process|memory|bottleneck|cuda|all)
                PROFILE_TYPE="$1"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    local profile_type=${PROFILE_TYPE:-"all"}
    
    # 设置输出目录
    setup_output_directory
    
    # 检查依赖
    check_dependencies
    
    # 收集基础信息
    collect_gpu_info
    
    # 根据分析类型运行相应分析
    case $profile_type in
        "basic")
            collect_gpu_info
            ;;
        "monitor")
            real_time_monitoring
            ;;
        "process")
            analyze_gpu_processes
            ;;
        "memory")
            analyze_gpu_memory
            ;;
        "bottleneck")
            real_time_monitoring
            analyze_performance_bottlenecks
            ;;
        "cuda")
            cuda_profiling
            ;;
        "all")
            real_time_monitoring
            analyze_gpu_processes
            analyze_gpu_memory
            analyze_performance_bottlenecks
            if [[ -n "$TARGET_COMMAND" ]]; then
                cuda_profiling
            fi
            ;;
        *)
            log_error "未知分析类型: $profile_type"
            show_help
            exit 1
            ;;
    esac
    
    # 生成报告
    generate_profiling_report
    
    # 清理临时文件
    cleanup_temp_files
    
    log_info "🎉 GPU性能分析完成！"
    log_info "分析结果保存在: $OUTPUT_DIR"
    log_info "详细报告: $REPORT_FILE"
    
    # 显示快速摘要
    if [[ -f "$OUTPUT_DIR/bottleneck_analysis.json" ]]; then
        log_info "\n📊 快速摘要:"
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/bottleneck_analysis.json', 'r') as f:
        data = json.load(f)
    print(f'GPU平均利用率: {data[\"gpu_utilization\"][\"mean\"]}%')
    print(f'内存平均利用率: {data[\"memory_utilization\"][\"mean\"]}%')
    if data.get('bottlenecks'):
        print('检测到瓶颈:', ', '.join(data['bottlenecks']))
except: pass
" 2>/dev/null || true
    fi
}

# 解析参数并运行主函数
parse_arguments "$@"
main