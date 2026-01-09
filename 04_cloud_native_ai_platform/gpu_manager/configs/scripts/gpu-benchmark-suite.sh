#!/bin/bash

# GPU 基准测试套件
# 提供全面的 GPU 性能基准测试，包括计算、内存、带宽等测试

set -e

# 配置参数
BENCHMARK_TYPE="${1:-all}"  # all|compute|memory|bandwidth|tensor
OUTPUT_DIR="${2:-./benchmark-results}"
ITERATIONS="${3:-3}"

# 日志函数
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 检查依赖
check_dependencies() {
    local deps=("nvidia-smi" "python3" "bc")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log "错误: 缺少依赖 $dep"
            exit 1
        fi
    done
    
    # 检查 CUDA 工具包
    if ! command -v "nvcc" &> /dev/null; then
        log "警告: 未找到 nvcc，部分测试可能需要 CUDA 工具包"
    fi
}

# 获取 GPU 信息
get_gpu_info() {
    log "获取 GPU 硬件信息..."
    
    local gpu_info=$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_capability --format=csv,noheader,nounits)
    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
    
    echo "=== GPU 硬件信息 ===" > "$OUTPUT_DIR/gpu-info.txt"
    echo "GPU 数量: $gpu_count" >> "$OUTPUT_DIR/gpu-info.txt"
    echo "详细信息:" >> "$OUTPUT_DIR/gpu-info.txt"
    echo "$gpu_info" >> "$OUTPUT_DIR/gpu-info.txt"
    
    # 显示信息
    cat "$OUTPUT_DIR/gpu-info.txt"
}

# 计算性能测试
run_compute_benchmark() {
    log "运行计算性能测试..."
    
    local results_file="$OUTPUT_DIR/compute-benchmark.csv"
    echo "Test,Iteration,Time(s),GFLOPS" > "$results_file"
    
    # 简单的矩阵乘法测试（使用 Python 和 NumPy）
    local python_script="
import numpy as np
import time

def matrix_multiply_benchmark():
    # 测试不同矩阵大小
    sizes = [512, 1024, 2048]
    results = []
    
    for size in sizes:
        # 创建随机矩阵
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        
        # 预热
        np.dot(a, b)
        
        # 计时
        start_time = time.time()
        c = np.dot(a, b)
        end_time = time.time()
        
        # 计算性能
        time_taken = end_time - start_time
        flops = 2 * size ** 3  # 2n^3 FLOPS for matrix multiplication
        gflops = flops / time_taken / 1e9
        
        results.append((size, time_taken, gflops))
    
    return results

if __name__ == "__main__":
    results = matrix_multiply_benchmark()
    for size, time_taken, gflops in results:
        print(f"{size},{time_taken:.3f},{gflops:.2f}")
"
    
    for ((i=1; i<=ITERATIONS; i++)); do
        log "计算测试迭代 $i/$ITERATIONS"
        
        # 运行 Python 基准测试
        local python_results=$(python3 -c "$python_script")
        
        # 解析结果
        while IFS=',' read -r size time_taken gflops; do
            echo "Matrix${size}x${size},$i,$time_taken,$gflops" >> "$results_file"
        done <<< "$python_results"
    done
    
    log "计算测试完成，结果保存到 $results_file"
}

# 内存带宽测试
run_memory_benchmark() {
    log "运行内存带宽测试..."
    
    local results_file="$OUTPUT_DIR/memory-benchmark.csv"
    echo "Test,Iteration,Bandwidth(GB/s)" > "$results_file"
    
    # 使用 nvidia-smi 监控内存带宽
    for ((i=1; i<=ITERATIONS; i++)); do
        log "内存带宽测试迭代 $i/$ITERATIONS"
        
        # 运行带宽测试（使用 dd 和 GPU 内存）
        local bandwidth_test="
#!/bin/bash
# 简单的内存带宽测试

# 测试内存拷贝速度
start_time=\$(date +%s.%N)

# 创建测试文件
if [ ! -f /tmp/testfile ]; then
    dd if=/dev/zero of=/tmp/testfile bs=1G count=1 status=none
fi

# 拷贝测试
for j in {1..10}; do
    cp /tmp/testfile /tmp/testfile_copy\$j
    sync
    rm /tmp/testfile_copy\$j
    sync
done

end_time=\$(date +%s.%N)
elapsed=\$(echo "\$end_time - \$start_time" | bc)

# 计算带宽（假设 1GB 数据 * 10 次拷贝）
total_data=10  # GB
bandwidth=\$(echo "scale=2; \$total_data / \$elapsed" | bc)

echo "\$bandwidth"
"
        
        # 运行测试
        local bandwidth=$(bash -c "$bandwidth_test" 2>/dev/null)
        
        if [ -n "$bandwidth" ]; then
            echo "MemoryBandwidth,$i,$bandwidth" >> "$results_file"
        else
            log "警告: 内存带宽测试失败"
        fi
    done
    
    log "内存带宽测试完成，结果保存到 $results_file"
}

# Tensor 性能测试（如果可用）
run_tensor_benchmark() {
    log "运行 Tensor 性能测试..."
    
    local results_file="$OUTPUT_DIR/tensor-benchmark.csv"
    echo "Test,Iteration,Time(s)" > "$results_file"
    
    # 检查是否安装 PyTorch
    if python3 -c "import torch" 2>/dev/null; then
        local pytorch_script="
import torch
import time

def tensor_operations_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试不同的张量操作
    operations = [
        ('MatrixMultiply', lambda x: torch.mm(x, x)),
        ('Conv2d', lambda x: torch.nn.functional.conv2d(x, torch.randn(32, 3, 3, 3).to(device))),
        ('ReLU', lambda x: torch.nn.functional.relu(x)),
    ]
    
    results = []
    
    for op_name, op_func in operations:
        # 创建测试张量
        if op_name == 'Conv2d':
            x = torch.randn(1, 3, 224, 224).to(device)
        else:
            x = torch.randn(1024, 1024).to(device)
        
        # 预热
        for _ in range(3):
            op_func(x)
        
        torch.cuda.synchronize()
        
        # 计时
        start_time = time.time()
        for _ in range(10):
            result = op_func(x)
        torch.cuda.synchronize()
        end_time = time.time()
        
        time_taken = (end_time - start_time) / 10  # 平均时间
        results.append((op_name, time_taken))
    
    return results

if __name__ == "__main__":
    if torch.cuda.is_available():
        results = tensor_operations_benchmark()
        for op_name, time_taken in results:
            print(f"{op_name},{time_taken:.6f}")
    else:
        print("CUDA not available")
"
        
        for ((i=1; i<=ITERATIONS; i++)); do
            log "Tensor 测试迭代 $i/$ITERATIONS"
            
            local torch_results=$(python3 -c "$pytorch_script" 2>/dev/null)
            
            if [ "$torch_results" != "CUDA not available" ]; then
                while IFS=',' read -r op_name time_taken; do
                    echo "$op_name,$i,$time_taken" >> "$results_file"
                done <<< "$torch_results"
            else
                log "警告: CUDA 不可用，跳过 Tensor 测试"
                break
            fi
        done
    else
        log "警告: PyTorch 未安装，跳过 Tensor 测试"
    fi
    
    log "Tensor 测试完成，结果保存到 $results_file"
}

# 生成测试报告
generate_report() {
    log "生成基准测试报告..."
    
    local report_file="$OUTPUT_DIR/benchmark-report.md"
    
    cat > "$report_file" << EOF
# GPU 基准测试报告

## 测试信息
- 测试时间: $(date)
- 测试类型: $BENCHMARK_TYPE
- 迭代次数: $ITERATIONS
- 输出目录: $OUTPUT_DIR

## GPU 硬件信息
$(cat "$OUTPUT_DIR/gpu-info.txt" | sed 's/^/    /')

## 测试结果摘要

### 计算性能
$(if [ -f "$OUTPUT_DIR/compute-benchmark.csv" ]; then
    echo "矩阵乘法性能 (GFLOPS):"
    awk -F, '\$1 ~ /Matrix/ {sum[\$1] += \$4; count[\$1]++} END {for (key in sum) printf "    - %s: %.2f GFLOPS\n", key, sum[key]/count[key]}' "$OUTPUT_DIR/compute-benchmark.csv"
fi)

### 内存带宽
$(if [ -f "$OUTPUT_DIR/memory-benchmark.csv" ]; then
    echo "内存带宽: "
    awk -F, '{sum += \$3; count++} END {printf "    - 平均带宽: %.2f GB/s\n", sum/count}' "$OUTPUT_DIR/memory-benchmark.csv"
fi)

### Tensor 性能
$(if [ -f "$OUTPUT_DIR/tensor-benchmark.csv" ]; then
    echo "Tensor 操作性能: "
    awk -F, '{sum[\$1] += \$3; count[\$1]++} END {for (key in sum) printf "    - %s: %.6f 秒\n", key, sum[key]/count[key]}' "$OUTPUT_DIR/tensor-benchmark.csv"
fi)

## 详细数据
详细测试数据请查看相应的 CSV 文件。

## 建议
基于测试结果，建议：
- 确保驱动程序为最新版本
- 检查系统散热和电源供应
- 根据工作负载特性优化配置
EOF
    
    log "测试报告生成完成: $report_file"
}

# 主函数
main() {
    log "启动 GPU 基准测试套件"
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    check_dependencies
    get_gpu_info
    
    # 运行指定的测试类型
    case "$BENCHMARK_TYPE" in
        "all")
            run_compute_benchmark
            run_memory_benchmark
            run_tensor_benchmark
            ;;
        "compute")
            run_compute_benchmark
            ;;
        "memory")
            run_memory_benchmark
            ;;
        "bandwidth")
            run_memory_benchmark
            ;;
        "tensor")
            run_tensor_benchmark
            ;;
        *)
            log "错误: 不支持的测试类型: $BENCHMARK_TYPE"
            exit 1
            ;;
    esac
    
    generate_report
    log "GPU 基准测试套件执行完成"
}

# 执行主函数
main "$@"