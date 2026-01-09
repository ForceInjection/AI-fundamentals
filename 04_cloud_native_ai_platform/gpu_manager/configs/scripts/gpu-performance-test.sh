#!/bin/bash
# GPU性能测试脚本
# 提供全面的GPU性能测试和基准测试功能

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置变量
TEST_DURATION=60
OUTPUT_DIR="gpu_performance_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_DIR/performance_test.log"
REPORT_FILE="$OUTPUT_DIR/performance_report.md"

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

# 创建输出目录
setup_output_directory() {
    mkdir -p "$OUTPUT_DIR"
    log_info "测试结果将保存到: $OUTPUT_DIR"
}

# 检查GPU环境
check_gpu_environment() {
    log_info "检查GPU环境..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi未找到，请安装NVIDIA驱动"
        return 1
    fi
    
    if ! command -v nvcc &> /dev/null; then
        log_warn "nvcc未找到，某些测试可能无法运行"
    fi
    
    # GPU基本信息
    log_info "GPU设备信息:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader | tee -a "$LOG_FILE"
    
    # 驱动和CUDA版本
    local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    log_info "驱动版本: $driver_version"
    
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "CUDA版本: $cuda_version"
    fi
    
    log_info "✅ GPU环境检查完成"
}

# GPU基础性能测试
basic_performance_test() {
    log_info "开始GPU基础性能测试..."
    
    # GPU利用率测试
    log_info "GPU利用率测试..."
    {
        echo "时间戳,GPU利用率(%),内存利用率(%),温度(C),功耗(W),时钟频率(MHz)"
        for ((i=0; i<30; i++)); do
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
            local mem_util=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits)
            local temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
            local power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
            local clock=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits)
            
            echo "$timestamp,$gpu_util,$mem_util,$temp,$power,$clock"
            sleep 1
        done
    } > "$OUTPUT_DIR/gpu_utilization.csv"
    
    # 内存带宽测试
    log_info "GPU内存带宽测试..."
    if command -v nvidia-smi &> /dev/null; then
        # 使用nvidia-smi进行内存测试
        nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv > "$OUTPUT_DIR/memory_info.csv"
    fi
    
    # GPU时钟频率测试
    log_info "GPU时钟频率测试..."
    {
        echo "时间戳,图形时钟(MHz),内存时钟(MHz),SM时钟(MHz)"
        for ((i=0; i<10; i++)); do
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            local gr_clock=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits)
            local mem_clock=$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits)
            local sm_clock=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)
            
            echo "$timestamp,$gr_clock,$mem_clock,$sm_clock"
            sleep 2
        done
    } > "$OUTPUT_DIR/clock_frequencies.csv"
    
    log_info "✅ GPU基础性能测试完成"
}

# CUDA性能测试
cuda_performance_test() {
    if ! command -v nvcc &> /dev/null; then
        log_warn "跳过CUDA性能测试（nvcc未找到）"
        return 0
    fi
    
    log_info "开始CUDA性能测试..."
    
    # 创建CUDA测试程序
    cat > "$OUTPUT_DIR/cuda_benchmark.cu" << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// 矩阵乘法核函数
__global__ void matrixMul(float *a, float *b, float *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

// 向量加法核函数
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 内存带宽测试
void memoryBandwidthTest() {
    printf("\n=== 内存带宽测试 ===\n");
    
    const int sizes[] = {1024, 2048, 4096, 8192};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i] * sizes[i];
        size_t size = n * sizeof(float);
        
        float *h_data, *d_data;
        CHECK_CUDA(cudaMallocHost(&h_data, size));
        CHECK_CUDA(cudaMalloc(&d_data, size));
        
        // 初始化数据
        for (int j = 0; j < n; j++) {
            h_data[j] = (float)rand() / RAND_MAX;
        }
        
        // H2D传输测试
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float h2d_time;
        CHECK_CUDA(cudaEventElapsedTime(&h2d_time, start, stop));
        float h2d_bandwidth = (size / 1e9) / (h2d_time / 1000.0);
        
        // D2H传输测试
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float d2h_time;
        CHECK_CUDA(cudaEventElapsedTime(&d2h_time, start, stop));
        float d2h_bandwidth = (size / 1e9) / (d2h_time / 1000.0);
        
        printf("矩阵大小: %dx%d, H2D: %.2f GB/s, D2H: %.2f GB/s\n", 
               sizes[i], sizes[i], h2d_bandwidth, d2h_bandwidth);
        
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFreeHost(h_data));
        CHECK_CUDA(cudaFree(d_data));
    }
}

// 计算性能测试
void computePerformanceTest() {
    printf("\n=== 计算性能测试 ===\n");
    
    const int sizes[] = {512, 1024, 2048, 4096};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        int width = sizes[i];
        int n = width * width;
        size_t size = n * sizeof(float);
        
        float *h_a, *h_b, *h_c;
        float *d_a, *d_b, *d_c;
        
        // 分配内存
        CHECK_CUDA(cudaMallocHost(&h_a, size));
        CHECK_CUDA(cudaMallocHost(&h_b, size));
        CHECK_CUDA(cudaMallocHost(&h_c, size));
        CHECK_CUDA(cudaMalloc(&d_a, size));
        CHECK_CUDA(cudaMalloc(&d_b, size));
        CHECK_CUDA(cudaMalloc(&d_c, size));
        
        // 初始化数据
        for (int j = 0; j < n; j++) {
            h_a[j] = (float)rand() / RAND_MAX;
            h_b[j] = (float)rand() / RAND_MAX;
        }
        
        // 复制数据到GPU
        CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
        
        // 矩阵乘法测试
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                      (width + blockSize.y - 1) / blockSize.y);
        
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        // 预热
        matrixMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, width);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 性能测试
        CHECK_CUDA(cudaEventRecord(start));
        matrixMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, width);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float elapsed_time;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
        
        // 计算GFLOPS
        double flops = 2.0 * width * width * width;
        double gflops = flops / (elapsed_time / 1000.0) / 1e9;
        
        printf("矩阵大小: %dx%d, 时间: %.2f ms, 性能: %.2f GFLOPS\n", 
               width, width, elapsed_time, gflops);
        
        // 清理资源
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFreeHost(h_a));
        CHECK_CUDA(cudaFreeHost(h_b));
        CHECK_CUDA(cudaFreeHost(h_c));
        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }
}

int main() {
    printf("CUDA性能基准测试\n");
    
    // 设备信息
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("多处理器数量: %d\n", prop.multiProcessorCount);
    printf("最大线程数/块: %d\n", prop.maxThreadsPerBlock);
    
    memoryBandwidthTest();
    computePerformanceTest();
    
    printf("\n✅ CUDA性能测试完成\n");
    return 0;
}
EOF
    
    # 编译和运行CUDA测试
    log_info "编译CUDA测试程序..."
    if nvcc -O3 -o "$OUTPUT_DIR/cuda_benchmark" "$OUTPUT_DIR/cuda_benchmark.cu"; then
        log_info "运行CUDA性能测试..."
        "$OUTPUT_DIR/cuda_benchmark" > "$OUTPUT_DIR/cuda_performance.log" 2>&1
        log_info "✅ CUDA性能测试完成"
    else
        log_error "CUDA测试程序编译失败"
    fi
}

# PyTorch性能测试
pytorch_performance_test() {
    log_info "开始PyTorch性能测试..."
    
    # 创建PyTorch测试脚本
    cat > "$OUTPUT_DIR/pytorch_benchmark.py" << 'EOF'
import torch
import time
import numpy as np

def test_pytorch_performance():
    print("PyTorch GPU性能测试")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过测试")
        return
    
    device = torch.device('cuda')
    print(f"GPU设备: {torch.cuda.get_device_name()}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 矩阵乘法测试
    print("\n=== 矩阵乘法性能测试 ===")
    sizes = [512, 1024, 2048, 4096]
    
    for size in sizes:
        # 创建随机矩阵
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # 预热
        torch.mm(a, b)
        torch.cuda.synchronize()
        
        # 性能测试
        start_time = time.time()
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        flops = 2 * size**3
        gflops = flops / avg_time / 1e9
        
        print(f"矩阵大小: {size}x{size}, 平均时间: {avg_time*1000:.2f} ms, 性能: {gflops:.2f} GFLOPS")
    
    # 卷积测试
    print("\n=== 卷积性能测试 ===")
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        # 创建输入和卷积层
        input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
        conv_layer = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
        
        # 预热
        conv_layer(input_tensor)
        torch.cuda.synchronize()
        
        # 性能测试
        start_time = time.time()
        for _ in range(100):
            output = conv_layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        throughput = batch_size / avg_time
        
        print(f"批次大小: {batch_size}, 平均时间: {avg_time*1000:.2f} ms, 吞吐量: {throughput:.2f} samples/s")
    
    # 内存带宽测试
    print("\n=== 内存带宽测试 ===")
    sizes = [1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024]
    
    for size in sizes:
        # 创建大张量
        data = torch.randn(size, device='cpu')
        
        # H2D传输测试
        start_time = time.time()
        gpu_data = data.to(device)
        torch.cuda.synchronize()
        h2d_time = time.time() - start_time
        
        # D2H传输测试
        start_time = time.time()
        cpu_data = gpu_data.to('cpu')
        torch.cuda.synchronize()
        d2h_time = time.time() - start_time
        
        data_size_gb = size * 4 / 1024**3  # float32 = 4 bytes
        h2d_bandwidth = data_size_gb / h2d_time
        d2h_bandwidth = data_size_gb / d2h_time
        
        print(f"数据大小: {data_size_gb:.2f} GB, H2D: {h2d_bandwidth:.2f} GB/s, D2H: {d2h_bandwidth:.2f} GB/s")

if __name__ == "__main__":
    test_pytorch_performance()
EOF
    
    # 运行PyTorch测试
    if command -v python3 &> /dev/null; then
        log_info "运行PyTorch性能测试..."
        python3 "$OUTPUT_DIR/pytorch_benchmark.py" > "$OUTPUT_DIR/pytorch_performance.log" 2>&1 || log_warn "PyTorch测试可能失败（需要安装PyTorch）"
    else
        log_warn "跳过PyTorch测试（python3未找到）"
    fi
    
    log_info "✅ PyTorch性能测试完成"
}

# 压力测试
stress_test() {
    log_info "开始GPU压力测试..."
    
    # 创建压力测试脚本
    cat > "$OUTPUT_DIR/gpu_stress_test.py" << 'EOF'
import subprocess
import time
import threading
import signal
import sys

class GPUStressTest:
    def __init__(self, duration=300):
        self.duration = duration
        self.running = True
        self.results = []
    
    def monitor_gpu(self):
        """监控GPU状态"""
        while self.running:
            try:
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,power.draw',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.results.append(result.stdout.strip())
                
                time.sleep(1)
            except Exception as e:
                print(f"监控错误: {e}")
                break
    
    def stress_computation(self):
        """GPU计算压力测试"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print("开始GPU计算压力测试...")
                
                while self.running:
                    # 大矩阵乘法
                    a = torch.randn(2048, 2048, device=device)
                    b = torch.randn(2048, 2048, device=device)
                    c = torch.mm(a, b)
                    
                    # 卷积操作
                    x = torch.randn(32, 3, 224, 224, device=device)
                    conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
                    y = conv(x)
                    
                    del a, b, c, x, y
                    torch.cuda.empty_cache()
        except ImportError:
            print("PyTorch未安装，跳过计算压力测试")
        except Exception as e:
            print(f"计算压力测试错误: {e}")
    
    def run_test(self):
        """运行压力测试"""
        print(f"开始GPU压力测试，持续时间: {self.duration} 秒")
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self.monitor_gpu)
        monitor_thread.start()
        
        # 启动压力测试线程
        stress_thread = threading.Thread(target=self.stress_computation)
        stress_thread.start()
        
        # 等待测试完成
        time.sleep(self.duration)
        
        # 停止测试
        self.running = False
        monitor_thread.join(timeout=5)
        stress_thread.join(timeout=5)
        
        # 保存结果
        with open('gpu_stress_results.csv', 'w') as f:
            f.write('timestamp,temperature,utilization,memory_used,power_draw\n')
            for result in self.results:
                f.write(result + '\n')
        
        print(f"压力测试完成，结果保存到: gpu_stress_results.csv")
        
        # 分析结果
        self.analyze_results()
    
    def analyze_results(self):
        """分析测试结果"""
        if not self.results:
            print("没有测试数据")
            return
        
        temps = []
        utils = []
        powers = []
        
        for line in self.results:
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    temps.append(float(parts[1]))
                    utils.append(float(parts[2]))
                    powers.append(float(parts[4]))
                except ValueError:
                    continue
        
        if temps and utils and powers:
            print("\n=== 压力测试分析结果 ===")
            print(f"平均温度: {sum(temps)/len(temps):.1f}°C")
            print(f"最高温度: {max(temps):.1f}°C")
            print(f"平均GPU利用率: {sum(utils)/len(utils):.1f}%")
            print(f"平均功耗: {sum(powers)/len(powers):.1f}W")
            print(f"最高功耗: {max(powers):.1f}W")

def signal_handler(sig, frame):
    print('\n测试被中断')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    duration = 60  # 默认60秒
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    
    test = GPUStressTest(duration)
    test.run_test()
EOF
    
    # 运行压力测试
    if command -v python3 &> /dev/null; then
        log_info "运行GPU压力测试 (60秒)..."
        cd "$OUTPUT_DIR"
        python3 gpu_stress_test.py 60
        cd - > /dev/null
        log_info "✅ GPU压力测试完成"
    else
        log_warn "跳过压力测试（python3未找到）"
    fi
}

# 生成性能报告
generate_performance_report() {
    log_info "生成性能测试报告..."
    
    cat > "$REPORT_FILE" << EOF
# GPU性能测试报告

## 测试环境
- 测试时间: $(date)
- 操作系统: $(uname -a)
- GPU驱动版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
EOF
    
    if command -v nvcc &> /dev/null; then
        echo "- CUDA版本: $(nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-)" >> "$REPORT_FILE"
    fi
    
    cat >> "$REPORT_FILE" << EOF

## GPU设备信息
\`\`\`
$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv)
\`\`\`

## 测试结果

EOF
    
    # 添加各项测试结果
    if [[ -f "$OUTPUT_DIR/cuda_performance.log" ]]; then
        echo "### CUDA性能测试" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        cat "$OUTPUT_DIR/cuda_performance.log" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    if [[ -f "$OUTPUT_DIR/pytorch_performance.log" ]]; then
        echo "### PyTorch性能测试" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        cat "$OUTPUT_DIR/pytorch_performance.log" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    if [[ -f "$OUTPUT_DIR/gpu_stress_results.csv" ]]; then
        echo "### 压力测试结果" >> "$REPORT_FILE"
        echo "压力测试数据保存在: gpu_stress_results.csv" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    log_info "✅ 性能测试报告生成完成: $REPORT_FILE"
}

# 清理测试文件
cleanup_test_files() {
    log_info "清理临时文件..."
    
    # 保留重要结果文件，删除临时文件
    find "$OUTPUT_DIR" -name "*.cu" -delete 2>/dev/null || true
    find "$OUTPUT_DIR" -name "cuda_benchmark" -delete 2>/dev/null || true
    
    log_info "✅ 清理完成"
}

# 显示帮助信息
show_help() {
    echo "GPU性能测试脚本"
    echo "用法: $0 [选项] [测试类型]"
    echo ""
    echo "测试类型:"
    echo "  basic      基础性能测试"
    echo "  cuda       CUDA性能测试"
    echo "  pytorch    PyTorch性能测试"
    echo "  stress     GPU压力测试"
    echo "  all        所有测试（默认）"
    echo ""
    echo "选项:"
    echo "  --duration SECONDS    测试持续时间（默认: 60秒）"
    echo "  --output-dir DIR      输出目录（默认: gpu_performance_TIMESTAMP）"
    echo "  --help               显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                           # 运行所有测试"
    echo "  $0 basic                     # 只运行基础测试"
    echo "  $0 --duration 120 stress     # 运行120秒压力测试"
    echo "  $0 --output-dir my_test all  # 指定输出目录"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --duration)
                TEST_DURATION="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                LOG_FILE="$OUTPUT_DIR/performance_test.log"
                REPORT_FILE="$OUTPUT_DIR/performance_report.md"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            basic|cuda|pytorch|stress|all)
                TEST_TYPE="$1"
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
    local test_type=${TEST_TYPE:-"all"}
    
    # 设置输出目录
    setup_output_directory
    
    # 检查环境
    check_gpu_environment
    
    # 根据测试类型运行相应测试
    case $test_type in
        "basic")
            basic_performance_test
            ;;
        "cuda")
            cuda_performance_test
            ;;
        "pytorch")
            pytorch_performance_test
            ;;
        "stress")
            stress_test
            ;;
        "all")
            basic_performance_test
            cuda_performance_test
            pytorch_performance_test
            stress_test
            ;;
        *)
            log_error "未知测试类型: $test_type"
            show_help
            exit 1
            ;;
    esac
    
    # 生成报告
    generate_performance_report
    
    # 清理临时文件
    cleanup_test_files
    
    log_info "🎉 GPU性能测试完成！"
    log_info "测试结果保存在: $OUTPUT_DIR"
    log_info "详细报告: $REPORT_FILE"
}

# 解析参数并运行主函数
parse_arguments "$@"
main