#!/bin/bash

# CUDA 调试工具脚本
# 用于 CUDA 环境诊断、调试和性能分析

set -euo pipefail

# 配置变量
OUTPUT_DIR="/tmp/cuda-debug-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$OUTPUT_DIR/cuda-debug.log"
REPORT_FILE="$OUTPUT_DIR/cuda-debug-report.md"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

# 初始化报告文件
init_report() {
    cat > "$REPORT_FILE" << EOF
# CUDA 调试报告

生成时间: $(date)
主机名: $(hostname)
用户: $(whoami)

## 系统信息

EOF
}

# 检查 CUDA 安装
check_cuda_installation() {
    log_info "检查 CUDA 安装..."
    
    echo "### CUDA 安装检查" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # 检查 nvcc
    if command -v nvcc &> /dev/null; then
        local nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
        log_success "NVCC 已安装: $nvcc_version"
        echo "- NVCC 版本: $nvcc_version" >> "$REPORT_FILE"
        
        # 保存完整的 nvcc 信息
        echo "\`\`\`" >> "$REPORT_FILE"
        nvcc --version >> "$REPORT_FILE" 2>&1
        echo "\`\`\`" >> "$REPORT_FILE"
    else
        log_error "NVCC 未安装"
        echo "- ❌ NVCC 未安装" >> "$REPORT_FILE"
    fi
    
    # 检查 CUDA 库路径
    if [ -d "/usr/local/cuda" ]; then
        log_success "CUDA 安装目录存在: /usr/local/cuda"
        echo "- CUDA 安装目录: /usr/local/cuda" >> "$REPORT_FILE"
        
        # 检查 CUDA 版本
        if [ -f "/usr/local/cuda/version.json" ]; then
            local cuda_version=$(cat /usr/local/cuda/version.json | grep '"version"' | cut -d'"' -f4)
            echo "- CUDA 版本: $cuda_version" >> "$REPORT_FILE"
        elif [ -f "/usr/local/cuda/version.txt" ]; then
            local cuda_version=$(cat /usr/local/cuda/version.txt)
            echo "- CUDA 版本: $cuda_version" >> "$REPORT_FILE"
        fi
    else
        log_warn "CUDA 安装目录不存在: /usr/local/cuda"
        echo "- ⚠️ CUDA 安装目录不存在" >> "$REPORT_FILE"
    fi
    
    # 检查环境变量
    echo "\n### CUDA 环境变量" >> "$REPORT_FILE"
    echo "\`\`\`" >> "$REPORT_FILE"
    echo "CUDA_HOME: ${CUDA_HOME:-未设置}" >> "$REPORT_FILE"
    echo "PATH: $PATH" >> "$REPORT_FILE"
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-未设置}" >> "$REPORT_FILE"
    echo "\`\`\`" >> "$REPORT_FILE"
    
    echo "" >> "$REPORT_FILE"
}

# 检查 NVIDIA 驱动
check_nvidia_driver() {
    log_info "检查 NVIDIA 驱动..."
    
    echo "### NVIDIA 驱动检查" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA 驱动已安装"
        echo "- ✅ NVIDIA 驱动已安装" >> "$REPORT_FILE"
        
        # 保存 nvidia-smi 输出
        echo "\`\`\`" >> "$REPORT_FILE"
        nvidia-smi >> "$REPORT_FILE" 2>&1
        echo "\`\`\`" >> "$REPORT_FILE"
        
        # 保存到单独文件
        nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt" 2>&1
    else
        log_error "NVIDIA 驱动未安装"
        echo "- ❌ NVIDIA 驱动未安装" >> "$REPORT_FILE"
    fi
    
    echo "" >> "$REPORT_FILE"
}

# 检查 GPU 设备
check_gpu_devices() {
    log_info "检查 GPU 设备..."
    
    echo "### GPU 设备信息" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # 使用 lspci 检查 GPU
    if command -v lspci &> /dev/null; then
        local gpu_count=$(lspci | grep -i nvidia | wc -l)
        log_info "检测到 $gpu_count 个 NVIDIA GPU"
        echo "- 检测到 GPU 数量: $gpu_count" >> "$REPORT_FILE"
        
        echo "\n#### PCI GPU 设备" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        lspci | grep -i nvidia >> "$REPORT_FILE" 2>&1
        echo "\`\`\`" >> "$REPORT_FILE"
    fi
    
    # 使用 nvidia-ml-py 检查（如果可用）
    if command -v python3 &> /dev/null; then
        python3 -c "
import sys
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f'NVML 检测到 {device_count} 个 GPU')
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f'GPU {i}: {name}')
        print(f'  内存: {memory_info.total // 1024**2} MB')
except ImportError:
    print('pynvml 未安装，跳过 NVML 检查')
except Exception as e:
    print(f'NVML 检查失败: {e}')
" >> "$OUTPUT_DIR/gpu-nvml-check.txt" 2>&1
    fi
    
    echo "" >> "$REPORT_FILE"
}

# 运行 CUDA 示例程序
run_cuda_samples() {
    log_info "运行 CUDA 示例程序..."
    
    echo "### CUDA 示例程序测试" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # 查找 CUDA 示例
    local samples_dirs=(
        "/usr/local/cuda/samples"
        "/usr/local/cuda-*/samples"
        "$HOME/NVIDIA_CUDA-*/samples"
    )
    
    local samples_found=false
    for samples_dir in "${samples_dirs[@]}"; do
        if [ -d "$samples_dir" ]; then
            samples_found=true
            log_info "找到 CUDA 示例目录: $samples_dir"
            echo "- CUDA 示例目录: $samples_dir" >> "$REPORT_FILE"
            
            # 运行 deviceQuery
            local device_query="$samples_dir/1_Utilities/deviceQuery/deviceQuery"
            if [ -f "$device_query" ]; then
                log_info "运行 deviceQuery..."
                echo "\n#### deviceQuery 输出" >> "$REPORT_FILE"
                echo "\`\`\`" >> "$REPORT_FILE"
                "$device_query" >> "$REPORT_FILE" 2>&1 || true
                echo "\`\`\`" >> "$REPORT_FILE"
                
                # 保存到单独文件
                "$device_query" > "$OUTPUT_DIR/deviceQuery.txt" 2>&1 || true
            else
                log_warn "deviceQuery 不存在，尝试编译..."
                if [ -d "$samples_dir/1_Utilities/deviceQuery" ]; then
                    cd "$samples_dir/1_Utilities/deviceQuery"
                    make > "$OUTPUT_DIR/deviceQuery-compile.log" 2>&1 || true
                    if [ -f "deviceQuery" ]; then
                        ./deviceQuery >> "$REPORT_FILE" 2>&1 || true
                    fi
                fi
            fi
            
            # 运行 bandwidthTest
            local bandwidth_test="$samples_dir/1_Utilities/bandwidthTest/bandwidthTest"
            if [ -f "$bandwidth_test" ]; then
                log_info "运行 bandwidthTest..."
                echo "\n#### bandwidthTest 输出" >> "$REPORT_FILE"
                echo "\`\`\`" >> "$REPORT_FILE"
                "$bandwidth_test" >> "$REPORT_FILE" 2>&1 || true
                echo "\`\`\`" >> "$REPORT_FILE"
                
                # 保存到单独文件
                "$bandwidth_test" > "$OUTPUT_DIR/bandwidthTest.txt" 2>&1 || true
            fi
            
            break
        fi
    done
    
    if [ "$samples_found" = false ]; then
        log_warn "未找到 CUDA 示例程序"
        echo "- ⚠️ 未找到 CUDA 示例程序" >> "$REPORT_FILE"
    fi
    
    echo "" >> "$REPORT_FILE"
}

# 检查 CUDA 库
check_cuda_libraries() {
    log_info "检查 CUDA 库..."
    
    echo "### CUDA 库检查" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # 检查关键库文件
    local cuda_libs=(
        "libcudart.so"
        "libcublas.so"
        "libcurand.so"
        "libcufft.so"
        "libcusparse.so"
        "libcusolver.so"
        "libnvrtc.so"
    )
    
    for lib in "${cuda_libs[@]}"; do
        if ldconfig -p | grep -q "$lib"; then
            local lib_path=$(ldconfig -p | grep "$lib" | awk '{print $NF}' | head -1)
            log_success "找到库: $lib -> $lib_path"
            echo "- ✅ $lib: $lib_path" >> "$REPORT_FILE"
        else
            log_warn "未找到库: $lib"
            echo "- ⚠️ $lib: 未找到" >> "$REPORT_FILE"
        fi
    done
    
    echo "" >> "$REPORT_FILE"
}

# 运行性能测试
run_performance_tests() {
    log_info "运行性能测试..."
    
    echo "### 性能测试" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # 创建简单的 CUDA 测试程序
    cat > "$OUTPUT_DIR/cuda_test.cu" << 'EOF'
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1000000;
    const int size = n * sizeof(float);
    
    // 分配主机内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 复制数据到设备
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 执行内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // 复制结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Vector addition completed in " << duration.count() << " microseconds" << std::endl;
    std::cout << "First 5 results: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;
    
    // 清理
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
EOF
    
    # 编译和运行测试程序
    if command -v nvcc &> /dev/null; then
        cd "$OUTPUT_DIR"
        if nvcc -o cuda_test cuda_test.cu > cuda_compile.log 2>&1; then
            log_success "CUDA 测试程序编译成功"
            echo "- ✅ CUDA 测试程序编译成功" >> "$REPORT_FILE"
            
            if ./cuda_test > cuda_test_output.txt 2>&1; then
                log_success "CUDA 测试程序运行成功"
                echo "- ✅ CUDA 测试程序运行成功" >> "$REPORT_FILE"
                
                echo "\n#### 测试输出" >> "$REPORT_FILE"
                echo "\`\`\`" >> "$REPORT_FILE"
                cat cuda_test_output.txt >> "$REPORT_FILE"
                echo "\`\`\`" >> "$REPORT_FILE"
            else
                log_error "CUDA 测试程序运行失败"
                echo "- ❌ CUDA 测试程序运行失败" >> "$REPORT_FILE"
            fi
        else
            log_error "CUDA 测试程序编译失败"
            echo "- ❌ CUDA 测试程序编译失败" >> "$REPORT_FILE"
            echo "\n#### 编译错误" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
            cat cuda_compile.log >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
        fi
    fi
    
    echo "" >> "$REPORT_FILE"
}

# 检查 Python CUDA 支持
check_python_cuda() {
    log_info "检查 Python CUDA 支持..."
    
    echo "### Python CUDA 支持" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    if command -v python3 &> /dev/null; then
        # 检查 PyTorch
        python3 -c "
import sys
try:
    import torch
    print(f'PyTorch 版本: {torch.__version__}')
    print(f'CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA 版本: {torch.version.cuda}')
        print(f'GPU 数量: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError:
    print('PyTorch 未安装')
except Exception as e:
    print(f'PyTorch 检查失败: {e}')
" > "$OUTPUT_DIR/pytorch_check.txt" 2>&1
        
        # 检查 TensorFlow
        python3 -c "
import sys
try:
    import tensorflow as tf
    print(f'TensorFlow 版本: {tf.__version__}')
    gpus = tf.config.list_physical_devices('GPU')
    print(f'GPU 设备: {len(gpus)}')
    for gpu in gpus:
        print(f'  {gpu}')
except ImportError:
    print('TensorFlow 未安装')
except Exception as e:
    print(f'TensorFlow 检查失败: {e}')
" > "$OUTPUT_DIR/tensorflow_check.txt" 2>&1
        
        # 添加到报告
        echo "#### PyTorch 检查" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        cat "$OUTPUT_DIR/pytorch_check.txt" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        
        echo "\n#### TensorFlow 检查" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        cat "$OUTPUT_DIR/tensorflow_check.txt" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
    else
        echo "- ⚠️ Python3 未安装" >> "$REPORT_FILE"
    fi
    
    echo "" >> "$REPORT_FILE"
}

# 生成诊断建议
generate_recommendations() {
    log_info "生成诊断建议..."
    
    echo "## 诊断建议" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # 基于检查结果生成建议
    if ! command -v nvcc &> /dev/null; then
        echo "### CUDA 工具包安装" >> "$REPORT_FILE"
        echo "- 安装 CUDA 工具包: https://developer.nvidia.com/cuda-downloads" >> "$REPORT_FILE"
        echo "- 设置环境变量:" >> "$REPORT_FILE"
        echo "  \`\`\`bash" >> "$REPORT_FILE"
        echo "  export CUDA_HOME=/usr/local/cuda" >> "$REPORT_FILE"
        echo "  export PATH=\$CUDA_HOME/bin:\$PATH" >> "$REPORT_FILE"
        echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> "$REPORT_FILE"
        echo "  \`\`\`" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "### NVIDIA 驱动安装" >> "$REPORT_FILE"
        echo "- 安装 NVIDIA 驱动: https://www.nvidia.com/drivers" >> "$REPORT_FILE"
        echo "- Ubuntu: \`sudo apt install nvidia-driver-xxx\`" >> "$REPORT_FILE"
        echo "- CentOS/RHEL: \`sudo yum install nvidia-driver\`" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    echo "### 常见问题解决" >> "$REPORT_FILE"
    echo "1. **CUDA 版本不匹配**: 确保 CUDA 工具包版本与驱动版本兼容" >> "$REPORT_FILE"
    echo "2. **库文件缺失**: 检查 LD_LIBRARY_PATH 设置" >> "$REPORT_FILE"
    echo "3. **权限问题**: 确保用户有访问 GPU 设备的权限" >> "$REPORT_FILE"
    echo "4. **内存不足**: 检查 GPU 内存使用情况" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "### 有用的调试命令" >> "$REPORT_FILE"
    echo "\`\`\`bash" >> "$REPORT_FILE"
    echo "# 查看 GPU 状态" >> "$REPORT_FILE"
    echo "nvidia-smi" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "# 查看 CUDA 版本" >> "$REPORT_FILE"
    echo "nvcc --version" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "# 查看库文件" >> "$REPORT_FILE"
    echo "ldconfig -p | grep cuda" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "# 检查设备文件" >> "$REPORT_FILE"
    echo "ls -la /dev/nvidia*" >> "$REPORT_FILE"
    echo "\`\`\`" >> "$REPORT_FILE"
}

# 清理函数
cleanup() {
    log_info "调试完成，报告保存在: $OUTPUT_DIR"
    log_info "主要文件:"
    log_info "  - 调试报告: $REPORT_FILE"
    log_info "  - 日志文件: $LOG_FILE"
    
    if [ -f "$OUTPUT_DIR/nvidia-smi.txt" ]; then
        log_info "  - nvidia-smi 输出: $OUTPUT_DIR/nvidia-smi.txt"
    fi
    
    if [ -f "$OUTPUT_DIR/deviceQuery.txt" ]; then
        log_info "  - deviceQuery 输出: $OUTPUT_DIR/deviceQuery.txt"
    fi
}

# 显示帮助信息
show_help() {
    cat << EOF
CUDA 调试工具脚本

用法: $0 [选项]

选项:
  full            执行完整的 CUDA 调试检查
  driver          仅检查 NVIDIA 驱动
  cuda            仅检查 CUDA 安装
  devices         仅检查 GPU 设备
  libraries       仅检查 CUDA 库
  samples         仅运行 CUDA 示例程序
  performance     仅运行性能测试
  python          仅检查 Python CUDA 支持
  help            显示此帮助信息

输出:
  所有结果保存在: $OUTPUT_DIR
  主要报告文件: cuda-debug-report.md

示例:
  $0 full
  $0 driver
  $0 performance

EOF
}

# 主函数
main() {
    local action=${1:-full}
    
    log_info "开始 CUDA 调试: $action"
    init_report
    
    case $action in
        full)
            check_nvidia_driver
            check_cuda_installation
            check_gpu_devices
            check_cuda_libraries
            run_cuda_samples
            run_performance_tests
            check_python_cuda
            generate_recommendations
            ;;
        driver)
            check_nvidia_driver
            ;;
        cuda)
            check_cuda_installation
            ;;
        devices)
            check_gpu_devices
            ;;
        libraries)
            check_cuda_libraries
            ;;
        samples)
            run_cuda_samples
            ;;
        performance)
            run_performance_tests
            ;;
        python)
            check_python_cuda
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "未知选项: $action"
            show_help
            exit 1
            ;;
    esac
    
    log_success "CUDA 调试完成"
}

# 信号处理
trap cleanup EXIT

# 执行主函数
main "$@"