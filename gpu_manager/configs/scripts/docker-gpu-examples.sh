#!/bin/bash

# Docker GPU 示例脚本
# 用于演示 Docker 容器中 GPU 的使用和配置

set -euo pipefail

# 日志配置
LOG_FILE="/tmp/docker-gpu-examples.log"
LOG_LEVEL="INFO"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

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

# 检查 Docker 和 NVIDIA Docker 运行时
check_docker_gpu() {
    log_info "检查 Docker GPU 支持..."
    
    # 检查 Docker 是否安装
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装"
        return 1
    fi
    
    # 检查 Docker 是否运行
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未运行"
        return 1
    fi
    
    # 检查 NVIDIA Docker 运行时
    if ! docker info | grep -q "nvidia"; then
        log_warn "NVIDIA Docker 运行时未配置"
    else
        log_success "NVIDIA Docker 运行时已配置"
    fi
    
    # 测试 GPU 访问
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log_success "Docker GPU 访问正常"
    else
        log_error "Docker GPU 访问失败"
        return 1
    fi
}

# 基础 GPU 容器示例
run_basic_gpu_container() {
    log_info "运行基础 GPU 容器示例..."
    
    docker run --rm --gpus all \
        nvidia/cuda:11.8-base-ubuntu20.04 \
        nvidia-smi
}

# CUDA 开发容器示例
run_cuda_dev_container() {
    log_info "运行 CUDA 开发容器示例..."
    
    docker run --rm --gpus all \
        -v "$(pwd)":/workspace \
        -w /workspace \
        nvidia/cuda:11.8-devel-ubuntu20.04 \
        bash -c "nvcc --version && nvidia-smi"
}

# PyTorch GPU 容器示例
run_pytorch_gpu_container() {
    log_info "运行 PyTorch GPU 容器示例..."
    
    docker run --rm --gpus all \
        pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
}

# TensorFlow GPU 容器示例
run_tensorflow_gpu_container() {
    log_info "运行 TensorFlow GPU 容器示例..."
    
    docker run --rm --gpus all \
        tensorflow/tensorflow:2.13.0-gpu \
        python -c "import tensorflow as tf; print(f'GPU devices: {tf.config.list_physical_devices('GPU')}')"
}

# 多 GPU 容器示例
run_multi_gpu_container() {
    log_info "运行多 GPU 容器示例..."
    
    # 使用所有 GPU
    docker run --rm --gpus all \
        nvidia/cuda:11.8-base-ubuntu20.04 \
        nvidia-smi
    
    # 使用指定数量的 GPU
    docker run --rm --gpus 2 \
        nvidia/cuda:11.8-base-ubuntu20.04 \
        nvidia-smi
    
    # 使用指定的 GPU
    docker run --rm --gpus '"device=0,1"' \
        nvidia/cuda:11.8-base-ubuntu20.04 \
        nvidia-smi
}

# GPU 内存限制示例
run_gpu_memory_limit_container() {
    log_info "运行 GPU 内存限制容器示例..."
    
    # 注意：Docker 本身不直接支持 GPU 内存限制
    # 需要在应用程序中实现内存管理
    docker run --rm --gpus all \
        -e CUDA_VISIBLE_DEVICES=0 \
        pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
        python -c "
import torch
import os

# 设置 GPU 内存增长
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5)  # 限制使用 50% GPU 内存
    print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
    print(f'GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')
"
}

# Docker Compose GPU 示例
create_docker_compose_gpu_example() {
    log_info "创建 Docker Compose GPU 示例..."
    
    cat > docker-compose-gpu.yml << 'EOF'
version: '3.8'

services:
  pytorch-gpu:
    image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
  tensorflow-gpu:
    image: tensorflow/tensorflow:2.13.0-gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python -c "import tensorflow as tf; print(f'GPU devices: {tf.config.list_physical_devices('GPU')}')"
    
  cuda-dev:
    image: nvidia/cuda:11.8-devel-ubuntu20.04
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./workspace:/workspace
    working_dir: /workspace
    command: bash -c "nvcc --version && nvidia-smi"
EOF
    
    log_success "Docker Compose GPU 示例已创建: docker-compose-gpu.yml"
}

# GPU 监控容器示例
run_gpu_monitoring_container() {
    log_info "运行 GPU 监控容器示例..."
    
    # 运行 nvidia-smi 监控容器
    docker run --rm --gpus all \
        --name gpu-monitor \
        -d \
        nvidia/cuda:11.8-base-ubuntu20.04 \
        bash -c "while true; do nvidia-smi; sleep 10; done"
    
    log_info "GPU 监控容器已启动，容器名称: gpu-monitor"
    log_info "查看监控输出: docker logs -f gpu-monitor"
    log_info "停止监控容器: docker stop gpu-monitor"
}

# 清理函数
cleanup() {
    log_info "清理临时资源..."
    
    # 停止监控容器
    if docker ps -q -f name=gpu-monitor | grep -q .; then
        docker stop gpu-monitor
        log_info "已停止 GPU 监控容器"
    fi
    
    # 清理未使用的镜像
    docker image prune -f
    
    log_success "清理完成"
}

# 显示帮助信息
show_help() {
    cat << EOF
Docker GPU 示例脚本

用法: $0 [选项]

选项:
  check           检查 Docker GPU 支持
  basic           运行基础 GPU 容器示例
  cuda-dev        运行 CUDA 开发容器示例
  pytorch         运行 PyTorch GPU 容器示例
  tensorflow      运行 TensorFlow GPU 容器示例
  multi-gpu       运行多 GPU 容器示例
  memory-limit    运行 GPU 内存限制容器示例
  compose         创建 Docker Compose GPU 示例
  monitor         运行 GPU 监控容器示例
  cleanup         清理临时资源
  all             运行所有示例
  help            显示此帮助信息

示例:
  $0 check
  $0 basic
  $0 all

EOF
}

# 主函数
main() {
    local action=${1:-help}
    
    log_info "开始执行 Docker GPU 示例: $action"
    
    case $action in
        check)
            check_docker_gpu
            ;;
        basic)
            check_docker_gpu && run_basic_gpu_container
            ;;
        cuda-dev)
            check_docker_gpu && run_cuda_dev_container
            ;;
        pytorch)
            check_docker_gpu && run_pytorch_gpu_container
            ;;
        tensorflow)
            check_docker_gpu && run_tensorflow_gpu_container
            ;;
        multi-gpu)
            check_docker_gpu && run_multi_gpu_container
            ;;
        memory-limit)
            check_docker_gpu && run_gpu_memory_limit_container
            ;;
        compose)
            create_docker_compose_gpu_example
            ;;
        monitor)
            check_docker_gpu && run_gpu_monitoring_container
            ;;
        cleanup)
            cleanup
            ;;
        all)
            check_docker_gpu
            run_basic_gpu_container
            run_cuda_dev_container
            run_pytorch_gpu_container
            run_tensorflow_gpu_container
            run_multi_gpu_container
            run_gpu_memory_limit_container
            create_docker_compose_gpu_example
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知选项: $action"
            show_help
            exit 1
            ;;
    esac
    
    log_success "Docker GPU 示例执行完成"
}

# 信号处理
trap cleanup EXIT

# 执行主函数
main "$@"