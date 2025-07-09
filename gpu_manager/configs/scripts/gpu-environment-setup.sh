#!/bin/bash
# GPU环境一键配置脚本
# 支持Ubuntu 20.04/22.04和CentOS 7/8

set -e

# 配置变量
NVIDIA_DRIVER_VERSION="550"
CUDA_VERSION="12.4"
DOCKER_VERSION="24.0"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检测操作系统
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    else
        log_error "无法检测操作系统"
        exit 1
    fi
    log_info "检测到操作系统: $OS $VER"
}

# 安装NVIDIA驱动
install_nvidia_driver() {
    log_info "开始安装NVIDIA驱动..."
    
    if [[ "$OS" == *"Ubuntu"* ]]; then
        # Ubuntu系统
        sudo apt update
        sudo apt install -y software-properties-common
        sudo add-apt-repository ppa:graphics-drivers/ppa -y
        sudo apt update
        sudo apt install -y nvidia-driver-$NVIDIA_DRIVER_VERSION
    elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
        # CentOS/RHEL系统
        sudo yum install -y epel-release
        sudo yum install -y dkms kernel-devel kernel-headers
        
        # 添加NVIDIA仓库
        sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
        sudo yum install -y nvidia-driver:$NVIDIA_DRIVER_VERSION
    else
        log_error "不支持的操作系统: $OS"
        exit 1
    fi
    
    log_info "NVIDIA驱动安装完成"
}

# 安装CUDA Toolkit
install_cuda() {
    log_info "开始安装CUDA Toolkit..."
    
    if [[ "$OS" == *"Ubuntu"* ]]; then
        # 添加CUDA仓库
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt update
        sudo apt install -y cuda-toolkit-$(echo $CUDA_VERSION | tr '.' '-')
    elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
        sudo yum install -y cuda-toolkit-$(echo $CUDA_VERSION | tr '.' '-')
    fi
    
    # 设置环境变量
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    log_info "CUDA Toolkit安装完成"
}

# 安装Docker
install_docker() {
    log_info "开始安装Docker..."
    
    if [[ "$OS" == *"Ubuntu"* ]]; then
        # 卸载旧版本
        sudo apt remove -y docker docker-engine docker.io containerd runc
        
        # 安装依赖
        sudo apt update
        sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release
        
        # 添加Docker GPG密钥
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        
        # 添加Docker仓库
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        # 安装Docker
        sudo apt update
        sudo apt install -y docker-ce docker-ce-cli containerd.io
    elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
        # 卸载旧版本
        sudo yum remove -y docker docker-client docker-client-latest docker-common docker-latest docker-latest-logrotate docker-logrotate docker-engine
        
        # 安装依赖
        sudo yum install -y yum-utils
        
        # 添加Docker仓库
        sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
        
        # 安装Docker
        sudo yum install -y docker-ce docker-ce-cli containerd.io
    fi
    
    # 启动Docker服务
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # 添加用户到docker组
    sudo usermod -aG docker $USER
    
    log_info "Docker安装完成"
}

# 安装NVIDIA Container Toolkit
install_nvidia_container_toolkit() {
    log_info "开始安装NVIDIA Container Toolkit..."
    
    if [[ "$OS" == *"Ubuntu"* ]]; then
        # 添加NVIDIA Container Toolkit仓库
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        sudo apt update
        sudo apt install -y nvidia-container-toolkit
    elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
        # 添加NVIDIA Container Toolkit仓库
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
            sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
        
        sudo yum install -y nvidia-container-toolkit
    fi
    
    # 配置Docker运行时
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    log_info "NVIDIA Container Toolkit安装完成"
}

# 验证安装
verify_installation() {
    log_info "开始验证安装..."
    
    # 验证NVIDIA驱动
    if nvidia-smi > /dev/null 2>&1; then
        log_info "✅ NVIDIA驱动验证成功"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    else
        log_error "❌ NVIDIA驱动验证失败"
        return 1
    fi
    
    # 验证CUDA
    if nvcc --version > /dev/null 2>&1; then
        log_info "✅ CUDA验证成功"
        nvcc --version | grep "release"
    else
        log_warn "⚠️ CUDA验证失败，可能需要重新登录或重启"
    fi
    
    # 验证Docker
    if docker --version > /dev/null 2>&1; then
        log_info "✅ Docker验证成功"
        docker --version
    else
        log_error "❌ Docker验证失败"
        return 1
    fi
    
    # 验证Docker GPU支持
    if docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        log_info "✅ Docker GPU支持验证成功"
    else
        log_error "❌ Docker GPU支持验证失败"
        return 1
    fi
    
    log_info "所有组件验证完成！"
}

# 显示帮助信息
show_help() {
    echo "GPU环境配置脚本"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示帮助信息"
    echo "  -d, --driver-only       仅安装NVIDIA驱动"
    echo "  -c, --cuda-only         仅安装CUDA Toolkit"
    echo "  -k, --docker-only       仅安装Docker"
    echo "  -t, --toolkit-only      仅安装NVIDIA Container Toolkit"
    echo "  -v, --verify-only       仅验证安装"
    echo "  --driver-version VER    指定驱动版本 (默认: $NVIDIA_DRIVER_VERSION)"
    echo "  --cuda-version VER      指定CUDA版本 (默认: $CUDA_VERSION)"
    echo ""
    echo "示例:"
    echo "  $0                      # 完整安装"
    echo "  $0 -d                   # 仅安装驱动"
    echo "  $0 --driver-version 535 # 安装指定版本驱动"
}

# 主函数
main() {
    local install_driver=true
    local install_cuda=true
    local install_docker=true
    local install_toolkit=true
    local verify_only=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--driver-only)
                install_cuda=false
                install_docker=false
                install_toolkit=false
                shift
                ;;
            -c|--cuda-only)
                install_driver=false
                install_docker=false
                install_toolkit=false
                shift
                ;;
            -k|--docker-only)
                install_driver=false
                install_cuda=false
                install_toolkit=false
                shift
                ;;
            -t|--toolkit-only)
                install_driver=false
                install_cuda=false
                install_docker=false
                shift
                ;;
            -v|--verify-only)
                verify_only=true
                shift
                ;;
            --driver-version)
                NVIDIA_DRIVER_VERSION="$2"
                shift 2
                ;;
            --cuda-version)
                CUDA_VERSION="$2"
                shift 2
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查是否为root用户
    if [[ $EUID -eq 0 ]]; then
        log_error "请不要以root用户运行此脚本"
        exit 1
    fi
    
    # 检测操作系统
    detect_os
    
    if [[ "$verify_only" == "true" ]]; then
        verify_installation
        exit 0
    fi
    
    log_info "开始GPU环境配置..."
    
    # 按顺序安装组件
    if [[ "$install_driver" == "true" ]]; then
        install_nvidia_driver
    fi
    
    if [[ "$install_cuda" == "true" ]]; then
        install_cuda
    fi
    
    if [[ "$install_docker" == "true" ]]; then
        install_docker
    fi
    
    if [[ "$install_toolkit" == "true" ]]; then
        install_nvidia_container_toolkit
    fi
    
    # 验证安装
    log_info "安装完成，开始验证..."
    if verify_installation; then
        log_info "🎉 GPU环境配置成功！"
        log_warn "请重新登录或重启系统以确保所有环境变量生效"
    else
        log_error "验证失败，请检查安装日志"
        exit 1
    fi
}

# 执行主函数
main "$@"