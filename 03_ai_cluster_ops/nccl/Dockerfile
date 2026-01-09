# =============================================================================
# NCCL 测试容器镜像
# 基于 NVIDIA PyTorch 官方镜像 24.10，支持 CUDA、NCCL 和 InfiniBand
# 
# 包含组件版本:
# - Ubuntu 22.04 + Python 3.10
# - NVIDIA CUDA 12.6.2
# - NVIDIA NCCL 2.22.3
# - PyTorch 2.5.0a0+e000cf0ad9
# - NVIDIA cuDNN 9.5.0.50
# - NVIDIA HPC-X 2.20
# - OpenMPI 4.1.7
# =============================================================================

# 使用 NVIDIA 官方 PyTorch 镜像作为基础镜像
ARG PYTORCH_VERSION=24.10
ARG CUDA_VERSION=12.6
ARG UBUNTU_VERSION=22.04

FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3

# 设置构建时环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 设置工作目录
WORKDIR /workspace

# 配置镜像源并安装系统依赖 (单层优化)
RUN set -eux; \
    # 备份原始源
    cp /etc/apt/sources.list /etc/apt/sources.list.bak; \
    # 配置阿里云源 (可通过构建参数覆盖)
    { \
        echo "deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse"; \
        echo "deb https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse"; \
        echo "deb https://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse"; \
        echo "deb https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list; \
    # 更新并安装包
    apt-get update; \
    apt-get install -y --no-install-recommends \
        # 网络工具
        net-tools \
        iproute2 \
        iputils-ping \
        netcat-openbsd \
        # InfiniBand 工具
        infiniband-diags \
        ibverbs-utils \
        librdmacm1 \
        libibverbs1 \
        # 系统监控工具
        htop \
        iotop \
        nvtop \
        # 开发工具
        vim \
        git \
        wget \
        curl \
        # 调试工具
        strace \
        gdb; \
    # 清理
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*; \
    rm -rf /tmp/*

# 复制 Python 依赖文件并安装
COPY requirements.txt /tmp/requirements.txt

RUN set -eux; \
    # 配置 pip 镜像源
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/; \
    pip config set install.trusted-host mirrors.aliyun.com; \
    # 安装 Python 包
    pip install --no-cache-dir -r /tmp/requirements.txt; \
    # 清理
    rm -f /tmp/requirements.txt

# 创建目录结构并复制文件 (单层优化)
RUN set -eux; \
    mkdir -p /workspace/nccl_test /workspace/logs /workspace/scripts

# 复制测试脚本
COPY nccl_python_template.py /workspace/nccl_test/
COPY nccl_benchmark.sh /workspace/nccl_test/

# 设置权限
RUN chmod +x /workspace/nccl_test/*.sh /workspace/nccl_test/*.py

# 设置 NCCL 调试环境变量 (运行时可覆盖)
# 注意: 不设置功能性环境变量，让 nccl_benchmark.sh 脚本完全控制 NCCL 配置
ENV NCCL_DEBUG=INFO \
    NCCL_DEBUG_SUBSYS=INIT,NET

# 设置默认命令
CMD ["/bin/bash"]

# 添加健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())" || exit 1

# 添加标签信息
LABEL maintainer="Grissom" \
      description="NCCL 测试容器" \
      version="2.0" \
      cuda.version="${CUDA_VERSION}" \
      pytorch.version="${PYTORCH_VERSION}" \
      org.opencontainers.image.title="NCCL Test Container" \
      org.opencontainers.image.description="Container for NCCL testing with InfiniBand support" \
      org.opencontainers.image.vendor="AI-Fundamentals" \
      org.opencontainers.image.created="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"