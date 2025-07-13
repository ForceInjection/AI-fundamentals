# 企业级多智能体AI系统 Docker 镜像
FROM python:3.11-slim

# 添加阿里云镜像源
RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ bookworm-backports main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security/ bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    # 添加备用镜像源
    echo "deb https://mirrors.163.com/debian/ bookworm main contrib non-free non-free-firmware" >> /etc/apt/sources.list

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    TZ=Asia/Shanghai

# 更新软件源并安装系统依赖（带重试机制）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates \
    || (apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates) \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 和安装构建工具
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用缓存目录）
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 复制项目文件
COPY . .

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 创建数据目录并设置权限
RUN mkdir -p /app/data/chroma \
    && chown -R app:app /app/data

# 暴露端口
EXPOSE 8000 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; import sys; sys.path.append('/app'); from main import MultiAgentSystem; print('Health check passed')" || exit 1

# 启动命令
CMD ["python", "main.py", "--config", "config.json"]