# 使用官方 Python 3.9 镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到容器的工作目录
COPY token_estimation.py .

RUN mkdir -p /app/models

# 使用阿里云镜像源安装依赖（确保 URL 是有效的）
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip && \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ tiktoken transformers hf_transfer

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME="/app/models"
#ENV HF_ENDPOINT="https://mirroring.huggingface.co"
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 运行主程序
CMD ["python", "token_estimation.py"]