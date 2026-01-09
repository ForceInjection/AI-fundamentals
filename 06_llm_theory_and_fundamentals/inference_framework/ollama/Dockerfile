# 基础镜像，使用 Python 3.10
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 升级 pip 工具
RUN pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip

# 安装相关依赖
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple ollama

# 复制 Python 脚本
COPY benchmark.py .

# 设置默认启动命令）
CMD ["python", "benchmark.py"]
