FROM python:3.13-slim-bullseye

USER root

# 设置阿里云 APT 源（适用于 Debian/Ubuntu）
RUN echo "deb http://mirrors.aliyun.com/debian bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bullseye-security main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian bullseye-updates main contrib non-free" >> /etc/apt/sources.list

ARG INSTALL_GIT=false
RUN if [ "$INSTALL_GIT" = "true" ]; then \
    apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*; \
    fi

# Runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple --upgrade pip

RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple markitdown

# Default USERID and GROUPID
ARG USERID=10000
ARG GROUPID=10000

USER $USERID:$GROUPID

ENTRYPOINT [ "markitdown" ]