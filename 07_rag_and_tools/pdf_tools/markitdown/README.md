# MarkItDown：Microsoft 通用文档转 Markdown

MarkItDown 是 Microsoft 开源的「万能格式转 Markdown」工具，覆盖 PDF、PowerPoint、Word、Excel、图片（EXIF + OCR）、音频（EXIF + 语音转写）、HTML、CSV / JSON / XML、ZIP 归档等 9 类输入，是一个典型的 RAG 前置 ETL 组件。本目录在官方用法之外，增补了一份加速国内构建的 Dockerfile。

## 1. 目录内容

- **[markitdown_intro.md](markitdown_intro.md)** — 快速入门：功能介绍、GitHub 地址、Docker 构建与运行命令（含示例 PDF 转换）。
- **[Dockerfile](Dockerfile)** — 本仓库定制版 Dockerfile，替换了官方镜像中的 `apt` 与 `pip` 源为国内镜像，显著加快国内网络下的构建速度。**使用时需将官方 repo 中的 Dockerfile 替换为本目录这一份。**
- **assets/deepseek_v3.pdf** — 用于 Docker 容器内试跑的样例 PDF（DeepSeek-V3 技术报告）。

## 2. 快速上手

```bash
# 克隆官方仓库
git clone https://github.com/microsoft/markitdown.git
cd markitdown

# 用本目录的加速版 Dockerfile 替换官方版
cp /path/to/this/Dockerfile ./Dockerfile

# 构建镜像（使用国内 apt/pip 源）
docker build --network=host -t markitdown:latest .

# 转换本目录的 sample PDF
docker run --network=host --rm -i markitdown:latest < assets/deepseek_v3.pdf > deepseek_v3.md
```

## 3. 相关资源

- [PDF 解析工具总览](../README.md) — MinerU / Marker / MarkItDown 三者的选型对比。
- [MarkItDown 官方仓库](https://github.com/microsoft/markitdown)
