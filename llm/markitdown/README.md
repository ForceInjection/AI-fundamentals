# 快速入门

## 介绍

MarkItDown is a utility for converting various files to Markdown (e.g., for indexing, text analysis, etc).
It supports:

- PDF
- PowerPoint
- Word
- Excel
- Images (EXIF metadata and OCR)
- Audio (EXIF metadata and speech transcription)
- HTML
- Text-based formats (CSV, JSON, XML)
- ZIP files (iterates over contents)

## URL

[**markitdown**](https://github.com/microsoft/markitdown)

## 快速运行

**下载源码**：

```bash
git clone https://github.com/microsoft/markitdown.git
```

> **注意**：请使用[`Dockerfile`](Dockerfile) 替换原 `repo` 中的 `Dockerfile`，增加了 `apt` 和 `pip` 的镜像，用于加快构建。

**构建并运行**：

```bash
docker build --network=host -t markitdown:latest .

docker run --network=host  --rm -i markitdown:latest < ~/deepseek_v3.pdf > deepseek_v3.md
```

---
