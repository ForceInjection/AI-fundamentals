# GPU 编程环境

本模块聚焦于现代 GPU 开发与部署的基础环境构建。主要介绍基于容器化的 NVIDIA GPU 环境配置原理，以及针对各种大模型框架的 CUDA 镜像构建实践。

## 1. [NVIDIA GPU 容器环境：原理与构建指南](01_nvidia_container_setup.md)

深入讲解宿主机与容器的职责分离，详细介绍如何通过 NVIDIA Container Toolkit 配置和管理 GPU 容器运行环境。

## 2. [大模型训练与推理框架的 GPU 镜像构建深度解析](02_cuda_image_build_analysis.md)

通过对 vLLM、TGI、Llama.cpp 以及 DeepSpeed 等开源框架 Dockerfile 的深度剖析，揭示在不同场景下如何构建与优化 CUDA 容器镜像。
