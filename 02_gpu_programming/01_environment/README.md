# GPU 编程环境

本模块讲 GPU 开发环境的搭建，覆盖容器化与裸金属两条路线。

## 1. [NVIDIA GPU 容器环境：原理与构建指南](01_nvidia_container_setup.md)

讲清宿主机与容器各自的职责，以及如何用 NVIDIA Container Toolkit 把 GPU 透传给容器。

## 2. [大模型训练与推理框架的 GPU 镜像构建深度解析](02_cuda_image_build_analysis.md)

对比 vLLM、TGI、Llama.cpp、DeepSpeed 四个项目的 Dockerfile：各自怎么分层、哪里能缓存、哪里容易踩坑。

## 3. [裸金属 CUDA 环境安装指南](03_cuda_metal_install_guide.md)

不依赖 Docker/SLURM 的裸金属路线：Ubuntu 22.04 + RTX 5090 + CUDA 12.8 的驱动、Toolkit、NCCL 安装与验收清单。
