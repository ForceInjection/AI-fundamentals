# 裸金属 CUDA 环境安装指南 (Ubuntu 22.04)

> 本文档基于 Ubuntu 22.04 + RTX 5090 + CUDA 12.8 实际环境编写，适用于不依赖 Docker/SLURM 的裸金属安装。如果你是 HPC 集群用户（如 Frontera），请参阅 [05_exer_device_query.md](../../01_hardware_architecture/nvidia/understand_gpu_architecture/05_exer_device_query.md) 中基于 `module load cuda` 的流程。

---

## 0. 背景：为什么需要"裸金属"安装

Docker 和 SLURM 在 AI 基础设施中广泛使用，但它们也引入了额外的抽象层：

- **Docker**：需要 `nvidia-container-toolkit` 做 GPU 透传，镜像体积动辄 5-20 GB，冷启动拉取镜像需要数分钟。
- **SLURM**：依赖集群调度器，节点间共享存储，单机使用反而增加配置复杂度。

裸金属安装直接在操作系统上部署驱动和 CUDA Toolkit，省去中间层。适用于：

| 场景 | 推荐方式 |
|------|----------|
| 个人工作站/开发机 | **裸金属安装**（本文） |
| 单 GPU 训练/推理 | 裸金属安装 |
| 多 GPU 集群 + 容器编排 | [NVIDIA Container Toolkit](01_nvidia_container_setup.md) |
| HPC 共享集群 | `module load cuda` + SLURM |

裸金属安装的核心链路：`内核驱动 → CUDA Toolkit → NCCL → 应用代码`。驱动是底座，CUDA Toolkit 是编译和运行时，NCCL 是多卡通信（单卡场景也建议安装以验证环境完整性）。

---

## 1. 环境验证

安装完成后，逐项执行以下命令验证。每一项覆盖链路中的一个关键组件：

```bash
# 1. 驱动状态
nvidia-smi

# 2. CUDA 编译器
nvcc --version

# 3. CUDA Runtime 库
ls /usr/local/cuda/lib64/libcudart.so*

# 4. NCCL 库
dpkg -l | grep libnccl

# 5. 编译 CUDA 程序测试
cat > /tmp/hello.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int count;
    cudaGetDeviceCount(&count);
    printf("CUDA devices: %d\n", count);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute: %d.%d | SMs: %d | Mem: %.1f GB\n",
        prop.major, prop.minor, prop.multiProcessorCount,
        (float)prop.totalGlobalMem / (1024*1024*1024));
    return 0;
}
EOF
nvcc -o /tmp/hello /tmp/hello.cu && /tmp/hello
```

---

## 2. NVIDIA 驱动安装

### 2.1 检查当前状态

```bash
# 检查是否已安装驱动
nvidia-smi

# 检查内核模块
lsmod | grep nvidia

# 查看可用驱动版本
apt search nvidia-driver | grep -E "^nvidia-driver-[0-9]+/"
```

### 2.2 通过 apt 安装

```bash
# 添加 NVIDIA 官方仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 安装驱动（以 595 版本为例）
sudo apt install -y nvidia-driver-595

# 重启生效
sudo reboot
```

---

## 3. CUDA Toolkit 安装

### 3.1 通过 apt (推荐)

```bash
# 安装 CUDA Toolkit 12.8
sudo apt install -y cuda-toolkit-12-8

# 设置环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

### 3.2 多版本共存

CUDA Toolkit 支持多版本安装到不同目录：

```bash
# 各版本路径
ls /usr/local/cuda-*/
# /usr/local/cuda-12.8  /usr/local/cuda-12  (符号链接)

# 编译时切换版本
PATH=/usr/local/cuda-12.8/bin:$PATH cmake ..
make -j$(nproc)
```

### 3.3 关键组件说明

| 组件              | 路径                                 | 说明              |
| ----------------- | ------------------------------------ | ----------------- |
| nvcc              | `/usr/local/cuda/bin/nvcc`           | CUDA 编译器       |
| Runtime           | `/usr/local/cuda/lib64/libcudart.so` | CUDA 运行时库     |
| 头文件            | `/usr/local/cuda/include/cuda.h`     | CUDA API 声明     |
| Compute Sanitizer | `/usr/local/cuda/compute-sanitizer/` | 内存/线程检查工具 |
| GDS               | `/usr/local/cuda/gds/`               | GPUDirect Storage |

---

## 4. NCCL 安装

```bash
# 安装 NCCL (与 CUDA 12.8 匹配的版本)
sudo apt install -y libnccl2 libnccl-dev

# 验证
ls /usr/lib/x86_64-linux-gnu/libnccl.so*
dpkg -l | grep libnccl
# 预期输出: libnccl2 2.25.1-1+cuda12.8
```

---

## 5. 快速参考：监控和性能工具

```bash
# GPU 监控
sudo apt install -y nvtop                # GPU 资源实时监控
sudo apt install -y datacenter-gpu-manager  # DCGM (可选)

# CUDA 开发
sudo apt install -y cuda-command-line-tools-12-8  # nsys, ncu 等性能分析工具
sudo apt install -y cuda-gdb-12-8                 # CUDA 调试器

# 带宽测试工具
git clone https://github.com/NVIDIA/nvbandwidth.git
cd nvbandwidth && apt install libboost-program-options-dev
cmake . && make -j$(nproc)
```

---

## 注意

- 驱动版本必须 >= CUDA Toolkit 版本要求。例如，驱动 595 支持 CUDA ≤ 13.2
- 消费级 GPU (GeForce RTX 系列) 不支持 NVLink 和 MIG
- 如果后续需要 Docker GPU 支持，需额外安装 NVIDIA Container Toolkit，参考 [01_nvidia_container_setup.md](01_nvidia_container_setup.md)
