# 裸金属 CUDA 环境安装指南 (Ubuntu 22.04)

> 本文档基于 Ubuntu 22.04 + RTX 5090 + CUDA 12.8 实际环境编写（2026-11 复核），适用于不依赖 Docker/SLURM 的裸金属安装。驱动与 NCCL 的版本时效性强，具体版本号以官方仓库当前状态为准。如果你是 HPC 集群用户（如 Frontera），请参阅 [05_exer_device_query.md](../../01_hardware_architecture/nvidia/understand_gpu_architecture/05_exer_device_query.md) 中基于 `module load cuda` 的流程。

---

## 1. 背景：为什么需要"裸金属"安装

Docker 和 SLURM 在 AI 基础设施中广泛使用，但它们也引入了额外的抽象层：

- Docker：需要 `nvidia-container-toolkit` 做 GPU 透传，镜像体积动辄 5-20 GB，冷启动拉取镜像需要数分钟。
- SLURM：依赖集群调度器，节点间共享存储，单机使用反而增加配置复杂度。

裸金属安装直接在操作系统上部署驱动和 CUDA Toolkit，省去中间层。适用于：

| 场景                   | 推荐方式                                                 |
| ---------------------- | -------------------------------------------------------- |
| 个人工作站/开发机      | **裸金属安装**（本文）                                   |
| 单 GPU 训练/推理       | 裸金属安装                                               |
| 多 GPU 集群 + 容器编排 | [NVIDIA Container Toolkit](01_nvidia_container_setup.md) |
| HPC 共享集群           | `module load cuda` + SLURM                               |

裸金属安装的核心链路：`内核驱动 → CUDA Toolkit → NCCL → 应用代码`。驱动是底座，CUDA Toolkit 是编译和运行时，NCCL 是多卡通信（单卡场景也建议安装以验证环境完整性）。

---

## 2. 环境验证

本节是装完后的验收清单——按第 3-5 节装完驱动、Toolkit、NCCL 后，逐项执行以下命令验证（每项对应 §1 链路中的一个组件）：

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

## 3. NVIDIA 驱动安装

### 3.1 检查当前状态

```bash
# 检查是否已安装驱动
nvidia-smi

# 检查内核模块
lsmod | grep nvidia

# 查看可用驱动版本
apt search nvidia-driver | grep -E "^nvidia-driver-[0-9]+/"
```

### 3.2 通过 apt 安装

```bash
# 添加 NVIDIA CUDA 仓库（Toolkit 从这里装；仓库内也有驱动，最高到 580 系）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 安装驱动——两个来源任选其一：
# a) Ubuntu 自带仓库（multiverse）：本文环境用的 595 系就来自这里
sudo apt install -y nvidia-driver-595
# b) NVIDIA CUDA 仓库：本体最高 nvidia-driver-580（595+ 只有 pinning 包，无驱动本体）
# sudo apt install -y nvidia-driver-580

# 重启生效
sudo reboot
```

> CUDA 12.8 要求的最低驱动是 570.26；595 是更新的驱动分支，向后兼容旧版 CUDA。CUDA 仓库里的驱动版本落后于 Ubuntu multiverse 时，驱动从 multiverse 装、Toolkit 从 CUDA 仓库装，两者不冲突。

---

## 4. CUDA Toolkit 安装

### 4.1 通过 apt (推荐)

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

### 4.2 多版本共存

CUDA Toolkit 支持多版本安装到不同目录：

```bash
# 各版本路径与符号链接
ls -d /usr/local/cuda*
# /usr/local/cuda-12.8   /usr/local/cuda -> cuda-12.8（符号链接，指向当前默认版本）

# 编译时切换版本
PATH=/usr/local/cuda-12.8/bin:$PATH cmake ..
make -j$(nproc)
```

### 4.3 关键组件说明

| 组件              | 路径                                 | 说明              |
| ----------------- | ------------------------------------ | ----------------- |
| nvcc              | `/usr/local/cuda/bin/nvcc`           | CUDA 编译器       |
| Runtime           | `/usr/local/cuda/lib64/libcudart.so` | CUDA 运行时库     |
| 头文件            | `/usr/local/cuda/include/cuda.h`     | CUDA API 声明     |
| Compute Sanitizer | `/usr/local/cuda/bin/compute-sanitizer` | 内存/线程检查工具 |
| GDS               | `/usr/local/cuda/gds/`               | GPUDirect Storage |

---

## 5. NCCL 安装

```bash
# 安装 NCCL (与 CUDA 12.8 匹配的版本)
sudo apt install -y libnccl2 libnccl-dev

# 验证
ls /usr/lib/x86_64-linux-gnu/libnccl.so*
dpkg -l | grep libnccl
# 预期：libnccl2 版本 ≥ 2.25.1 且后缀为 +cuda12.8。
# apt 默认装仓库最新版（如 2.32.x-1+cuda12.9），不要拿本文的版本号逐字比对；
# 需要锁定版本时写 sudo apt install -y libnccl2=2.25.1-1+cuda12.8
```

---

## 6. 快速参考：监控和性能工具

```bash
# GPU 监控
sudo apt install -y nvtop                # GPU 资源实时监控
sudo apt install -y datacenter-gpu-manager  # DCGM (可选)

# CUDA 开发
sudo apt install -y cuda-command-line-tools-12-8  # nsys, ncu 等性能分析工具
sudo apt install -y cuda-gdb-12-8                 # CUDA 调试器

# 带宽测试工具
git clone https://github.com/NVIDIA/nvbandwidth.git
cd nvbandwidth && sudo ./debian_install.sh   # 官方脚本装 build-essential 与 cmake
cmake . && make -j$(nproc)
```

---

## 7. 注意事项

- 驱动版本必须 ≥ CUDA Toolkit 版本要求。例如，驱动 595 支持 CUDA ≤ 13.2。
- 消费级 GPU（GeForce RTX 系列）不支持 MIG；NVLink 仅 RTX 3090 等少数卡支持，RTX 40/50 系均无。
- 如果后续需要 Docker GPU 支持，需额外安装 NVIDIA Container Toolkit，参考 [01_nvidia_container_setup.md](01_nvidia_container_setup.md)
