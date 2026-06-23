# NCCL 分布式通信

NCCL 的性能直接决定多机多卡训练的扩展效率。本目录覆盖从理论到压测的完整链路：算法原理 → 安装验证 → 基准测试 → Debug 排障 → 通信路径逐层压测，支持容器化和 Kubernetes 两种部署方式。

## 文档

| #   | 文档                                            | 内容                                                      |
| --- | ----------------------------------------------- | --------------------------------------------------------- |
| 01  | [NCCL 技术理论](./01_nccl_theory.md)            | AllReduce 算法、Ring/Tree 拓扑、RDMA 机制、性能建模       |
| 02  | [单卡验证指南](./02_nccl_helloworld.md)         | 无需多卡的 NCCL 安装验证                                  |
| 03  | [详细使用教程](./03_nccl_tutorial.md)           | 安装、配置、单/多节点测试、容器化部署                     |
| 04  | [基准测试方法论](./04_nccl_benchmark.md)        | `allreduce_perf` 编译运行，A100 实测，MIG+NVLink 异常案例 |
| 05  | [Debug 输出解读](./05_nccl_debug_output.md)     | `NCCL_DEBUG=INFO` 逐段拆解，正常/异常速查表               |
| 06  | [通信路径逐层压测](./06_nccl_path_benchmark.md) | H100 实测：NVLink (316 GB/s) → P2P Disable (24 GB/s)      |
| —   | [Kubernetes 部署](./k8s/README.md)              | K8s 多节点 NCCL 测试方案                                  |

---

## 工具脚本

| 文件                         | 用途                            |
| ---------------------------- | ------------------------------- |
| `nccl_benchmark.sh`          | 主基准测试脚本（NVLink/IB/PXN） |
| `nccl_container_manager.sh`  | 容器化测试管理                  |
| `nccl_multinode_launcher.sh` | 原生多节点启动                  |
| `gpu_topology_detector.sh`   | GPU 拓扑检测                    |
| `nccl_python_template.py`    | PyTorch 分布式测试模板          |

---

## 快速开始

```bash
# 容器化（推荐）
./nccl_container_manager.sh --build
./nccl_container_manager.sh --gpus all --size 100M --time 60

# 拓扑检测
./gpu_topology_detector.sh

# 运行测试套件（验证脚本正确性）
./test/run_all_tests.sh --quick
```

多节点和 K8s 部署见[详细教程](./03_nccl_tutorial.md)和 [K8s 指南](./k8s/README.md)。

---

## 前置条件

- NVIDIA GPU + CUDA 驱动
- Docker + NVIDIA Container Toolkit（容器化部署）
- InfiniBand 或高速以太网（多节点测试）

---

## 参考

- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [GPUDirect RDMA 跨节点验证](../../01_hardware_architecture/gpudirect/04_gpudirect_rdma_verification.md)
