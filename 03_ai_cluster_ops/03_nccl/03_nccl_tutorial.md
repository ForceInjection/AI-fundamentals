# NCCL 测试验证工具说明文档

本教程覆盖 NCCL 测试套件的完整使用流程：单节点 → 容器化 → 多节点。理论原理见 [01](01_nccl_theory.md)，基准方法论见 [04](04_nccl_benchmark.md)，Debug 输出解读见 [05](05_nccl_debug_output.md)，通信路径压测见 [06](06_nccl_path_benchmark.md)。

---

## 1. 快速开始

### 1.1 单节点

```bash
# 拓扑检测
./gpu_topology_detector.sh

# 容器化基础测试
./nccl_container_manager.sh --build
./nccl_container_manager.sh --gpus all --size 100M --time 60

# 原生环境（需先安装 PyTorch + NCCL）
torchrun --nproc_per_node=4 nccl_python_template.py --size 256M --iters 20
```

### 1.2 多节点

```bash
# 原生多节点（需 SSH 互信）
# Master
./nccl_multinode_launcher.sh 0 192.168.1.100 --world-size 4 --nproc-per-node 2
# Worker
./nccl_multinode_launcher.sh 1 192.168.1.100 --world-size 4 --nproc-per-node 2

# Kubernetes
cd k8s/ && ./deploy.sh deploy --gpus 4 --test-size 1G --test-duration 120
```

详细部署参数见 [K8s 指南](k8s/README.md)。

---

## 2. 工具与脚本

| 脚本                         | 用途             | 关键参数                                             |
| ---------------------------- | ---------------- | ---------------------------------------------------- |
| `nccl_benchmark.sh`          | 主基准测试       | `--gpus`, `--size`, `--network` (auto/nvlink/ib/pxn) |
| `nccl_container_manager.sh`  | 容器化测试管理   | `--build`, `--gpus`, `--size`, `--time`              |
| `nccl_multinode_launcher.sh` | 原生多节点启动   | `<rank> <master-addr> --world-size --nproc-per-node` |
| `gpu_topology_detector.sh`   | GPU 拓扑检测     | 无参数                                               |
| `nccl_python_template.py`    | PyTorch 测试模板 | `--size`, `--iters`, 通过 torchrun 启动              |

---

## 3. 网络后端选择

NCCL 自动检测（`--network auto`，默认）在大多数场景下是最优选择。手动指定仅在以下情况需要：

| 后端       | `--network` | 适用场景                      |
| ---------- | ----------- | ----------------------------- |
| NVLink     | `nvlink`    | 纯单机 NVSwitch 环境，禁用 IB |
| InfiniBand | `ib`        | 多节点 RDMA                   |
| PXN        | `pxn`       | 多节点 NVLink + IB 组合路径   |
| Socket     | `socket`    | 无 RDMA 硬件的 fallback       |

> PXN 模式下支持三种优化级别：`conservative`（稳定性优先）、`balanced`（默认）、`aggressive`（最大性能）。用 `--optimization` 指定。

---

## 4. 关键环境变量

日常运行只需关注以下 5 个。完整列表见 [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)。

| 变量                 | 作用         | 常用值                                               |
| -------------------- | ------------ | ---------------------------------------------------- |
| `NCCL_DEBUG`         | 日志级别     | `WARN`（日常）/ `INFO`（排查）/ `TRACE`（深层调试）  |
| `NCCL_P2P_LEVEL`     | P2P 策略     | `NVL`（优先 NVLink）/ `PIX`（允许 PCIe P2P）         |
| `NCCL_IB_DISABLE`    | 禁用 IB      | `1`（纯单机）/ `0`（多节点 RDMA）                    |
| `NCCL_SOCKET_IFNAME` | 指定网络接口 | 多节点时必须设（如 `ibp25s0f0`）                     |
| `NCCL_ALGO`          | 强制算法     | 通常不设，让 NCCL 自动选。排查时可设 `Ring` / `Tree` |

---

## 5. 单节点测试详解

### 5.1 基础测试流程

```bash
# 1. 确认 GPU 拓扑
./gpu_topology_detector.sh
# 输出: NVLink / PCIe P2P / SYS 连接矩阵

# 2. 运行 AllReduce 基准
./nccl_container_manager.sh --gpus all --size 100M --time 60

# 3. 对照期望带宽
# NVLink (NVSwitch): > 250 GB/s @ 512MB
# PCIe P2P: ~24 GB/s @ 512MB
# CPU fallback: < 10 GB/s
```

### 5.2 测试 GPU 子集

```bash
# 仅测 GPU 0,1,2,3
./nccl_container_manager.sh --gpus 0,1,2,3 --size 256M --time 30

# 验证 NVSwitch 是否屏蔽 NUMA 差异：测同 NUMA vs 跨 NUMA pair
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 nccl_python_template.py
CUDA_VISIBLE_DEVICES=0,4 torchrun --nproc_per_node=2 nccl_python_template.py
```

---

## 6. 故障排除

### 6.1 常见问题

| 现象                            | 可能原因                          | 排查方向                                                                                                   |
| ------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| bus_bw < 10 GB/s                | 走了 CPU 中转而非 NVLink/PCIe P2P | `nvidia-smi topo -m` 检查连接类型                                                                          |
| 某个 GPU 拖慢全组               | NVLink 故障                       | `nvidia-smi nvlink --status -i <ID>`                                                                       |
| `ncclSystemError`               | GPU 显存不足或驱动不匹配          | `nvidia-smi` 检查空闲显存                                                                                  |
| 多节点 `Network is unreachable` | IB 接口无 IP                      | 配置 IPoIB（参考 [RDMA 验证](../../01_hardware_architecture/gpudirect/04_gpudirect_rdma_verification.md)） |

### 6.2 诊断流程

```text
出现问题 → NCCL_DEBUG=INFO 重跑 → 查 [Debug 输出解读](05_nccl_debug_output.md) §3 速查表
         → 确认 Channel XX via 是 P2P/NVLink 还是 P2P/PCIe
         → 对照 [通信路径压测](06_nccl_path_benchmark.md) 的期望带宽
         → 对照 [基准测试](04_nccl_benchmark.md) 的 A100/H100 实测数据
```

---

## 7. 测试套件

`test/` 目录包含 `nccl_benchmark.sh` 脚本的单元测试，验证参数解析、配置注入、网络后端选择等脚本层面的正确性。

```bash
./test/run_all_tests.sh           # 全部 4 个套件
./test/run_all_tests.sh --quick   # 快速模式
./test/run_all_tests.sh --suite config  # 单个套件
```

| 套件           | 测试内容                                                                |
| -------------- | ----------------------------------------------------------------------- |
| `syntax`       | 脚本语法、参数验证、帮助信息                                            |
| `config`       | 20 项：`set_nccl_config` / `setup_network_config` / `cache_system_info` |
| `optimization` | 三种优化级别（conservative / balanced / aggressive）                    |
| `pxn`          | PXN 模式参数和配置                                                      |

> **注意**：这是脚本单元测试，不验证 NCCL 通信性能。真实带宽测试见 [通信路径逐层压测](06_nccl_path_benchmark.md)。

---

## 8. 参考

- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL 环境变量完整列表](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [nccl-tests GitHub](https://github.com/NVIDIA/nccl-tests)
- [GPUDirect RDMA 技术详解](../../01_hardware_architecture/gpudirect/01_gpudirect_technology.md)
- [NVLink 技术入门](../../01_hardware_architecture/nvlink/nvlink_intro.md)
