# GPU 进程与资源管理

> 基于 8 × A100-SXM4-80GB (NVSwitch Gen2) 生产环境实测。GPU 是共享资源——多用户、多任务同时提交时，如何确保正确的人在正确的卡上运行，如何隔离显存、如何避免 OOM、如何排查残留进程。本文从 Compute Mode 到 NUMA 亲和性完整覆盖。

---

## 1. Compute Mode：GPU 的共享策略

GPU 的 Compute Mode 决定多个进程能否同时在该 GPU 上执行计算：

```bash
nvidia-smi --query-gpu=index,name,compute_mode --format=csv
```

当前环境输出（全部 Default）：

```text
0, NVIDIA A100-SXM4-80GB, Default
3, NVIDIA A100-SXM4-80GB, Default
7, NVIDIA A100-SXM4-80GB, Default
```

| Mode                  | 含义                                                  | 适用场景                     |
| --------------------- | ----------------------------------------------------- | ---------------------------- |
| **Default**           | 多个进程可以在同一 GPU 上同时运行 CUDA 代码           | 共享集群、开发环境           |
| **EXCLUSIVE_PROCESS** | 同一时刻只有一个进程的 CUDA context 能在该 GPU 上存在 | 专机专用训练、性能 benchmark |
| **PROHIBITED**        | 该 GPU 禁止任何 CUDA 程序使用                         | 预留 GPU（如留给 MIG 实验）  |
| **EXCLUSIVE_THREAD**  | 已废弃，同 EXCLUSIVE_PROCESS                          | —                            |

```bash
# 设置 GPU 3 为独占模式（需要 root）
nvidia-smi -i 3 -c 1    # 1=EXCLUSIVE_PROCESS

# 恢复默认可共享
nvidia-smi -i 3 -c 0    # 0=Default
```

> **Default 模式的隐患**：两个训练任务同时占用同一 GPU 时不会报错，各自认为自己拥有全部显存，直到 OOM。Default 模式适合开发环境，生产训练建议 EXCLUSIVE_PROCESS。

---

## 2. 查看 GPU 上的进程

### 2.1 nvidia-smi 进程列表

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_name --format=csv
```

当前环境（摘录）：

```text
pid, process_name, used_gpu_memory [MiB], gpu_name
2159821, sglang::scheduler_TP1, 9752 MiB, NVIDIA A100-SXM4-80GB
2160317, sglang::scheduler_TP0, 9752 MiB, NVIDIA A100-SXM4-80GB
2332308, VLLM::EngineCore, 75172 MiB, NVIDIA A100-SXM4-80GB
1105912, ada_be, 4770 MiB, NVIDIA A100-SXM4-80GB
2095211, VLLM::EngineCore, 75260 MiB, NVIDIA A100-SXM4-80GB
```

- sglang 占用 GPU 0/1 各 ~9.5 GB — TP=2 推理
- vllm 占用 GPU 2/6 各 ~75 GB — 大模型推理
- ada_be 占用 GPU 6 的 4.7 GB — 开发环境
- GPU 3/4/5 空闲（仅 4-130 MiB 驱动占用）

### 2.2 fuser：排查残留进程

`nvidia-smi` 列出的都是有进程名的活跃进程。但有时进程已退出、显存却未释放：

```bash
fuser -v /dev/nvidia*
```

正常输出会列出每个 `/dev/nvidia*` 设备文件的打开进程。如果列出的 PID 在 `nvidia-smi` 中看不到 → 僵尸 GPU context（进程退出但 CUDA context 未释放）。

**清理僵尸进程**：

```bash
# 1. 确认进程已退出
ps aux | grep <PID>

# 2. 如果进程已退出但 nvidia-smi 仍显示
#    等待 CUDA context 自动回收（通常 < 30s）
#    或重启 nvidia-persistenced: systemctl restart nvidia-persistenced
```

> **常见场景**：Python 脚本 `Ctrl+C` 中断后，PyTorch 的 CUDA context 可能延迟数秒才释放。在此期间 `nvidia-smi` 仍显示显存占用。

---

## 3. CUDA_VISIBLE_DEVICES：进程级 GPU 选择

这是最常用的 GPU 隔离方式——设置环境变量，让进程只能"看到"指定的 GPU：

```bash
# 只使用 GPU 3
CUDA_VISIBLE_DEVICES=3 python train.py

# 使用 GPU 0,3,5（映射为 device 0,1,2）
CUDA_VISIBLE_DEVICES=0,3,5 python train.py

# 排除 GPU 7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train.py
```

**映射规则**：

```text
物理 GPU:  0  1  2  3  4  5  6  7
设置:      3,5,7
映射后:    ─  ─  ─  0  ─  1  ─  2  ← 进程内的 device ID
                (不可见)
```

**常见用法**：

| 场景                 | 命令                                                       | 说明                                |
| -------------------- | ---------------------------------------------------------- | ----------------------------------- |
| 每个用户分配专属 GPU | `CUDA_VISIBLE_DEVICES=3`                                   | 进程只看得到 GPU 3，映射为 device 0 |
| 训练避开生产 GPU     | `CUDA_VISIBLE_DEVICES=3,4,5`                               | 用空闲的 GPU 3/4/5                  |
| 多 GPU 分布式        | `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4` | 在 4 GPU 上并行                     |
| 排除故障 GPU         | 排除 GPU 7（NVLink 故障）                                  | 避免 NCCL fallback 到 PCIe          |

> **限制**：`CUDA_VISIBLE_DEVICES` 只对单个进程生效，不影响其他用户。多个用户仍可通过 `CUDA_VISIBLE_DEVICES=3` 同时使用同一 GPU（除非设置了 EXCLUSIVE_PROCESS）。

---

## 4. GPU-NUMA 亲和性绑定

A100 通过 PCIe 连接到特定的 NUMA node。如果 GPU 和 CPU 线程不在同一 NUMA node，H2D/D2H 传输需要跨 socket 经 QPI/UPI，延迟显著增加。

### 4.1 查看 GPU 的 NUMA 归属

```bash
nvidia-smi topo -m
```

```text
        GPU0  ...  GPU3  GPU4  ...  GPU7  CPU Affinity    NUMA Affinity
GPU3    ...       X    NV12  ...  NV12  0-23,48-71       0
GPU7    ...      SYS    PXB  ...   X    24-47,72-95      1
```

- GPU 0-3 → NUMA node 0, CPUs 0-23, 48-71
- GPU 4-7 → NUMA node 1, CPUs 24-47, 72-95

### 4.2 NUMA 绑定：taskset + numactl

```bash
# 将进程绑定到 GPU 3 所属的 NUMA node 0 的 CPU
taskset -c 0-23,48-71 CUDA_VISIBLE_DEVICES=3 python train.py

# 或用 numactl
numactl --cpunodebind=0 --membind=0 CUDA_VISIBLE_DEVICES=3 python train.py
```

**为什么重要**：

| 场景                      | 无 NUMA 绑定            | 有 NUMA 绑定 | 影响     |
| ------------------------- | ----------------------- | ------------ | -------- |
| GPU 3 + CPU 线程在 node 1 | H2D 跨 socket (距离 32) | —            | 延迟翻倍 |
| GPU 3 + CPU 线程在 node 0 | ✅ 本地访问 (距离 10)   | 可进一步固化 | 最优     |

```bash
# 查看 NUMA 距离
numactl --hardware
# node 0 → node 1: 32  (跨 socket, 慢)
# node 0 → node 0: 10  (本地, 快)
```

> **经验法则**：单 GPU 推理无需 NUMA 绑定（H2D 只发生一次）。用 `taskset` 绑定；多 GPU 分布式训练建议用 `torchrun` / `mpirun` 的 NUMA 感知启动器。

---

## 5. 显存管理

### 5.1 检查显存使用

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv
```

GPU 3/4/5 的显存状态：

```text
3, 130 MiB, 81790 MiB, 81920 MiB
4, 4 MiB, 81916 MiB, 81920 MiB
5, 4 MiB, 81916 MiB, 81920 MiB
```

- 130 MiB（GPU 3）：之前跑过测试后 CUDA context 可能未完全释放
- 4 MiB（GPU 4/5）：纯空闲，仅驱动开销

### 5.2 PyTorch 显存控制

```python
import torch

# 限制 PyTorch 使用的显存比例
torch.cuda.set_per_process_memory_fraction(0.5)  # 只用 50%

# 或在环境变量中限制
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
```

### 5.3 清理显存

```python
# Python 侧：手动清空 cache
torch.cuda.empty_cache()

# PyTorch DDP 中释放未被引用的梯度
import gc
gc.collect()
torch.cuda.empty_cache()
```

---

## 6. 常见问题与排查

| 问题                   | 现象                                      | 排查                                             | 解决                                                     |
| ---------------------- | ----------------------------------------- | ------------------------------------------------ | -------------------------------------------------------- |
| 显存泄漏               | `nvidia-smi` 显示显存占用但无进程         | `fuser -v /dev/nvidia*` 找僵尸 PID               | `kill -9 <PID>` 或重启 persistenced                      |
| 多用户冲突             | 两个训练任务 OOM                          | `nvidia-smi --query-compute-apps` 看是否同一 GPU | 设置 EXCLUSIVE_PROCESS 或用 `CUDA_VISIBLE_DEVICES`       |
| 跨 NUMA 延迟高         | H2D 带宽远低于预期 (~15 GB/s vs ~28 GB/s) | `numactl --hardware` 看 GPU 和 CPU 的 NUMA 归属  | `taskset` 绑定 CPU 到对应 NUMA node                      |
| CUDA out of memory     | 训练中途 OOM                              | `nvidia-smi` 看剩余显存                          | 减小 batch size / 启用 gradient checkpointing            |
| GPU 3 (130 MiB) 不是 0 | CUDA context 未完全释放                   | `nvidia-smi` 无活跃进程但 `memory.used > 0`      | `nvidia-smi -pm 0 && nvidia-smi -pm 1` 重置 persistenced |

---

## 7. 与已有文档的联动

- [`03_nvidia_smi_guide.md`](03_nvidia_smi_guide.md)：nvidia-smi 基础命令详解
- [`06_gpu_health_check.md`](06_gpu_health_check.md)：进程残留检查是 L1 健康检查的一部分
- [NVLink 诊断与实操](../../01_hardware_architecture/nvlink/nvlink_diagnostics.md)：GPU 7 NVLink 故障 → 建议在分布式训练中排除
- [GPU P2P 带宽实测](../../02_gpu_programming/04_profiling/08_p2p_bandwidth.md)：NUMA 亲和性直接影响 P2P 带宽

## 参考

- [NVIDIA nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/)
- [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)
