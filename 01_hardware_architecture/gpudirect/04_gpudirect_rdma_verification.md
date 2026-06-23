# GPUDirect RDMA 跨节点验证

## 概述

验证两台 H100 服务器之间的 GPUDirect RDMA 可用性，为 PD 分离（Prefill-Decode Disaggregation）提供硬件基础确认。

## 节点配置

两台服务器硬件基本相同，唯一差异是 GPU 数量和容器运行时：

| 组件           | 配置                                           |
| -------------- | ---------------------------------------------- |
| GPU            | 8× / 7× H100 80GB                              |
| CPU            | 2× Xeon Platinum 8468                          |
| 内存           | 2 TB                                           |
| NIC            | 8× Mellanox ConnectX-7 (HDR 200 Gb/s × 8 端口) |
| NVLink         | 18× 26.5 GB/s                                  |
| GPU↔NIC 拓扑   | GPU0↔NIC0 PIX（同一 PCIe switch）              |
| CUDA           | 12.8                                           |
| nvidia-peermem | ✅                                             |
| 容器运行时     | Docker / containerd+nerdctl                    |

## 验证步骤

### 1. 前置检查

```bash
# nvidia-peermem 内核模块（RDMA 注册 GPU 内存必需）
lsmod | grep nvidia_peermem

# IB 设备状态
ibv_devinfo mlx5_0 | grep -E "state|link_layer"

# GPU↔NIC 拓扑（PIX = 同一 PCIe switch，最优）
nvidia-smi topo -m | grep "GPU.*NIC"
```

### 2. IPoIB 配置

IB 接口默认无 IP，NCCL 控制面握手需要 IP：

```bash
# Server A（192.168.1.10）
ip addr add 10.0.0.2/24 dev ibp25s0f0 && ip link set ibp25s0f0 up

# Server B（192.168.1.11）
ip addr add 10.0.0.1/24 dev ibp25s0f0 && ip link set ibp25s0f0 up
```

### 3. IB 物理层带宽验证（ib_write_bw）

```bash
# Server B
ib_write_bw -d mlx5_0 -a -F --report_gbits -n 1000 -s 65536

# Server A
ib_write_bw -d mlx5_0 -a -F --report_gbits -n 1000 -s 65536 192.168.1.11
```

结果：**196 Gb/s (24.5 GB/s)**，HDR 满速。

### 4. NCCL 跨节点 GPU-to-GPU（核心验证）

关键配置：

| 参数                       | 值          | 说明                  |
| -------------------------- | ----------- | --------------------- |
| `NCCL_SOCKET_IFNAME`       | `ibp25s0f0` | 指定 IB 接口          |
| `NCCL_IB_CUDA_SUPPORT`     | `1`         | 启用 GPUDirect        |
| `--privileged`             | ✅          | GPU 内存注册到 IB HCA |
| `--network host`           | ✅          | 使用主机网络栈        |
| `--device /dev/infiniband` | ✅          | 挂载 IB 设备          |
| `--ulimit memlock=-1`      | ✅          | RDMA 内存锁定         |

测试脚本（512MB tensor, all_reduce）：

```python
import os, time
import torch, torch.distributed as dist

r = int(os.environ['RANK'])
torch.cuda.set_device(0)
dist.init_process_group(backend='nccl',
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    world_size=2, rank=r)

t = torch.ones(134217728, dtype=torch.float32, device='cuda:0')
# 预热
for _ in range(3): dist.all_reduce(t, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
# 计时
t0 = time.perf_counter()
for _ in range(10): dist.all_reduce(t, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

# 2 节点 ring all_reduce: Bus BW = 2 × 单向带宽
bus_bw = (t.numel() * 4 * 2 * 1 / 2) / (elapsed / 10) / 1e9
print(f'GPUDirect RDMA PASSED - Bus BW: {bus_bw:.1f} GB/s')
dist.destroy_process_group()
```

运行（两侧使用各自容器运行时）：

```bash
# Master (Server B — containerd+nerdctl)
nerdctl run --rm --gpus 1 --network host --privileged --ulimit memlock=-1 \
  -e MASTER_ADDR=10.0.0.1 -e MASTER_PORT=29510 -e WORLD_SIZE=2 -e RANK=0 \
  -e NCCL_SOCKET_IFNAME=ibp25s0f0 -e NCCL_IB_CUDA_SUPPORT=1 \
  -v /tmp/test.py:/tmp/test.py:ro --entrypoint python3 \
  lmcache/vllm-openai:v0.4.7-cu129 /tmp/test.py

# Worker (Server A — docker)
docker run --rm --gpus all --network host --privileged --ulimit memlock=-1 \
  -e MASTER_ADDR=10.0.0.1 -e MASTER_PORT=29510 -e WORLD_SIZE=2 -e RANK=1 \
  -e NCCL_SOCKET_IFNAME=ibp25s0f0 -e NCCL_IB_CUDA_SUPPORT=1 \
  -v /tmp/test.py:/tmp/test.py:ro --entrypoint python3 \
  lmcache/vllm-openai:v0.4.7-cu129 /tmp/test.py
```

### 验证结果

| 测试                  | 带宽                     | 说明                           |
| --------------------- | ------------------------ | ------------------------------ |
| `ib_write_bw` (host)  | **196 Gb/s (24.5 GB/s)** | 单端口 HDR 满速                |
| NCCL all_reduce (GPU) | **Bus BW: 45.8 GB/s**    | 2 节点 ring，等价单向 ~23 GB/s |

> 2 节点 ring all-reduce 下 Bus BW = 2 × 单向带宽（每个节点同时收发）。实际单向 ~23 GB/s，达到单端口 HDR 200 Gb/s 理论极限。瓶颈不在 IB 端口数（共 8×200 Gb/s），而在 GPU↔NIC 的 PCIe Gen5 x16 ≈ 64 GB/s。

## 踩坑记录

| #   | 现象                                  | 原因                                | 修复                                           |
| --- | ------------------------------------- | ----------------------------------- | ---------------------------------------------- |
| 1   | `ib_write_bw --use_cuda` 不可用       | perftest 未链接 CUDA                | 用 host 内存测物理层，GPU 层面用 NCCL 验证     |
| 2   | `Failed to initialize NET plugin`     | 容器未挂载 IB 设备                  | `--device /dev/infiniband --ulimit memlock=-1` |
| 3   | `Network is unreachable`              | IB 接口无 IP                        | 配置 IPoIB                                     |
| 4   | `vendor err 81` (Remote Access Error) | 缺 `--privileged`，GPU 内存注册失败 | 加 `--privileged`                              |
| 5   | 两侧容器运行时不一致                  | 一侧 Docker，一侧 containerd        | 命令格式略有差异，功能等价                     |

## 结论

两台 H100 节点之间的 GPUDirect RDMA 可用，单向带宽 ~23 GB/s，满足 PD 分离场景的 KV cache 跨节点传输需求。
