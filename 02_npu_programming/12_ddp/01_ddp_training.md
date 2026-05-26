# 多卡分布式训练 (DDP + HCCL) on Ascend NPU

## 1. 背景

### 1.1 为什么需要多卡训练

之前的 12 个 Phase 全部在单张 NPU（NPU 7）上完成——7B LoRA 单卡 16.4 GB HBM，64 GB 绰绰有余。但遇到以下场景时单卡不够：

- **更大模型**：14B/70B 模型单卡放不下
- **更大 batch**：单卡 batch_size=1 限制梯度稳定性
- **更快训练**：8 卡并行可将训练时间缩短到 ~1/8

DDP（Distributed Data Parallel）是最基础的多卡训练方案——每张卡有完整模型副本，处理不同数据，梯度同步后更新。

### 1.2 HCCL vs NCCL

| 概念       | NVIDIA | Ascend                                             |
| ---------- | ------ | -------------------------------------------------- |
| 集合通信库 | NCCL   | **HCCL** (Huawei Collective Communication Library) |
| 卡间互联   | NVLink | **HCCS** (Huawei Cache Coherence System)           |
| 分布式后端 | `nccl` | **`hccl`**                                         |

HCCL 提供 AllReduce、AllGather、Broadcast、ReduceScatter 等标准集合通信原语。8 张 910B3 通过 HCCS 全互联（每两张卡直连），通信延迟低。

### 1.3 DDP 工作原理

```text
单卡训练流程:
  数据 → forward → loss → backward → optimizer.step()

DDP 训练流程 (2 卡):
  Rank 0:  数据子集A → forward → lossA → backward → gradA ─┐
                                                           ├─ AllReduce(gradA, gradB) → avg → optimizer.step()
  Rank 1:  数据子集B → forward → lossB → backward → gradB ─┘
```

每个 rank 独立计算 forward/backward，得到各自的梯度。AllReduce 将所有 rank 的梯度求和取平均，然后每个 rank 用相同的平均梯度更新参数——保证训练一致。

---

## 2. 实现

### 2.1 分布式初始化

```python
def setup_distributed():
    rank = int(os.environ["RANK"])          # 全局 rank
    world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
    local_rank = int(os.environ["LOCAL_RANK"])  # 节点内 rank

    if world_size > 1:
        dist.init_process_group(backend="hccl")  # 初始化 HCCL
        torch.npu.set_device(local_rank)         # 绑定 NPU

    return rank, world_size, local_rank
```

关键点：

- `RANK`/`WORLD_SIZE`/`LOCAL_RANK` 由 `torch.distributed.run` 自动设置
- `backend="hccl"` 指定使用 HCCL 通信库
- `set_device(local_rank)` 确保每个进程绑定到不同的 NPU（local_rank=0→NPU0, local_rank=1→NPU1）

### 2.2 数据分发

```python
sampler = DistributedSampler(dataset, num_replicas=world_size,
                              rank=rank, shuffle=True)
dataloader = DataLoader(dataset, sampler=sampler, ...)
```

`DistributedSampler` 自动将数据集按 world_size 等分，每个 rank 处理不重叠的子集。每个 epoch 开始时调用 `sampler.set_epoch(epoch)` 打乱数据分布。

等效 batch_size 的计算：

```text
effective_batch = per_gpu_batch × grad_accum × world_size

例如: batch_size=1 × grad_accum=2 × 2 GPUs = 等效 batch=4
```

### 2.3 DDP 包装模型

```python
model = DDP(model, device_ids=[local_rank], output_device=local_rank,
             find_unused_parameters=False)
trainable_model = model.module  # 访问底层非 DDP 模型（用于保存）
```

DDP 包装后：

- `model.forward()` 自动在 backward 时触发梯度 AllReduce
- `find_unused_parameters=False` 提升性能（LoRA 所有参数都参与梯度计算）
- 通过 `model.module` 访问原始模型（保存 checkpoint 时使用）

### 2.4 梯度同步细节

```python
loss.backward()  # ← DDP 在此处自动插入 AllReduce：
                 #   各 rank 的梯度自动求和取平均
```

DDP 使用 `backward` hook 注入梯度同步——对用户代码完全透明。无需手动调用 AllReduce。

### 2.5 checkpoint 保存

```python
# 只在 rank 0 保存，其他 rank 等待
if world_size > 1:
    dist.barrier()  # 所有 rank 到达此点才继续
if rank == 0:
    trainable_model.save_pretrained(save_path)
if world_size > 1:
    dist.barrier()  # rank 0 保存完成后其他 rank 才继续
```

`barrier()` 确保所有 rank 同步——如果 rank 0 保存耗时较长而其他 rank 已进入下一轮训练并开始通信，可能导致死锁。

---

## 3. 训练结果

### 3.1 环境

- 2 × Ascend 910B3, CANN 8.0.1, torch_npu 2.1.0
- Qwen2.5-7B-Instruct + LoRA r=8, BF16
- 380 QA 指令数据, max_length=512

### 3.2 训练指标

| 指标           | 单卡 (基线) | 2 卡 DDP |
| -------------- | ----------- | -------- |
| world_size     | 1           | 2        |
| per_gpu_batch  | 1           | 1        |
| grad_accum     | 4           | 2        |
| **等效 batch** | **4**       | **4**    |
| Epochs         | 2           | 1        |
| Steps (总)     | 91          | 91       |
| HBM/卡         | 16.4 GB     | 16.4 GB  |
| Loss           | ~25→9.1     | ~25→9.1  |
| 训练时间       | ~4 min      | ~2 min   |

### 3.3 关键观察

- **HBM 不变**：DDP 不增加单卡显存——每卡仍是独立的模型副本（16.4 GB）
- **梯度同步开销低**：LoRA 可训练参数仅 5M，梯度数据量极小（5M × 2 bytes = 10 MB），AllReduce 几乎无开销
- **等效 batch 翻倍**：2 卡时相同 grad_accum 下等效 batch_size ×2——梯度方向更稳定
- **通信正常**：训练过程无死锁、无超时，barrier 机制正确工作

---

## 4. 启动命令

### 2 卡训练

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --nproc_per_node=2 --nnodes=1 \
  ddp_train.py --epochs 2 --batch-size 1 --grad-accum 2
```

### 8 卡训练

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run \
  --nproc_per_node=8 --nnodes=1 \
  ddp_train.py --epochs 2 --batch-size 1 --grad-accum 1
```

等效 batch = 1 × 1 × 8 = 8，与单卡 `grad_accum=8` 等效但训练速度 ~8×。

---

## 5. 代码结构

```text
12_ddp/
└── ddp_train.py    # DDP 多卡 LoRA 微调脚本 (~320 行)
    ├── setup_distributed()     — HCCL 初始化 + NPU 绑定
    ├── SFTDataset              — 指令微调数据集（同 lora_finetune.py）
    ├── train()                 — DDP 训练循环
    │   ├── DistributedSampler  — 数据分发
    │   ├── DDP 模型包装         — 自动梯度同步
    │   ├── barrier 同步         — checkpoint 保存保护
    │   └── rank 0 单点日志      — 避免重复输出
    └── cleanup_distributed()   — 销毁进程组
```

---

## 6. 后续扩展

- **8 卡全量训练**：world_size=8，等效 batch 最大可达 8 × 4 = 32
- **FSDP (Fully Sharded Data Parallel)**：将模型参数也分片到多卡，支持 70B+ 模型（单卡放不下完整模型时必需）
- **混合并行**：数据并行 + 模型并行 + Pipeline 并行的组合（如 Megatron-LM 风格）
- **多机多卡**：`--nnodes=2 --node_rank=0/1` 跨服务器训练
- **vLLM-Ascend 推理服务化**：已验证当前环境不支持——vLLM-Ascend 要求 CANN ≥9.0.0 + PyTorch ≥2.10.0，而服务器驱动 24.1.0.3 仅配套 CANN 8.0.1。Docker 容器化（共享主机驱动）同样不可行（NPU 初始化失败 drvRet=87）。需升级主机驱动后才能启用。

## 参考链接

- [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html)
- [HCCL 集合通信库](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha003/apiref/appdevg/aclpythondevg/aclpythondevg_01_012.html)
