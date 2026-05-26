# 12. 多卡分布式训练 (DDP + HCCL)

在 Ascend NPU 上使用 DDP（Distributed Data Parallel）+ HCCL 进行多卡 LoRA 微调。

## 文件

| 文件                 | 说明                                     |
| -------------------- | ---------------------------------------- |
| `01_ddp_training.md` | DDP 多卡训练文档（原理、实现、训练结果） |
| `ddp_train.py`       | DDP 多卡 LoRA 微调脚本（HCCL 通信）      |

## 关键发现

- 2 卡 DDP 训练正常，HCCL 通信无死锁
- 等效 batch = per_gpu_batch × grad_accum × world_size
- HBM 16.4 GB/卡（与单卡一致），梯度同步开销极低（LoRA 仅 5M 参数）
- `dist.barrier()` 保护 checkpoint 保存，防止通信死锁
