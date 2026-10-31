# 第 3 讲：观察一次 preempt / swap

30 分钟（5 分钟环境检查 + 25 分钟任务）。目标是看清请求被抢占时 KV 去了哪里，以及元数据和真实 token 输出分别由谁验证。

`L03_scheduler.py` 只驱动 `Scheduler` / `BlockManager` 的元数据，不生成 token；它没有单场景开关，§1–§7 会一次跑完，场景 E 和 E2 在脚本后段自动执行。token 一致性由可选的 GPU 脚本 `tests/verify_swap.py` 检查，不在 L03 的范围内。

## 准备

- nano-vllm 本地仓库，HEAD `b9dd55a`；
- 一个 Qwen3-0.6B 的 config 目录（含 `config.json`），设置 `NANOVLLM_MODEL_PATH` 指向它；
- 纯 CPU 部分需要 `transformers`、`xxhash`、`numpy`，不需要模型权重和 CUDA。

## 任务 0：环境检查（5 min）

```bash
cd ~/Project/study/nano-vllm
git rev-parse --short HEAD
python -m pytest tests/ -v
```

当前基线是 43 个 CPU 测试通过；换版本后以全部通过为准，不要硬记数字。如果测试失败，先确认 Python 版本、`transformers`/`xxhash`/`numpy` 是否装好，再继续。

## 任务 1：跑 L03（5 min）

```bash
export NANOVLLM_MODEL_PATH=/path/to/Qwen3-0.6B
PYTHONPATH=. python docs/llm-inference-visual/scripts/L03_scheduler.py "$NANOVLLM_MODEL_PATH"
```

记录四段输出：

- §4 的 decode / preempt 场景；
- §5 的抢占状态机；
- §6 的 BlockManager 元数据往返；
- §7 的场景 E / E2。

## 任务 2：核对 E / E2（5 min）

下表是 L03 §7 的预期行为，运行后逐项核对。脚本参数为 `block_size=256`、两条 256-token 序列、`num_kvcache_blocks=2`。

| 场景 | 输入 | 预期抢占路径 | 预期 block 数/状态 |
| --- | --- | --- | --- |
| E | GPU 块 2，CPU 块 8；A、B 各 256 token，prefill 后各采样 1 个 token（len=257，`num_cached_tokens=256`）；`eos=0` | A 下次 decode 跨块边界，`can_append(A)` 因空闲块为 0 返回 False；抢占队尾 B；`can_swap_out(B)` 为真（CPU 8 ≥ 1，且 B 的块 `ref_count == 1`），走 `swap_out` | 抢占后 B：GPU 0 块、CPU 1 块、`block_table=[]`、`num_cached_tokens=256`、状态 SWAPPED；A：GPU 2 块；`num_swapped_out_blocks=1`。A 命中 EOS 后 B `swap_in`：恢复 1 个 GPU 块，decode 再追加 1 块；`num_swapped_in_blocks=1`，`num_recompute_preemptions=0` |
| E2 | 同 E，但 CPU 块为 0 | 同样的 `can_append(A)` 失败触发抢占；`can_swap_out(B)` 因 CPU 空闲块不足返回 False，退回 `_recompute` | B：GPU 0 块、`block_table=[]`、`num_cached_tokens=0`、状态 WAITING，回到 `waiting`；`blocks_to_swap_out` 为空；`num_recompute_preemptions=1`；不出现 `blocks_to_swap_in` |

两条路径的差别只在 KV 是否保留：E 把 KV 搬到 CPU，`num_cached_tokens` 保留；E2 直接丢弃，回到 waiting 重算。

## 任务 3：填时序表（10 min）

下表由学员填写，前两行给示例，后三行按 L03 输出补全。最后一列只判断元数据和脚本断言是否一致；token 一致性属于可选 GPU 步骤。

| 步骤 | 请求状态 | block 操作 | KV 位置 | 元数据/断言是否一致 |
| --- | --- | --- | --- | --- |
| prefill 完成 | A、B 均为 RUNNING | `allocate` 各分配 1 个 GPU 块 | GPU 2/2，空闲 0 | 一致：`free_block_ids` 为空，两条 `block_table` 长度均为 1 |
| 下一次 decode 调度 | A RUNNING，B SWAPPED | `preempt(B)` → `swap_out` 释放 B 的 GPU 块；A `may_append` 拿到这 1 块 | B 的 KV 在 CPU，A 在 GPU | 一致：`blocks_to_swap_out` 非空，B 的 `block_table` 为空 |
| A 继续 decode | 待填 | 待填 | 待填 | 待填 |
| A 命中 EOS | 待填 | 待填 | 待填 | 待填 |
| B 迁回并继续 decode | 待填 | 待填 | 待填 | 待填 |

表后写一句：这次抢占的代价主要落在显存、CPU-GPU 带宽还是重算上。场景 E 的代价是拷贝带宽，场景 E2 的代价是重算。

## 任务 4：解释结果（5 min）

1. 为什么被抢占的是 B，不是 A？
2. 为什么 `swap_out` 后 `num_cached_tokens` 还保留？
3. 为什么 E2 只能重算？
4. 为什么 `swap_in` 之后要重建前缀哈希链？
5. L03 的模拟场景 D 里“自身抢占”之后还能继续跑；真实 `Scheduler` 会怎样？

## 常见失败

- `NANOVLLM_MODEL_PATH` 指错：必须是一个包含 `config.json` 的目录；L03 §7 会调用 `AutoConfig.from_pretrained`。
- 没有 `transformers`：§7 在构造 `Config` 时就会失败；§1–§6 不需要它。
- 用了别的 nano-vllm 版本：行号和断言可能变化，先确认 `b9dd55a`。
- 把 E2 当成 bug：CPU 块为 0 时本来就不会走 swap，`num_recompute_preemptions=1` 是预期结果。
- 把 L03 的模拟场景 D 当成真实引擎行为：真实 `Scheduler` 在自身抢占后会在 `assert scheduled_seqs` 处失败，脚本里的模拟分支只用于展示抢占逻辑。

## 产出

一张填完的时序表，加一张 E/E2 对照表，并标出每个断言对应的计数器或字段：

- `blocks_to_swap_out` / `blocks_to_swap_in`；
- `num_swapped_out_blocks` / `num_swapped_in_blocks` / `num_recompute_preemptions`；
- `Sequence.status`、`Sequence.block_table`、`Sequence.num_cached_tokens`。

可选 GPU 步骤：`python tests/verify_swap.py <模型权重目录>` 检查 token 一致性（B1 逐字节、B2 单序列逐 token）。这一步需要 CUDA 和权重，不是本讲必做。

`trace_nanovllm.py` 是 swap 之前的参考，不跟踪 `swap_out/swap_in`；它可以用来看早期版本的单路径 trace，但主实验以 L03 为准。
