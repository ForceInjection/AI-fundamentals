# 第 2 讲：走一遍 block table

35 分钟（5 分钟自检 + 30 分钟任务），目标是把逻辑 token 映射到物理 block，并看到三种浪费的差别。主脚本是 `code/block_table_demo.py`，只用 Python 标准库。

## 准备

```bash
cd 10_ai_related_course/kv_cache_course/02-kv-cache-layout
python3 code/block_table_demo.py --self-check
python3 code/block_table_demo.py
```

脚本默认跑三个场景：单请求映射、共享前缀、A/B 碎片对照。也可以显式指定参数：

```bash
python3 code/block_table_demo.py --block-size 4 --seq-len 10 --prefix-tokens 6
```

## 步骤

1. **场景 1**：一个 10-token 请求，`block_size=4`。记录 token → block → slot、block table 和内部碎片；挑一行手算 `slot = physical_block_id × block_size + offset`，和脚本对答案。
2. **场景 2**：第二个请求前 6 个 token 与第一个相同，第 7 个 token 起不同。记录两个请求的 block table 和每个物理块的 `ref_count`，回答“为什么不是共享 6 个 token”。
3. **场景 3**：A/B 对照。
   - A 连续分配 `max_model_len=16`，4 个 10-token 请求，池容量 72 token。算 reservation waste 和 external fragmentation。
   - B 固定 block 池，容量 72 token，即 18 个 block。只算 internal fragmentation，说明 external fragmentation 为什么是 0。
   - 核对：A 的尾隙是 8 token，总和够用，但不是 16 token 的连续段，新请求放不进去，所以记 external fragmentation。
4. **扩展任务**：改 `--block-size 8 --seq-len 20 --prefix-tokens 12 --max-model-len 24 --arena-tokens 96`，重复上面三步，观察 block 数、slot 范围和碎片如何变化。
5. **选做**：跑 nano-vllm 的 `L02_sequence.py` 和 `L04_block_manager.py` 对照 block table。注意这两个脚本不打印 `slot_mapping`；`slot_mapping` 在 `nanovllm/engine/model_runner.py:153-186` 的 prefill 路径里计算。

## 默认预期

| 项目 | 预期 |
| --- | --- |
| 10 token、block_size=4 | 3 块，分配 24 KiB，有效 20 KiB，内部碎片 2 token / 4 KiB |
| 共享前 6 token | 只共享 block 0（4 token），共享块 `ref_count=2` |
| A 连续分配 | reservation waste 48 KiB，external 16 KiB |
| B 固定 block 池 | internal 16 KiB，external 0，剩余 6 块 |

## 产出

下表由学员填写，第一行是示例。玩具模型按 2 层、每层每 token 1 KB 计算，所以每 token KV 为 2 KB，`block_size=4` 时 `block_bytes=8 KB`。

| 请求 | seq_len | block_size | block table | 每 token KV | block 字节 | 分配块数 | 分配字节 | 有效 KV | 内部碎片 | 共享 block / ref_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 10 | 4 | [0, 1, 2] | 2 KB | 8 KB | 3 | 24 KB | 20 KB | 4 KB | 无 |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |

字节数示例：`2 层 × 4 token × 1 KB/层/token = 8 KB/block`；3 块共 24 KB，有效 KV 为 10 token × 2 KB = 20 KB，内部碎片为 4 KB。

A/B 对照另起一张表，这张表也由学员填写：

| 方案 | 池容量 | 已分配 | 有效 KV | reservation waste | internal frag | external frag | 可用余量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A 连续 max_model_len=16 |  |  |  |  |  |  |  |
| B 固定 block 池 |  |  |  |  |  |  |  |

核对：A 的 reservation waste = 4 × (16 − 10) × 2 KB = 48 KB，external = 8 token × 2 KB = 16 KB；B 的 internal = 4 × 2 token × 2 KB = 16 KB，external = 0。

填完后写一句结论：这组分页策略在本例中浪费了多少空间，浪费主要来自 reservation、internal 还是 external。
