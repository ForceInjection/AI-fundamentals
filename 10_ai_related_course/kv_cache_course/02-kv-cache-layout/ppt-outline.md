# 第 2 讲：KV Cache 怎么放 — PPT 大纲

## 设计说明

- **总页数**：约 20 页
- **演讲时长**：120 分钟（30 min 概念 + 45 min 公式/演示 + 35 min 实验 + 10 min 答疑）
- **建议风格**：深蓝 + 暖灰；逻辑 token 用蓝色，物理 block 用橙色，碎片用红色标记
- **字体**：标题 32–36 pt，正文 18–22 pt，等宽字体用于公式、block table 和命令
- **教学策略**：先画一条 token → block → slot 的映射，再算碎片，最后跑脚本验证
- **可视化策略**：
  - 逻辑序列与物理池用一张“连续 token 条形图 → 离散 block 方格”对照
  - block table 和 slot mapping 用同一组 10-token 例子贯穿
  - 实验页直接用 `code/block_table_demo.py` 的输出，不另造一套数字
- **动手环节**：第 15–17 页，5 min 自检 + 30 min 任务

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：KV Cache 怎么放
- **副标题**：从 token 到 block / page / slot
- **课程信息**：KV Cache 八讲 · 第 2 讲 · 120 分钟
- 日期、讲师信息
- **一张图**：左侧一条连续的 token 序列，右侧一组离散的物理 block 方格

### 第 2 页：本讲议程（标注动手环节）

- **第一部分**：从 token 到 block（30 min）— 第 3–8 页
  - 逻辑序列 / 术语 / block table / slot mapping / 两种池形状 / 容量公式
- **第二部分**：碎片、共享与 block 大小（45 min）— 第 9–14 页
  - 三种浪费 / A/B 对照 / 10-token 例子 / block 增长 / prefix 共享 / block_size 取舍
- **第三部分**：动手实验（35 min）— 第 15–17 页
  - 5 min 自检 + 30 min 任务
- **总结与答疑**：10 min — 第 18–20 页
- 本讲只回答三个问题：block/page/slot 是什么；一个 token 怎么定位；分页省了什么

---

## 第一部分：从 token 到 block（30 min，第 3–8 页）

### 第 3 页：逻辑序列与物理 KV 池

- 请求看到的是连续 token：`0, 1, 2, ..., T−1`
- 池里实际存放的是一组固定大小的 block
- block table 负责把“逻辑块”映射到“物理块”

  ```text
  逻辑：  [ block 0 ][ block 1 ][ block 2 ]
              │          │          │
              ▼          ▼          ▼
  物理：  [ block 7 ][ block 2 ][ block 5 ]
  ```

- 逻辑连续不等于物理连续；分页把两者解耦
- **一句话**：分页改变的是布局，不是 KV 总量

### 第 4 页：术语表 — 先把名字分清

| 术语 | 含义 | 容易混的点 |
| --- | --- | --- |
| block / page | 固定大小的 KV 存储单元 | vLLM 多叫 block，论文/部分系统叫 page |
| block_size | 一个 block 装多少 token | 作业用 4；nano-vllm 默认 256；vLLM 默认 16 |
| logical block index | 请求内部第几个 block | `token_index // block_size` |
| physical block id | 池里的实际块号 | 不要求连续 |
| slot | 一个 token 在池里的位置 | `block_id × block_size + offset` |
| slot mapping | 当前 batch 每个 token 的 slot 列表 | 不是 block table |
| block table | 请求的逻辑块 → 物理块 | 每个请求一张 |
| ref_count | 物理块被几个请求引用 | 共享前缀时大于 1 |

- `num_blocks` 有两个含义：请求需要多少块；池里总共有多少块
- **检查**：block table 是“块 → 块”，slot mapping 是“token → 槽位”

### 第 5 页：block table 与 slot mapping

- 设 `block_size=4`，请求的 block table 为 `[7, 2, 5]`
- 逻辑块 0 的 token 0–3 落在物理块 7：`slot = 7 × 4 + offset`

  | logical block | token | physical block | slot |
  | --- | --- | --- | --- |
  | 0 | 0–3 | 7 | 28–31 |
  | 1 | 4–7 | 2 | 8–11 |
  | 2 | 8–9 | 5 | 20–21 |

- `slot_mapping` 是“本步要写的 token → slot”；prefill 和 decode 都按它散写
- nano-vllm 的 `ModelRunner.prepare_prefill` 在 `nanovllm/engine/model_runner.py:153-186` 构造 slot mapping
- **一句话**：block table 负责找块，slot mapping 负责写 token

### 第 6 页：两种 KV 池形状

- nano-vllm：一张大张量

  ```text
  [2, L, num_blocks, block_size, H_kv, D]
   │  │      │           │         │    │
   K/V 层数  池块数     每块 token  KV头  head维
  ```

- vLLM FlashAttention 后端：每层一个张量

  ```text
  [num_blocks, num_kv_heads, block_size, 2 × head_size]
  ```

- 层维度不在同一张量里；不同 backend 的布局可能不同
- **一句话**：先确认引擎的池形状，再谈 block table 的位置

### 第 7 页：block_bytes 与 num_blocks

```text
per_token_bytes = 2 × b_kv × L × H_kv × D
block_bytes     = per_token_bytes × block_size
request_blocks  = ceil(T_total / block_size)
pool_blocks     = available_kv_bytes // block_bytes
```

- 10 token、`block_size=4`：需要 `ceil(10/4)=3` 块
- 2 层、每层每 token 1 KiB：每 token 2 KiB，`block_bytes=8 KiB`
- 3 块分配 24 KiB；有效 20 KiB；内部碎片 4 KiB
- `block_bytes` 不要漏掉公式开头的 `2` 和层数 `L`
- **一句话**：先算一块多大，再算池里能放几块

### 第 8 页：第一部分小结 — 逻辑到物理的三步

- 第一步：`logical = token_index // block_size`
- 第二步：`physical = block_table[logical]`
- 第三步：`slot = physical × block_size + offset`
- 这三步把任意 token 定位到池里的唯一位置
- 下一部分：看分页到底省了什么、没省什么
- **过渡**：能定位之后，才谈碎片和共享

---

## 第二部分：碎片、共享与 block 大小（45 min，第 9–14 页）

### 第 9 页：三种浪费

| 浪费类型 | 定义 | 连续分配 `max_model_len` | 固定 block 池 |
| --- | --- | --- | --- |
| reservation waste | 预留但没用 | `(max_model_len − seq_len) × per_token_bytes` | 不预留，按需分配 |
| internal fragmentation | 分配单元内部没用满 | 不单独统计 | `(blocks × block_size − seq_len) × per_token_bytes` |
| external fragmentation | 空闲总量够，但没有足够大的连续块 | 尾部或空洞小于 `max_model_len` 时出现 | 按构造为 0 |

- 固定 block 池的碎片只发生在最后一个 block 内部
- 连续分配的 reservation waste 和 external fragmentation 是两笔账
- **一句话**：分页消除的是外部碎片，不是内部碎片

### 第 10 页：A/B 碎片对照

- 玩具模型：2 层、每层每 token 1 KiB、每 token 2 KiB、`block_size=4`
- A：连续分配 `max_model_len=16`，4 个 10-token 请求，池 72 token
  - 预留 64 token，有效 40 token
  - reservation waste = 24 token = 48 KiB
  - 尾隙 8 token，不够下一个 16-token 请求 → external = 16 KiB
- B：固定 block 池，72 token = 18 block
  - 每个请求 3 块，4 个请求用 12 块
  - 内部碎片 = 4 请求 × 2 token = 8 token = 16 KiB
  - external = 0，剩余 6 块可服务任何请求
- **一句话**：A 浪费在“预留 + 尾隙”，B 浪费在“最后一个块内部”

### 第 11 页：一个 10-token 的完整例子

- `block_size=4`，物理块 `[7, 2, 5]`，每 token 2 KiB

  ```text
  token 0–3 → logical 0 → physical 7 → slot 28–31
  token 4–7 → logical 1 → physical 2 → slot 8–11
  token 8–9 → logical 2 → physical 5 → slot 20–21
  ```

- 分配 3 块 = 24 KiB
- 有效 KV = 10 token × 2 KiB = 20 KiB
- 内部碎片 = 3×4 − 10 = 2 token = 4 KiB
- **检查**：物理块号不连续，但 slot 仍然唯一
- **一句话**：逻辑块按顺序，物理块可以跳

### 第 12 页：block table 怎么随请求增长

- prefill：一次分配 `ceil(T_prompt / block_size)` 块，写满的块登记前缀哈希
- decode：每步只追加 1 个 token；只有当前块写满时，才申请新块
- block table 的长度随逻辑块数增长，不随 token 逐个增长
- nano-vllm 里对应 `Sequence.num_blocks`、`BlockManager.allocate`、`can_append` / `may_append`
- 释放时按 `ref_count` 回收，不能直接覆盖共享块
- **一句话**：block table 按块增长，slot mapping 按 token 增长

### 第 13 页：prefix 共享与 ref_count

- 两个请求前 6 个 token 相同，`block_size=4`
- 共享以完整 block 为粒度：只共享 block 0（4 个 token）
- block 1 覆盖 token 4–7；第 7 个 token 起不同，整块不能共享
- 共享块 `ref_count=2`，私有块 `ref_count=1`
- 释放：每个请求释放自己的引用，归零后才回收物理块
- **一句话**：前缀共享按块对齐，未满/未对齐的尾部不共享

### 第 14 页：block_size 的取舍

| block_size | 10,000 token 需要 | 最大内部碎片 | 适用 |
| --- | --- | --- | --- |
| 4 | 2,500 块 | 3 token | 手算、教学演示 |
| 16 | 625 块 | 15 token | vLLM 默认 |
| 256 | 40 块 | 255 token | nano-vllm 默认 |

- 小 block：内部碎片少，但 block table 更长、查表和 kernel gather 开销更大
- 大 block：元数据少，但最后一个块浪费更多
- 生产值由引擎、kernel 和负载共同决定；作业的 4 不是生产默认
- **一句话**：block_size 是碎片和元数据之间的交换

---

## 第三部分：动手实验（35 min，第 15–17 页）

### 第 15 页：[动手] 先跑默认映射

- **目标**：用 `code/block_table_demo.py` 打印 token → block → slot、block table、内部碎片和共享前缀
- 只依赖 Python 标准库

  ```bash
  cd 10_ai_related_course/kv_cache_course/02-kv-cache-layout
  python3 code/block_table_demo.py --self-check
  python3 code/block_table_demo.py
  ```

- 默认预期：
  - 10 token、`block_size=4` → 3 块、24 KiB、有效 20 KiB、内部碎片 4 KiB
  - 共享前缀 6 token → 只共享 1 块（4 token），`ref_count=2`
  - A 方案 reservation 48 KiB、external 16 KiB；B 方案 internal 16 KiB、external 0
- 预计 5 min 自检 + 30 min 任务

### 第 16 页：[动手] 三个任务

- 任务 1：改 `--block-size`、`--seq-len`，重算 block 数、slot 和内部碎片
- 任务 2：改 `--prefix-tokens`，观察共享块数和 `ref_count`
- 任务 3：改 `--max-model-len`、`--arena-tokens`，观察 A/B 的 reservation、internal、external 变化

  ```bash
  python3 code/block_table_demo.py --block-size 4 --seq-len 10 --prefix-tokens 6
  python3 code/block_table_demo.py --block-size 8 --seq-len 20 --prefix-tokens 12 --max-model-len 24 --arena-tokens 96
  python3 code/block_table_demo.py --max-model-len 32 --arena-tokens 144
  ```

- 填一张表：请求、block table、slot 范围、分配字节、有效字节、内部碎片、共享块/ref_count
- 产出：一张 token → block → slot 图 + 一句“这组参数浪费在哪”

### 第 17 页：[动手] 结果讨论

- 为什么 10 token 需要 3 块？最后一块只用了 2 个 slot
- 为什么共享 6 个 token 实际只共享 4 个？block 1 里第 7 个 token 不同
- 为什么固定 block 池的 external fragmentation 是 0？
- 可选：跑 nano-vllm 的 `L02_sequence.py`、`L04_block_manager.py` 对照；它们不打印 slot_mapping
- **过渡**：第 3 讲看这套 block table 在 nano-vllm 里怎么被调度和回收

---

## 总结与答疑（10 min，第 18–20 页）

### 第 18 页：本讲小结

- 三步定位：`logical → physical → slot`
- 容量公式：`block_bytes`、`request_blocks`、`pool_blocks`
- 三种浪费：reservation / internal / external
- 分页不减少 KV 总量，只减少浪费并让共享、换出、扩容成为可能
- 课后作业：画出 10-token 请求的 block table 与 slot，并解释共享边界

### 第 19 页：六个常见错误

- 把 `num_blocks`（请求）和 `num_blocks`（池）混用
- 把 `slot` 当成请求内偏移，忘了加 `physical_block_id × block_size`
- 认为固定 block 池没有内部碎片
- 认为固定 block 池还有外部碎片
- `block_bytes` 漏乘 `2` 或层数 `L`
- 以为未满/未对齐的前缀块也能共享

### 第 20 页：下次预告 + 参考资料

- 下次：第 3 讲《KV Cache 在 nano-vllm 里怎么实现》— block manager、前缀哈希、ref_count、preempt/swap
- 课后练习：用自己的 `block_size` 复算一遍 10-token 例子
- 参考资料：
  - [第 2 讲实验脚本](code/block_table_demo.py)
  - [AI-fundamentals KV Cache 技术体系](../../../09_inference_system/kv_cache/README.md)
  - [PagedAttention 技术详解](../../../09_inference_system/kv_cache/01_concepts/basic/paged_attention.md)
  - `~/Project/study/nano-vllm/nanovllm/engine/block_manager.py`
  - `~/Project/study/nano-vllm/docs/llm-inference-visual/scripts/L02_sequence.py`
  - `~/Project/study/nano-vllm/docs/llm-inference-visual/scripts/L04_block_manager.py`
- Q&A：留 3 分钟，重点检查 `num_blocks`、slot、外部碎片、共享粒度
