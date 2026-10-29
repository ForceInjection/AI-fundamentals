# 第 2 讲：KV Cache 怎么放

> 120 分钟 | 课程第 2 讲 | 前置：第 1 讲的 token 账

第 1 讲算的是“要存多少”，这一讲回答“存在哪里、怎么找”。核心是把逻辑上的 token 序列映射到物理 block/page，再用 block table 在 decode 时找回历史 K/V。

## 学习目标

- 说清 token、block/page、physical block id、logical block index、block table、slot、slot mapping、ref_count 各自是什么。
- 画出 nano-vllm 的 `[2, L, num_blocks, block_size, H_kv, D]` KV 池，算出单块和整池字节数。
- 用 `block_bytes` 和 `num_blocks` 做容量账，区分“请求需要多少块”和“池里有多少块”。
- 区分 reservation waste、internal fragmentation、external fragmentation，知道固定 block 池的外部碎片按构造为 0。
- 理解 prefix 共享以完整 block 为粒度，以及 `ref_count` 的基本机制。

## 术语表

| 术语                 | 含义                                                                        | 容易混的点                                               |
| -------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------- |
| token                | 模型处理的最小单位                                                          | 一个 token 在每层各产生一份 K、一份 V                    |
| block / page         | 固定大小的 KV 存储单元，装 `block_size` 个 token 的 K/V                     | vLLM 多叫 block，论文和部分系统叫 page，本课程视为同义词 |
| physical block id    | KV 池里的物理下标，`0..num_blocks-1`                                        | 决定 slot 的起始位置                                     |
| logical block index  | 请求内第几个 block，`token_index // block_size`                             | 只是逻辑编号，不直接出现在池里                           |
| block table          | 请求级数组，`block_table[logical_block_index] = physical_block_id`          | 每个请求一张，长度随序列增长                             |
| slot                 | 池内一个 token 的写入位置，`slot = physical_block_id × block_size + offset` | 全池唯一，不是请求内偏移                                 |
| slot mapping         | 本步要写的 token 对应的 slot 列表，长度等于本步 token 数                    | prefill 和 decode 都按它散写；L02/L04 不打印它           |
| ref_count            | 一个物理 block 被多少个请求引用                                             | 共享前缀时大于 1，减到 0 才能回收                        |
| `num_blocks`（请求） | 请求需要的块数，`ceil(seq_len / block_size)`                                | 代码里是 `Sequence.num_blocks`                           |
| `num_blocks`（池）   | 池容量，`available_kv_bytes // block_bytes`                                 | 代码里是 `config.num_kvcache_blocks`，两个含义不要混     |

## 公式与形状

`per_token_bytes` 沿用第 1 讲：`2 × b_kv × L × H_kv × D`。

`block_bytes = 2 × b_kv × L × block_size × H_kv × D`

`num_blocks = available_kv_bytes // block_bytes`

`block_bytes` 乘了 `L`，所以是跨所有层的单块字节数；它等于 `per_token_bytes × block_size`。`available_kv_bytes` 是权重、激活和框架开销之外留给 KV 池的显存。

nano-vllm 的池形状是 `[2, L, num_blocks, block_size, H_kv, D]`，第一维的 2 是 K/V；代码见 `nanovllm/engine/model_runner.py:113-119`。vLLM 的 FlashAttention 后端不用这一张大张量，每层单独一个 `[num_blocks, num_kv_heads, block_size, 2 * head_size]`，层维度不在同一张量里；第 4 讲展开。

## 三种浪费

| 浪费类型               | 定义                                 | 连续分配 `max_model_len`                           | 固定 block 池                                                                       |
| ---------------------- | ------------------------------------ | -------------------------------------------------- | ----------------------------------------------------------------------------------- |
| reservation waste      | 预留了但没用到                       | `(max_model_len − seq_len) × per_token_bytes`      | 不预留，按需分配                                                                    |
| internal fragmentation | 分配单元内部没用满                   | 不单独统计；预留段未用部分已记入 reservation waste | `(blocks × block_size − seq_len) × per_token_bytes`，最多 `block_size − 1` 个 token |
| external fragmentation | 空闲总量够，但没有足够大的连续空闲块 | 尾部或空洞小于 `max_model_len` 时出现              | 按构造为 0：所有空闲 block 都能服务任何请求                                         |

固定 block 池不是“没有碎片”，而是碎片只发生在最后一个 block 内部；外部碎片为 0。连续分配方案里，reservation waste 和 external fragmentation 是两笔账，不能只算其中一笔。

## block_size 的取舍

- 作业里的 `block_size=4` 只为手算，把 token → block → slot 看清楚。
- nano-vllm 默认 256（`nanovllm/config.py:17`，且要求是 256 的倍数），vLLM 默认 16，参考文章里的例子常用 16。
- block 越小，最后一个 block 的内部碎片越少，但同样序列需要更多 block，block table 更长，元数据和查表开销更大。

## 本讲产出

一张 token → block → slot 的映射图，附一个请求的 block table，并写清内部碎片、外部碎片和共享前缀的 `ref_count`。

## 目录

- [课程总览](../README.md)
- [教学大纲](syllabus.md)
- [PPT 提纲](ppt-outline.md)
- [课堂实验](hands-on-exercise.md)
- [课后作业](homework.md)
- [实验环境](lab-environment.md)

## 前后衔接

第 1 讲建立了 token 账；本讲把它变成 page/slot 账。第 3 讲进入 nano-vllm，看这套映射怎么在代码里跑起来。
