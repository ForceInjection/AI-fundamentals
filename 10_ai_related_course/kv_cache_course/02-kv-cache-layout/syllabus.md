# 第 2 讲：KV Cache 怎么放 — 教学大纲

## 基本信息

| 项目     | 内容                                                    |
| -------- | ------------------------------------------------------- |
| 课时     | 120 分钟                                                |
| 先修     | 第 1 讲的 KV 公式与 token 账                            |
| 本讲定位 | 从逻辑 token 序列走到物理 block/page，建立 page/slot 账 |

## 教学目标

- **知识目标**：掌握 KV 池张量形状、block 分配、slot mapping、block table、`ref_count` 和三种浪费。
- **能力目标**：给定 `block_size` 和序列长度，算出需要的 block 数、内部碎片，并说明外部碎片在固定池中为什么为 0。
- **素养目标**：讨论容量时区分“逻辑 token 数”和“物理块数”，也区分请求的 `num_blocks` 和池的 `num_blocks`。

## 时间安排

| 时间   | 环节       | 内容                                                                                                                                                                                                  |
| ------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 30 min | 概念       | 术语表；逻辑序列与物理 KV 池；`block_bytes` 与 `num_blocks` 的两个含义；slot mapping；block table；内部/外部碎片；prefix 共享与 `ref_count`                                                           |
| 45 min | 源码与演示 | 用 `code/block_table_demo.py` 打印 token → block → slot；对照 `Sequence.num_blocks`、`BlockManager.allocate` 和 `ModelRunner.prepare_prefill` 的 slot_mapping；讲清 L02/L04 为什么不打印 slot_mapping |
| 35 min | 实验       | 跑 A/B 碎片对照：连续分配 `max_model_len` vs 固定 block 池；观察共享前缀的 `ref_count`                                                                                                                |
| 10 min | 答疑       | 回答 block 大小取舍、前缀共享边界、碎片统计口径                                                                                                                                                       |

## 重点与难点

| 类型 | 内容                                       | 处理方式                                                                      |
| ---- | ------------------------------------------ | ----------------------------------------------------------------------------- |
| 重点 | token → block → slot 的映射                | 用一个小序列手算，再对照 `block_table_demo.py`                                |
| 重点 | block table 的作用                         | 画两个请求共享前缀时的表项                                                    |
| 难点 | reservation / internal / external 三种浪费 | 用同一组玩具数字做 A/B 对照：A 连续分配 `max_model_len`，B 固定 block 池      |
| 难点 | 固定池的外部碎片为什么是 0                 | 所有空闲 block 大小相同、都能服务任何请求；浪费只剩最后一个 block 的内部碎片  |
| 难点 | prefix 共享后的引用计数                    | 共享以完整 block 为粒度；先讲共享，再讲释放时机，第 3 讲落到 `ref_count`      |
| 难点 | 两种 KV 池形状                             | nano-vllm 把层放进一张张量，vLLM FlashAttention 后端每层一个张量；第 4 讲展开 |

## 本讲产出

一张 token → block → slot 的映射图，标注：

- `block_size`；
- 每个 token 的 `slot`；
- 请求的 `block_table`；
- 内部碎片字节数；
- 共享前缀的 block 编号与 `ref_count`；
- 外部碎片为什么为 0。

## 参考材料

- [第 2 讲实验脚本](code/block_table_demo.py)
- [nano-vllm](https://github.com/ForceInjection/nano-vllm)
- `~/Project/study/nano-vllm/nanovllm/engine/block_manager.py:79-96`（`allocate`）
- `~/Project/study/nano-vllm/nanovllm/engine/model_runner.py:153-186`（prefill 的 `slot_mapping`）
- `~/Project/study/nano-vllm/docs/llm-inference-visual/scripts/L02_sequence.py`
- `~/Project/study/nano-vllm/docs/llm-inference-visual/scripts/L04_block_manager.py`
- [AI-fundamentals KV Cache 技术体系](../../../09_inference_system/kv_cache/README.md)

源码行号按 nano-vllm `b9dd55a` 标注，不同版本会漂移。
