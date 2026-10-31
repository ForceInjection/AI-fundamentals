# 第 4 讲：对照 vLLM 的 KV 生命周期

30 分钟（5 分钟准备 + 25 分钟任务）。目标是把 nano-vllm 的一条路径按生命周期映射到 vLLM V1，并读懂一次 block table 和 prefix cache 统计。

本讲大部分任务读源码即可完成；跑小模型是可选项，需要 GPU。没有 GPU 时使用 [code/README.md](code/README.md) 里的示例日志和字段表。

## 任务 0：准备（5 min）

- vLLM 本地 checkout，HEAD `8b365ff949`，只读，不复制源码到课程仓库；
- 先读 [code/README.md](code/README.md) 的字段表和 instrumentation 片段；
- 选择一条轨道：
  - CPU / 只读源码：文本编辑器 + Python 3.10+（可选，用于跑日志解析）；
  - GPU / 跑小模型：CUDA、vLLM 环境、一个小模型。

## 任务 1：对象图与 `allocate_slots`（8 min）

按对象图读入口，不按整文件读：

- `KVCacheManager.get_computed_blocks`（`vllm/v1/core/kv_cache_manager.py:264`）
- `KVCacheManager.allocate_slots`（同文件 L371）
- `KVCacheCoordinator.get_num_blocks_to_allocate`（`kv_cache_coordinator.py:169`）
- `KVCacheCoordinator.allocate_new_computed_blocks`（同文件 L231）
- `KVCacheCoordinator.allocate_new_blocks`（同文件 L277）
- `SingleTypeKVCacheManager.allocate_new_blocks`（`single_type_kv_cache_manager.py:366`）
- `BlockPool.get_new_blocks` / `free_blocks`（`block_pool.py:668` / L776）

写下 prefill/admission 的调用顺序，并标出哪一步可能返回 `None`。

## 任务 2：prefix 命中与 BlockPool（7 min）

读下面几处，回答三个问题：

- `get_computed_blocks` 里的 `max_cache_hit_length` 为什么是 `request.num_tokens - 1`？
- `find_longest_cache_hit` 命中后，`add_local_computed_blocks` 为什么先 touch 命中块？
- `free_blocks` 为什么把非缓存块前插、缓存块后插？

把 BlockPool 的三种状态填进下表：

| 状态 | 判据 | 空闲队列位置 | 分配/回收行为 |
| --- | --- | --- | --- |
| 请求持有 |  |  |  |
| 缓存可驱逐 |  |  |  |
| 未缓存空闲 |  |  |  |

## 任务 3：block table 与日志（5 min）

无 GPU 时先跑解析脚本：

```bash
cd 10_ai_related_course/kv_cache_course/04-vllm-kv-cache
python3 code/parse_stats_log.py --demo
```

应看到三行示例记录：两行 Engine 统计和一行 `[prefix]` 统计。解释 `requests / queries / hits` 和命中率。

可选 GPU 路径按 [code/README.md](code/README.md) 第二节打印 `input_batch.block_table[0].get_numpy_array()`，并记录有效列和 0 填充。

## 任务 4：映射表与结论（5 min）

填完下面的映射表，并写一句“两套实现最大的差异”。

## 映射表

下表由学员填写；第一行是示例，其余三行按阅读结果补全。

| 生命周期 | nano-vllm | vLLM | 作用 | 差异 |
| --- | --- | --- | --- | --- |
| prefill / admission | `can_allocate` + `allocate` | `get_computed_blocks` + `allocate_slots` | 查前缀、分配首批块 | 示例：nano 先判断再分配，vLLM 在 `allocate_slots` 内一次完成，失败返回 `None` |
| decode 增长 | `can_append` + `may_append` | `allocate_slots(num_new_tokens=1)` | 跨块边界时追加块 | 待填 |
| 释放 | `deallocate` | `free` | 归还块 | 待填 |
| 缓存登记 | `hash_blocks` | `cache_blocks` | 把满块登记进前缀缓存 | 待填 |
| 末块 | 满的末块也不复用 | `max_cache_hit_length = request.num_tokens - 1` | 保证最后 token 重算 | nano-vllm 连满的末块也不进复用；vLLM 最后 token 必重算，按块对齐后可能整块重算 |

## 实验记录表

下表由学员填写，第一行是示例：

| 观察点 | 记录值 | 解释 |
| --- | --- | --- |
| 请求的 block table 行 | 例如 `[12, 13, 0, 0]`，有效长度 2 | 第一个请求在 batch 的第 0 行，`get_numpy_array()[0]` 的前 `num_blocks_per_row[0]` 列有效，其余列是 0 填充 |
| prefix cache 统计 | 例如 `requests=2, queries=64, hits=48` | `hits / queries` 是命中率；只在 `log_stats` 打开时有值 |
| prefill 到 decode 的块增长 | 待填 | 待填 |
| 请求结束后的块状态 | 待填 | 待填 |

## 常见失败

- `make_prefix_cache_stats()` 返回 `None`：`LLM` 入口默认关闭统计，需要 `LLM(..., disable_log_stats=False)`。
- 把 block table 里的 0 当成有效块：0 是 null block，有效长度看 `num_blocks_per_row`。
- 单 group 模型只打印 `block_table[0]`；混合模型要逐 group 打印。
- 连续调用两次 `make_prefix_cache_stats()`：第一次读取并重置，第二次可能为空。
- 没有 GPU 却把示例日志当成实测：示例只用于理解字段，结论要标明来源。

## 产出

一张填完的映射表，附：

- 一次请求的 block table 快照，注明哪些列有效、哪些是 0 填充；
- prefix cache 的 `requests / queries / hits` 和命中率；
- 一句结论：两套实现最大的差异在“查询与分配的组织方式”“驱逐顺序”还是“抢占后 KV 是否保留”。

## 注意

- `get_numpy_array()` 返回的是整块预分配 buffer，未使用的列是 0，不是 `-1`；有效长度看 `num_blocks_per_row`。
- `MultiGroupBlockTable` 按 group 组织，单 group 模型取 `[0]`；混合模型要逐 group 打印。
- `make_prefix_cache_stats()` 会读取并重置统计，正常日志路径也在调用它；instrumentation 只用于实验，不要留在生产代码里。
