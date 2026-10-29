# 第 4 讲：对照 vLLM 的 KV 生命周期

30 分钟，目标是把 nano-vllm 的一条路径按生命周期映射到 vLLM V1，并读懂一次 block table 和 prefix cache 统计。

本讲大部分任务读源码即可完成；跑小模型是可选项，需要 GPU。没有 GPU 时使用 [code/README.md](code/README.md) 里的示例日志和字段表。

## 准备

- vLLM 本地 checkout，HEAD `8b365ff949`，只读，不复制源码到课程仓库；
- 可选 GPU 路径：按 vLLM 官方说明安装，记录实际版本；
- 可选 GPU 路径需要一个小模型和一块显卡；CPU 路径只需要文本编辑器。

## 步骤

1. 按对象图读一遍入口，不按整文件读：

   - `KVCacheManager.get_computed_blocks`（`vllm/v1/core/kv_cache_manager.py:264`）
   - `KVCacheManager.allocate_slots`（同文件 L371）
   - `KVCacheCoordinator.get_num_blocks_to_allocate`（`kv_cache_coordinator.py:169`）
   - `KVCacheCoordinator.allocate_new_computed_blocks`（同文件 L231）
   - `KVCacheCoordinator.allocate_new_blocks`（同文件 L277）
   - `SingleTypeKVCacheManager.allocate_new_blocks`（`single_type_kv_cache_manager.py:366`）
   - `BlockPool.get_new_blocks` / `free_blocks`（`block_pool.py:668` / L776）

2. 跟一次 prefill/admission：从 `get_computed_blocks` 到 `allocate_slots`，记下命中块怎样加引用、新块从哪个队列取、失败时返回什么。

3. 跟一次 decode：确认 `allocate_slots(num_new_tokens=1)` 只在跨块边界时真正取新块，并找到 `SingleTypeKVCacheManager.req_to_blocks` 里对应的那一条。

4. 可选 GPU：用 [code/README.md](code/README.md) 的 instrumentation 片段打印 `input_batch.block_table[0].get_numpy_array()`（MultiGroup 取第 0 组）和 `kv_cache_manager.make_prefix_cache_stats()`。短请求设 `VLLM_LOG_STATS_INTERVAL=1`，让日志按秒输出。

5. 没有 GPU：直接读 [code/README.md](code/README.md) 的示例日志，按字段表解释每一列。

6. 填下面的映射表和记录表。

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

## 产出

一张填完的映射表，附：

- 一次请求的 block table 快照，注明哪些列有效、哪些是 0 填充；
- prefix cache 的 `requests / queries / hits` 和命中率；
- 一句结论：两套实现最大的差异在“查询与分配的组织方式”“驱逐顺序”还是“抢占后 KV 是否保留”。

## 注意

- `get_numpy_array()` 返回的是整块预分配 buffer，未使用的列是 0，不是 `-1`；有效长度看 `num_blocks_per_row`。
- `MultiGroupBlockTable` 按 group 组织，单 group 模型取 `[0]`；混合模型要逐 group 打印。
- `make_prefix_cache_stats()` 会读取并重置统计，正常日志路径也在调用它；instrumentation 只用于实验，不要留在生产代码里。
