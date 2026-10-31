# 第 4 讲：KV Cache 在 vLLM 里怎么调度与回收 — 教学大纲

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 课时 | 120 分钟 |
| 先修 | 第 3 讲的 block 生命周期；了解 vLLM 基本请求流程 |
| 代码基线 | 本地 vLLM，HEAD `8b365ff949` |
| 本讲定位 | 把 nano-vllm 的 KV 路径映射到 vLLM V1 的分层管理与回收 |

## 教学目标

- **知识目标**：理解 `KVCacheManager`、`KVCacheCoordinator`、`SingleTypeKVCacheManager`、`BlockPool`、block table、prefix cache 和 recompute 抢占。
- **能力目标**：能沿一次请求找到块查询、分配、缓存登记、驱逐和回收的代码路径。
- **素养目标**：比较实现时先对齐生命周期，再比较类名；不把 CPU offload connector 误当成抢占中的 swap。

## 时间安排

| 时间 | 环节 | 内容 |
| --- | --- | --- |
| 30 min | 概念 | 对象图：Scheduler → `KVCacheManager` → `KVCacheCoordinator` → `SingleTypeKVCacheManager` → `BlockPool`；`allocate_slots`；prefix 命中与末块重算；BlockPool 三态与 LIFO/LRU；recompute 抢占；offload connector 的边界 |
| 50 min | 源码 | 按函数级锚点读 `kv_cache_manager.py`、`kv_cache_coordinator.py`、`single_type_kv_cache_manager.py`、`block_pool.py`、`block_table.py`、`sched/scheduler.py`；对照 nano-vllm 的生命周期 |
| 30 min | 实验 | 用 instrumentation 打印 block table 与 prefix 统计；没有 GPU 时用示例日志和字段表；填映射表 |
| 10 min | 答疑 | 回答缓存块与空闲块的区别、LIFO/LRU 的队列位置、recompute 的触发条件、offload 与抢占的区别 |

## 阅读清单（函数级）

| 文件 | 函数 / 类 | 行号 | 看什么 |
| --- | --- | --- | --- |
| `vllm/v1/core/kv_cache_manager.py` | `make_prefix_cache_stats` | L236 | 统计什么时候可用；关闭 `log_stats` 时返回 `None` |
| 同上 | `get_computed_blocks` | L264 | prefix 查询；`max_cache_hit_length = request.num_tokens - 1` |
| 同上 | `allocate_slots` | L371 | 一次分配的主流程；返回 `KVCacheBlocks` 或 `None` |
| 同上 | `free` | L610 | 请求结束或抢占时释放块 |
| 同上 | `cache_blocks` | L802 | 把满块登记进 prefix cache |
| `vllm/v1/core/kv_cache_coordinator.py` | `get_num_blocks_to_allocate` | L169 | 按 group 汇总需要的新块数 |
| 同上 | `allocate_new_computed_blocks` | L231 | 命中块加引用，先 touch 再分配外部块 |
| 同上 | `allocate_new_blocks` | L277 | 真正向 BlockPool 要新块 |
| 同上 | `cache_blocks` | L333 | 逐个 SingleType manager 登记缓存 |
| 同上 | `free` | L380 | 按 manager 释放 |
| 同上 | `find_longest_cache_hit` | L457 | 抽象入口；`UnitaryKVCacheCoordinator` 的实现在 L574 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `get_num_blocks_to_allocate` | L180 | 单 group 的块数计算 |
| 同上 | `add_local_computed_blocks` | L267 | 命中块 `ref_cnt += 1` |
| 同上 | `allocate_new_blocks` | L366 | 从 BlockPool 取块并写入 `req_to_blocks` |
| 同上 | `cache_blocks` | L473 | 按块边界登记哈希 |
| 同上 | `free` | L578 | 逆序归还，尾部先释放 |
| 同上 | `FullAttentionManager` | L741 | full attention 的缓存实现 |
| `vllm/v1/core/block_pool.py` | `get_cached_block` | L197 | 按哈希查缓存块 |
| 同上 | `cache_full_blocks` | L225 | 建立哈希到块的映射 |
| 同上 | `get_new_blocks` | L668 | 从空闲队列取块，驱逐旧哈希 |
| 同上 | `unpin_blocks` | L710 | 讲 offload 时才提，不是通用 pin API |
| 同上 | `_maybe_evict_cached_block` | L731 | 分配时清理旧哈希 |
| 同上 | `free_blocks` | L776 | 非缓存块前插、缓存块后插 |
| `vllm/v1/worker/block_table.py` | `BlockTable.get_numpy_array` | L277 | 打印 block table 的入口 |
| 同上 | `MultiGroupBlockTable.__getitem__` | L393 | 多 group 时按 `[0]` 取第一组 |
| `vllm/v1/core/sched/scheduler.py` | `_preempt_request` | L1539 | V1 抢占：释放块、PREEMPTED、计数清零、回 waiting |
| 同上 | `allocate_slots` 调用点 | L743 / L1214 | prefill/admission 与 decode 各自从哪进 |

## 重点与难点

| 类型 | 内容 | 处理方式 |
| --- | --- | --- |
| 重点 | 分层对象的分工 | 先画对象图，再沿 `allocate_slots` 跟一次分配 |
| 重点 | prefix 命中与引用 | 用共享前缀请求演示 `get_computed_blocks` 和 `add_local_computed_blocks` |
| 难点 | 驱逐顺序 | 讲清 BlockPool 的三种块状态，以及 `free_blocks` 为什么把非缓存块前插、缓存块后插 |
| 难点 | V1 抢占 | 只讲 recompute，不编造 swap；offload 单独作为 connector 讲 |
| 难点 | 末块差异 | nano-vllm 连满的末块也不复用；vLLM 保留最后 token 重算，可能整块重算 |

## 本讲边界

| 边界 | 代码位置 | 说明 |
| --- | --- | --- |
| V1 抢占只有 recompute | `sched/scheduler.py:1539` | 释放块、置 `PREEMPTED`、`num_computed_tokens=0`、回 waiting；没有 swapped 队列 |
| offload 不是抢占 | `config/cache.py:256`、`config/vllm.py:1123` | `kv_offloading_size` 开启的是 KV transfer connector，不改变请求状态 |
| 最后 token 必重算 | `kv_cache_manager.py:295` | `max_cache_hit_length = request.num_tokens - 1`；块对齐后可能整块重算 |
| BlockPool 复用顺序 | `block_pool.py:776` | 非缓存块前插（LIFO），缓存块后插（FIFO/LRU） |
| 没有通用 pin API | `block_pool.py:710` | `unpin_blocks` 只服务外部读取者，不是请求生命周期的常规接口 |
| Worker block table 的 0 | `vllm/v1/worker/block_table.py:277` | 0 是 null block；有效长度看 `num_blocks_per_row` |
| prefix 统计会重置 | `kv_cache_manager.py:236` | `make_prefix_cache_stats()` 读取并重置；`log_stats=False` 时返回 `None` |

## 本讲产出

一张按生命周期成对写的映射表。下表由学员填写；第一行和最后一行给出示例与关键差异，中间三行待填。

| 生命周期 | nano-vllm | vLLM | 作用 | 差异 |
| --- | --- | --- | --- | --- |
| prefill / admission | `can_allocate` + `allocate` | `get_computed_blocks` + `allocate_slots` | 查前缀、分配首批块 | 示例：nano 先 `can_allocate` 再 `allocate`；vLLM 的查询和分配都在 `allocate_slots` 内完成，失败返回 `None` |
| decode 增长 | `can_append` + `may_append` | `allocate_slots(num_new_tokens=1)` | 跨块边界时追加块 | 待填 |
| 释放 | `deallocate` | `free` | 归还块 | 待填 |
| 缓存登记 | `hash_blocks` | `cache_blocks` | 把满块登记进前缀缓存 | 待填 |
| 末块 | 满的末块也不复用 | `max_cache_hit_length = request.num_tokens - 1` | 保证最后一个 token 重算 | vLLM 最后 token 必重算，按块对齐后可能整块重算；nano-vllm 的 `can_allocate` 只查到 `num_blocks - 1` |

## 参考材料

- [vLLM](https://github.com/vllm-project/vllm)
- `~/Project/ai-infra/vLLM/vllm/v1/core/kv_cache_manager.py`
- `~/Project/ai-infra/vLLM/vllm/v1/core/kv_cache_coordinator.py`
- `~/Project/ai-infra/vLLM/vllm/v1/core/single_type_kv_cache_manager.py`
- `~/Project/ai-infra/vLLM/vllm/v1/core/block_pool.py`
- `~/Project/ai-infra/vLLM/vllm/v1/worker/block_table.py`
- `~/Project/ai-infra/vLLM/vllm/v1/core/sched/scheduler.py`
- [nano-vllm](https://github.com/ForceInjection/nano-vllm)
