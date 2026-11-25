# 第 3 讲 · KV cache 管理：PagedAttention 的生与死

> 120 分钟 | 基线：nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`(v0.34.1rc0-84,2026-11-25)

本讲回答一个核心问题：**nano 用一个 `BlockManager` 管所有 KV，生产版为什么长成四层？以及——`PagedAttention` 这个名字，为什么正在从 vLLM 的代码里消失？**

知识链：请求 → `KVCacheManager.allocate_slots` → Coordinator（按组数三选一）→ SingleTypeKVCacheManager（按注意力类型）→ BlockPool（全引擎唯一(HiSparse 的辅助池 `v1/hisparse/block_pool.py` 为特性内例外)块池 + LRU 链）→ 命中查找按 `(hash, group)` 联合键 → 分页思想活在 `block_table` 里，kernel 本体已退役，读取交给各 attention 后端的原生 page-table API。

## 一、nano 已学形态（回顾）

先修课第 4 课你走过的单一 `BlockManager`（`nanovllm/engine/block_manager.py`）：

- **一个类管所有事**：块元数据列表 + `free_block_ids`/`used_block_ids`（`:26-37`）+ 哈希映射 `hash_to_block_id`；
- **链式哈希**：`can_allocate`（`:62-77`）逐块算链式哈希查命中，`token_ids` 全等校验防碰撞；`hash_blocks`（`:114-124`）在 `postprocess` 开头写回——「刚写完的块尽快可复用」；
- **`Block` 三字段**（`:8-24`）：`ref_count`（几个 seq 在用）、`hash`、`token_ids`；
- 你 fork 里加的 CPU 池（`:34-37`）与 swap（`can_swap_out` 要求全部块 `ref_count == 1`，`:135-140`；`swap_in` 后重建哈希链，`:161-180`）。

对照生产版之前，记住 nano 的三个「单一」：**一个管理器、一种注意力类型、一个哈希键维度(hash→block)**。V1 的四层结构，就是这三个「单一」各被打破一次的结果。

## 二、演进：一层变四层

初始化链（第 1 讲已定位入口）：`EngineCore._initialize_kv_caches`（`core.py:289-419`）→ worker profile 显存（`gpu_worker.py:740`）→ `get_kv_cache_configs`（`kv_cache_planning.py:1476-1600`，`num_blocks = available_memory // bytes_per_block`（公式 `kv_cache_planning.py:675`）；跨 rank 取 min）→ `Scheduler` 持有 `KVCacheManager`（`kv_cache_manager.py:131-216`）。四层结构：

| 层   | 类                                                                   | 职责                                                                                                                                                                                                     |
| ---- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 门面 | `KVCacheManager`（`kv_cache_manager.py:131`）                        | 调度器唯一入口：`allocate_slots`（`:391-639`）、`get_computed_blocks`（`:278-341`）                                                                                                                      |
| 协调 | `get_kv_cache_coordinator` 工厂（`kv_cache_coordinator.py:828-878`） | **按组数三选一**：关缓存→`NoPrefixCache`；单组→`Unitary`；多组→`Hybrid`（`:870`）                                                                                                                        |
| 类型 | `SingleTypeKVCacheManager` 系（`single_type_kv_cache_manager.py`）   | 每种注意力一个：Full / SlidingWindow / DFlashSlidingWindow / Circular / ChunkedLocal / RSWA / Mamba / CrossAttention(交叉注意力)；HiSparse 辅助管理器在 `v1/hisparse/`(特性内)；各自实现 `find_longest_cache_hit`（基类 `:742`，Full `:873`、SlidingWindow `:1149`、Mamba `:1956`） |
| 池   | `BlockPool`（`block_pool.py:107-173`）                               | **全引擎唯一(HiSparse 的辅助池 `v1/hisparse/block_pool.py` 为特性内例外)**块池：`free_block_queue` 双向链表（`:153`，头部即 LRU 逐出序）+ 哈希表 `cached_block_hash_to_block`（`:156`）                                                                              |

为什么分层？两个动力：一是**混合注意力**——同一模型内 sliding window 与 full attention 交替（如 Gemma-2，`models/gemma2.py:156` 逐层二选一）、GDN 线性注意力与 full 混合（`models/qwen3_next.py:526-533`）——每类的命中算法完全不同，所以「按类型」一层；二是**显存只有一个池**——所有组 overlay 同一块 allocation（`kv_cache_interface.py:638-641`），所以「池」必须全局唯一。层的申报来自模型侧：每层 attention 声明自己的 manager（`attention.py:738`；GDN 走 MambaSpec，`gdn_attn.py:93`）。

## 三、前缀缓存的三个升级

对照 nano 的链式哈希，HEAD 上有三个升级：

1. **哈希键加了 group 维度**：`BlockHashWithGroupId`（`kv_cache_utils.py:50`，哈希字节串后拼 group 后缀，`block_pool.py:262`）。nano 是 `hash → block`，HEAD 是 `(hash, group) → block`——同一段 token 在滑窗组和 full 组是独立缓存、独立逐出的。
2. **哈希计算前移**：nano 在 postprocess 时写回；HEAD 在**请求构造时就算好全前缀哈希**（`Request.__init__` 存 `block_hashes` 并立即 `update_block_hashes()`，`request.py:243-248`；append token 时增量延长，`:305`）。`schedule` 只查表——这也是第 2 讲 `cache_aware_admission_window` 能在准入阶段就知道「谁命中了缓存」的前提。哈希算法默认 sha256（`config/cache.py:268`）。
3. **部分块缓存出现**：nano「只缓存整块」，HEAD 有 `cache_partial_block`（`block_pool.py:459`）与 partial→full 提升（`:291-300`）——Mamba/线性注意力与细粒度命中场景需要。

prefix caching 默认开（`enable_prefix_caching: bool = True`，`config/cache.py:271`）；CLI 上**没有** `--no-prefix-caching`，否定形式是 `--no-enable-prefix-caching`。共享语义与 nano 同构：命中即 `touch`（`block_pool.py:788-810`，`ref_cnt` 0→1 时从 free 队列摘除），释放后 `ref_cnt` 归 0 进队尾，真正逐出发生在被重新分配时清哈希（`_maybe_evict_cached_block`，`:765`）。

## 四、混合协调：多组不动点

多组时 `HybridKVCacheCoordinator` 的前缀命中是**迭代不动点算法**（`:704-810`）：每组要么接受当前候选命中长度、要么缩短；有缩短就全组重查，直到收敛。full attention 组排最前且只查一次（`:750-757`，左→右首 miss 即停给出上界）。容量侧还有一个反直觉分支：局部注意力(swa/chunked-local)在可共享 block table 时**直接晋升为全量分配**（`kv_cache_planning.py:727-789`）——反正省不下内存，干脆按 full 管理，省掉双组协调的复杂度。

## 五、PagedAttention 的生与死

**生(2023-06)**：OS 虚拟内存类比，自研 CUDA kernel（`attention_kernels.cu`），解决碎片——这部分你们在先修课已经学透。

**死（2026 年核实，证据链三步）**：

1. 自研 CUDA kernel 已删除：`csrc/attention/` 下没有 `attention_kernels.cu`；官方 `docs/design/paged_attention.md:8-9` 自己声明「no longer exists in the codebase」。
2. `vllm/v1/attention/ops/paged_attn.py` 只剩 **53 行**布局辅助（`split_kv_cache` / `write_to_paged_cache`），唯一使用方是 ROCm 后端。
3. 分页 KV 的实际读取交由各后端原生 page-table API:FlashAttention 吃 `block_table` 参数（`flash_attn.py:1387`、`:1583`）；FlashInfer 走 `paged_kv_indices` 系原生接口（`flashinfer.py:454-456`）；Triton 同（`triton_attn.py:700-724`）。残留的 `paged_attention` 命名只在平台专用路径(ROCm/AITER/B12x)。

**生（思想永生）**：调度器仍在分配 `block_id`、维护 per-request `block_table`、前缀缓存仍在按块共享——「分页」思想已经从 kernel 下沉为引擎的 `BlockPool` + `block_table`。写简历时的措辞建议：写「分页 KV 管理」比「精通 PagedAttention kernel」更经得起核对。

## 六、旧文档 vs HEAD

| 说法（旧文档/旧文）                                               | HEAD 事实                                                                |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `KVCacheGroupId` / `KVCacheGroupSpec`                             | 已不存在；统一为 `KVCacheGroup`（`kv_cache_interface.py:682`）           |
| 「只缓存整块」（`prefix_caching.md` Note 1）                      | `cache_partial_block` 出现，Note 1 滞后                                  |
| 「Hybrid 仅支持恰好 2 组」（`hybrid_kv_cache_manager.md:225`）    | 任意组数 + 部分哈希命中 + sparse retention；该文档自述基于旧 commit      |
| 「哈希在 postprocess 写回」（nano 做法）                          | `add_request` 时即算好全前缀哈希，schedule 只查表                          |
| 「request→blocks 映射在 BlockPool」（`prefix_caching.md` 组件图） | 在各 SingleTypeKVCacheManager 的 `req_to_blocks`；BlockPool 只管池与哈希 |
| 容量换算在 `kv_cache_utils` 单文件                                | 拆到 `kv_cache_planning.py` + 平台可替换的 `KVCacheConfigBuilder`        |

## 七、实验与产出

见 [hands-on-exercise.md](hands-on-exercise.md)：实验 1 同前缀请求的命中观察（`code/prefix_cache_probe.py`，纯推理负载）；实验 2 混合注意力模型的分组日志（需能跑 Gemma-2/gpt-oss 级模型）；实验 3（可选）`--kv-cache-dtype` 对照。课后作业见 [homework.md](homework.md)。

## 八、边界

| 内容                                                                       | 本讲                                  | 去处                                                      |
| -------------------------------------------------------------------------- | ------------------------------------- | --------------------------------------------------------- |
| KVCacheManager/BlockPool 对象图逐字段、容量账本（每 token 多少、并发上限） | 只看结构与演进                        | [KV Cache 八讲](../../kv_cache_course/README.md)第 4 讲（对象图）/第 1-2、8 讲（账本） |
| 注意力 kernel 数学（MLA/NSA 怎么算）                                       | 不展开                                | 第 4 讲                                                   |
| offloading(cpu/fs/tiering)                                                 | 不展开（第 2 讲已交代「移出调度器」） | 第 9 讲                                                   |

## 九、前后衔接

第 2 讲的 `allocate_slots` 失败触发抢占——本讲打开了它的内部：三层下探、None 在哪层判定（`kv_cache_manager.py:560-563` 准入门与 `:595-599` 常规路径）。第 4 讲讲「块找到了，attention kernel 怎么读它」。
