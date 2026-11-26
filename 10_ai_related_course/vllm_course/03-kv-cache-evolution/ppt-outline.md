# 第 3 讲：KV cache 管理 — PPT 大纲

## 设计说明

- **总页数**：约 23 页
- **演讲时长**：120 分钟（nano 回顾 10 min + 演进 15 min + 现状走读 45 min + PagedAttention 生死 15 min + 实验 25 min + 答疑 10 min）
- **核心视觉**：一层→四层结构演化图（本讲锚点，第 6 页）；PagedAttention 退役证据链（第 19 页，三步）；`BlockHashWithGroupId` 联合键对照（nano 单键 vs HEAD 双维）
- **教学策略**：三个「单一」各被打破一次作为叙事骨架（一个管理器→四层；一种类型→七类管理器；一个哈希键维度→group 联合键）；退役部分用证据链方式（每条证据都可在 HEAD 上 grep 验证）
- **配色**：nano 橙色系、四层结构蓝色系（每层不同深浅）、退役证据红色系、prefix caching 绿色系
- **边界**：对象图逐字段与容量账本导流 KV Cache 八讲；kernel 数学留第 4 讲；offloading 留第 9 讲

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：KV cache 管理：PagedAttention 的生与死
- **副标题**：vLLM 十讲 · 第 3 讲
- **基线信息**：nano-vllm `b9dd55a` + vLLM `2f5d8a2b1` | 核对日期 2026-11-24
- **一张图**：墓碑式构图——「PagedAttention kernel(2023-2026)」+ 背景里活着的 `block_table`/BlockPool

### 第 2 页：本讲核心问题与议程

- **核心问题**：nano 用一个 `BlockManager` 管所有 KV，生产版为什么长成四层？「PagedAttention」这个名字为什么正从代码里消失？
- **四段议程**：nano 回顾（10 min，第 3–4 页）| 演进：一层变四层（15 min，第 5–9 页）| 现状走读（45 min，第 10–17 页）| PagedAttention 生死（15 min，第 18–21 页）+ 实验（第 22 页）

## 第一部分：nano 回顾（10 min，第 3–4 页）

### 第 3 页：nano 回顾：三个「单一」

- **复用先修课第 4 课锚点图**（BlockManager 全景）
- **三个单一**（橙色标注）：
  - 一个管理器：`BlockManager`（`block_manager.py:26-37`）管元数据 + free/used + 哈希
  - 一种注意力类型：所有层同构(GQA)，一套分配逻辑
  - 一个哈希键维度：`hash → block_id`（`hash_to_block_id`）
- **口播**：「先修课你们走过它的全链；今天这三个『单一』各被打破一次」

### 第 4 页：nano 的哈希链与你的 fork

- **上稿**（三锚点）：
  - 链式哈希三件套：`can_allocate`（`:62-77`，查命中+防碰撞）→ `allocate`（`:79-96`，命中挂表）→ `hash_blocks`（`:114-124`，postprocess 写回）
  - `Block` 三字段（`:8-24`）：`ref_count` / `hash` / `token_ids` 全等校验
  - 你 fork 的增量：CPU 池（`:34-37`）、`can_swap_out` 的 `ref_count==1` 约束（`:135-140`）、`swap_in` 重建哈希链（`:161-180`）
- **口播**：「`swap_in` 重建哈希链这个细节，第 9 讲对照 vLLM offloading 时还会回来」

## 第二部分：演进——一层变四层（15 min，第 5–9 页）

### 第 5 页：演进总览

- **一张图**：左 nano 单层（橙）→ 右 vLLM 四层（蓝，四段深浅）：门面 KVCacheManager / 协调 Coordinator 三选一 / 类型 SingleType×7 / 池 BlockPool 唯一
- **初始化链**（底部小字）：`_initialize_kv_caches`（`core.py:289`）→ profile（`gpu_worker.py:740`）→ `get_kv_cache_configs`（`kv_cache_planning.py:1476`，`num_blocks = available_memory // bytes_per_block`，`:675`）→ `KVCacheManager.__init__`（`kv_cache_manager.py:131`）
- **口播**：「四层不是过度设计，是两个动力逼出来的——下一页」

### 第 6 页：四层结构图（本讲锚点）

- **一张大图**：四层堆叠图（蓝色系四段深浅），每层标注类名与 file:line：
  - 门面：`KVCacheManager`（`kv_cache_manager.py:131`）——调度器唯一入口
  - 协调：工厂三分支（`kv_cache_coordinator.py:828-878`）——NoPrefixCache / Unitary / Hybrid
  - 类型：`SingleTypeKVCacheManager` 系——Full（`:873`）/ SlidingWindow（`:1149`）/ Circular / ChunkedLocal / RSWA / Mamba（`:1956`）
  - 池：`BlockPool`（`block_pool.py:107`）——全引擎唯一，LRU 双向链
- **口播**：后续每页在图中高亮一层；这张图第 2 讲 schedule() 地图的姊妹篇

### 第 7 页：为什么分层

- **上稿**（两个动力，左右分栏）：
  - 动力一（混合注意力）：同一模型 sliding window + full 交替（Gemma-2，`models/gemma2.py:156`）；GDN 线性注意力 + full（`models/qwen3_next.py:526-533`，GDN 走 MambaSpec）——每类命中算法完全不同 → 「类型」层
  - 动力二（显存只有一个）：所有组 overlay 同一 allocation（`kv_cache_interface.py:638-641`）→ 「池」必须全局唯一
  - 层的申报来自模型侧：每层 attention 声明 manager（`attention.py:738`）
- **口播**：「nano 只有一种类型，所以类型层和池层合并成了一个类；类型一多，就必须拆」

### 第 8 页：容量换算与 auto-fit

- **上稿**（`kv_cache_planning.py:1476-1600`）：
  - `num_blocks = available_memory // bytes_per_block`（（`kv_cache_planning.py:675`，公式主体在 `get_kv_cache_configs`，`:1476` 起））；跨 rank 取 min（`:1577-1588`）
  - `_auto_fit_max_model_len` 二分搜索最大可容纳长度（`:1283` 起）
  - 平台可替换：`KVCacheConfigBuilder`（`kv_cache_config_builder.py:19-105`）
- **口播**：`--gpu-memory-utilization` 打进来的是这里，不是调度器——容量账本的完整推导在 KV 八讲第 8 讲（每 token 公式在第 1 讲）

### 第 9 页：Coordinator 三选一

- **上稿**（工厂，`kv_cache_coordinator.py:828-878`）：
  - `enable_caching=False` → `NoPrefixCache`（`:843`）
  - 组数 == 1 → `UnitaryKVCacheCoordinator`（`:856`）
  - 组数 > 1 → `HybridKVCacheCoordinator`（`:870`）
- **辨析框**：「HMA」是生态叫法（混合内存注意力），HEAD 类名是 `HybridKVCacheCoordinator`；旧文 `hybrid_kv_cache_manager_deep_dive.md`（HMA 口径）作演进素材，类名以本讲为准

## 第三部分：现状走读（45 min，第 10–17 页）

### 第 10 页：`allocate_slots` 全链

- **一张图**：三层下探路径图（scheduler `:897`/`:1487` → `KVCacheManager.allocate_slots`（`kv_cache_manager.py:391-639`）→ coordinator → BlockPool）
- **上稿**（三阶段，`:478-486` docstring）：①释放滑窗外冗余块并查余量 ②前缀命中块 touch+外部块 ③新 token 分配
- **两处 None 判定**：`:560-563`（准入门 `full_sequence_must_fit`）与 `:607-614`（常规路径）——第 2 讲的「返回 None 触发抢占」就发生在这

### 第 11 页：BlockPool：唯一池与一条 LRU 链

- **上稿**（`block_pool.py:107-173`）：
  - 预分配全部块对象（`:147`）；`free_block_queue` 双向链表（`:153`，**头部 = LRU 逐出序**）；哈希表 `cached_block_hash_to_block`（`:156`，自实现 map，`:38-102`）
  - `get_new_blocks`（`:681`）：从队首弹 N 块；弹出带哈希的块 → `_maybe_evict_cached_block`（`:712-713`，**这就是逐出时刻**）
- **口播**：「分配和逐出是同一个动作的两面——弹队首时，如果它有哈希，前缀缓存里就少了它」

### 第 12 页：touch 与 free 的双语义

- **上稿**（`block_pool.py`）：
  - `touch`（`:788-810`）：命中块 `ref_cnt` 0→1，从 free 队列摘除
  - `free_blocks`（`:812-850`）：`ref_cnt` 归 0 入队——**无哈希块 prepend 队首（LIFO 复用，利好 GPU 局部性）**，**有哈希块 append 队尾(FIFO ⇒ LRU)**
- **口播**：一条链两种入队方向，是两条淘汰策略的合流；nano 的 free/used 集合没有这个区分

### 第 13 页：升级一：哈希键加 group

- **一张对照图**：nano `hash → block`（橙）vs HEAD `(hash, group_id) → block`（蓝；`BlockHashWithGroupId`，`kv_cache_utils.py:50`；后缀拼接 `block_pool.py:262`）
- **上稿**：同一段 token 在滑窗组和 full 组独立缓存、独立逐出；链式哈希本体不变（`hash_block_tokens`，`kv_cache_utils.py:747-774`，父哈希+本块 token+`extra_keys`，默认 sha256）

### 第 14 页：升级二：哈希前移

- **上稿**：
  - nano:postprocess 时写回（`hash_blocks`，`:114-124`）
  - HEAD：`Request.__init__` 存 `block_hashes` 并立即 `update_block_hashes()`（`request.py:243-248`）；append token 增量延长（`:305`）；schedule 只查表
  - 连锁：`cache_aware_admission_window`（第 2 讲第 19 页）之所以能在准入阶段知道命中，因为哈希早就绪
- **绿色系**：前缀缓存相关页全绿

### 第 15 页：升级三：partial block

- **上稿**：
  - nano「只缓存整块」；HEAD `cache_partial_block`（`block_pool.py:459`）与 partial→full 提升（`:291-300`）
  - 场景：Mamba/线性注意力状态、细粒度命中
  - 文档滞后：`prefix_caching.md` Note 1「only cache full blocks」已不严格
- **口播**：「只缓存整块」曾是教程金句——核对日期过了它就过期了

### 第 16 页：混合协调：多组不动点

- **一张图**：Hybrid 命中的迭代不动点流程（`kv_cache_coordinator.py:686-808`）：每组接受或缩短 → 有缩短则全组重查 → 收敛；full 组排最前且只查一次（`:750-757`）
- **上稿**：`prefix_lookup_groups` 分桶（`:665-694`，full attention 组排最前）；模型例子：Gemma-2、gpt-oss、Jamba、Llama4
- **文档滞后框**：「仅支持恰好 2 组」（`hybrid_kv_cache_manager.md:225`）已失效——HEAD 支持任意组数 + 部分哈希命中 + sparse retention

### 第 17 页：反直觉：局部注意力晋升全量

- **上稿**（`kv_cache_planning.py:727-789`）：
  - 当局部注意力(swa/chunked-local)可共享 block table 时，直接晋升为 FullAttentionManager 配置（`:783`）——「反正省不下内存，干脆全量分配」
  - 这是文档没写的现代演进；判断「省不省得下」的账本逻辑属 KV 八讲第 6-7 讲
- **口播**：优化不是永远加分支，有时是删分支——前提是算过账

## 第四部分：PagedAttention 生死（15 min，第 18–21 页）

### 第 18 页：PagedAttention 之生(2023)

- **上稿**：SOSP 2023(arXiv:2309.06180)——OS 虚拟内存类比；自研 CUDA kernel `attention_kernels.cu`；解决内外碎片（利用率 20-40% → 90%+）
- **口播**：先修课与 KV 八讲已学透机制；本讲只问「它现在在哪」

### 第 19 页：退役证据链（三步，每步可 grep 验证）

- **上稿**（红色系，三张证据卡）：
  1. `csrc/attention/` 下无 `attention_kernels.cu`；官方文档自述「no longer exists」（`paged_attention.md:8-9`）
  2. `vllm/v1/attention/ops/paged_attn.py` 只剩 **53 行**布局辅助（`split_kv_cache`/`write_to_paged_cache`），唯一使用方 ROCm 后端
  3. 分页读取交给各后端原生 API:FA 吃 `block_table`（`flash_attn.py:1387/1583`）；FlashInfer 原生 `paged_kv_*`（`flashinfer.py:454-456`）；Triton（`triton_attn.py:700-724`）
- **残留说明**：`paged_attention` 命名只在 ROCm/AITER/B12x 平台路径

### 第 20 页：思想永生

- **一张图**：分页思想的三个「永生落点」——BlockPool（块分配/逐出）、`block_table`（per-request 页表）、prefix caching（按块共享）全部仍在，且升级为四层主干
- **上稿**：「kernel 退役 ≠ 分页退役」；调度器仍在分配 `block_id`、维护页表、按块共享——它们不再依赖某个特定 kernel
- **简历措辞框**：写「分页 KV 管理」比「精通 PagedAttention kernel」经得起核对

### 第 21 页：旧文档 vs HEAD

- **六行对照表**（README 同款）：KVCacheGroupId 不存在 / 只缓存整块滞后 / 仅 2 组失效 / 哈希前移 / `req_to_blocks` 位置 / 容量换算新文件
- **口播**：三篇设计文档里，`paged_attention.md` 最诚实（文档自述为历史），另两篇滞后在「能力边界」

## 第五部分：实验与收尾（第 22–23 页）

### 第 22 页：本讲实验

- **三行实验卡**：实验 1(GPU)`prefix_cache_probe.py` 同前缀命中观察；实验 2（GPU，大显存）混合模型分组日志；实验 3（可选）`--kv-cache-dtype fp8` 对照
- **产出要求**：冷热两轮 TTFT 对照表；分组日志标注（组数/GPU blocks）；dtype 对照数字

### 第 23 页：结课与预告

- **本讲你应能**：画四层结构；复述退役证据链；独立验证任何一条「旧说法」
- **作业**：对照题（group 键为什么必须）+ 退役证据链独立复验
- **预告**：块在显存里排好了，attention kernel 怎么读它——第 4 讲：注意力后端矩阵
- **下节材料**：第 4 讲 syllabus 阅读清单

---

## 图资产清单（自制）

| 图                          | 页    | 说明                                    |
| --------------------------- | ----- | --------------------------------------- |
| 墓碑封面图                  | 1     | kernel 2023-2026 + 背景活体 `block_table` |
| nano 三单一点图             | 3     | 先修课第 4 课复用                       |
| 一层→四层演化图             | 5-6   | 锚点图；四段深浅蓝                      |
| 初始化链路图                | 5     | 底部小字流程                            |
| `allocate_slots` 三阶段下探图 | 10    | 两处 None 判定标注                      |
| BlockPool 一条链两语义      | 11-12 | LIFO/LRU 入队方向                       |
| 联合键对照图                | 13    | nano 单键 vs HEAD 双维                  |
| Hybrid 不动点流程           | 16    | 多组互约束收敛                          |
| 退役证据链三卡              | 19    | 红色系；每卡带 grep 命令                |
| 思想永生三落点              | 20    | BlockPool/block_table/prefix caching    |
