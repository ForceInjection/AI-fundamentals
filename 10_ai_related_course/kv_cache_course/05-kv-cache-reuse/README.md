# 第 5 讲：KV Cache 怎么复用与分层

> 120 分钟 | 课程第 5 讲 | 前置：第 2–4 讲的 block 与引擎

第 3、4 讲在 block 和哈希层面做前缀复用：nano-vllm 用前缀哈希链让多个请求指向同一批 block，并带请求级 swap；vLLM 在显存不足时 preempt 单个请求，走 recompute，不是 swap。这两套机制都以 block 为单位，复用跟着块的生命周期走，没有回答“哪些前缀值得留、留不下时先扔谁、能不能放到 HBM 之外”。

本讲把这件事系统化：用 radix tree 管理前缀，讲清命中条件、引用保护和驱逐策略，再往下一层存储走。相同前缀怎么在树里命中，命中后靠什么字段保护，满了以后驱逐谁，HBM 放不下时怎么备份到主机内存和存储，是这一讲的主线。

先分开两个容易混的概念。nano-vllm 的请求级 swap 是调度器把一个运行中请求的 KV 整体搬到 CPU，等它重新调度时再搬回来，搬的是这个请求自己的块；vLLM V1 没有这条 swap 路径，它的 preemption 是 recompute。HiCache 的 host tier 是按前缀保存 KV 副本，供之后任意请求命中，命中后仍要回载到 HBM 才能参与计算。前者是请求维度的让路，后者是内容维度的缓存。

## 本讲回答三个问题

1. 前缀缓存省的是哪部分计算？一条请求要命中前缀，需要满足哪些条件？
2. radix tree 怎么匹配、分裂、加锁和驱逐？`lock_ref` 与 block 级的 `ref_count` 差在哪？
3. HBM 放不下时，HiCache 怎么在 host 和 storage 之间搬运？哪些动作会释放 device，哪些不会？

## 一句话

前缀复用解决“算过的不要再算”，分层解决“放不下的放到哪”。两者都靠命中条件和引用计数保证正确性；缓存命中不等于 device 命中，换出也不等于释放。

## 学习目标

- 说清 prefix caching 省的是哪部分计算，列出收益与代价。
- 画出 `UnifiedRadixCache → UnifiedTreeCore → UnifiedTreeNode → ComponentData` 的对象关系，以及 `HostPoolGroup` 的位置。
- 区分 nano-vllm `ref_count`、vLLM `ref_cnt`、SGLang `lock_ref` 三套引用语义。
- 用四状态模型判断节点能否被驱逐，走通 split、加锁、驱逐、解锁。
- 区分 HiCache 的 hit、loadback、backup、demote / evict 与 write policy。
- 说明 `host_value` 存的是什么，命中为什么要看 `extra_key`、`cache_salt` 和 `page_size`。

## 先看对象关系

前缀缓存的元数据和 KV 数据分开存放。树节点只记“这段 token 的 KV 在哪里”，真正的 K/V 在 device pool 和 host pool 里。

```text
UnifiedRadixCache（对外接口）
├── UnifiedTreeCore（树结构与 LRU、可驱逐叶集合）
│   ├── UnifiedTreeNode
│   │   ├── key：这段边覆盖的 token 区间
│   │   ├── children / parent / hash_value
│   │   └── component_data[ComponentType]
│   │       ├── value：device pool 的槽位索引
│   │       ├── host_value：host pool 的槽位索引
│   │       ├── lock_ref / host_lock_ref
│   │       └── session_ref / session_ids
│   └── components：FULL / SWA / MAMBA / C128
├── HostPoolGroup（host pool 的分配门面）
│   ├── anchor pool
│   └── side pools（混合模型才有）
└── UnifiedSessionRefTracker（会话软保护）
```

`UnifiedRadixCache` 管对外接口、组件注册和 HiCache 初始化；`UnifiedTreeCore` 管树结构、LRU 链和可驱逐叶集合。`UnifiedTreeNode.component_data` 按组件索引，普通 MHA 只有 FULL 组件；SWA、MAMBA、C128 是混合模型才会出现的组件。

`value` 和 `host_value` 都是槽位索引张量，不是裸指针。`value` 指向 device KV pool，`host_value` 指向 host pool；`host_value` 不覆盖 L3，L3 的元数据由存储后端按 key 实时查询。`HostPoolGroup` 本身不存 KV，它只是 host pool 的分配门面：普通 MHA 只有 anchor，混合模型才有 SWA / MAMBA / C128 等 side pool。

经典 `RadixCache` 的 `TreeNode` 用单个 `value` 和 `lock_ref` 表示 KV 与锁，适合建立树结构直觉；HEAD 默认走 `UnifiedRadixCache` / `UnifiedTreeCore`，两者字段和锁语义不同，读行号前先确认打开的是哪个文件。

## 前缀缓存省什么、不省什么

前缀缓存省的是 prefill：命中部分不再做 QKV 投影，也不再对这些位置做注意力计算。decode 仍然要逐 token 生成，不受影响。命中越长，TTFT 和 prefill 总算力下降越明显。

收益：

- 高并发、长 system prompt 的场景下，prefill 总算力下降，吞吐上升。
- 共享前缀在 device 上只存一份 KV，多个请求的 block table 指向同一批槽位。
- 命中部分直接复用已有 KV，TTFT 下降。

代价：

- 缓存常驻内存。命中率不高时，它挤占可用 KV，降低并发上限。
- 每次请求都要 lookup；insert 和 split 有元数据开销，split 还要切 `value` / `host_value`。
- 驱逐后马上又命中就是抖动，重算成本可能高于省下的计算。
- page 对齐会截掉尾部 token，命中长度不是想要多少就有多少。
- host / storage 命中要搬运，loadback 延迟可能吃掉 TTFT 收益。
- 跨租户共享有侧信道风险，靠 `extra_key` / `cache_salt` 隔离，不是默认安全。

## radix tree：匹配、分裂、插入

一条前缀在树里的键不只是 token 序列。`RadixKey` 同时带 `token_ids`、`extra_key` 和 `cache_salt`，匹配前还会按 `page_size` 向下对齐：

```python
key = key.page_aligned(self.page_size)
if len(key) == 0:
    return self._empty_match_result
```

`match_prefix()` 从根节点开始，按 `child_key` 找子节点，逐段比较 token。命中可能停在节点中间，也可能停在某个节点的末尾。返回结果里最重要的是 `device_indices`、`host_hit_length` 和 `best_match_node`：前者是 device 上可直接复用的槽位，`host_hit_length` 是需要从 host 回载的 Full-KV token 数，`best_match_node` 是 L2 loadback 的锚点。

匹配停在一条边的中间时，要分裂节点：

```text
原节点：ABCD
匹配 AB
分裂后：AB（新父节点）→ CD（原子节点）
```

`_split_node()` 新建一个父节点，把原节点的前缀挂到新节点上，原节点保留后缀。新节点复制 `lock_ref`，`value` / `host_value` 按 split 长度切片；哈希链也要按 split 长度重新切分。分裂不是拷贝 KV，只是把元数据切成两段。

插入发生在 prefill 结束后。新块按 `hash_value` 链挂到树上，只有 page 对齐的 span 才会登记。树的默认实现是 `UnifiedRadixCache`，`registry.py:80` 的 `default_radix_cache_factory` 在常规路径最后返回 `create_unified_radix_cache`；经典 `RadixCache` 主要出现在 `PureSWARadixCache`、`FlexKVRadixCache` 和单测里。

## 三套引用语义

| 字段        | 实现                         | 粒度     | 语义                                                      | 释放                                                             |
| ----------- | ---------------------------- | -------- | --------------------------------------------------------- | ---------------------------------------------------------------- |
| `ref_count` | nano-vllm `Block`            | block 级 | 0 或 1+；共享同一 block 的请求各加一                      | 最后一个请求释放时归零，块回 free list                           |
| `ref_cnt`   | vLLM `KVCacheBlock` metadata | block 级 | 0 或 1+；被请求引用时加一                                 | 归零后进 free list，`_block_hash` 保留时仍可被 prefix cache 命中 |
| `lock_ref`  | SGLang radix node            | node 级  | 沿节点到根的路径累加，root 不参与释放；split 时新节点复制 | 沿同一路径减一，归零且是 leaf 才进入驱逐候选                     |

`ref_count` 和 `ref_cnt` 数的是“谁持有这个 block”，`lock_ref` 数的是“这条路径上有多少活跃请求”。三套字段不能互相替代，读代码时先看它挂在哪一层。

`lock_ref` 的累加方式按组件不同。普通 MHA 的 FULL 组件是 path-lock：从节点走到根，每个祖先都加一；root 的初值是 1，请求释放不会把它减到 0。SWA 是 segment-lock，只锁到滑动窗口边界；Mamba 是单节点锁，只锁注册的叶子。第 5 讲以 FULL 为主，第 6 讲再展开混合模型。

## 四状态模型

| 状态           | 判定                                                    | 能否驱逐 | 说明                                                                |
| -------------- | ------------------------------------------------------- | -------- | ------------------------------------------------------------------- |
| locked         | `lock_ref > 0`                                          | 否       | 路径上有活跃请求；祖先被锁不阻止后代被驱逐，后代看自己的 `lock_ref` |
| evictable leaf | 没有 device KV 子节点，`lock_ref == 0`，`value != None` | 是       | 驱逐候选只有这一类                                                  |
| internal       | 至少有一个 device KV 子节点                             | 否       | 要等子节点先被删掉                                                  |
| evicted        | `value == None`                                         | 不适用   | 可能还有 `host_value`；有 host 副本就从 host 命中并回载，没有则重算 |

四条规则：

- 只有 evictable leaf 进候选集，internal 和 locked 都不进。
- `lock_ref` 沿路径累加，root 不参与释放。
- split 后新节点复制 `lock_ref`，`value` / `host_value` 按 split 长度切片。
- leaf 被删除后，父节点只有在没有 device KV 子节点且 `lock_ref == 0` 时才回到候选。

ABCD 四步：

| 步骤 | 动作                      | 树            | `lock_ref` | 可驱逐节点 |
| ---- | ------------------------- | ------------- | ---------- | ---------- |
| 1    | 插入 ABCD                 | 一个节点 ABCD | 全 0       | ABCD       |
| 2    | 匹配 AB，节点在匹配点分裂 | AB → CD       | 全 0       | CD         |
| 3    | 对 AB 加锁                | AB → CD       | AB=1，CD=0 | CD         |
| 4    | 驱逐 CD，再解锁 AB        | AB            | AB 回到 0  | AB         |

第 3 步说明锁的保护范围：锁 AB 只保护 AB，CD 的 `lock_ref` 仍是 0，照样能被驱逐。第 4 步说明回退条件：CD 删掉后 AB 没有子节点，`lock_ref` 归零，才重新成为候选。

## 命中条件

一条请求要命中前缀，需要同时满足：

1. 从第 0 个 token 起逐 token 相同。
2. `extra_key` 相同。它是调用方定义的请求分类键，可用于把不同 LoRA / adapter 或业务命名空间隔开。
3. `cache_salt` 相同，用于租户或请求隔离。
4. 匹配长度按 `page_size` 向下对齐。

反例：两条请求的 prompt 一模一样，一条带 LoRA A，一条带 LoRA B。调用方给它们设了不同的 `extra_key`，`RadixKey.child_key` 生成的字典键不同，两条 prompt 落在不同子树，命中长度为 0。换 `cache_salt` 同理。`page_size=16` 时，60 token 的共享前缀只能命中 48，尾部 12 token 重算。

## 驱逐

候选集合只有 evictable leaves。一个叶被删掉后，父节点若满足条件才加入候选；这个从叶向根的回退是树结构维护的一部分，不是一种驱逐策略，也不与会话保护并列。

策略由启动参数 `--radix-eviction-policy` 决定，可选 `lru`、`lfu`、`slru`、`priority`、`tlru`。策略只决定候选之间的先后，不改变候选集合。

`session_ref` 是 `UnifiedRadixCache` 在 `--enable-session-radix-cache` 下给会话打的软保护。它让未被会话引用的节点先被驱逐，但被引用的 KV 在内存不足时仍可驱逐，不等同于 `lock_ref` 的硬锁。

真正释放 device 时，`evict_device_leaf()` 分两种情况：节点已有 host 副本，就 `demote`，只释放 device，节点留在树里；节点没有 host 副本，write-through / write_through_selective 策略下直接删除，write-back 策略下先做 D→H backup，再 demote。

## HiCache：三层与动作

HiCache 把 KV 分成三层：

| 层  | 介质            | 共享范围                                                                                     |
| --- | --------------- | -------------------------------------------------------------------------------------------- |
| L1  | GPU HBM         | 单实例私有                                                                                   |
| L2  | host DRAM       | 单实例私有，同一节点上的两个实例也不互相读取                                                 |
| L3  | storage backend | 取决于 backend；`file` 默认是节点本地目录，Mooncake / HF3FS / NIXL / AIBrix 才可能跨实例共享 |

几个动作的边界：

| 动作         | 方向              | 释放 device | 说明                                                                                        |
| ------------ | ----------------- | ----------- | ------------------------------------------------------------------------------------------- |
| device hit   | 无搬运            | 否          | 前缀 KV 已在 HBM，直接进 prefill                                                            |
| host hit     | H→D loadback      | 否          | 计入 `host_hit`，回载完成后才能算命中                                                       |
| storage hit  | Storage→H→D       | 否          | 先 prefetch 到 host，再 loadback                                                            |
| backup       | D→H               | 否          | 拷贝一份到 host pool，device 仍保留                                                         |
| demote       | D 释放，节点保留  | 是          | 有 host 副本的节点只释放 device 部分                                                        |
| evict        | 删节点或释放 host | 是          | write-through / write_through_selective 下没有副本的叶直接删除                              |
| write policy | 触发时机          | 不适用      | `write_through` 插入即备份；`write_through_selective` 只备份热点；`write_back` 驱逐时才备份 |
| L3 prefetch  | Storage→H         | 否          | 按 `--hicache-storage-prefetch-policy` 决定等多少                                           |

一次 host 命中的数据流：

```text
请求 → match_prefix 得到 device 段和 host 段
     → host 段分配 host 槽位，host_lock_ref 加一
     → loadback H2D 到新分配的 device 槽位
     → 拼进 prefix_indices，剩余后缀做 prefill
     → 请求结束释放 lock_ref 和 host_lock_ref
```

storage 命中在这条链路前面多一步：storage 查到 hash，prefetch 到 host staging，之后按 host 命中处理。

## `host_value` 与 `HostPoolGroup`

`host_value` 挂在 `UnifiedTreeNode.component_data[ct]` 上，每个组件各一份。它是一个 host pool 的槽位索引张量，不是 DRAM 指针，也不覆盖 L3。split 时按长度切片；host 驱逐时置 `None`；loadback 期间由 `host_lock_ref` 保护。

普通 MHA 只有 FULL 组件，`host_value` 指向 anchor pool 的槽位。混合模型才有 SWA / MAMBA / C128 等 side pool，每个组件各有一份 `host_value`。`HostPoolGroup` 是这些 host pool 的分配门面，本身不存 KV；它的分配与回收细节留到第 6 讲。

## 本讲边界

- 前缀缓存不减少 KV 总量。命中只省计算，缓存本身仍占 device 和 host 内存。
- 命中长度按 `page_size` 向下对齐，尾部不足一页的 token 一定会重算。
- `lock_ref` 的语义按组件不同：FULL 是路径锁，SWA 是窗口段锁，Mamba 是单节点锁。本讲只展开 FULL。
- L1 / L2 是单实例私有，只有 L3 可能跨实例共享；`file` backend 默认不是共享存储。
- `host_value` 是 host pool 槽位索引，不是 DRAM 指针；`HostPoolGroup` 是分配门面，不是 KV 存储。
- `session_ref` 是软保护，不是 `lock_ref` 那样的硬锁。
- 驱逐策略只改变候选顺序，不改变候选集合；只有 evictable leaf 能被驱逐。
- 经典 `RadixCache` 与 `UnifiedTreeCore` 的节点字段不同，不能把单测里的 `TreeNode` 当成默认实现。

## 本讲产出

一张统一观察表，列为：请求、prompt tokens、device_hit、host_hit、storage_hit、实际 prefill tokens、TTFT、load_back ms、evicted tokens、备注。表后固定写三句：TTFT 变化、吞吐变化、是否值得换出。

| 请求 | prompt tokens | device_hit | host_hit | storage_hit | 实际 prefill tokens | TTFT   | load_back ms | evicted tokens | 备注                               |
| ---- | ------------- | ---------- | -------- | ----------- | ------------------- | ------ | ------------ | -------------- | ---------------------------------- |
| R1   | 2200          | 0          | 0        | 0           | 2200                | 410 ms | 0            | 0              | 冷启动，建树                       |
| R2   | 2200          | 2000       | 0        | 0           | 200                 | 180 ms | 0            | 0              | device 命中 system prompt          |
| R3   | 2200          | 0          | 2000     | 0           | 200                 | 260 ms | 42 ms        | 2000           | device 副本被驱逐后重放，host 命中 |
| R4   | 2200          | 0          | 0        | 2000        | 200                 | 320 ms | 58 ms        | 2000           | 重启后从 storage 命中              |

示例行只示意填法，其余行由学员填写，至少覆盖冷启动、device 命中、host 命中、storage 命中、强制驱逐五种情况。

## 目录

- [课程总览](../README.md)
- [教学大纲](syllabus.md)
- [PPT 提纲](ppt-outline.md)
- [课堂实验](hands-on-exercise.md)
- [课后作业](homework.md)
- [实验环境](lab-environment.md)
- [实验脚本](code/prefix_cache_sim.py)

## 前后衔接

第 3、4 讲解决了“块怎么分配、怎么在请求内复用”；本讲解决“块怎么跨请求复用、满了往哪放”。第 6 讲会把复用和分层带到 GLM-5.3-Flash 的混合注意力结构里，那时 `host_value` 不再只有 FULL 组件一份，KDA state 和 indexer 也各有自己的生命周期。

## 源码索引

| 文件                                                                  | 关键内容                                                                           |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `python/sglang/srt/mem_cache/registry.py:80`                          | `default_radix_cache_factory`，默认落到 `create_unified_radix_cache`               |
| `python/sglang/srt/mem_cache/unified_radix_cache.py:162`              | `UnifiedRadixCache`，对外接口、组件注册与 HiCache 初始化                           |
| `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:414`  | `UnifiedTreeCore`，树结构、LRU 链与可驱逐叶集合                                    |
| `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:110`  | `UnifiedTreeNode`                                                                  |
| `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:812`  | `match_prefix`                                                                     |
| `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1390` | `_split_node`                                                                      |
| `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1610` | `evict_device_leaf`                                                                |
| `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1864` | `demote`                                                                           |
| `python/sglang/srt/mem_cache/unified_cache/components/base.py:48`     | `ComponentData`：`value`、`host_value`、`lock_ref`、`host_lock_ref`、`session_ref` |
| `python/sglang/srt/mem_cache/pool_host/group.py:35`                   | `HostPoolGroup`，anchor + side pool 的分配门面                                     |
| `python/sglang/srt/mem_cache/radix_cache.py:58`                       | `RadixKey`，带 `extra_key` / `cache_salt` / `page_aligned`                         |
| `python/sglang/srt/mem_cache/radix_cache.py:354`                      | 经典 `RadixCache.match_prefix`                                                     |
| `python/sglang/srt/mem_cache/radix_cache.py:665`                      | 经典 `RadixCache._split_node`                                                      |

行号按 SGLang `f4de6abee6` 核对；换版本后行号会漂移，读之前先 `git rev-parse --short HEAD`。
