# 第 5 讲：KV Cache 怎么复用与分层 — 教学大纲

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 课时 | 120 分钟 |
| 先修 | 第 3、4 讲的 block 生命周期与 prefix cache；能读 Python |
| 代码基线 | 本地 SGLang，HEAD `f4de6abee6`；第 3 讲 nano-vllm `b9dd55a` 作引用语义对照 |
| 本讲定位 | 从 block / 哈希级前缀复用与 nano-vllm 的请求级 swap，走到系统化的跨请求复用与 HiCache 分层 |

## 教学目标

- 知识目标：掌握 prefix caching 的收益与代价、radix tree 的匹配与分裂、三套引用字段、四状态模型、命中条件和 HiCache 的分层边界。
- 能力目标：给定一条请求，能算出命中长度和实际 prefill token 数，并判断换出是否值得。
- 素养目标：把“缓存命中”和“存储换出”分开讨论，报数字时同时说明命中层级和是否含 loadback。

## 时间安排

| 时间 | 环节 | 内容 |
| --- | --- | --- |
| 30 min | 概念 | 前缀缓存省什么、代价清单；radix tree 匹配与分裂；`ref_count` / `ref_cnt` / `lock_ref`；四状态模型与 ABCD 例子；命中条件；HiCache 的 hit / loadback / backup / demote / evict |
| 40 min | 源码与演示 | 读 `UnifiedTreeCore` 的 `match_prefix`、`_split_node`、`evict_device_leaf`、`demote`；对照经典 `RadixCache`；跑 `code/prefix_cache_sim.py` 的 ABCD 与驱逐演示 |
| 40 min | 实验 | CPU 模拟、server 打开 HiCache 看三层命中、强制驱逐，填统一观察表 |
| 10 min | 答疑 | split 后 value 怎么切；`lock_ref` 何时归零；host / storage 命中为什么要回载；写策略怎么选 |

## 重点与难点

| 类型 | 内容 | 处理方式 |
| --- | --- | --- |
| 重点 | 前缀匹配与分裂 | 用 ABCD 例子手推一遍，再跑脚本看状态 |
| 重点 | 三套引用语义 | 按 block 级 / node 级 / 路径累加三种粒度对照 |
| 重点 | HiCache 各动作边界 | hit、loadback、backup、demote / evict 分开画 |
| 难点 | `lock_ref` 的路径累加 | 先讲谁持有节点，再讲 split 为什么复制 `lock_ref` |
| 难点 | `host_value` 的含义 | 它是指向 host pool 的槽位索引，不是 DRAM 指针 |
| 难点 | 命中条件的命名空间 | 用同 prompt 换 LoRA / salt 的反例 |

## 引用语义对照

| 字段 | 实现 | 粒度 | 语义 | 释放 |
| --- | --- | --- | --- | --- |
| `ref_count` | nano-vllm `Block` | block 级 | 0 或 1+；共享同一 block 的请求各加一 | 最后一个请求释放时归零，块回 free list |
| `ref_cnt` | vLLM `KVCacheBlock` metadata | block 级 | 0 或 1+；被请求引用时加一 | 归零后进 free list，`_block_hash` 保留时仍可被 prefix cache 命中 |
| `lock_ref` | SGLang radix node | node 级 | 沿节点到根的路径累加，root 不参与；split 时新节点复制 | 沿同一路径减一，归零且是 leaf 才进入驱逐候选 |

`ref_count` 和 `ref_cnt` 数的是“谁持有这个 block”，`lock_ref` 数的是“这条路径上有多少活跃请求”。三套字段不能互相替代，读代码时先看它挂在哪一层。

## 默认实现与源码锚点

HEAD 默认走 `UnifiedRadixCache` / `UnifiedTreeCore`。`registry.py:80` 的 `default_radix_cache_factory` 在常规路径最后返回 `create_unified_radix_cache`。经典 `RadixCache` 主要出现在三处：`PureSWARadixCache` 和 `FlexKVRadixCache` 继承它，单测直接构造它。课堂上用它建立树结构直觉，主线代码读 unified。

| 实现 | 文件 | 锚点 |
| --- | --- | --- |
| unified（默认） | `unified_cache/unified_tree_core.py` | `UnifiedTreeCore.match_prefix`（`:812`）、`_split_node`（`:1390`）、`evict_device_leaf`（`:1610`）、`demote`（`:1864`） |
| 经典（直觉与单测） | `radix_cache.py` | `RadixCache.match_prefix`（`:354`）、`_split_node`（`:665`）、`evict`（`:553`）、`inc_lock_ref`（`:583`）、`dec_lock_ref`（`:598`） |

两套实现的节点字段不同：unified 的 `UnifiedTreeNode.component_data` 是按组件索引的列表，经典 `TreeNode` 只有单个 `value` 和 `lock_ref`。读行号时先确认自己打开的是哪个文件。

## 四状态模型

| 状态 | 判定 | 能否驱逐 | 说明 |
| --- | --- | --- | --- |
| locked | `lock_ref > 0` | 否 | 路径上有活跃请求；祖先被锁不阻止后代被驱逐，后代看自己的 `lock_ref` |
| evictable leaf | 没有 device KV 子节点，`lock_ref == 0`，`value != None` | 是 | 驱逐候选只有这一类 |
| internal | 至少有一个 device KV 子节点 | 否 | 要等子节点先被删掉 |
| evicted | `value == None` | 不适用 | 可能还有 `host_value`；有 host 副本就从 host 命中并回载，没有则重算 |

四条规则：

- 只有 evictable leaf 进候选集，internal 和 locked 都不进。
- `lock_ref` 沿路径累加，root 不参与释放。unified 实现里 root 的初值是 1，请求释放不会减到它。
- split 后新节点复制 `lock_ref`，`value` / `host_value` 按 split 长度切片。
- leaf 被删除后，父节点只有在没有 device KV 子节点且 `lock_ref == 0` 时才回到候选。

ABCD 四步：

| 步骤 | 动作 | 树 | `lock_ref` | 可驱逐节点 |
| --- | --- | --- | --- | --- |
| 1 | 插入 ABCD | 一个节点 ABCD | 全 0 | ABCD |
| 2 | 匹配 AB，节点在匹配点分裂 | AB → CD | 全 0 | CD |
| 3 | 对 AB 加锁 | AB → CD | AB=1，CD=0 | CD |
| 4 | 驱逐 CD，再解锁 AB | AB | AB 回到 0 | AB |

第 3 步说明锁的保护范围：锁 AB 只保护 AB，CD 的 `lock_ref` 仍是 0，照样能被驱逐。第 4 步说明回退条件：CD 删掉后 AB 没有子节点，`lock_ref` 归零，才重新成为候选。

## 命中条件

一条请求要命中前缀，需要同时满足：

1. 从第 0 个 token 起逐 token 相同。
2. `extra_key` 相同。LoRA / adapter 会拼进这个字段，调用方也可以自定义命名空间。
3. `cache_salt` 相同，用于租户或请求隔离。
4. 匹配长度按 `page_size` 向下对齐。

反例：两条请求的 prompt 一模一样，一条带 LoRA A，一条带 LoRA B。`extra_key` 不同，`RadixKey.child_key` 生成的字典键不同，两条 prompt 落在不同子树，命中长度为 0。换 `cache_salt` 同理。`page_size=16` 时，60 token 的共享前缀只能命中 48，尾部 12 token 重算。

## 驱逐

策略是启动参数 `--radix-eviction-policy`，可选 `lru`、`lfu`、`slru`、`priority`、`tlru`。策略只决定候选之间的先后，不改变候选集合。

候选集合只有 evictable leaves。一个叶被删掉后，父节点若满足条件才加入候选，这个从叶向根的回退遍历就是平时说的“分支”。它不是一个策略，也不与会话保护并列。`session_ref` 是 `UnifiedRadixCache` 在 `--enable-session-radix-cache` 下给会话打的软保护，会推迟驱逐，但不等同于 `lock_ref` 的硬锁。

## 收益与代价

收益：

- 省掉前缀的 QKV 投影和自注意力计算，TTFT 下降。
- 共享前缀在 device 上只存一份 KV，多个请求的 block table 指向同一批槽位。
- 高并发、长 system prompt 的场景下，prefill 总算力下降，吞吐上升。

代价：

- 缓存常驻内存。命中率不高时，它挤占可用 KV，降低并发上限。
- 每次请求都要 lookup；insert 和 split 有元数据开销，split 还要切 value。
- 驱逐后马上又命中就是抖动，重算成本可能高于省下的计算。
- page 对齐会截掉尾部 token，命中长度不是想要多少就有多少。
- host / storage 命中要搬运，loadback 延迟可能吃掉 TTFT 收益。
- 跨租户共享有侧信道风险，靠 `extra_key` / `cache_salt` 隔离，不是默认安全。

## HiCache 的边界

| 动作 | 方向 | 释放 device | 说明 |
| --- | --- | --- | --- |
| device hit | 无搬运 | 否 | 前缀 KV 已在 HBM，直接进 prefill |
| host hit | H→D loadback | 否 | 计入 `host_hit`，回载完成后才能算命中 |
| storage hit | Storage→H→D | 否 | 先 prefetch 到 host，再 loadback |
| backup | D→H | 否 | 拷贝一份到 host pool，device 仍保留 |
| demote | D 释放，节点保留 | 是 | 有 host 副本的节点只释放 device 部分 |
| evict | 删节点或释放 host | 是 | write_through 下没有副本的叶直接删除 |
| write policy | 触发时机 | 不适用 | write_through 插入即备份；write_through_selective 只备份热点；write_back 驱逐时才备份 |
| L3 prefetch | Storage→H | 否 | 按 `--hicache-storage-prefetch-policy` 决定等多少 |

L1 HBM 和 L2 host DRAM 是单实例私有。L3 是否共享取决于 backend：Mooncake、HF3FS 这类跨实例共享，file 只是本地目录。存储层保存的是 KV 数据，命中判断由本地维护的 hash 链发起，存储层只负责按 key 查数据。

一次 host 命中的数据流：

```text
请求 → match_prefix 得到 device 段和 host 段
     → host 段分配 host 槽位，host_lock_ref 加一
     → loadback H2D 到新分配的 device 槽位
     → 拼进 prefix_indices，剩余后缀做 prefill
     → 请求结束释放 lock_ref 和 host_lock_ref
```

storage 命中在这条链路前面多一步：storage 查到 hash，prefetch 到 host staging，之后按 host 命中处理。

## `host_value`

- 位置：`UnifiedTreeNode.component_data[ct].host_value`，每个组件各一份。
- 类型：host pool 的槽位索引张量，不是 DRAM 指针，也不覆盖 L3。
- 经典 `TreeNode` 没有这个字段。
- split 时按长度切片；host 驱逐时置 `None`；loadback 期间由 `host_lock_ref` 保护。
- 普通 MHA 只有 FULL 组件；SWA / MAMBA / C128 是混合模型的组件，各自有独立的 `host_value`。

## `HostPoolGroup` 的位置

`HostPoolGroup` 是 host pool 的分配门面，本身不存 KV。
它由一个 anchor pool 和若干 side pool 组成。
普通 MHA 只有 anchor，anchor 就是 KV host pool，`host_value` 指向它的槽位。
混合模型才有 SWA / MAMBA / C128 等 side pool，每个组件各有一份 `host_value`。
它的分配与回收细节留到第 6 讲。

## 本讲产出

一张统一观察表：

| 请求 | prompt tokens | device_hit | host_hit | storage_hit | 实际 prefill tokens | TTFT | load_back ms | evicted tokens | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 2200 | 0 | 0 | 0 | 2200 | 410 ms | 0 | 0 | 冷启动，建树 |
| R2 | 2200 | 2000 | 0 | 0 | 200 | 180 ms | 0 | 0 | device 命中 system prompt |

示例行只示意填法，其余行由学员填写，至少覆盖冷启动、device 命中、host 命中、storage 命中、强制驱逐五种情况。

表后固定写三句：

TTFT 变化：……
吞吐变化：……
是否值得换出：……

## 参考材料

主参考：

- [Prefix Caching 技术详解](../../../09_inference_system/kv_cache/01_concepts/prefix_caching/prefix_caching.md)：第 5.1 节的简化模型给出 P=2000、Q=200 时约 5.5× 的 TTFT 加速比，作业按这个模型算，不按命中 token 线性折算。
- [HiCache 深入详解](../../../09_inference_system/sglang/hicache_deep_dive.md)：L1 / L2 / L3 的划分、prefetch 三阶段与三种写策略。

源码（SGLang `f4de6abee6`）：

- `python/sglang/srt/mem_cache/registry.py:80` `default_radix_cache_factory`：默认落到 `create_unified_radix_cache`。
- `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:110` `UnifiedTreeNode`，`:812` `match_prefix`，`:1390` `_split_node`，`:1610` `evict_device_leaf`，`:1864` `demote`。
- `python/sglang/srt/mem_cache/radix_cache.py:354` 经典 `match_prefix`，`:553` `evict`，`:583` `inc_lock_ref`，`:598` `dec_lock_ref`，`:665` `_split_node`。
- `python/sglang/srt/mem_cache/unified_cache/components/base.py:48` `ComponentData`：`value`、`host_value`、`lock_ref`、`host_lock_ref`、`session_ref`。
- `python/sglang/srt/mem_cache/pool_host/group.py:35` `HostPoolGroup`。

延伸：

- [KV Cache 复用系列](../../../09_inference_system/kv_cache/06_kv_reuse/kv-reuse-overview.md)：KVLink 与 DroidSpeak，跨出同前缀、同模型之后的复用路线。
- [RadixAttention 技术详解](../../../09_inference_system/kv_cache/01_concepts/prefix_caching/radix_attention.md)：适合建立树结构直觉。文中的 `RadixNode.ref_count` 是教学示意，与 HEAD 有四处差异：字段实际叫 `lock_ref`；锁沿路径累加且 root 不参与释放；`inc_lock_ref` 的 delta 是 token 数，不是节点数；节点的 `value` 是 KV pool 槽位索引张量，不是 block 对象列表。
