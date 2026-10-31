# 第 5 讲：KV Cache 怎么复用与分层 — PPT 大纲

## 设计说明

- **总页数**：20 页
- **演讲时长**：120 分钟（30 min 概念 + 40 min 源码/演示 + 40 min 实验 + 10 min 答疑）
- **建议风格**：深蓝 + 暖灰；树节点和 device 用蓝色，host 用橙色，storage 用绿色，驱逐和风险用红色
- **字体**：标题 32–36 pt，正文 18–22 pt，代码和函数名用等宽字体
- **教学策略**：先用第 3、4 讲的 block 复用做对照，再画 radix tree，最后把树放到 L1/L2/L3 三层里
- **可视化策略**：
  - 对象图固定四层：`UnifiedRadixCache → UnifiedTreeCore → UnifiedTreeNode → ComponentData`
  - ABCD 例子用同一棵树贯穿 split、加锁、驱逐、解锁四步
  - HiCache 用三层阶梯图，箭头标清 backup、loadback、prefetch 的方向
- **动手环节**：第 15–17 页，40 分钟（CPU 模拟 15 min，server 观测 15 min，强制驱逐 10 min）

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：KV Cache 怎么复用与分层
- **副标题**：从 block 级复用，到 radix tree 与 HiCache 三层
- **课程信息**：KV Cache 八讲 · 第 5 讲 · 120 分钟
- 日期、讲师信息
- **一张图**：左边是第 3、4 讲的 block table，中间长成一棵 radix tree，右边是 L1 / L2 / L3 三层阶梯

### 第 2 页：本讲议程

- **第一部分**：从 block 复用到树级复用（30 min）— 第 3–8 页
  - 前缀缓存省什么 / 对象图 / 匹配 / 分裂 / 插入 / 默认实现
- **第二部分**：引用、状态与驱逐（30 min）— 第 9–13 页
  - 三套引用语义 / 四状态 / ABCD / 命中条件 / 驱逐策略
- **第三部分**：HiCache 与动手实验（50 min）— 第 14–17 页
  - L1/L2/L3 / host 命中数据流 / CPU 模拟 / server 观测
- **总结与答疑**：10 min — 第 18–20 页
- 本讲只回答三个问题：前缀缓存省什么、命中要满足什么；radix tree 怎么匹配和驱逐；HBM 放不下时怎么分层

---

## 第一部分：从 block 复用到树级复用（30 min，第 3–8 页）

### 第 3 页：block 级复用解决了什么，没解决什么

- 第 3 讲：nano-vllm 用前缀哈希链让多个请求指向同一批 block，带请求级 swap
- 第 4 讲：vLLM 在显存不足时 preempt 单个请求，走 recompute，不是 swap
- 两套机制都以 block 为单位，复用跟着块的生命周期走
- 没回答的问题：哪些前缀值得留、留不下时先扔谁、能不能放到 HBM 之外
- **一句话**：block 级复用管“同一条前缀在不在”，树级复用要管“整棵前缀树怎么组织、怎么淘汰、往哪放”

### 第 4 页：前缀缓存省的是 prefill

- 命中部分不再做 QKV 投影，也不再对这些位置做注意力计算
- decode 仍然逐 token 生成，不受影响
- 收益：TTFT 下降、prefill 总算力下降、共享前缀只存一份 KV
- 代价：常驻内存、lookup / split 开销、驱逐抖动、page 对齐截断、loadback 延迟、跨租户侧信道
- **一句话**：前缀缓存省计算，不省显存；命中率不高时，它可能反而降低并发上限

### 第 5 页：对象图：树、节点、组件、host pool

```text
UnifiedRadixCache
├── UnifiedTreeCore
│   └── UnifiedTreeNode
│       └── component_data[FULL / SWA / MAMBA / C128]
│           ├── value（device 槽位）
│           ├── host_value（host 槽位）
│           ├── lock_ref / host_lock_ref
│           └── session_ref
├── HostPoolGroup（anchor + side pools）
└── UnifiedSessionRefTracker
```

- `UnifiedRadixCache`（`unified_radix_cache.py:162`）管对外接口、组件注册和 HiCache 初始化
- `UnifiedTreeCore`（`unified_tree_core.py:414`）管树结构、LRU 链和可驱逐叶集合；`UnifiedTreeNode` 在 `:110`
- `ComponentData`（`components/base.py:48`）按组件保存 `value`、`host_value`、`lock_ref`、`host_lock_ref`、`session_ref`
- `value` / `host_value` 都是槽位索引张量，不是裸指针
- **一句话**：树节点只记“KV 在哪里”，真正的 K/V 在 device pool 和 host pool 里

### 第 6 页：匹配：key 不只是 token

- `RadixKey` 同时带 `token_ids`、`extra_key`、`cache_salt`
- 匹配前先 `page_aligned(page_size)`，尾部不足一页直接截掉
- `match_prefix()`（`unified_tree_core.py:812`）从根开始按 `child_key` 找子节点
- 返回 `device_indices`、`host_hit_length`、`best_match_node`
- `device_indices` 是 device 上可直接复用的槽位；`host_hit_length` 是需要回载的 Full-KV token 数
- **一句话**：命中长度不是 prompt 长度，先过 token、命名空间和 page 对齐三道关

### 第 7 页：分裂：匹配停在一条边中间

```text
原节点：ABCD
匹配 AB
分裂后：AB（新父节点）→ CD（原子节点）
```

- `_split_node()`（`unified_tree_core.py:1390`）新建父节点，原节点保留后缀
- 新节点复制 `lock_ref`，`value` / `host_value` 按 split 长度切片
- 哈希链也按 split 长度重新切分
- 分裂只切元数据，不拷贝 KV
- **一句话**：split 是树的结构操作，KV 槽位不动，动的是节点边界和引用

### 第 8 页：插入与默认实现

- 插入发生在 prefill 结束后，新块按 `hash_value` 链挂到树上
- 只有 page 对齐的 span 才会登记
- `registry.py:80` 的 `default_radix_cache_factory` 在常规路径最后返回 `create_unified_radix_cache`
- 经典 `RadixCache` 主要出现在 `PureSWARadixCache`、`FlexKVRadixCache` 和单测里
- 经典 `TreeNode` 用单个 `value` 和 `lock_ref` 表示 KV 与锁，用来建立直觉；不要把它当成默认实现
- **一句话**：HEAD 默认走 unified，经典实现是教学和单测的参照

---

## 第二部分：引用、状态与驱逐（30 min，第 9–13 页）

### 第 9 页：三套引用语义

| 字段        | 实现                | 粒度     | 语义                        | 释放                                   |
| ----------- | ------------------- | -------- | --------------------------- | -------------------------------------- |
| `ref_count` | nano-vllm `Block`   | block 级 | 共享同一 block 的请求各加一 | 归零回 free list                       |
| `ref_cnt`   | vLLM `KVCacheBlock` | block 级 | 被请求引用时加一            | 归零进 free list，哈希保留时仍可命中   |
| `lock_ref`  | SGLang radix node   | node 级  | 沿节点到根的路径累加        | 沿同一路径减一，归零且是 leaf 才可驱逐 |

- `ref_count` / `ref_cnt` 数“谁持有这个 block”，`lock_ref` 数“这条路径上有多少活跃请求”
- FULL 组件是 path-lock，从节点走到根；root 初值是 1，请求释放不会减到它
- SWA 是 segment-lock，只锁到滑动窗口边界；Mamba 是单节点锁
- **一句话**：三套字段挂在不同层，读代码先看它数的是 block、节点还是路径

### 第 10 页：节点四状态

| 状态           | 判定                                                  | 能否驱逐 | 说明                  |
| -------------- | ----------------------------------------------------- | -------- | --------------------- |
| locked         | `lock_ref > 0`                                        | 否       | 路径上有活跃请求      |
| evictable leaf | 无 device KV 子节点，`lock_ref == 0`，`value != None` | 是       | 唯一候选              |
| internal       | 至少一个 device KV 子节点                             | 否       | 等子节点先删          |
| evicted        | `value == None`                                       | 不适用   | 可能还有 `host_value` |

- 只有 evictable leaf 进候选集，internal 和 locked 都不进
- leaf 删除后，父节点满足条件才回到候选
- `evicted` 不等于没数据：有 host 副本就从 host 命中并回载，没有才重算
- **一句话**：驱逐看的是“叶 + 未锁 + device 有值”，不是“树上最老的节点”

### 第 11 页：ABCD：一次 split、加锁、驱逐、解锁

| 步骤 | 动作             | 树      | `lock_ref` | 可驱逐节点 |
| ---- | ---------------- | ------- | ---------- | ---------- |
| 1    | 插入 ABCD        | ABCD    | 全 0       | ABCD       |
| 2    | 匹配 AB，分裂    | AB → CD | 全 0       | CD         |
| 3    | 对 AB 加锁       | AB → CD | AB=1，CD=0 | CD         |
| 4    | 驱逐 CD，解锁 AB | AB      | AB 回到 0  | AB         |

- 第 3 步：锁 AB 只保护 AB，CD 的 `lock_ref` 仍是 0，照样能被驱逐
- 第 4 步：CD 删掉后 AB 没有子节点，`lock_ref` 归零，才重新成为候选
- **一句话**：锁保护的是路径，不是整棵树；驱逐候选随叶的增删动态变化

### 第 12 页：命中条件：token、命名空间、page 对齐

1. 从第 0 个 token 起逐 token 相同
2. `extra_key` 相同：调用方用它分隔 LoRA / adapter 或业务命名空间
3. `cache_salt` 相同：租户或请求隔离
4. 匹配长度按 `page_size` 向下对齐

- 反例：同 prompt 换 LoRA A / B，调用方设置不同 `extra_key`，命中长度为 0
- 反例：同 prompt 换 `cache_salt`，落在不同子树，命中长度为 0
- `page_size=16` 时，60 token 的共享前缀只能命中 48，尾部 12 token 重算
- **一句话**：前缀缓存不是“文本一样就命中”，命名空间和页对齐都会改变结果

### 第 13 页：驱逐：候选、策略与回退

- 候选集合只有 evictable leaves
- `--radix-eviction-policy` 可选 `lru`、`lfu`、`slru`、`priority`、`tlru`
- 策略只决定候选之间的先后，不改变候选集合
- `session_ref` 是 `--enable-session-radix-cache` 下的软保护，不等于 `lock_ref` 的硬锁
- `evict_device_leaf()`（`:1610`）：有 host 副本就 `demote`；没有副本，write-through / write_through_selective 直接删，write-back 先 backup 再 demote
- **一句话**：先定候选集合，再谈淘汰顺序；软保护只能推迟，不能阻止

---

## 第三部分：HiCache 与动手实验（50 min，第 14–17 页）

### 第 14 页：HiCache 三层：L1 / L2 / L3

| 层  | 介质            | 共享范围                         |
| --- | --------------- | -------------------------------- |
| L1  | GPU HBM         | 单实例私有                       |
| L2  | host DRAM       | 单实例私有，同节点两实例也不互读 |
| L3  | storage backend | 取决于 backend                   |

- `file` backend 默认写 `/tmp/hicache`，是节点本地目录
- Mooncake / HF3FS / NIXL / AIBrix 才可能跨实例共享，前提是所有实例指向同一命名空间
- L2 不能跨实例共享；需要跨实例复用时只能走 L3
- **一句话**：L1 / L2 私有，只有 L3 可能共享；共享范围由 backend 配置决定

### 第 15 页：[动手] CPU 模拟：命中、加锁、驱逐

- 15 分钟，只用 Python 标准库
- 跑四个命令：

  ```bash
  python3 code/prefix_cache_sim.py --demo abcd
  python3 code/prefix_cache_sim.py --demo share
  python3 code/prefix_cache_sim.py --demo evict
  python3 code/prefix_cache_sim.py --self-check
  ```

- `abcd`：看 split、加锁、驱逐、解锁四步
- `share`：看 LoRA / salt 不命中，以及 `page_size=16` 时 60 token 只命中 48
- `evict`：12 token 容量下锁住 R2，再插入 R5，看谁被驱逐
- **完成标准**：能解释每一步的 `lock_ref` 和可驱逐节点

### 第 16 页：一次 host 命中要走几步

```text
请求 → match_prefix 得到 device 段和 host 段
     → host 段分配 host 槽位，host_lock_ref 加一
     → loadback H2D 到新分配的 device 槽位
     → 拼进 prefix_indices，剩余后缀做 prefill
     → 请求结束释放 lock_ref 和 host_lock_ref
```

- storage 命中多一步：storage 查到 hash，prefetch 到 host staging，再按 host 命中处理
- 三层命中要分开看；只看 device 会低估
- 关注 `sglang:` 下的 `cache_hit_rate`、`hicache_backup_tokens_total`、`load_back_tokens_total`、`load_back_duration_seconds`、`hicache_host_used_tokens`
- **一句话**：命中层级越低，回载链路越长；TTFT 要把 loadback 算进去

### 第 17 页：[动手] server 观测与强制驱逐

- 15 分钟 server 观测 + 10 分钟强制驱逐
- 启动参数：`--enable-hierarchical-cache`、`--hicache-ratio 2`、`--hicache-write-policy write_through`、`--hicache-storage-backend file`、`--page-size 1`
- 发四条请求：R1 冷启动、R2 同 prompt 重放、S1 换 `cache_salt`、E1 换 `extra_key`
- host / storage 命中用 R1 重放观察，分别记作 R3 / R4
- 用 `/metrics` 看三层命中和 loadback
- 把 `--max-total-tokens` 压到 4096，观察 `evicted` 和命中层级变化
- 没有 GPU 时用示例日志填表，CPU 部分照常完成
- **完成标准**：观察表至少覆盖冷启动、device 命中、host 命中、storage 命中、强制驱逐

---

## 总结与答疑（10 min，第 18–20 页）

### 第 18 页：`host_value` 与 `HostPoolGroup`

- `host_value` 挂在 `component_data[ct]`，每个组件各一份
- 它是 host pool 的槽位索引张量，不是 DRAM 指针，也不覆盖 L3
- split 时按长度切片；host 驱逐时置 `None`；loadback 期间由 `host_lock_ref` 保护
- 普通 MHA 只有 anchor pool；混合模型才有 SWA / MAMBA / C128 side pool
- `HostPoolGroup` 是分配门面，本身不存 KV
- **一句话**：`host_value` 是“host 上哪个槽”，不是“内存里哪个地址”

### 第 19 页：本讲边界与常见错误

- 以为前缀缓存会减少 KV 总量：它只省计算，缓存仍占内存
- 以为命中长度等于 prompt 长度：page 对齐会截掉尾部
- 以为 `lock_ref` 都是路径锁：FULL 是路径锁，SWA 是窗口段锁，Mamba 是单节点锁
- 以为 L2 可以跨实例共享：L1 / L2 都是单实例私有
- 把 `host_value` 当成 DRAM 指针：它是 host pool 槽位索引
- 把 `session_ref` 当成硬锁：它是软保护，内存不足时仍可驱逐
- 把驱逐策略当成候选集合：策略只排序，候选只有 evictable leaf
- 把单测里的经典 `TreeNode` 当成默认实现：HEAD 默认走 unified
- **一句话**：先分清层、锁和命中边界，再谈调参

### 第 20 页：小结、下次预告与参考资料

- 三个带走：
  1. 前缀复用省 prefill，不省显存；命中要过 token、命名空间、page 对齐三道关
  2. 树节点只记槽位，`value` / `host_value` 分别指向 device 和 host；驱逐候选只有 evictable leaf
  3. HiCache 的 L1 / L2 私有，L3 才可能共享；命中层级越低，回载链路越长
- 下次：第 6 讲《KV Cache 在 GLM-5.3-Flash 里怎么变成三本账》— KDA state、indexer、full KV
- 课后练习：统一观察表 + 三句结论；完成 `prefix_cache_sim.py --self-check`
- 参考资料：
  - [第 5 讲实验](hands-on-exercise.md)
  - [第 5 讲教学大纲](syllabus.md)
  - [Prefix Caching 技术详解](../../../09_inference_system/kv_cache/01_concepts/prefix_caching/prefix_caching.md)
  - [HiCache 深入详解](../../../09_inference_system/sglang/hicache_deep_dive.md)
  - `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py`
  - `python/sglang/srt/mem_cache/radix_cache.py`
  - `python/sglang/srt/mem_cache/pool_host/group.py`
- Q&A：留 3 分钟，重点检查 split 后 `lock_ref` / `host_value` 怎么变、host 命中为什么要回载、驱逐候选为什么只有叶节点
