# KV Cache L1↔L2 数据流深度分析

> 2026-07-28 | 基于 SGLang v0.5.14 源码（`sglang/srt/mem_cache/hiradix_cache.py`），行号指向该文件。
> 本文是 [SGLang KV Pool 管理](./sglang-kv-pool-management.md) §6（HiCache 多层存储）的深度配套文章——§6 给出架构全景，本文给出 L1↔L2 之间三个核心操作的完整代码级流程与决策逻辑。

SGLang 的 KV cache 以 radix tree 组织。每个 `TreeNode` 代表一个 prefix 匹配点，持有若干 device KV pages（`node.value`）。L1（GPU HBM）与 L2（DRAM）之间的数据流动包含三个操作：**write_backup**（HBM → L2）、**eviction**（释放 HBM）、**load_back**（L2 → HBM）。三个操作都以 `TreeNode` 为操作单位，串在一起构成了推理引擎在"显存有限 vs 序列无限"这个根本矛盾上的完整应对策略。

---

## 一、为什么需要 L2 —— GPU HBM 的容量天花板

KV cache 是 attention 计算的副产品，每生成一个 token，K 和 V 就追加一行。在标准 MHA 下，一个 token 的 KV 需要 $2 \times \mathrm{num\ layers} \times \mathrm{num\ kv\ heads} \times \mathrm{head\ dim} \times \mathrm{dtype\ size}$ 字节——以 Llama-3-8B (BF16) 为例，每 1K tokens 产生约 128MB 的 KV cache，128K context 就是 16GB。而 GPU HBM 留给 KV cache 的空间是有限的：H100 80GB 中模型权重占 ~16GB，加上中间激活值和框架开销，留给 KV cache 的容量在大量并发请求下迅速吃紧。

这引出一个根本性的选择：HBM 满了，旧的 KV 数据往哪里去？

第一个选项是**丢弃**——释放旧数据，后续共享相同 prefix 的请求必须重新 prefill。这在小规模推理中可以接受，但在多轮对话或共享系统提示词的场景下，丢弃意味着大量重复计算。

第二个选项是**搬走**——把旧数据搬到更大但更慢的存储层（CPU DRAM），等后续请求需要时再搬回来。"空间换时间"：用廉价的大容量存储，换取未来 prefill 计算量的减少。

这就是 L2 存在的根本原因。但"搬走"和"搬回来"不是简单的 memcpy——三个操作（什么时候搬、搬谁、怎么搬回来）组成了 L1↔L2 数据流的完整生命周期，每一步都有精细的约束和决策逻辑。

---

## 二、操作一：write_backup —— HBM → L2

**write_backup 将 node 的 KV 数据从 GPU HBM 拷贝到 host DRAM，使后续请求可以通过 load_back 恢复，避免重算。** 它是"空间换时间"的起点——如果这一步不做，后面两步（eviction 和 load_back）就没有存在的意义。

### 2.1 触发时机：两种策略，两种哲学

write_backup 的触发不是"HBM 快满时"——那是 eviction。write_backup 的触发是由 **写策略** 决定的：

| 策略              | 触发条件                                                    | 时机                        |
| ----------------- | ----------------------------------------------------------- | --------------------------- |
| **write_through** | `hit_count >= write_through_threshold`（默认 1）            | node 被第二次访问时立即备份 |
| **write_back**    | HBM 满 → eviction 触发 → `write_backup(x, write_back=True)` | 延迟到被迫逐出时才备份      |

> 关于两种策略的完整对比见 [第五章](#五策略权衡write_through-vs-write_back)。本章先聚焦于 write_backup 操作本身的流程，以 write_through 模式为主线，write_back 模式的差异在小节 2.3 中单独说明。

### 2.2 write_through 模式下的完整流程

write_through 模式下，备份的触发路径是：请求 prefill → `match_prefix` 匹配到 TreeNode → `_inc_hit_count` → `write_backup`。每一步都有明确的保护逻辑。

**第一步：命中计数与阈值判断：**

`_inc_hit_count(node)`（line 896-905）统计 node 的命中次数。这个计数器是"这个 prefix 值得缓存"的粗略信号——被命中越多次，越可能有后续请求复用。两个 guard 条件保护这个判断的质量：

- `write_policy == "write_back"` → 直接跳过。write_back 模式不在这里触发备份，而是推迟到 eviction。
- chunked prefill 的中间 chunk → 跳过。chunked prefill 将长 prompt 拆成多个 chunk 分步执行，中间的 chunk node 的"命中"是因为同一个请求的续传而非新请求的复用，不应触发备份。

满足条件后 `hit_count += 1`。如果 node 尚未 backup 且 `hit_count >= write_through_threshold`（默认 1，即第二次访问），调用 `write_backup(node)`。

**第二步：连续前缀不变量：**

```python
# hiradix_cache.py:763-766
if not write_back and not node.parent.backuped:
    return 0
```

write_backup 要求父节点已经完成备份，否则直接返回 0 跳过。这个不变量的动机是**确保备份的节点在 radix tree 中形成从 root 开始的连续前缀，避免 gap**。

为什么 gap 是问题？load_back 沿树向上恢复 evicted 节点（见[第四章](#四操作三load_back--l2-hbm)），如果树中存在"祖先未备份、子节点已备份"的断点，恢复时祖先节点的 host 数据不存在，子节点的 host 数据也无法和 GPU 端数据拼接成连续的前缀——恢复链路断裂。连续前缀不变量从根本上防止了这个场景。

**第三步：DMA 搬运与空间竞争：**

```python
# hiradix_cache.py:768-772
host_indices = self.cache_controller.write(
    device_indices=node.value)
```

`cache_controller.write()` 发起 DMA 将 GPU pages 拷贝到 host pool，返回 host page indices。host pool 以 page 为单位管理，返回的 indices 是 host 端的"地址"，供后续 eviction 和 load_back 定位。

但 host pool 也有容量上限。如果 DMA 失败（host pool 满），触发 `evict_host(num_tokens)`（line 773-779）——从 `evictable_host_leaves` 中选出优先级最低的节点，释放其 host pool pages，并从 radix tree 中删除该节点，然后重试 `cache_controller.write()`。

**第四步：记录 host 位置：**

```python
# hiradix_cache.py:781
node.host_value = host_indices.clone()
```

host indices 记录在 node 上后，`node.backuped` property（检查 `self.host_value is not None`）自动变为 True。这是后续所有操作的关键状态标记——eviction 根据它决定"直接删除还是仅释放 device pages"，load_back 根据它定位 host 端数据。

**第五步：异步跟踪与锁定保护：**

```python
# hiradix_cache.py:783-785
self._track_write_through_node(node)
self.inc_lock_ref(node)
```

`_track_write_through_node` 将 node 加入 `ongoing_write_through` 字典，等待 DMA 完成 ack。DRAM 的 DMA 是异步的——`write()` 返回只表示命令被提交，实际数据搬运可能尚未完成。`ongoing_write_through` 字典让 eviction 流程在处理这些 node 时等待 ack 完成。

`inc_lock_ref(node)` 沿 parent 链向上，增加 node 及其所有祖先的 `lock_ref`。`lock_ref > 0` 的节点不会被 eviction 选中——这保证了写入 L2 的过程中，node 不会同时被另一个线程 evict。

### 2.3 write_back 模式下的差异

write_back 模式下的 `write_backup(node, write_back=True)` 有三处关键差异：

**跳过 backup 不变式检查**。write_back 模式下的 eviction 从 `evictable_leaves` 底部向上选择候选节点，此时祖先可能尚未 backup（甚至祖先节点可能根本不会被 evict），不适用"父节点必须已 backup"的不变式。代码中 `if not write_back and not node.parent.backuped` 的 guard 为 write_back 模式单独放行。

**不调用 `inc_lock_ref`**。write_back 模式下的 write_backup 发生在 eviction 流程内部，eviction 流程对 lock_ref 有统一的处理策略（末尾统一 wait ack + 统一 evict），不需要在 write_backup 这一步加锁。

**DMA ack 在 eviction 末尾统一等待**。`evict()` 末尾调用 `writing_check(write_back=True)` 等待所有 DMA 完成，然后统一调用 `_evict_backuped` 释放已完成备份的节点的 device pages（line 1072-1076）。

---

## 三、操作二：eviction —— 释放 HBM 空间

**eviction 在 GPU HBM 的 KV pool 接近耗尽时，选择低优先级的 node，释放其 device pages（`node.value`），腾出空间给新请求。** 它是容量天花板真正被触及时的应对机制——不是预防性的，是响应性的。

### 3.1 选谁逐出：优先级堆与 evictable_leaves

eviction 的候选池是 `evictable_leaves`——一个最小堆维护的叶子节点集合。进入这个集合的节点必须同时满足三个条件：

1. **`lock_ref == 0`**：没有被活跃请求或 write_backup/load_back 锁定
2. **所有子节点已 evicted**（`child.value == None`）：保证逐出是自底向上的叶子优先
3. **`node.value != None`**：持有 device pages（已 evicted 的节点不需要再 evict）

优先级由 `eviction_strategy` 计算（LRU / FIFO / priority），最小堆保证每次弹出的是"最不重要"的节点。

### 3.2 逐出的三条分支

`evict()` 主循环（line 1033-1079）从堆中弹节点，逐节点处理。每个节点的处理方式取决于它的 **backup 状态 × write policy** 组合：

```text
check_decode_mem → evict_from_tree_cache → evict()
  → 从 evictable_leaves 构建优先级堆
  → 循环弹堆，逐节点处理:
      │
      ├─ lock_ref > 0 → 跳过
      │   （被 write_backup/load_back 锁定，不能动）
      │
      ├─ 非 backuped + write_through → _evict_regular → 直接删除
      │
      ├─ 非 backuped + write_back → write_backup(x, write_back=True) → 先备份
      │
      └─ backuped（任意策略）→ _evict_backuped → 仅释放 device pages
```

**分支一：`_evict_regular`（line 1057-1058）——非 backuped + write_through**

节点从未被第二次访问过，意味着它只被当前请求使用了一次。这类数据在 write_through 模式下被视为"低价值数据"——既然第二次访问都没有触发，说明没有其他请求共享这个 prefix。直接调用 `cache_controller.mem_pool_device_allocator.free(node.value)` 释放 device pages，然后 `_delete_leaf(node)` 将节点从 radix tree 中彻底删除。**数据永久丢失。**

**分支二：`write_backup(x, write_back=True)`（line 1050-1056）——非 backuped + write_back**

write_back 模式下，备份被推迟到了 eviction 这一刻。"这个数据是否值得保留"的决策不是在看它被共享了多少次（命中计数），而是在看它在 eviction 时的相对优先级——如果优先级仍足够高（堆中还有更差的节点先被 evicted），就给它一个备份到 L2 的机会。备份完成后节点变为 backuped 状态，不在此轮释放 device pages（由 eviction 末尾统一处理）。

**分支三：`_evict_backuped`（line 1059-1060）——backuped（任意策略）**

节点已经在 L2 中有备份，HBM 中的数据可以安全释放。关键操作：

```python
# hiradix_cache.py:1086-1091
cache_controller.evict_device(node.value)  # 释放 device pages 回 pool
node.value = None                           # 标记为已 evict
_update_leaf_status(node)                   # 刷新 evictable_leaves
_update_host_leaf_status(node)              # 加入 evictable_host_leaves
```

`node.value = None` 是 eviction 的核心标记。设置后，`node.evicted` property（`return self.value is None`）自动变为 True，后续请求匹配到此节点时触发 `host_hit` → `needs_host_load_back()` → `load_back`。

节点**保留在 radix tree 中**（不删除），因为 L2 中仍有数据。这一步体现了 eviction 与删除的本质区别：eviction 只释放 HBM 副本，节点仍"活着"——它的 host_value 在 L2 中，可以通过 load_back 重生。删除则是从树中移除节点，host_value 随节点一起释放，不可恢复。

### 3.3 evict_host：L2 也满时的补充机制

host pool 的容量虽然远大于 HBM，但在极端场景下仍可能耗尽。`evict_host`（line 774 触发，host pool 内部管理）从 `evictable_host_leaves` 中选择节点，释放其 host pool pages，并从 radix tree 中彻底删除（`parent.children.pop`）。

`evictable_host_leaves` 中的节点满足：`value == None`（HBM 已释放）+ `host_value != None`（L2 仍有数据）+ `lock_ref == 0`（无活跃引用）。这形成了一个清晰的逐出链：

```text
HBM 满 → evict_device（释放 HBM，保留 L2）
L2 满   → evict_host（释放 L2，删除节点，永久丢失）
```

下一步恢复只能依赖 L3（NVMe / 远程存储）或重新 prefill。

---

## 四、操作三：load_back —— L2 → HBM

**load_back 在后续请求匹配到已 evict 的 prefix node 时，从 L2 将 KV 数据恢复到 GPU HBM，使 node 重新变为 `evicted=False`，后续 decode 可以直接使用。** 它是"空间换时间"的兑现——如果没有这一步，前面 write_backup 付出 L2 空间代价就没有机会转化为 prefill 节省。

### 4.1 触发：match_prefix 发现 host_hit

load_back 的触发嵌入在 prefill 流程中：

```text
init_next_round_input → tree_cache.match_prefix()
  → 沿 radix tree 匹配 prefix
  → 遇到 evicted==True 但有 host_value 的节点 → host_hit_length 累加
  → needs_host_load_back() 返回 True
  → init_load_back() → load_back(node)
```

这里的 `host_hit_length` 与 `device_hit_length`（L1 命中）是独立的两段匹配长度。`host_hit_length > 0` 表示有 token 的 KV 在 host 中——不需要重新 prefill，只需要从 host 加载。

### 4.2 恢复整条链

被 evict 的节点可能形成一条链。radix tree 的 split 操作将一个节点在匹配边界分裂为多个子节点——如果这些子节点随后都被 evicted，就形成了一条从被 evict 的子节点到第一个 non-evicted 祖先的"断裂链"。load_back 必须恢复整条链上的所有节点，否则断点处的 prefix 不完整。

```python
# hiradix_cache.py:1147-1155
nodes_to_load = []
cur_node = best_match_node
while cur_node is not None and cur_node.evicted:
    nodes_to_load.append(cur_node)
    cur_node = cur_node.parent
nodes_to_load.reverse()  # 从祖先到叶子顺序加载
```

沿 `parent` 链向上遍历，收集所有 `evicted == True` 的节点，直到第一个 `evicted == False` 的祖先（持有有效 HBM device pages）。然后将列表反转为从祖先到叶子的顺序——这是恢复的加载顺序，保证祖先先恢复。

### 4.3 阈值与配额：两道经济性检查

不是所有 host hit 都值得 load_back。代码中有两道经济性检查：

```python
# hiradix_cache.py:1163-1168
if len(host_indices) < self.load_back_threshold:
    return None  # 太小，prefill 重算更快

if len(host_indices) > self.mem_quota + self.mem_quota_delta:
    return None  # 太大，新请求可能没有足够 HBM
```

**阈值检查**：太小的 prefix 直接 prefill 重算更经济。DMA 搬运有固定开销（CPU→GPU 的 PCIe 延迟），小数据量下 I/O 开销可能超过重算开销。

**配额检查**：过大的 host 数据一次加载会占用大量 HBM space，可能导致新请求的 KV cache 没有足够空间。`mem_quota + delta` 提供了缓冲——允许略超配额，但不能太多。

### 4.4 恢复与衔接

通过检查后，进入实际恢复：

```python
# hiradix_cache.py:1170-1181
device_indices = self.cache_controller.load(
    host_indices=host_indices)
```

`cache_controller.load()` 执行两步：先从 device pool 分配新 pages，再用 GPU kernel 将 host 端数据搬运到 device pages。

如果分配失败（HBM 满），`evict(num_tokens)` 先驱逐腾空间（line 1175-1181），重试 `load()`。这与 write_backup 中 host pool 满时的 `evict_host → 重试` 完全对称——**每个方向的搬运都可能在目标端触发空间竞争，每个竞争都由对应的逐出机制解决。**

加载成功后，`device_indices` 是一段连续的 slot 索引，需要按 node 边界切分：

```python
# hiradix_cache.py:1196-1198
for node in nodes_to_load:
    num = len(node.host_value)
    node.value = device_indices[offset:offset + num].clone()
    offset += num
```

设置 `node.value` 后，`node.evicted` property（检查 `self.value is None`）自动变为 False——从这一刻起，后续请求的 `match_prefix` 将直接在 L1 中命中该节点，不再触发 load_back。

恢复完成后执行关键的衔接操作：

1. **`inc_lock_ref(last_hit_node)`**（line 1203）：锁定目标节点，防止刚恢复到 HBM 的 KV 立即被 evict。当前请求即将使用这些 KV 做 forward——如果在这一步和 forward 之间被 evict，等于白恢复。

2. **`_record_store_event(node, medium=StorageMedium.GPU)`**（line 1201）：记录"block 已恢复到 GPU"事件，供 downstream 的事件系统（如 PD 分离的 indexer）感知 KV 的物理位置变化。

3. **拼接 prefix**：`init_load_back` 将返回的 `device_indices` 写入 `req_to_token_pool`，与祖先已有的 device pages 拼接成完整 prefix。attention kernel 按 `req_to_token_pool` 的索引读写 KV Pool，不关心 slot 是来自 L1 命中还是刚从 L2 恢复——它只按索引寻址。

---

## 五、策略权衡：write_through vs write_back

write_through 和 write_back 是两种截然不同的备份理念。它们的差异不在于"备份数据的格式"（底层的 `cache_controller.write()` 完全一致），而在于**"何时决定值得备份"**：write_through 用命中次数作为信号（"被第二次访问就值得备份"），write_back 用 eviction 优先级作为信号（"被逐出时优先级足够高就值得备份"）。

### 5.1 行为差异矩阵

| 维度                       | write_through                         | write_back                                        |
| -------------------------- | ------------------------------------- | ------------------------------------------------- |
| **备份时机**               | node 第二次被访问时立即触发           | HBM 满时，eviction 触发                           |
| **信号来源**               | 命中次数（频率信号）                  | eviction 优先级（时效信号）                       |
| **`_inc_hit_count`**       | 递增 hit_count，达阈值调 write_backup | **跳过**（line 898：不在此处触发备份）            |
| **evict 非 backuped node** | `_evict_regular` — 直接删除           | `write_backup(x, write_back=True)` — 先备份再处理 |
| **数据丢失风险**           | 低：第二次访问即备份                  | 较高：首次访问后可能被 evict 而不备份             |
| **HBM 占用**               | 无差异                                | 无差异                                            |
| **写放大**                 | 高（每次命中都写）                    | 低（只写最终被 evict 的）                         |
| **适合场景**               | 共享 prefix 多、可预测的复用模式      | 写入压力大、HBM 换手快、复用模式不确定            |

---

## 六、总结：三个操作的因果关系链

```text
write_backup:  "空间换时间"——用 L2 空间换取未来的 prefill 节省
      │
      ▼
eviction:      "容量天花板到了"——释放 HBM，依赖 write_backup 先保存数据
      │
      ▼
load_back:     "已付出的空间，兑现为时间"——避免重算，恢复 KV 到 HBM
```

三个操作不是独立的。write_backup 的质量（备份了多少节点、何时备份、用什么策略）直接决定了 load_back 的命中率和恢复成本——write_through 保证 L2 中有"最热"的数据（第二次访问即备份），write_back 只保证 L2 中有"被淘汰之前优先保留"的数据。eviction 的节点选择（LRU/FIFO/priority）决定了 L2 中保留的是"最近被使用"还是"最常被使用"的数据——这个选择与 write_through 的频率信号（hit_count）在逻辑上是互补的：前者管理"空间有限时淘汰谁"，后者管理"什么值得从 L1 复制到 L2"。

从更宏观的视角看，这三个操作是在一个多级存储层次上实现了一个**非对称的读写路径**：写入方向（HBM → L2）由命中频率或空间压力触发，读取方向（L2 → HBM）由请求匹配触发。写入是"投资"——用 L2 空间和 I/O 带宽换取未来的 prefill 加速；读取是"兑现"——当后续请求需要这些数据时，投资产生回报。**整个系统的效率取决于投资决策的质量**：备份了不值得备份的数据（L2 空间浪费），或者没备份值得备份的数据（prefill 重算浪费），都是损失。

---

## 七、相关代码位置

| 文件                                              | 方法                | 作用                               |
| ------------------------------------------------- | ------------------- | ---------------------------------- |
| `sglang/srt/mem_cache/hiradix_cache.py:759-789`   | `write_backup()`    | HBM → L2 备份核心逻辑              |
| `sglang/srt/mem_cache/hiradix_cache.py:896-905`   | `_inc_hit_count()`  | 命中计数 → 触发 write_through 备份 |
| `sglang/srt/mem_cache/hiradix_cache.py:1033-1079` | `evict()`           | HBM 逐出主循环（三条分支）         |
| `sglang/srt/mem_cache/hiradix_cache.py:1083-1091` | `_evict_backuped()` | 释放 device pages，保留 L2         |
| `sglang/srt/mem_cache/hiradix_cache.py:1057-1058` | `_evict_regular()`  | 直接删除低价值节点                 |
| `sglang/srt/mem_cache/hiradix_cache.py:1141-1211` | `load_back()`       | L2 → HBM 恢复核心逻辑              |
| `sglang/srt/mem_cache/hiradix_cache.py:1163-1168` | (load_back 内部)    | 阈值与配额检查                     |
| `sglang/srt/mem_cache/cache_controller.py:203`    | `HiCacheController` | 双 CUDA stream（write/load）协调   |

---

## 延伸阅读

- [SGLang KV Pool 管理](./sglang-kv-pool-management.md) — HiCache 架构全景：三层存储体系、Host Pool 布局、L3 后端与 page_size 全栈贯穿
- [HiCache 深入详解](./hicache_deep_dive.md) — 演进背景、HiRadixTree 元数据拓扑、预取与写回策略、存储后端热插拔
- [SGLang 调度器](./sglang-scheduler.md) — Prefill/Decode 调度决策、chunked prefill、TTFT 拆解
