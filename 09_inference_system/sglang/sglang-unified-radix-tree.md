# SGLang UnifiedRadixTree：一棵树，四种注意力

> 2026 年 7 月 25 日发布的 SGLang v0.5.16 让 UnifiedRadixTree 成为 SWA、Mamba、DSA 和全注意力四种 attention 类型的默认缓存路径（PR #30468）。
> 这不是性能优化，而是架构简化——把三棵各自为政的缓存树合并为一棵树 + 可插拔组件。
> 四种注意力对应三个组件 + 一个组合：FULL / MAMBA / SWA 是独立组件，DSA 是 FULL + SWA 的组合（§4.4）。
>
> 相关文章：[SGLang 0.5.16 发布解读](sglang-0.5.16-release.md) ｜ [SGLang KV Pool 管理](sglang-kv-pool-management.md)

---

## 一、一棵树曾经够用：RadixAttention 的前缀缓存

SGLang 的差异化竞争力，最早来自 RadixAttention：用一棵 Radix Tree 把请求的公共前缀的 KV 缓存组织起来，命中即复用，省掉重复 prefill。理解 UnifiedRadixTree 之前，需要先理解这棵原生的树——它定义了"前缀缓存"的全部核心机制。

### 1.1 树如何工作

Radix Tree 按 **token 序列**组织 KV pool 的 slot 索引。树中每个节点存自己覆盖的 token span 与对应的显存 slot：

```text
Radix Tree 结构:
  root
   └── [1,2,3]          ← 前缀 [1,2,3] 的 KV 在 slots [a, b, c]
        ├── [4,5]        ← 扩展 [4,5] 后 KV 在 slots [d, e]
        │     └── [6]    ← 再扩展 [6] 后 KV 在 slot [f]
        └── [6,7]        ← 另一分支: 扩展 [6,7] 后 KV 在 slots [f, g]
             └── [8]     ← ... slot [h]
```

三个核心机制支撑它的正确性与效率：

**共享**：请求 `[1,2,3,4,5]` 和 `[1,2,3,6,7]` 共享节点 `[1,2,3]` 的 slots `[a,b,c]`——公共前缀只算一次 prefill，后续请求直接复用。

**分裂**：当匹配只命中节点的一部分时，节点在边界分裂。分裂只调整索引结构（把节点切成两段），不复制 KV 数据——KV 的物理存储不受影响。

**引用计数**：`lock_ref` 沿 parent 链保护整条路径。正在被活跃请求使用的 slot 不可逐出；请求结束后 `cache_finished_req` 把它的 KV 索引插入树中，并释放与树中已有节点重复的 slot。

TreeNode 的核心字段（`radix_cache.py:217`，省略了 hit_count 等统计字段）：

```python
class TreeNode:
    key: RadixKey            # 此节点覆盖的 token 序列
    value: torch.Tensor      # KV pool slot 索引 (GPU, int64)
    host_value: torch.Tensor # KV pool slot 索引 (CPU host pool)
    lock_ref: int            # 引用计数 (>0 = 被活跃请求使用，不可逐出)
    parent: TreeNode         # 父节点
    children: defaultdict    # 子节点
```

### 1.2 为什么它如此重要

前缀复用是 SGLang 相对其他推理引擎的核心差异化：多轮对话（同一对话的累积前缀）、few-shot 提示（共享的示例前缀）、批量同前缀请求——这些场景下命中树缓存的请求跳过 prefill，TTFT 大幅下降。

这棵树的语义假设很朴素：**token 序列即缓存键**。这个假设成立的前提是模型的缓存状态只有 KV 一种。模型架构开始混合后，这个前提不再成立。

---

## 二、混合架构模型：树的第一次分裂

2024-2026 年，模型架构不再纯粹。SWA（滑动窗口注意力）只在窗口内做注意力，Mamba 用递推状态替代注意力，DSA（DeepSeek 稀疏注意力）按需选择稀疏 KV。其中 SWA 和 Mamba 的缓存语义与标准 KV 完全不同，各自需要一棵树；DSA 的稀疏选择不改变 KV 存储格式，不需要新树（见 §4.4）。

### 2.1 SWA 需要自己的树

滑动窗口注意力中，每个 token 只关注窗口内的历史。这意味着：**窗口外的旧 KV 已经无意义**——它既不会被新 token 读取，也占用显存。标准 RadixCache 的语义是"只增不减"（前缀永久有效），对 SWA 不成立。SGLang 需要一棵 `SWARadixCache`：窗口滚动时旧 KV 要能失效、可复用。

### 2.2 Mamba 需要自己的树

Mamba/SSM 模型没有 KV Cache——它有一个**固定大小的递推状态**，每个 token 一步更新。这个状态同样需要前缀缓存（相同前缀的状态可以直接继承），但它的语义完全不同：状态不是"一段 token 对应的存储"，而是"整个前缀压缩出来的摘要"。`mamba_radix_cache.py` 的做法是在标准 TreeNode 上硬塞扩展字段：

```python
# mamba_radix_cache.py:65 — TreeNode 扩展字段
self.mamba_value: Optional[torch.Tensor] = None
self.mamba_host_value: Optional[torch.Tensor] = None
self.full_lock_ref = 0
self.mamba_lock_ref = 0
# mamba_evicted: property, mamba_value is None
```

这棵树的 lock 语义还有一个显式的不变量注释（`mamba_radix_cache.py:77`，原文为英文）：

> invariant: 如果 mamba_lock_ref 被锁，full_lock_ref 必须被锁；full_lock_ref 总是 >= mamba_lock_ref。

一棵树塞两套字段（full 一套、mamba 一套），每套有自己的 lock_ref 和 evicted 状态——正确性依赖这条不变量，而约束只是注释，没有机制强制。

### 2.3 三棵树并存的代价

到 0.5.15 为止，SGLang 的 mem_cache 目录里有三棵树：`RadixCache`（全注意力）、`MambaRadixCache`（SSM）、`SWARadixCache`（滑动窗口）。SGLang 的 roadmap issue #20415 对这个问题做了精准的自我诊断：

> RadixCache、MambaRadixCache 和 SWARadixCache 共享大部分逻辑，但作为独立副本分别维护——导致代码重复、行为不一致，扩展新模型类型时维护成本极高。

三个具体代价：

1. **代码重复**：树的 match/insert/evict/LRU 核心逻辑在三处复制，任何一处修复都要同步到另外两处，而"同步"在实践中总是会漏。
2. **行为不一致**：三棵树各自漂移——同一前缀在不同树上的命中行为、逐出行为可能不同，排障时难以定位。
3. **无法扩展**：每支持一种新模型类型（比如 DSA），都要再复制一棵树。0.5.16 支持四种注意力类型，这条路显然走不下去。

树的核心逻辑本不该关心注意力类型，却被注意力类型绑死了。

---

## 三、UnifiedRadixTree：一棵树，可插拔组件

答案不是"再写一棵树"，而是**把树的核心抽出来，让注意力类型成为可插拔的组件**。这就是 PR #21206 引入的 HybridRadixCache V2（后改名 UnifiedRadixTree）的设计。

### 3.1 统一树核心

核心洞察：树只负责一件事——**按 token 序列组织索引**。match（前缀匹配）、insert（插入）、evict（逐出）、LRU（最近最少使用）这些操作与"节点里存的是什么"无关。无论节点存的是 KV slot、SSM 状态还是滑动窗口索引，树的遍历逻辑完全一样。

于是 UnifiedRadixTree 把树核心做成 **component-agnostic**：核心的树遍历逻辑不关心节点承载的状态是什么，状态的读写、锁、逐出全部下放给组件（组件通过接口回调接入树的 match/insert/evict 流程）。注意，这里的"核心"仍然操作 TreeNode 对象——节点句柄化（`NodeId`）是后续 RadixTreeCore 拆分的设计，见 §6.4。

### 3.2 组件注册表

`COMPONENT_REGISTRY` 把具体组件挂到树核心上：

```python
# unified_radix_cache.py:270
COMPONENT_REGISTRY = {
    ComponentType.FULL:   FullComponent,
    ComponentType.MAMBA:  MambaComponent,
    ComponentType.SWA:    SWAComponent,
}
```

`ComponentType` 枚举（FULL / MAMBA / SWA）是节点的"状态维度"。**新增注意力类型 = 实现一个组件接口 + 注册**——树的 match/insert/evict 一行不用改。三个组件的实现在独立的 `unified_cache_components/` 子目录（`full_component.py` / `mamba_component.py` / `swa_component.py`）——树核心、注册表、组件实现三层分离。

### 3.3 节点结构

`UnifiedTreeNode` 不再为每种状态硬编码字段，而是用一个按 `ComponentType` 索引的数组承载（结构示意，字段为简化）：

```python
class UnifiedTreeNode:
    component_data: List[ComponentData]  # 按 ComponentType 索引
    # 每个 ComponentData 包含: device value / host value / lock_ref / metadata
    lru_prev / lru_next: 每组件独立的 LRU 指针
    last_access_time:     最近访问时间（LRU 逐出用）
```

对比第二章的旧版 `mamba_radix_cache.py`（一棵树塞两套字段），这里的设计是**一棵树，N 套字段，按组件类型索引**——新增状态类型不需要改节点类。

### 3.4 每组件独立 LRU

**每个组件维护自己的 LRU 列表**（`UnifiedLRUList`），FULL、SWA、Mamba 因此有不同的逐出优先级和 lock scope。

不同状态的复用价值不同：全注意力的 KV 可以被任意新请求复用（应久留）；滑动窗口的 KV 只对窗口内的请求有效（可早逐出）；SSM 状态是压缩摘要（价值高但更新频繁）。共用一条 LRU 时，一种状态的访问会挤掉另一种状态——独立 LRU 让每种状态的逐出策略互不干扰。

---

## 四、三个组件怎么工作

统一树没有消除差异，只是把差异搬进组件。每个组件实现一类状态的完整缓存语义。

### 4.1 FULL（标准 KV）

继承原 RadixCache 的全部语义：token 序列 → KV slot，共享、分裂、lock_ref 逐出保护。行为与 0.5.15 之前完全一致——这是迁移的兼容性基线。

### 4.2 SWA：tombstone + in-window recovery

滑动窗口的缓存语义是"窗口内的有效，窗口外的失效"。SWAComponent 用 **tombstone（墓碑）标记**处理失效：窗口滚动导致 KV 索引失效时，节点标记为 tombstone（代码注释的原话：`swa_tombstone is used to indicate the kv indices have been freed for swa layers`），失效的索引可被复用，节点也不再参与 SWA LRU 的正常逐出。

窗口之后滑回时，分两种情况：失效的 KV 未被其他请求占用，索引直接复用；已被占用，则重新计算。tombstone 的价值是让"失效"成为节点的显式状态，避免在每次窗口滑动时做激进的索引回收。

### 4.3 Mamba：SSM 状态 + copy-on-write

SSM 状态按 `mamba_track_interval` 对齐缓存——只在固定的状态追踪间隔上保存状态快照，避免每个 token 都存一份。状态被多个请求共享时，用 **copy-on-write** 语义处理：任何请求要修改共享状态，先复制一份，不影响其他持有者。

旧版 MambaRadixCache 用 `mamba_lock_ref` 的组合来维持这两个语义的正确性；组件化之后，对齐快照和 copy-on-write 成为组件内部显式的实现，不再依赖 lock_ref 的排列组合。

### 4.4 Eagle 与 DSA 的特殊处理

统一树还需要兼容 SGLang 的投机解码（Eagle）和稀疏注意力（DSA）：

- **Eagle**：启用投机解码时，缓存键转换为 bigram key（两两组合的 token 对）以匹配草稿模型的行为；Mamba 作为树组件时 `is_eagle` 被 gate 掉（SSM 状态不适合 bigram 键）。
- **DSA**：DeepSeek 稀疏注意力不是单一组件，而是 **FULL + SWA 组件的组合**——稀疏选择本身有滑动窗口特性，KV 存储是标准格式。DSA 模型通过组合已有组件获得缓存支持，不需要新组件。

DSA 的缓存支持完全来自组件组合，没有新增组件——组件化设计因此得到了真实模型的验证。

---

## 五、0.5.16 的落地：默认化与三个子特性

统一树在 0.5.16 从"可选实验"（`--enable-hybrid-radix-tree`，默认关闭）变成**默认路径**（PR #30468）。三个新特性以它为前提。

### 5.1 默认化与破坏性变更

PR #30468 让 UnifiedRadixTree 成为 SWA、Mamba、DSA 模型的默认缓存实现。release notes 明确标注这是**行为变化**（breaking change）：缓存命中逻辑改变，升级后需要重新验证这些架构的缓存行为。

默认化之前，PR #21206 做过正确性验证：AIME 25 精度测试中统一树与旧版每模型树基本持平（如 Qwen3-next 64.38% vs 64.79% pass@1、GPT-OSS-20B 73.33% vs 71.67%）——统一没有牺牲正确性。

### 5.2 ReplaySSM 同步

ReplaySSM（投机解码的状态重放）与 Mamba int8 检查点被同步到统一树上（#30636、#30626）。在旧架构下，这两个特性需要为每棵旧树单独实现；统一树让它们只需实现一次。

### 5.3 命中只重置实际用到的状态

统一树带来一个此前不可能实现的优化：**缓存命中时只重置该请求真正用到的组件状态**（#31643、#31648）。旧版 MambaRadixCache 命中时需要处理整棵树的 mamba 状态；统一树按组件精确追踪，命中后只重置实际使用的状态。

DSpark 的逐请求动态验证窗口依赖这个特性：每次命中后的状态管理必须精确到请求级，统一树的组件级追踪让这成为可能。

### 5.4 GLM-5.2 DSA 缓存层拆分

以统一树为前提，GLM-5.2 在 prefill context parallelism 下实现了 DSA 缓存层按 CP rank 分片（#29421）：每个 rank 只拥有不重叠的层范围。8192 token、78 层、cp_size=4 条件下，每 rank KV 从 0.77GB 降至 0.20GB（约 -74%）。这个优化在"所有状态类型共享一棵树"的抽象下才成立——否则拆分逻辑需要为每种注意力类型分别实现。

---

## 六、统一抽象的代价与边界

统一树有代价。它把"每棵树各自简单"换成了"一棵树精心设计"——以下四条是具体代价。

### 6.1 统一 vs 专用：逐出决策的并行化

组件化解决了代码重复，但把逐出决策从"一棵树的单一 LRU"变成"每个组件类型各一条 LRU 链"（`UnifiedLRUList` 按 component_type 组织）。独立 LRU 给了每种状态独立的逐出策略（这是收益），但代价是调优面从 1 变成 N——生产环境需要分别观测每种状态的命中率和逐出率，逐出参数的调优也从"调一棵树"变成"调 N 条链"，排障复杂度上升。

### 6.2 lock 语义的复杂性

旧版 Mamba 树有独立的 `full_lock_ref / mamba_lock_ref`——两种状态可以有不同的驻留保证，且代码注释明确约束了二者关系（`full_lock_ref` 总是 >= `mamba_lock_ref`，`mamba_radix_cache.py:77`）。统一树把这个语义收进组件内部后，正确性保证依赖每个组件的 lock 实现不互相干扰。这是组件接口的隐式契约：**一个组件不能破坏另一个组件的驻留保证**。这个契约在代码里没有显式机制强制（旧版至少还有注释声明不变量），只能靠组件实现自律——这是统一的隐性风险。

### 6.3 行为变化风险

0.5.16 明确把 UnifiedRadixTree 默认化标记为 breaking change。对使用 SWA/Mamba/DSA 模型的团队，升级后缓存命中逻辑改变——必须重新验证命中率、TTFT 和精度。release notes 的已知问题里，Mamba overlap scheduler 的 seqlen 问题修复后又被回滚（#31369 → #31622），说明混合架构的缓存行为在统一之后仍然存在未解决的角落。

### 6.4 未来：从统一树到 Rust 核心

统一树是中间站，不是终点。roadmap #20415 的 Stage 3 是把树核心用 Rust 重写——为此，UnifiedRadixCache 正在被拆分为 `RadixTreeCore`（纯树结构，NodeId 句柄）+ `TreeComponents`（每组件 KV 逻辑）+ controller（缓存控制）三层。拆分完成后，树核心与注意力类型彻底解耦，Rust 重写只动核心、不动组件。

---

## 七、总结

SGLang 的缓存架构走了三步：从一棵 RadixCache 服务全注意力模型，到 SWA、Mamba 各复制一棵树，再到 UnifiedRadixTree 把树核心与注意力类型解耦、以 FULL/MAMBA/SWA 三个可插拔组件承载差异。

这次重构的直接收益在 0.5.16 落地：ReplaySSM 状态重放、命中只重置实际状态、GLM-5.2 缓存层拆分（每 rank KV -74%）都只需要实现一次，而不是为每棵树各写一遍；DSpark 的逐请求动态验证窗口也依赖组件级的状态追踪。

0.5.16 之后，旧的 `radix_cache.py` / `mamba_radix_cache.py` / `swa_radix_cache.py` 仍然保留在代码库里——统一树是默认路径，但不是唯一路径。后续的 RadixTreeCore 拆分与 Rust 重写（roadmap #20415 Stage 3）还在继续。

SGLang 的顺序是先统一抽象（0.5.16 默认化），再推进实现优化（Rust 重写）——注意力类型还在增加，抽象层先稳定下来，后续的优化才不用为每种类型重复做。

---

## 参考

- [PR #30468：Using UnifiedRadixTree by default for SWA, Mamba, and DSA models](https://github.com/sgl-project/sglang/pull/30468)
- [Issue #20415：Unified Hybrid Radix Cache Refactor（roadmap）](https://github.com/sgl-project/sglang/issues/20415)
- [PR #21206：Support Unified HybridRadixTree V2](https://github.com/sgl-project/sglang/pull/21206)
- [unified_radix_cache.py 实现（v0.5.16）](https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/srt/mem_cache/unified_radix_cache.py)
- [SGLang v0.5.16 Release Notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.16)
- [SGLang KV Pool 管理：Radix Tree 索引与请求视图](sglang-kv-pool-management.md)
- [SGLang 0.5.16 发布解读](sglang-0.5.16-release.md)
