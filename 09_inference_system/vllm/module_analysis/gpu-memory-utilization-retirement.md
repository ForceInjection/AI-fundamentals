# gpu-memory-utilization 的谢幕：vLLM 如何用「实测」替代「估算」

> 2026-08-20 | 基于 vLLM PR [#50779](https://github.com/vllm-project/vllm/pull/50779)（Extensible KV cache，njhill，draft）与 [#51718](https://github.com/vllm-project/vllm/pull/51718)（KV cache layout 标准化，LucasWilkinson）的 patch 分析。**两个 PR 均未合入 main**——#50779 明确标注 draft，且依赖 #51718 先落地。文中所有 `file:line` 行号以 2026-08-20 的 PR head 分支为准，合入后可能漂移。

一张 H100 的 80GB 显存里，模型权重、KV Cache、激活值、CUDA Graph 池各分走多少——这个问题在 vLLM 里由一个数字回答：`--gpu-memory-utilization`。它是教程里出现频率最高的参数，也是一个把「运行时才确定的事实」提前到「启动前必须猜对」的接口。猜大一点，warmup 阶段 OOM；猜小一点，8% 的 HBM 常年闲置。而 vLLM 正在消灭这个猜谜游戏：PR #50779 用驱动级虚拟内存管理（VMM）把 KV Cache 变成**可增长的**——先预留虚拟地址空间，warmup 后实测真实内存占用，再一次性补齐物理页。**sizing 从「profiling 估算 + 手工 margin」变成「warmup 后实测」**，合入后参数默认值将从 0.92 变为 1.0。本文拆解这个转变：参数为什么必然猜错、VMM 机制如何让"先留地址后付页"成为可能、layout 标准化（#51718）为什么是前置依赖，以及权衡与已知缺口。

## 一、显存预算这道必答题

推理部署的第一约束是显存。但「显存够不够」不是一个静态问题——它由四个内存行为完全不同的消费者组成，而 `gpu-memory-utilization` 把它们的总和压成一个启动时必须确定的数字。

### 1.1 显存账本：四个消费者，四种内存行为

| 消费者                    | 内存行为                                | 何时确定         |
| ------------------------- | --------------------------------------- | ---------------- |
| 模型权重                  | 静态，加载即固定                        | 加载时           |
| KV Cache                  | 随 batch 与序列长度伸缩，推理的主战场   | 分配时（一次性） |
| 激活值                    | 随 batch / 序列形状波动，峰值在 Prefill | 运行时逐迭代     |
| CUDA Graph 池 / workspace | 录制时固化、后续按需增长                | **warmup 之后**  |

前两个是「账面上算得清」的，后两个是「只有跑起来才知道」的。问题在于：参数把后两者的不确定性也一并要求用户提前买单。

### 1.2 gpu-memory-utilization 做了什么

vLLM 的启动流程是：先跑一轮 profiling（估算模型权重 + 激活的占用），然后用 `gpu-memory-utilization` 乘以剩余可用显存，把结果一次性切给 KV Cache。文档里的定义是 per-instance 的预算声明：

> The fraction of GPU memory to be used for the model executor... If unspecified, will use the default value of `DEFAULT_GPU_MEMORY_UTILIZATION`（此时为 0.92）. This is a per-instance limit, and only applies to the current vLLM instance.（`vllm/config/cache.py:118-128`）

在 #50779 之前，默认值 0.92 意味着：**启动时就有 8% 的 HBM 被刻意闲置**，作为 profiling 估算误差的「安全税」。

### 1.3 它隐含的三个假设

这个参数之所以存在，是因为它背后的分配模式立基于三个假设：

1. **启动时一次性分配**——KV Cache 的大小在启动时定死，之后不可变；
2. **profiling 能代表运行时**——估算时刻的内存行为与真实运行时一致；
3. **设备由本实例独占**——计算可用显存时不需要考虑其他进程。

真实生产环境里，三个假设全部不成立。下一章看它们如何让同一个参数产生两种相反的灾难。

## 二、一个参数，两种死法

这个参数只需要回答一个数字，但错误的答案有两种：调大 OOM、调小浪费。更隐蔽的是第三种——数字本身没错，但系统运行时的内存行为变了，让「曾经对的数字」变成错的。

### 2.1 死法一：OOM——估算链在时间点上断裂

问题不在估算精度，而在估算的**时间点**。vLLM 的 profiling 发生在 warmup（启动时用假请求跑真实前向流程的预热阶段，用于触发算子初始化与 CUDA Graph 录制）之前，而有一批显存分配发生在 profiling 之后、warmup 与运行时期间：

- **CUDA Graph 池**（把一串 GPU 内核录制为可重放的对象，录制时按最大 batch 形状固化内存）：大小取决于录制配置而非 profiling 估算；
- **投机解码的 logits all-gather**：draft 与 target 模型的 logits 拼接缓冲区，随投机步数增长；
- **workspace 按需增长**：算子后端的工作空间第一次真实运行时才扩张；
- **多模态 frame staging**：首帧到达时的临时显存。

这些分配在 profiling 时**根本不存在**，因此无论把 `gpu-memory-utilization` 设成多少，都只能靠「猜 margin」来覆盖。PR #50779 的测试里有一个教科书级的用例：Qwen3.5-4B TP4 + MTP 投机 100 个 token（MTP：Multi-Token Prediction，训练时植入的多 token 预测头）+ chunk 256，`gpu-memory-utilization=0.92`——**传统分配路径直接 OOM**（eager——逐迭代直接执行——与 cudagraph 录制回放双模式皆如此，见 #50779 Testing 一节）。这不是配置错误，而是估算模式的系统性盲区。

### 2.2 死法二：浪费——安全税没有反馈闭环

OOM 的另一面是浪费。0.92 的默认值意味着每张 80GB 的 H100 有 6.4GB HBM 常驻闲置；被 OOM 吓过的用户会主动调低到 0.8、0.7，闲置进一步扩大。关键是这个浪费**没有反馈闭环**：系统从不告诉用户「你其实还剩 X GB」，于是保守一旦开始就没有动力停止。8% 的闲置是"安全税"，而它的代价随集群规模线性放大。

### 2.3 为什么猜不对：margin 是拍脑袋的

把两种失败放在一起看，根因是同一个：**估算链的三处不确定性**。

1. **时间点错位**：profiling 早于真实运行时，warmup 之后才发生的分配无法被估算；
2. **margin 不可验证**：headroom 设多少、覆盖什么，没有机制能确认「设对了」，只能靠事故来证伪；
3. **共享 GPU 完全失准**：参数是 per-instance 的，但当另一个进程占着同一张卡时，计算出的"可用显存"与真实可用之间差着一个未知数。

本质问题：**这个参数把系统的职责推给了用户**——让用户在启动时对运行时才确定的事实做出承诺。而系统明明可以自己测量。

## 三、拆开「分配」：地址空间与物理页

与其要求用户猜对数字，不如让系统自己量。第一步是把「分配显存」这个笼统动作拆成两个独立决策——**预留多大地址空间**与**commit 多少物理页**。这一步由驱动级的虚拟内存管理（VMM）提供，也是 CUDA 早已有之、却很少被推理引擎用上的能力。

### 3.1 VMM：先留地址，后付页

CUDA 的虚拟内存管理 API 把一次分配拆成三个动作：`cuMemCreate`（创建物理内存句柄）、`cuMemMap`（映射到虚拟地址范围）、`cuMemSetAccess`（授予访问权限）。**预留一大段虚拟地址空间几乎不消耗显存**——显存只在物理页被 commit 时实际占用。这意味着「分配」和「占用」可以完全解耦。（这里的虚拟地址是驱动管理的 GPU 统一地址空间保留区，与操作系统的换页机制无关。）

PR #50779 用 ctypes 直接绑定这套驱动 API（`vllm/utils/vmm_driver.py`，398 行），并同时覆盖 ROCm 的镜像接口 `hipMem*`。工程细节里有一个值得注意的坑：ROCm 的 `hipMemSetAccess` 在单个预留区包含多个不同大小分配时存在非确定性失败，修复要等到 TheRock 7.12+——因此代码里硬编码了最低版本门槛：

```python
# vmm_driver.py:30-35 — ROCm 版本门槛
# First HIP runtime carrying rocm/rocm-systems#2451. Earlier runtimes fail
# hipMemSetAccess non-deterministically once a reservation holds several
# differently-sized allocations, which is exactly how a growable KV cache is
# committed. The fix landed on TheRock release branches (7.12+) and is absent
# from every release/rocm-rel-7.x branch.
_HIP_MEM_SET_ACCESS_FIXED_IN = (7, 12)
```

### 3.2 ExtensibleTensor：视图是权威，`nbytes()` 不可信

机制的内核是 `ExtensibleTensor`（`vllm/utils/extensible_tensor.py`，595 行）与其内部的 `_VirtualBuffer`。`_VirtualBuffer` 在构造时预留整个容量的地址空间，但只按 commit granule（物理页提交的最小粒度，ROCm 上为 4 KiB）逐步映射物理页：

```python
# extensible_tensor.py:39-58（节选）— 预留容量，按需 commit
class _VirtualBuffer:
    """...Physical memory is committed incrementally, at granularity-sized
    granules, via `ensure_committed_range`..."""
    def __init__(self, ...):
        self.reserved_size = _round_up(max(max_bytes, 1), self.granularity)
        self.base_ptr = self._driver.reserve(self.reserved_size)
```

`ensure_committed_range(start, end)` 只映射指定区间的物理页（`extensible_tensor.py:76`），`release_physical()` 可以整体释放物理页而保留地址空间与视图（`extensible_tensor.py:170-175`——base pointer（预留地址空间的基址指针）与上层 tensor 视图保持有效，只是暂时无物理支撑）。

这个设计带来一个**契约变化**，是 PR 作者明言「值得评审注意」的一点：在可增长缓存下，未类型化的 storage 跨越的是**预留容量**，而只有每个 layout segment 的块前缀有物理支撑——layout segment 指 KV Cache 物理地址空间按层与块维度划分的连续段，每段有独立的 block 几何（详见 3.3 的 layout 标准化）；块前缀即每段起始的少量 block。任何从 `untyped_storage().nbytes()` 推导 block 几何或注册范围的代码，都会算出一个包含未 commit 页面的虚高大小。因此 connectors（连接外部 KV Cache 传输系统的组件，如 PD 分离的 NIXL、异构存储的 mooncake、CPU offload 等）全部改为从视图推导注册范围，KV 传输初始化也推迟到 `extend_kv_cache` 之后——**connectors 只注册已 commit 的内存**。

### 3.3 为什么 layout 标准化是前置依赖（#51718）

直接对 main 分支做这件事有多难？PR 描述里给出了答案：集成代码量在 #51718 的支撑下缩小了约 **30%**——buffer 分段从「按后端探测 shape/stride/block 维度」变成 `KVCacheLayout.stride_order` 的纯函数（PR 描述称新增 `num_outer_segments()` 辅助函数；该符号在 head 分支源码中未检索到，可能已改名或并入别处），packed 与 `block_stride` 特例全部消失。

`#51718（[6/N] KV-cache layout 重构系列）`做的正是这件事：把每个 KV Cache 分配统一到逻辑 `[L, B, H, N, C]` 词汇表（RFC #42082），物理布局由 6 种 stride 排列之一的 `KVCacheLayout` 枚举描述（`vllm/v1/kv_cache_layout.py:16-30`，58 行新增）：

```python
# kv_cache_layout.py:16-30（节选）— 六种物理 stride 排列
class KVCacheLayout(Enum):
    """The logical shape is always [L, B, H, N, <content>] (RFC #42082).
    Each member's value is a stride permutation that maps logical axes
    to physical (memory) order."""
    LBHNC = (0, 1, 2, 3, 4)  # [L, B, H, N, C] (identity)
    LBNHC = (0, 1, 3, 2, 4)  # [L, B, N, H, C]
    LHBNC = (0, 2, 1, 3, 4)  # [L, H, B, N, C]
    BLHNC = (1, 0, 2, 3, 4)  # [B, L, H, N, C]
    BLNHC = (1, 0, 3, 2, 4)  # [B, L, N, H, C]
    BHLNC = (1, 2, 0, 3, 4)  # [B, H, L, N, C]
```

以 `LBNHC` 为例：逻辑维度 `[L, B, H, N, C]` 中的 N 与 H 互换，物理内存里同一 layer、同一 block 的 N 个 head 紧邻、C 连续——块的 `[N, C]` 是一整段连续访问；`BLHNC` 则是 block 排在最外层，同一 block 的所有 layer 物理上连续。六种排列代表六种「先排谁」的选择，attention 后端各取所需。

layout 解析有了**唯一写者**——同一时刻只有一处决定物理布局，其余组件只读：attention 后端选择后，把结果发布到 `CacheConfig.kv_cache_layout`，解析顺序依次是：

1. 测试覆盖（测试可强制指定）；
2. 后端强制（某些后端只支持特定布局）；
3. `VLLM_KV_CACHE_LAYOUT` 环境变量；
4. connector 偏好（外部 KV 传输系统的要求）；
5. 默认 `LBNHC`。

所有消费者都从 `CacheConfig` 读取，`indexes_kv_by_block_stride` 和每个后端的 cache-shape/stride-order hooks 被整体删除。**先有标准、再做优化**——layout 标准化的意义不在它自己，而在于让后续的机制演进不必在 6 个后端里各打一份补丁。

## 四、warmup 后实测：让系统自己量显存

地址空间与物理页分离之后，分配时机从「启动时猜」变成「warmup 后测」：先以极小物理占用跑完真实流程（warmup + CUDA Graph capture），再测量剩余空间，最后一次补齐。这是 PR #50779 的核心流程。

### 4.1 四步流程

```text
1. 预留虚拟地址（VA）空间，每个 layout segment 只 commit 1 个 block
2. 跑 warmup + CUDA Graph capture（物理占用极小）
3. 实测 post-warmup 空闲内存，重算 KV Cache 大小
4. extend_kv_cache() 在同一 base pointer 下映射剩余页
```

用地址空间的视角看这四步，物理页的追加不移动任何已有内容：

```text
预留的虚拟地址空间（base pointer 恒定不变）
┌────────────────────────────────────────────────────┐
│  segment 1   │  segment 2   │  尚未提交物理页的区间   │
│  (1 block)   │  (1 block)   │                      │
└────────────────────────────────────────────────────┘
      │              │
      ▼  第 2 步：warmup + CUDA Graph capture（物理占用 ≈ 0）
      ▼  第 3 步：实测空闲内存 → 重算 KV Cache 大小
      ▼  第 4 步：extend_kv_cache() 在既有 block 之后追加物理页
┌────────────────────────────────────────────────────┐
│  segment 1   │  segment 2   │  全部物理页已提交      │
│  (所有 block) │  (所有 block) │                     │
└────────────────────────────────────────────────────┘
```

第一步的「每 segment 只 commit 一个 block」是精心设计的：它让 warmup 和 CUDA Graph capture 以真实形状跑过完整流程——所有后 profiling 期的分配（graph 池、投机 logits、workspace）此时真实发生并被计入——而物理占用几乎为零，**warmup 期 OOM 从根源上消失**。第三步的实测逻辑在 `post_warmup_available_memory()`（`vllm/v1/core/kv_cache_utils.py:2311-2343`）：

```python
# kv_cache_utils.py:2311-2343（节选）— 从 warmup 实际用量反推 KV Cache 预算
def post_warmup_available_memory(
    vllm_config: VllmConfig,
    available_memory: list[int],
    warmup_memory: list[int],
    transient_peak_headroom: list[int],
) -> list[int] | None:
    """Per-worker KV cache memory re-derived from what warmup actually used.

    Returns None when the first-pass sizing should stand: with an explicit
    `kv_cache_memory_bytes` the requested size is committed as-is, and when
    warmup was skipped (VLLM_ELASTIC_EP_SCALE_UP_LAUNCH) there is no
    measurement to re-derive from."""
    if vllm_config.cache_config.kv_cache_memory_bytes is not None or not warmup_memory:
        return None
    final = [
        max(available - used - _warmup_memory_buffer(peak, available), 0)
        for available, used, peak in zip(...)
    ]
```

### 4.2 为什么增长不破坏已捕获的 Graph

整套方案的可行性支点在一个不变式上：**每个 block 在 layout segment 内的 offset 固定，base pointer 从不移动**。由此，layer views 与已捕获的 CUDA Graphs 在增长后依然有效——`extend_kv_cache()` 只是把新的物理页映射到已预留、已录制的地址区间，**无需 re-view、无需 re-capture**。这是「先预留地址」这个设计的最直接回报：如果显存块在增长时需要搬家，CUDA Graph 捕获的地址固化特性会让整个方案失去意义。

### 4.3 新的 margin 账本

实测不是零 margin。warmup 仍然看不到三类内存：碎片、warmup 未覆盖的激活形状、运行期才增长的 workspace。它们由 `_warmup_memory_buffer()`（`vllm/v1/core/kv_cache_utils.py:97-118`）兜底：

```python
# kv_cache_utils.py:97-117（节选）— 实测之后的 margin，不再是拍脑袋的 8%
_WARMUP_MEMORY_BUFFER_FLOOR_BYTES = 150 * (1 << 20)
_WARMUP_MEMORY_BUFFER_PEAK_FRACTION = 0.20
_WARMUP_MEMORY_BUFFER_MIN_BUDGET_FRACTION = 0.02
_WARMUP_MEMORY_BUFFER_MAX_BUDGET_FRACTION = 0.10

def _warmup_memory_buffer(transient_peak_headroom: int, available_memory: int) -> int:
    buffer = max(
        _WARMUP_MEMORY_BUFFER_FLOOR_BYTES,
        int(_WARMUP_MEMORY_BUFFER_PEAK_FRACTION * transient_peak_headroom),
        int(_WARMUP_MEMORY_BUFFER_MIN_BUDGET_FRACTION * available_memory),
    )
    return min(buffer, int(_WARMUP_MEMORY_BUFFER_MAX_BUDGET_FRACTION * available_memory))
```

三个分量各司其职：

- **150MB 下限**兜住最小运行成本；
- **峰值激活的 20%**覆盖 warmup 没压到的激活形状；
- **预算的 2% 下限与 10% 上限**夹住 buffer 占比——注释明确说 margin 只额外承担一小份 KV 预算，远低于旧固定默认值留下的约 8% 闲置（"still far below the ~8% the old fixed-utilization default left idle"），上限防止 floor 在小预算场景（共享设备、低 utilization）下反噬 KV Cache 本身。

实测还有一个失败保护：如果 post-warmup 实测显示没有剩余空间（例如同进程内共驻了 reference model 消耗了预算），不放弃启动，而是**保留第一轮通过 warmup 证明可行的大小**并告警（`kv_cache_utils.py:2332-2342`）。

### 4.4 默认开启与配套清理

`--enable-extensible-kv-cache` 参数**默认开启**——只要驱动支持 VMM（以下均为 #50779 合入后的行为）：

```python
# cache.py:77-84 — 三个默认值的联动
DEFAULT_GPU_MEMORY_UTILIZATION: ClassVar[float] = 0.92
# Sizing from measured post-warmup memory needs no unused fraction as a
# margin; the explicit post-warmup buffer is its margin instead.
DEFAULT_GPU_MEMORY_UTILIZATION_EXTENSIBLE: ClassVar[float] = 1.0
DEFAULT_EXTENSIBLE_KV_CACHE: ClassVar[bool] = True
```

配套变化：`gpu_memory_utilization` 字段默认值从字面量 0.92 改为 None（「未指定」），构造时按是否启用 extensible 模式解析——启用则取 1.0，不启用则取 0.92（`cache.py:_apply_defaults`）；新增 `user_specified_*` 标志追踪用户是否显式给出；CI 侧有一个专门 commit，让不需要的测试不再 pin `gpu_memory_utilization`。生态层面的态度已经明确。

## 五、参数退居二线：边界与权衡

gpu_memory_utilization 没有被删除——它从「必填的预算声明」降级为「可选的显式覆盖」。保留它是有理由的：有些信息系统确实无法自测。

### 5.1 参数没有死：两类必须显式设置的场景

**共享 GPU** 是第一类。默认值假设独占设备，文档在 #50779 里新增了明示：

> Sharing a GPU with another process requires setting it explicitly, since the default assumes exclusive use of the device.（`cache.py:131-132`）

两个 vLLM 实例（或 vLLM + 其他负载）共驻一张卡时，post-warmup 实测看到的是「别人也在用」的显存——这时唯一合理的预算来源就是用户显式给出的数字。第二类是显式覆盖预算上限：例如刻意限制 KV Cache 大小以给其他业务留空间。

fallback 路径同样完整：当 extensible 模式不可用（无 VMM 驱动的老环境、或配置冲突）时，`disable_extensible_kv_cache()`（`cache.py:331-340`）把 `enable_extensible_kv_cache` 置 False，并把未显式指定的 `gpu_memory_utilization` 恢复为 0.92——**因为回退到的估算路径需要那个 unused fraction 当 margin**。这是「默认值」与「语义」强耦合的一个干净例子：utilization=1.0 只有在实测 sizing 下才成立。

### 5.2 实测的盲区

实测替代估算并没有消灭 margin，只是把 margin 从「拍脑袋的 8%」变成「有测量依据、可收敛的 3 分量」。代码注释里有一句诚实的自省：buffer 的系数尚未拟合（"TODO: the fraction is unfitted; it wants a sweep against a long-running workload"）。碎片化、warmup 未覆盖的形状、运行期 workspace 增长——这些仍要靠 margin 覆盖，只是基数从「全部显存的 8%」变为「峰值激活 + KV 预算份额」。

### 5.3 已知缺口与成熟度

PR 状态是 draft，依赖 #51718 先落地（#51718 自身堆叠在 #51704 → #51612 之上）。作者列出的已知缺口：

| 缺口                                                         | 影响                  |
| ------------------------------------------------------------ | --------------------- |
| ROCm `hipMem*` 后端未在 AMD 实机验证                         | ROCm 路径仅代码级就绪 |
| 多节点 NIXL（GDR + POSIX-FD 分配）未验证                     | 验证环境仅 localhost  |
| encoder-decoder 混合 layout 直接 raise 而非回退              | 特定架构不可用        |
| V2 encoder-cache profiling 预留未在最终分支重跑              | 多模态配置缺验证      |
| `commit(defragment=True)` 与 connector sleep gate 无单元测试 | 仅 e2e NIXL 覆盖      |

**方向确定，工程未完成**——这是对这两个 PR 最准确的成熟度评估。

### 5.4 无损的证据链

sizing 方式变了，正确性不能变。PR 提供了两层证据：**byte-identical 输出对比**（标准生成、sleep/wake、`--kv-cache-memory-bytes` 三组配置下，输出与不可增长基线逐字节一致）与 **gsm8k 基准**（P/D 分离——Prefill 与 Decode 分机部署——四种配置下 0.415~0.422，与基线 ~0.41 一致）。压力用例的 sizing 数据也值得记录：Qwen3.5-4B TP4 + MTP100 + chunk256 下，实测重算把 KV Cache 从 35,277 tokens 调到 34,721（eager）、从 35,243 调到 34,592（cudagraph）——**margin 收窄后的数字，而不是拍脑袋的百分比**。

## 六、结语：从「猜」到「测」的系统哲学

一个参数的谢幕背后，是推理系统把「运行时才知道的事实」从用户接口移回系统内部——**凡能自测的，就不该让用户猜**。

这不是孤例。PD 分离把「什么时候传 KV」从用户的部署决定变成系统的调度决策；KV Cache 量化把「省多少显存」从用户的手工权衡变成引擎的默认策略。gpu-memory-utilization 的演进是同一趋势在显存预算上的投影：当系统能够自己测量自己，参数的默认值就从「保险系数」变成「1.0」。

对部署者的行动指南分两段：

- **合入之前**：共享 GPU 场景仍需显式设置 `--gpu-memory-utilization`（这是它唯一的必填场景），独占设备保持默认即可；
- **合入之后**：该参数对大多数用户变为可省略——OOM 与闲置两个方向都有系统兜底。

跟进验证路径：#51718 先合入 main（layout 标准化），#50779 随后 retarget 到 main 并转正；届时用 `vllm serve --enable-extensible-kv-cache` 跑一次自己的压力配置即可确认效果——该参数本就默认开启，显式传参只是为了确认当前环境走的是 extensible 路径。

## 附：源码与证据索引

### PR 关系

- **#50779**（Extensible KV cache，njhill，draft）：堆叠于 #51718 之上，产品化 #47363（zhuohan123 的 demo 实现）；依赖 #44458（= #51718 的前身，restacked 后被取代）
- **#51718**（[6/N] KV-cache layout 重构，LucasWilkinson）：堆叠于 #51704 → #51612 之上，取代 #44458

### 关键文件索引

| 文件（PR head 分支）                                                  | 关键类 / 函数                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vllm/config/cache.py`                                                | `CacheConfig`：`DEFAULT_GPU_MEMORY_UTILIZATION`(0.92) / `DEFAULT_GPU_MEMORY_UTILIZATION_EXTENSIBLE`(1.0) / `DEFAULT_EXTENSIBLE_KV_CACHE`(True)；`_apply_defaults`；`disable_extensible_kv_cache`；`get_resolved_kv_cache_layout`；`user_specified_*` 标志 |
| `vllm/config/vllm.py`                                                 | `_validate_extensible_kv_cache`：不兼容配置的校验与回退                                                                                                                                                                                                   |
| `vllm/v1/core/kv_cache_utils.py`                                      | `_WARMUP_MEMORY_BUFFER_*` 常量；`_warmup_memory_buffer`；`post_warmup_available_memory`；`may_override_num_blocks`                                                                                                                                        |
| `vllm/utils/vmm_driver.py`（新增，398 行）                            | ctypes 绑定 `cuMem*` / `hipMem*`；`_HIP_MEM_SET_ACCESS_FIXED_IN = (7, 12)`                                                                                                                                                                                |
| `vllm/utils/extensible_tensor.py`（新增，595 行）                     | `_VirtualBuffer`（`reserve` / `ensure_committed_range` / `release_physical` / `free`）；`ExtensibleTensor`；DLPack 桥接                                                                                                                                   |
| `vllm/v1/kv_cache_interface.py`                                       | `create_kv_cache_views`；`compute_layout_strides`（`extend_kv_cache` 的实现在 `vllm/v1/worker/gpu_worker.py:865` 的 `GPUWorker.extend_kv_cache`）                                                                                                         |
| `vllm/v1/kv_cache_layout.py`（新增，58 行）                           | `KVCacheLayout` 枚举（6 种 stride 排列 + 4 个布尔布局谓词）                                                                                                                                                                                               |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py` 等 | connector 注册范围从视图推导（`NixlBaseConnectorWorker.register_kv_caches`，`base_worker.py:955`；`get_kv_cache_block_regions` 仅见于 PR 描述，head 分支未检索到该符号，可能已改名）                                                                      |

### 测试数据（#50779 Testing 一节）

- Unit：`test_extensible_kv_cache.py` + `test_extensible_tensor.py` + `test_engine_args.py` → 29/29 通过（GB200, aarch64）
- 压力用例：Qwen3.5-4B TP4 + MTP 100 tokens + chunk 256 + utilization 0.92（传统路径 OOM）→ extensible 模式通过；sizing 35,277→34,721 tokens（eager）、35,243→34,592（cudagraph）
- 正确性：标准生成 / sleep-wake / `--kv-cache-memory-bytes` 三组输出 byte-identical；gsm8k P/D 分离 0.415~0.422 vs 基线 ~0.41

### 诚实声明

本文全部源码引用来自 GitHub API 获取的 PR patch（2026-08-20 head），**非本地仓库源码**；行号以 patch hunk 标注为准，两个 PR 合入 main 后必然漂移。机制描述以 PR 描述 + patch 为准，未运行任何验证。
