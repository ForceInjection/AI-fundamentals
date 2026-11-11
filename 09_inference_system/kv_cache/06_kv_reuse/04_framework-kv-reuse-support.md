# 四条路线，一条落地：vLLM 与 SGLang 的 KV 复用支持现状

> **元信息**：核对基线为本地源码 HEAD：vLLM `c2320e37f1`（2026-11-09）、SGLang `f4de6abee6`（2026-09-28）、LMCache `f93cafe6`（2026-11-08），核对与写作时点 2026-11-11。核对方式：两仓逐条源码检索（前缀语义、LoRA 交互、connector 生态、非前缀复用、跨权重复用、PD 传输），关键论断另做人工抽查。框架移动快，引用结论前先对 commit。

[KV 复用总纲](01_kv-reuse-overview.md)把复用条件放宽划成四条路线：前缀缓存（零对齐代价的基线）、CacheBlend（免训练的非前缀选择性重算）、KVLink（文档级预计算 + 联合微调）、DroidSpeak（跨微调变体共享）。论文给的是机制与账本，引擎给的是落地。本文按同一框架盘点两大引擎的现状，结论先行：路线 1 两家原生默认开且已卷到跨实例纵深；路线 2 只剩 LMCache 一条接入路径，且与前缀缓存互斥；路线 3/4 在两仓中零实现。

## 一、路线 1：默认开，且严格前缀

### 1.1 vLLM：APC 默认开，哈希链保证「断链即停」

V1 引擎的前缀缓存默认开启（`enable_prefix_caching: bool = True`，`vllm/config/cache.py:260`）。复用的严格前缀语义写进了哈希构造：子块的哈希里包含父块哈希（`hash_block_tokens()`，`vllm/v1/core/kv_cache_utils.py:703-734`），哈希器 docstring 自述「每个哈希唯一指纹化了到其边界为止的完整前缀」（kv_cache_utils.py:957）。查找端从第一个块顺序匹配，中间缺失即终止（`find_longest_cache_hit`，`vllm/v1/core/single_type_kv_cache_manager.py:798-802`，注释原话：A missing block implies every later block misses too）。

### 1.2 SGLang：RadixTree 常开，UnifiedRadixTree 不改语义

RadixAttention 前缀缓存常开：`disable_radix_cache` 默认 False（`python/sglang/srt/arg_groups/fields/memory.py:64-66`），只有显式关闭才落到不做复用的 ChunkCache（`srt/mem_cache/registry.py:88-96`）。匹配从 root 沿 children 逐节点做最长前缀（含节点内 split，`srt/mem_cache/radix_cache.py:639-646`）。0.5.16 起默认的 UnifiedRadixCache 把 FULL/SWA/MAMBA 做成同一棵树上的组件（`srt/mem_cache/unified_cache/component_type.py:8-14`），匹配仍从 key[0] 起沿树走（`unified_cache/unified_tree_core.py:896`）；组件校验只会让命中变短（如 SWA 段被逐出则该段不命中），不会产生非前缀命中。

### 1.3 跨实例纵深：两条不同的生态路径，同一套前缀语义

|            | vLLM                                                                            | SGLang                                                                                                |
| ---------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 接入面     | 通用 KVConnector 插件接口（`KVConnectorBase_V1`，factory 注册 18 个 connector） | 无通用插件接口；两个面：HiCache L3 存储后端注册表 + radix-cache 整体替换注册表                        |
| 跨实例共享 | LMCache/Mooncake/HF3FS 等 connector 承担分层与共享存储                          | L3 层集群共享（L1/L2 实例私有，KV 进 L3 才对其他实例可见），mooncake/hf3fs/nixl/aibrix 等 12 个后端   |
| 共享语义   | 键为 token 前缀哈希                                                             | L3 页哈希即从序列起点起的前缀哈希链（`srt/mem_cache/hicache_storage.py:47-48`），匹配返回「连续前缀」 |

vLLM 侧（`vllm/distributed/kv_transfer/kv_connector/factory.py:158-317`）：Nixl/Mooncake 走 PD 传输，LMCache 走分层存储，HF3FS/UMBP/FlexKV 走共享或第三方 KV 存储。SGLang 侧（`srt/mem_cache/storage/backend_factory.py:202-270`）：L3 后端注册表管纯 KV 字节层的存取；`enable_lmcache` 与 FlexKV 则走另一条路：radix-cache 整体替换（registry.py:105-117、131-142），第三方直接接管整个缓存层。方向相反，落地形态一致：跨实例复用依旧严格前缀。

## 二、LoRA：路线 4 的反面，框架选择全隔离

两家的解法同构。vLLM 把 LoRA 的 name 与 path 混进块哈希的 extra keys（`_gen_lora_extra_hash_keys()`，kv_cache_utils.py:586-602，docstring 注明 adapter path 入键是为了防止同名重指向时误复用旧 KV）；SGLang 把 `lora_id` 拼进 radix 树的 extra_key（`srt/managers/schedule_batch.py:1122-1125`），radix_cache.py:357-363 的 docstring 说得直白：不同 adapter 的同 token 序列被有意隔离、永不共享前缀节点。

LoRA 与前缀缓存都不互斥（vLLM 全仓无「开 LoRA 即禁用 APC」的限制），但微调变体之间的 KV 共享在两家都是零：框架对「变体」的处理是命名空间隔离，防串而不促通。DroidSpeak 要做的事（profile 出可信层组后定向借），在框架里连接口都不存在。

## 三、路线 2：LMCache 一条独木桥，且与前缀缓存互斥

vLLM 核心没有任何非前缀复用逻辑（全仓搜索 blend 只命中无关的 hidden-state 残差与注释）。唯一的入口在 LMCache connector：`enable_blending` 打开后，LMCache 的 `LMCBlender` 在逐层保存前执行 blend（`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py:515、626-629、874-882`），即 CacheBlend 的 token 级选择性重算（重算比例参数 `blend_recompute_ratios`，LMCache `lmcache/v1/compute/blend/blender.py:19、49`；MP 模式已演进到 CacheBlend V3，`lmcache/v1/multiprocess/modules/blend.py`，config 区分 `blend` 与 `blend_legacy`）。

但接线处留着一个 TODO，原话是「Need to make prefix caching and blending compatible」（vllm_v1_adapter.py:876）：blending 目前在层写入路径上前置执行、绕开 APC，两套机制并未打通。使用前提还有一条：需要在不同文档块之间插入分隔符让 LMCache 识别边界（机制与源码见 [CacheBlend 专篇](../02_systems/01_lmcache/14_cache_blend.md)）。

SGLang 侧更彻底：全仓含 git 历史无任何 CacheBlend 实现或 PR 痕迹。LMCache 虽然可接入，但走的是 radix-cache 整体替换，blend 能力在这个接入面里没有暴露。

## 四、路线 3/4：空白，以及反向工程

路线 3（KVLink）没有任何引擎落地。「按文档预计算 KV、推理时任意拼接」需要「文档」作为一等复用单元，两家的复用单元都只有 token 块与树节点；KVLink 官方仓库是训练与研究工件，不含 serving 集成（总纲已注明其存储形态是论文自认的未解项）。

路线 4（DroidSpeak）同样为零，且框架在做反向工程：vLLM 的 connector API（`get_num_new_matched_tokens`，base.py:523）以 token 哈希为键，没有模型身份或权重指纹的校验位；MooncakeStore 反而用模型指纹构造命名空间前缀（`cache_prefix='auto'`，按 build、model config、hf_commit 哈希隔离），注释原话是新镜像或模型版本不得读到别的部署写的 KV，目的是防止跨权重误读，方向与路线 4 相反。全仓唯一的「cross-model KV sharing」字样在 vLLM 的 spec-decode 里（draft 与 target 模型共享部分层的 KV），那是同进程的助手模型机制，不是微调变体间的复用。

## 五、容易混淆的两个概念：传输与事件不是复用

PD 分离的 KV transfer（vLLM 的 `start_load_kv` 拉取接口、SGLang 的 MooncakeKVSender/Receiver）搬的是**同一个请求**的 KV，从 prefill 实例到 decode 实例一次；`kv_events`（BlockStored/Removed 事件广播）是给 KV 感知路由（如 NVIDIA Dynamo）做调度参考。两者都与「跨请求、跨模型借用 KV」无关，引用时不要与四条路线混排。

## 六、总评与适用边界

判断一：路线 1 已经卷到纵深，代价口径是「跨实例也只认前缀」。vLLM 靠 18 个 connector 的插件生态、SGLang 靠 L3 后端注册表加 radix 替换注册表，两条生态路径的共享键全部是前缀哈希。

判断二：路线 2 的高度取决于 LMCache 一家。vLLM 有接线但互斥（TODO 未解），SGLang 未接线；blending 要进入主路径，需要引擎侧给「非前缀块的定点重算」一个正式位置，而不是在 connector 的层写入回调里绕行。

判断三：路线 3/4 的空白不是遗漏，是接口缺口。总纲 §六 提过现状是「条件刻在缓存键里」的点对点形态，本文给的是它的源码证据：vLLM 的键是 token 哈希（含 LoRA 身份），SGLang 的键是 extra_key（含 lora_id），除此之外没有任何声明兼容性的字段。要在框架里落路线 2-4，缺的正是出处声明、对齐代价可知性、准入决策接口三样。

适用边界：本篇结论绑定文首三个 commit；vLLM 与 LMCache 基线距写作仅两三天，SGLang 基线早六周（当时快进拉取超时），其后版本可能已变化，引用前建议重新核对。分析只覆盖 vLLM、SGLang 与 LMCache 生态，NVIDIA Dynamo、AIBrix 自身的 KV 共享实现未在盘点范围内；「路线 3/4 零支持」指引擎与 connector 生态，研究原型的仓库状态以总纲与单篇为准。

## 源文件索引

| 仓库（HEAD）         | 文件                                                                                                  | 关键内容                                   | 本文引用 |
| -------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------ | -------- |
| vLLM（c2320e37f1）   | vllm/config/cache.py:260                                                                              | APC 默认开启                               | 1.1      |
| vLLM                 | vllm/v1/core/kv_cache_utils.py:586-602, 703-734, 957                                                  | LoRA extra keys；块哈希链构造              | 1.1、二  |
| vLLM                 | vllm/v1/core/single_type_kv_cache_manager.py:798-802                                                  | 顺序匹配、断链即停                         | 1.1      |
| vLLM                 | vllm/distributed/kv_transfer/kv_connector/factory.py:158-317                                          | 18 个 connector 注册                       | 1.3      |
| vLLM                 | vllm/distributed/kv_transfer/kv_connector/v1/base.py:523                                              | connector 拉取接口签名（无模型身份位）     | 四       |
| vLLM                 | vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py:515, 626-629, 876 | blending 接线与互斥 TODO                   | 三       |
| vLLM                 | vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:221-249                         | 模型指纹命名空间隔离                       | 四       |
| SGLang（f4de6abee6） | python/sglang/srt/arg_groups/fields/memory.py:64-66, 237-239, 248-254                                 | radix 默认开；LMCache/FlexKV 接入          | 1.2、1.3 |
| SGLang               | python/sglang/srt/mem_cache/radix_cache.py:354-363, 639-646                                           | match_prefix 前缀语义；LoRA 隔离 docstring | 1.2、二  |
| SGLang               | python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:896                                    | UnifiedRadixTree 匹配仍前缀                | 1.2      |
| SGLang               | python/sglang/srt/mem_cache/hicache_storage.py:47-48                                                  | L3 页哈希=前缀链                           | 1.3      |
| SGLang               | python/sglang/srt/mem_cache/storage/backend_factory.py:202-270                                        | L3 后端注册表（12 后端）                   | 1.3      |
| SGLang               | python/sglang/srt/managers/schedule_batch.py:1122-1125                                                | lora_id 拼入 extra_key                     | 二       |
| LMCache（f93cafe6）  | lmcache/v1/compute/blend/blender.py:19, 49                                                            | LMCBlender 与重算比例参数                  | 三       |
| LMCache              | lmcache/v1/multiprocess/modules/blend.py                                                              | CacheBlend V3（MP 模式）                   | 三       |

## 参考资料

- 本系列：[KV 复用总纲](01_kv-reuse-overview.md) · [KVLink](02_kvlink.md) · [DroidSpeak](03_droidspeak.md)
- 本仓：[CacheBlend 专篇（LMCache 实现）](../02_systems/01_lmcache/14_cache_blend.md) · [SGLang UnifiedRadixTree 解读](../../sglang/sglang-0.5.16-release.md) · [HiCache 深入详解](../../sglang/hicache_deep_dive.md) · [vLLM 原生 KV Offloading](../../vllm/module_analysis/native_kv_offloading.md)
- 上游：[vLLM](https://github.com/vllm-project/vllm) · [SGLang](https://github.com/sgl-project/sglang) · [LMCache](https://github.com/LMCache/LMCache)
