# 第 6 讲：课后作业

- 模型 config：[zai-org/GLM-5.3-Flash config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json)
- SGLang 基线：`f4de6abee6`
- 预读：[三本账：图解 GLM-5.3-Flash 的 KV Cache 布局](../../../09_inference_system/sglang/glm53-flash-kv-layout.md)

> 文章口径：单 rank、attn_tp=1、scaled FP8、64-token page 预留、无 MTP。
>
> 每个数字都要写清四项：模型变体、精度、缩放因子、统计范围。缺一项，这个数字不算完成。第 1、2 题按文章示例算，不要先把口径改成 SGLang 的默认值；第 5 题再单独比较。

## 基础题

1. 按文章示例复算 7,260 B/token（scaled FP8）。
   - 写出 528 和 132 的来历；
   - 说明 5,808 和 1,452 各覆盖几层；
   - 说明为什么这是 64-token page 的容量预留，不是有效写入量；
   - 给出 H100/H200 上 `auto` 可能落到 BF16 时的 12,716 B/token，以及 TRT-LLM raw FP8 的 7,084 B/token，说明差异来自哪里。

2. 按文章示例复算 147.6 MB/state slot。
   - 写出 conv 和 temporal 的 shape、dtype；
   - 说明 147.6 MB 是 34 层合计；如果把它再乘 34，会得到约 5.0 GB，这是错的；
   - 如果 conv 也用 fp32，结果是多少？

3. 默认 5 slots/request 的 738 MB/request；同时给出 4 slots 和 3 slots 的值及触发条件（fp32 下约 738.1 / 590.5 / 442.9 MB/request）。三个值都要写，不能只算默认 5。
   - 5：默认 triton + overlap + extra_buffer，base 3 + overlap 2；
   - 4：`--disable-overlap-schedule`，base 3 + 1；
   - 3：`no_buffer` 或非 triton；
   - 每个值注明源码锚点（`kv_cache_configurator.py:174-184`、`:2232-2261`）。

4. 用 `--mamba-ssm-dtype bfloat16` 重算第 2、3 题：76.3 MB/slot，5/4/3 slots 分别约 382/305/229 MB。

5. 对照三种 KV 口径：scaled FP8 7,260 B/token、TRT-LLM raw FP8 7,084 B/token、BF16 12,716 B/token。说明每种口径下 528、512、1,024 三个行宽分别怎么来。

## 源码题

1. 画 `HybridLinearKVPool → full_kv_pool / mamba_pool → kv_buffer / IndexKeyCache` 三级图，标出哪个是池、哪个是 buffer；明确 indexer 不是池。
2. 说明 token KV、indexer、KDA state 分别放在哪里；`HybridLinearKVPool` 和 `MambaPool` 是否在同一个源文件。
3. 哪些对象有 HiCache host 副本，哪些没有？限定本模型的 CUDA hybrid 路径；补一句 bare DSA 的 `PoolName.INDEXER` sidecar 和 NPU 的 `page_first_kv_split` 分别改变了什么（前者是 indexer sidecar，后者是 Mamba host layout 回退）。
4. KDA state 的 backup/loadback 与 token KV 是否走同一条路径？相同点是什么，不同点是什么？提示：`host_value` 契约相同，transfer 形状不同；KV 可沿 evicted ancestors 拼 page，Mamba 只加载一个 state slot。

## 思考题

为什么 indexer 和 compress tail（请求级尾缓冲，不进 HiCache）在本模型的 HiCache host pool 里没有 entry？如果强行给它们加 host 副本，收益和代价各是什么？可以从 `HostPoolGroup` 的 entry、host buffer 布局、DRAM 带宽和实现复杂度四个角度讨论。

## 提交

容量表、三级对象图和源码说明，不超过三页。每个数字按题首的四项要求标注。
