# 第 6 讲：KV Cache 在 GLM-5.3-Flash 里怎么变成三本账 — PPT 提纲

每讲 10–12 页。每页只写标题和一句落点，不写逐页讲稿。

1. **封面：45 个文本层，不是模型全部**
   落点：先分清 45 层文本塔和 24 层 vision encoder，本讲只算文本塔的三本账。

2. **同一批层，两种注意力**
   落点：34 层 KDA 加 11 层 full；full 的注意力是 DSA，KV 布局是 MLA，DSA 与 MLA 不是两种层。

3. **三本账与口径表**
   落点：先定单位（B/token、MB/state slot、MB=10^6 B）和「文章口径」，再看数字；scaled FP8 只约束前两行，KDA 的 dtype 是 conv bf16 + temporal fp32。

   | 账本 | 单位 | 文章口径下的单值 | 随什么增长 |
   | --- | --- | --- | --- |
   | token KV | B/token | 5,808（scaled FP8） | 上下文长度 |
   | indexer | B/token | 1,452 | 上下文长度 |
   | KDA state | MB/state slot | 147.6（34 层合计） | state slot 数 |

4. **第一本账：11 层 MLA latent KV**
   落点：528 B/层/token × 11 = 5,808 B/token；528 是 scaled FP8 行宽，不是 512。

5. **第二本账：11 层 DSA indexer**
   落点：132 B/层/token × 11 = 1,452 B/token；indexer 是 `IndexKeyCache` buffer，不是独立池。

6. **第三本账：34 层 KDA state**
   落点：`S_t = f(S_{t-1}, x_t)`，状态形状由 heads 和 head_dim 决定；序列从 10 token 到 10K token，state 大小不变。

7. **KDA 的 slot 账**
   落点：147.6 MB 是一个 state slot 的 34 层合计，conv 默认 bf16、temporal 默认 fp32；全部 fp32 约 152.6 MB。

8. **5 slots 从哪来**
   落点：默认 triton + overlap + extra_buffer 是 base 3 加 overlap 2 等于 5；关 overlap 是 4，no_buffer 或非 triton 是 3；5 × 147.6 MB 是预留上限。

9. **Indexer 与 compress tail 的边界**
   落点：compress tail 是请求级尾缓冲，不进 HiCache；本模型 CUDA hybrid 路径的 host pool 只有 KV 和 MAMBA 两个 entry，indexer 和 tail 不在其中，bare DSA 的 `PoolName.INDEXER` sidecar 与 NPU 的 `page_first_kv_split` 是两条不同例外。

10. **一个 wrapper，两个子池**
    落点：`HybridLinearKVPool` 下挂 `full_kv_pool` 和 `mamba_pool`；token KV 在 `kv_buffer`，indexer 在 `IndexKeyCache`，KDA state 在 `mamba_pool`。

11. **一个 wrapper，两个子池的双预算**
    落点：调度器分别用 `max_total_num_tokens` 和 `max_mamba_cache_size // slots` 约束两个子池，这是两条容量约束，不是两个池对象。

12. **实验与小结：先爆哪本账**
    落点：用公式复算和启动日志对照，判断 token 池还是 KDA 池先限制并发；多卡先确认 attn_tp。
