# 第 6 讲：GLM-5.3-Flash 的三本账

40 分钟。无 GPU 必做约 30 分钟，有 GPU 选做约 10 分钟起。

> 文章口径：单 rank、attn_tp=1、scaled FP8、64-token page 预留、无 MTP。下面没标其他口径的数字都按这组条件。scaled FP8 只约束 11 层 latent KV 和 indexer，KDA state 的 dtype 单独看 conv/temporal。
>
> 单位：MB = 10^6 B。启动日志里的 `GB` 是 `1024^3` B（`memory_pool.py:107`），读日志时分开算。

## 一、无 GPU 必做（30 分钟）

### 1. 手算三本账（12 分钟）

先按公式算，再用 `code/calc_glm53_kv.py` 对答案。脚本只用标准库，不依赖 GPU。

token 账（文章口径，11 层 full）：

```text
528 B = 512 + 512/128 × 4
5,808 B = 528 × 11
132 B = 128 + 128/128 × 4
1,452 B = 132 × 11
7,260 B = 5,808 + 1,452
```

7,260 B/token 是 64-token page 的容量预留，不是有效写入量。`index_kpool=4` 加 compress 时，indexer 每 4 个 token 写一个 pooled K，有效 payload 约 33 B/token/层，11 层约 363 B/token；容量仍按 132 B/token/层预留。换口径时：H100/H200 上 `auto` 可能落到 BF16，约 12,716 B/token；TRT-LLM raw FP8 约 7,084 B/token。

KDA 账（文章口径，34 层合计）：

```text
conv（bf16）= 3 × 24,576 × 2 B = 147,456 B/层   # 3 = short_conv_kernel_size - 1
temporal（fp32）= 64 × 128 × 128 × 4 B = 4,194,304 B/层
每层 = 4,341,760 B
34 层 = 147,619,840 B = 147.6 MB/state slot
```

147.6 MB 已经乘过 34，不要再乘 34。全部 fp32 是 152.6 MB；加 `--mamba-ssm-dtype bfloat16` 后是 76.3 MB/slot，5 slots 约 382 MB/request。

slots 账（文章口径）：

```text
默认 triton + overlap + extra_buffer：base 3 + overlap 2 = 5 slots/request
--disable-overlap-schedule：base 3 + 1 = 4
no_buffer 或非 triton：3
5 × 147.6 MB ≈ 738 MB/request
```

738 MB 是调度器按请求预留的上限，不等于每请求常驻 738 MB；池子按 `max_mamba_cache_size` 整块分配，未用 slot 也在池里。池大小由 `max_mamba_cache_size` 或 `mamba_full_memory_ratio`（默认 0.9）决定；prefix checkpoint 还会占 slot。

> 核对：5/4/3 slots 在 temporal fp32 下约 738.1 / 590.5 / 442.9 MB/request；加 `--mamba-ssm-dtype bfloat16` 后约 381.6 / 305.3 / 229.0 MB/request。

### 2. 读源码，画三级对象图（10 分钟）

源码在 `~/Project/ai-infra/sglang`，HEAD `f4de6abee6`。先确认位置：

- `HybridLinearKVPool` 和 `MambaPool` 都在 `python/sglang/srt/mem_cache/memory_pool.py`，前者约 `:3862`，后者约 `:401`；不要去 `mamba_pool.py` 找。
- `IndexKeyCache` 在 `python/sglang/srt/mem_cache/index_key_cache.py:14`，它是 `DSATokenToKVPool` 内部的 buffer 管理器，不是池。
- host 侧 KDA state 在 `python/sglang/srt/mem_cache/pool_host/mamba.py`。

画出下面三级结构，并标出每个对象的 HBM 位置：

```text
HybridLinearKVPool
├── full_kv_pool: DSATokenToKVPool
│   ├── kv_buffer[11]              # token KV，11 层 full
│   └── IndexKeyCache.buffer[11]   # indexer，不是独立池
└── mamba_pool: MambaPool
    └── mamba_cache.conv / temporal # 34 层 KDA state
```

回答：

1. 哪些对象是池，哪些只是池里的 buffer？
2. 147.6 MB 是每层还是 34 层合计？
3. `full_kv_pool` 和 `mamba_pool` 是平级的两个池对象，还是同一个 wrapper 的两个成员？

### 3. 读示例启动快照（8 分钟）

完整文件在 `code/sample-startup.log`。关键两行：

```text
max_total_num_tokens=131072, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=64, context_len=1048576, available_gpu_mem=64.00 GB
Mamba Cache is allocated. max_mamba_cache_size: 320, conv_state size: 1.49GB, ssm_state size: 42.50GB
```

这是教学快照，字段名来自 `f4de6abee6`，数值按公式构造。`conv_state size` 和 `ssm_state size` 是 34 层 × 320 个 slot 的池总量，日志里的 GB 是 `1024^3` B。`max_total_num_tokens` 的单位是 token，`max_mamba_cache_size` 的单位是 state slot，两者单位不同，不能直接相除或相加；只能分别用每请求的 token 数和 slots 数估算并发。

问题：

1. `max_total_num_tokens` 和 `max_mamba_cache_size` 各是什么单位？
2. 5、4、3 slots/request 下，只看 mamba 池容量，各允许多少并发请求（先不管 `max_running_requests`）？
3. 平均请求长度 2,048 token 时，token 池允许多少并发？和 mamba 池比，谁先到 64？
4. 平均请求长度 4,096 token 时，结论变了吗？
5. 如果把 temporal 换成 `--mamba-ssm-dtype bfloat16`，`ssm_state size` 大约变成多少？

> 核对：320 // 5 = 64，320 // 4 = 80，320 // 3 = 106；131072 // 2048 = 64，131072 // 4096 = 32；bf16 后 `ssm_state size` 约 21.25GB。

## 二、有 GPU 选做（10 分钟起）

没有 GPU 就跳过这一段，用上面的快照完成同样的读数练习。

启动模板（`--tp-size` 按你的卡数改）：

```bash
sglang serve \
  --model-path zai-org/GLM-5.3-Flash \
  --tp-size <按卡数> \
  --mem-fraction-static 0.80 \
  --max-total-tokens 131072 \
  --max-mamba-cache-size 320 \
  --max-running-requests 64
```

先跑默认，抓 `max_total_num_tokens=` 和 `Mamba Cache is allocated.` 两行。然后依次改：

1. 加 `--mamba-ssm-dtype bfloat16`：看 `ssm_state size` 是否约减半，算每 slot 是否接近 76.3 MB。
2. 加 `--disable-overlap-schedule`：看 mamba cap 警告里的 slots/request 是否变成 4。
3. 把 `--mamba-full-memory-ratio` 改成 0.5：看 `max_mamba_cache_size` 和可并发请求数怎么变。

注意：官方 cookbook 的 TP4/TP2 recipe 用 bf16 KV，TP4 还开 EAGLE MTP；不要拿它的日志直接对文章的单 rank scaled FP8 无 MTP 数字。多卡时先确认 `attn_tp = tp_size // attn_dp_size // attn_cp_size`。

## 三、TP/attn_tp 对照（3 分钟）

下表按文章口径的单 rank 值在 attention TP 组内均分估算，未计复制、padding 和对齐。

| 部署形态 | attn_tp | MB/state slot | MB/request（5 slots） |
| --- | --- | --- | --- |
| TP1 | 1 | 147.6 | 738.1 |
| TP2 | 2 | 73.8 | 369.0 |
| TP4 | 4 | 36.9 | 184.5 |
| TP8 | 8 | 18.45 | 92.3 |

`attn_tp=1` 只在 `attn_dp_size == tp_size` 且 `attn_cp_size=1` 时成立。TP8/DP4/cp1 的 attn_tp 是 2，应按 TP2 行；TP8 非 DP 是 8，才按 TP8 行。官方 cookbook 的 TP4/TP2 recipe 也不能直接套文章的单 rank 数字：它们的 KV dtype 和 MTP 设置不同，先读实际命令。

## 四、产出

下表由学员填写；第一行是示例，示范单位和填法。

| 项目 | 单 token / 单 slot | 默认配置 | 可用并发上限 |
| --- | --- | --- | --- |
| token KV（示例） | 5,808 B/token（scaled FP8） | `max_total_num_tokens × 5,808` | `max_total_num_tokens ÷ 平均请求长度` |
| indexer | 待填 | 待填 | 待填 |
| KDA state（temporal fp32） | 待填 | 待填 | 待填 |
| KDA state（bf16 flag） | 待填 | 待填 | 待填 |

表后写一句：默认配置下先限制并发的是哪个池，依据是哪两行日志或哪两个参数。结论要落到“一个 `HybridLinearKVPool`，两个子池的双预算”，不要说成两个池对象。
