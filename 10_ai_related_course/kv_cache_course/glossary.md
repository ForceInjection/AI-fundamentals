# KV Cache 八讲 — 术语表

首次出现时以本表为准；各讲正文可以展开，但不要换用另一套叫法。

## 基础

| 术语     | 含义                                                                         | 首次出现   |
| -------- | ---------------------------------------------------------------------------- | ---------- |
| KV Cache | 自回归生成中缓存历史 token 的 Key/Value，避免每步重算                        | 第 1 讲    |
| token    | 模型处理的最小单位；一个请求的 token 数记为 `T`                              | 第 1 讲    |
| Q / K / V | 注意力里的 query / key / value；Q 是当前 token 的查询，K 是被匹配的键，V 是被加权取回的内容 | 第 1 讲 |
| attention | 用 Q 与所有 K 算相关性，softmax 后对 V 加权求和 | 第 1 讲 |
| head | 注意力里的一个并行子空间；每个 head 有自己的 Q/K/V 投影 | 第 1 讲 |
| `H_q` | Q head 数 | 第 1 讲 |
| `H_kv` | KV head 数；MHA 时等于 `H_q`，GQA/MQA 时小于 `H_q` | 第 1 讲 |
| `head_dim` | 每个 head 的维度 `D`；不要用 `hidden_size / H_q` 代替 | 第 1 讲 |
| prefill  | 一次处理整段 prompt，通常计算密集                                            | 第 1 讲    |
| decode   | 每步生成一个 token，典型小 batch 下受显存带宽限制                            | 第 1 讲    |
| TTFT     | Time To First Token，包含排队、prefill 和首 token 采样                       | 第 1 讲    |
| TPOT     | Time Per Output Token，decode 阶段每个输出 token 的平均时间                  | 第 1 讲    |
| 吞吐     | 单位时间处理的 token 数或请求数；要和延迟指标分开看                          | 第 1 讲    |
| MHA      | Multi-Head Attention，每个 query head 有独立 K/V                             | 第 1 讲    |
| GQA      | Grouped-Query Attention，多个 query head 共享一组 K/V                        | 第 1 讲    |
| MQA      | Multi-Query Attention，所有 query head 共享一组 K/V                          | 第 1 讲    |
| MLA      | Multi-head Latent Attention，存共享 latent 和 RoPE K，不按 `2 × H_kv × D` 存 | 第 1、6 讲 |
| KDA      | linear attention 的一类，维护固定大小的循环状态，不按 token 存 KV            | 第 6 讲    |
| DSA      | DeepSeek Sparse Attention；GLM-5.3-Flash 的 11 层 full attention 用它        | 第 6 讲    |
| SWA      | Sliding Window Attention，只保留最近 `W` 个位置                              | 第 7 讲    |
| CED      | V4.1 的压缩编码器-解码器结构，文中按 20 encoder + 20 decoder 层讲            | 第 7 讲    |
| CSA2     | V4.1 的跨层稀疏注意力结构，文中按 4 Full + 4 Reindex + 30 Reuse 层讲         | 第 7 讲    |
| MTP      | Multi-Token Prediction；容量口径要注明是否包含 MTP                           | 第 6 讲    |

## 分页与复用

| 术语                   | 含义                                                                                         | 首次出现   |
| ---------------------- | -------------------------------------------------------------------------------------------- | ---------- |
| block / page           | 固定大小的 token 组；vLLM 多叫 block，SGLang 多叫 page                                       | 第 2 讲    |
| block_size / page_size | 一个 block/page 容纳的 token 数；nano-vllm 默认 256，vLLM 默认 16，GLM 的 SGLang 物理页为 64 | 第 2 讲    |
| logical block index    | 请求内部第几个 block                                                                         | 第 2 讲    |
| physical block id      | KV 池里实际分配的块编号                                                                      | 第 2 讲    |
| block table            | 每个请求的 logical block → physical block 映射                                               | 第 2 讲    |
| slot                   | 一个 token 在 KV 池中的具体位置：`block_id × block_size + offset`                            | 第 2 讲    |
| slot mapping           | 当前 batch 每个 token 对应的 slot 列表                                                       | 第 2 讲    |
| 内部碎片               | 最后一个 block 未用满的 slot；最多 `block_size - 1` 个 token                                 | 第 2 讲    |
| 外部碎片               | 连续分配下不同长度请求释放后留下的空洞；固定 block 池按构造为 0                              | 第 2 讲    |
| prefix caching         | 相同前缀的 KV 只算一次、跨请求复用                                                           | 第 2、5 讲 |
| ref_count / ref_cnt    | nano-vllm 的 block 级引用计数；vLLM 写作 `ref_cnt`                                           | 第 3、4 讲 |
| lock_ref               | SGLang 树节点的锁引用，沿路径累加；root 不参与释放                                           | 第 5 讲    |
| RadixTree              | 按 token 前缀组织 KV 的树；节点可分裂、加锁、驱逐                                            | 第 5 讲    |
| UnifiedTreeCore        | SGLang 当前默认的统一树核心，替代经典 `RadixCache` 的主路径                                  | 第 5 讲    |
| HiCache                | SGLang 的层次化 KV 缓存，包含 HBM、host DRAM、存储层                                         | 第 5 讲    |
| L1 / L2 / L3           | HiCache 的 device / host / storage 三层；L1/L2 私有，L3 是否共享取决于 backend               | 第 5 讲    |
| backup                 | device → host 的拷贝；不等于释放 device                                                      | 第 5 讲    |
| loadback               | host/storage 命中后把 KV 回载到 device                                                       | 第 5 讲    |
| demote / evict         | 释放 device 上的 KV；demote 通常指 host 副本已存在后的下沉                                   | 第 5 讲    |
| host_value             | `UnifiedTreeNode` 组件上的 host-pool slot 索引，不是 DRAM 指针                               | 第 5 讲    |
| host_lock_ref          | 保护 host 节点的锁引用                                                                       | 第 5 讲    |

## 模型与容量

| 术语          | 含义                                                                | 首次出现   |
| ------------- | ------------------------------------------------------------------- | ---------- |
| KDA state     | KDA 每层维护的循环状态；一个请求在一个 KDA 层占一个 state slot      | 第 6 讲    |
| state slot    | 存放一个请求在一个 KDA 层的 state 的槽位                            | 第 6 讲    |
| global KV     | DeepSeek-V4.1-Flash 的全局 KV，包含 main KV 和 indexer K            | 第 7 讲    |
| SWA KV        | 最近 128 个位置的滑动窗口 KV，FP8 存储                              | 第 7 讲    |
| indexer       | 稀疏注意力的索引缓存；在 global KV 里是组成部分，不是独立池         | 第 7 讲    |
| compress tail | 请求级压缩尾缓冲，不进入 HiCache                                    | 第 6 讲    |
| PD 分离       | Prefill / Decode 分离部署，KV 通过 RDMA/NVLink 传输                 | 第 8 讲    |
| TP / DP / CP  | Tensor / Data / Context Parallel                                    | 第 6、8 讲 |
| attn_tp       | attention 部分的 TP 大小；文章里的 147.6 MB 等数字按 `attn_tp=1` 计 | 第 6 讲    |

## 部署参数

| 参数                       | 作用                                                             |
| -------------------------- | ---------------------------------------------------------------- |
| `max_total_tokens`         | SGLang token 池上限；超过 profiled 值会被压回，主要用于开发/调试 |
| `max_running_requests`     | 最大并发请求数；受 token 容量和 state slot 容量双重限制          |
| `max_mamba_cache_size`     | SGLang KDA/mamba state slot 上限                                 |
| `mamba_full_memory_ratio`  | SGLang 在 token 池与 state 池之间分配 HBM 的比例旋钮             |
| `--max-num-seqs`           | vLLM 的最大并发序列数                                            |
| `--max-num-batched-tokens` | vLLM 每步 token 预算，不是 KV 池容量                             |
| `--gpu-memory-utilization` | vLLM 允许使用的 GPU 显存比例                                     |
| `--kv-cache-memory-bytes`  | vLLM 直接指定 KV 池字节数                                        |
| `--max-model-len`          | 单请求最大上下文长度                                             |
| MB                         | 本课程按 `10^6 B` 计；不是 MiB                                   |
