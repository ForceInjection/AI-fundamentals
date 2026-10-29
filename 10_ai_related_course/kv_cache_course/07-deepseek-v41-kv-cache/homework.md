# 第 7 讲：课后作业

## 基础题

1. 从 DeepSeek-V4.1-Flash 的架构复算 890 B/token。写出 2.5 个 entry 来自哪四个 Full 层、各自的压缩率，main KV 288 B 与 indexer K 68 B 的逐项组成。
2. 上下文从 8K 增加到 128K：
   - main KV 增长多少；
   - indexer K 增长多少；
   - SWA KV 增长多少。
   三个答案分别给公式和数值，并说明哪一个不随长度增长。
3. 为什么 V4.1 选择不持久化 SWA KV、而用 bounded replay 兜底？在什么配置下 vLLM 会反过来缓存 SWA？答案要覆盖访问模式、持久层容量、replay 代价，以及关闭 bounded replay 的三种条件。

## 源码题

读本地 vLLM（HEAD `8b365ff949`）：

- `vllm/config/cache.py`：`swa_bounded_replay` 是字段还是函数？默认值是什么？对应的 CLI flag 在哪？
- `vllm/v1/core/sched/scheduler.py`：`_mark_prefix_replay` 在哪些路径被调用？它如何设置 `request.replay_start`？
- `vllm/v1/attention/backends/mla/sparse_swa.py`：`DeepseekV4SWACache` 的 `block_size` 是多少？它和 token 级 KV 的分配粒度是否一致？
- `vllm/v1/kv_cache_interface.py`：`SlidingWindowMLASpec.prefix_cacheable` 返回什么？`bounded_replay` 为 True 和 False 时，SWA 是否进 prefix cache？

回答时必须区分：配置字段、调度器触发点、缓存实现类，三者不是同一个东西。

## 口径题

列一张 V4-Pro / V4-Flash / V4.1-Flash 的对照表，至少包含：

| 列 | 要求 |
| --- | --- |
| 模型层数 | V4-Pro 61 层，V4.1-Flash 40 层；V4-Flash 按公开 config 查 |
| global KV 精度 | bf16 / FP8 / NVFP4 分别注明 |
| 每 token 字节 | 890 只算 V4.1 的 global KV |
| 统计范围 | 是否含 SWA、是否含 indexer、是否只算 global |
| 数据来源 | 报告、vLLM 博客或 config，逐项注明 |

表后回答：V4-Pro 的 9.62 GiB/1M 序列估算，能不能除以 890 B/token 得到“V4.1 比 V4-Pro 省多少倍”？为什么？

提示：9.62 GiB/1M 是 V4-Pro、61 层、bf16 的估算，统计范围含 main KV、c4a 索引器和 SWA 窗口；890 是 V4.1-Flash、40 层、NVFP4 main、global-only。模型、精度、统计范围三项都不同，相除没有意义。要做跨代比较，用报告给的 1/4 和 1/8。

## 提交

分层表 + 源码说明 + 口径表，不超过三页。所有数字附公式或源码位置。
