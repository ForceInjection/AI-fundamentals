# 第 8 讲：KV Cache 怎么算容量、怎么选引擎 — PPT 提纲

每讲 10–12 页，只列页面主题和讲解落点，不写完整讲稿。

1. 封面：两层账本，一个 HBM 预算
   落点：本讲不算新模型，只把前七讲的账合成一张能执行的容量表。

2. 先纠正一个说法：三本账不是三个池
   落点：page/slot 是分配粒度，DRAM 是 host cache；只有 token 池和 request state 池共享 HBM。

3. token 池账：per-token KV + indexer，按 page 对齐
   落点：给出 `ceil(len/page) × page` 的修正，说明为什么小 page 省碎片、大 page 省元数据。

4. request state 池账 A：GLM 的 KDA
   落点：147.6 MB/slot × 5 slots = 738 MB/request；用 32K 和 128K 两个长度说明瓶颈会切换。

5. request state 池账 B：DeepSeek 的 SWA
   落点：40 × 128 × 528 ≈ 2.7 MB/request；短上下文占 27%，128K 时只占 2.3%。

6. 联合求解：`mem_fraction_static` + `mamba_full_memory_ratio`
   落点：SGLang 用 `r/(1+r)` 切 mamba 预算；vLLM 用 `gpu_memory_utilization` 或 `kv_cache_memory_bytes` 反推。

7. 参数映射：SGLang ↔ vLLM ↔ nano-vllm
   落点：`max_running_requests` ↔ `--max-num-seqs`；`max_total_tokens` 是上限不是承诺；`--max-num-batched-tokens` 是每步预算。

8. `max_running_requests` 的双重上限
   落点：`token_capacity // 2` 与 `max_mamba_cache_size // ratio` 取小；ratio 基础 3，GLM 默认策略 5。

9. page size 表：256 / 16 / 64
   落点：nano-vllm 256、vLLM 16、GLM SGLang 64；碎片公式和 len=1,000 的算例。

10. DRAM 账按对象列
    落点：GLM 的 KV + KDA 进 HiCache，indexer 和 tail 不进；DeepSeek 的 SWA 走独立 host 池，global 才进持久层。

11. PD / offload / prefix cache 判断树
    落点：PD 看 prefill/decode 时长比与传输带宽，offload 看 host 带宽与换出频率，prefix cache 看复用率与 host 容量；没有阈值。

12. 实验与收尾：选一个模型 + 一个引擎
    落点：GLM 算 token/KDA/page，DeepSeek 算 global/SWA，nano-vllm 只算 token/block；先算哪本账先爆，再谈引擎。
