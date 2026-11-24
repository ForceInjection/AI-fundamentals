# 第 9 讲课后作业

## 作业（必做两题）

**题 1（必做，计入考核——作业 3）**：写一份正式对照分析《从 kv_swap 到 HEAD offloading：三代演进》:

- 第一代：V0 CacheEngine swap（决策在主 Scheduler 的 SWAPPED 队列）——给出课程内证据（第 2 讲讲义引用）或 vLLM git 历史；
- 第二代：你 fork 的 kv_swap（设计文档 `kv-offload.md`；三队列；CPU 块不参与前缀缓存；3090 实测）;
- 第三代：HEAD 的双路径（OffloadingConnector 分层后端 / SimpleCPUOffloadConnector）——决策位置、搬运方式、淘汰策略、CPU 块的前缀缓存语义、观测口径五个维度逐一对照，全部带 `file:line`；
- 结论：三代各自的「决策者/搬运者/淘汰策略」填入第 5 页对照表，并回答「如果再让你重写 kv_swap，你会把 swap 决策放在哪里?为什么?」

**题 2（PD 分离方案）**：为一组负载（P=长文档分析 8K prompt,D=每请求 128 token 输出，尾延迟敏感）写 PD 分离方案：prefill 实例与 decode 实例各自的 TP/资源配置、NixlConnector 配置 JSON、一条「为什么这个拆分能压尾延迟」的机制解释（引用 disagg 文档的动机声明与第 2 讲 chunked prefill）。

## 附加题（不计考核）

读 `docs/design/nixl_kv_cache_lease.md`（租约与心跳续期），回答：没有租约机制时，什么时序会导致 D 端读到已被覆盖的 KV?（推演一个具体的竞态时序）租约的「初始 30 秒、心跳续 2/3」参数与什么负载特征相关?

## 自测题（不提交）

1. 为什么 swap 决策移到 connector 侧后，主调度器的抢占仍然只有 recompute?（提示：offload 的换入由谁在什么时机触发——它不需要调度器预先决策）
2. `--kv-offloading-backend lmcache` 与 native(OffloadingConnector)的区别在哪一层?（决策/搬运/存储位置）
3. 两条路径的 metrics 名不同（`vllm:kv_offload_*` vs `vllm:simple_kv_offload_*`）——如果你的 Grafana 面板想兼容两种配置，应该抓哪个更上层的指标?（提示：第 10 讲的 cached_by_source）
