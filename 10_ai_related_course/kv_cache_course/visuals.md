# KV Cache 八讲 — 可复用配图

课程本身是提纲级材料，不重复造图。下面这些图已经在仓库或本地课程仓库里，讲课时直接打开对应文件即可。

| 讲次 | 配图 | 用途 |
| --- | --- | --- |
| 第 1 讲 | `~/Project/study/ai-infra-course/06-kvcache-optimization/visuals/kv-cache-formula.html` | KV 公式逐参数拆解 |
| 第 1 讲 | `~/Project/study/ai-infra-course/06-kvcache-optimization/visuals/kv-cache-tensor.html` | 一个 token 的 K/V 张量形状 |
| 第 1 讲 | `~/Project/study/ai-infra-course/06-kvcache-optimization/visuals/kv-cache-precision.html` | FP16 / FP8 / INT8 / INT4 的字节数 |
| 第 2 讲 | `~/Project/study/nano-vllm/docs/llm-inference-visual/diagrams/L04-block-manager-prefix-cache.png` | block manager 与前缀缓存 |
| 第 2 讲 | `~/Project/study/nano-vllm/docs/llm-inference-visual/diagrams/L06-decode-block-tables.png` | decode 时的 block table |
| 第 3 讲 | `~/Project/study/nano-vllm/docs/llm-inference-visual/diagrams/L03-swap-queues.png` | waiting / running / swapped 三队列与抢占 |
| 第 3 讲 | `~/Project/study/nano-vllm/docs/llm-inference-visual/diagrams/L04-block-manager-prefix-cache.png` | 前缀哈希链与块复用 |
| 第 5 讲 | [`sglang-kv-pool-three-relation.html`](../../09_inference_system/sglang/assets/sglang-kv-pool-three-relation.html) | SGLang KV pool 与树/池关系；讲 HiCache 前建立对象关系 |
| 第 6 讲 | [`glm53flash-01-layers.png`](../../09_inference_system/sglang/assets/glm53flash-01-layers.png) | 45 层文本塔：34 KDA + 11 full |
| 第 6 讲 | [`glm53flash-02-bytes.png`](../../09_inference_system/sglang/assets/glm53flash-02-bytes.png) | 三本账的字节拆分 |
| 第 6 讲 | [`glm53flash-03-device-pools.png`](../../09_inference_system/sglang/assets/glm53flash-03-device-pools.png) | 一个 wrapper、两个子池 |
| 第 6 讲 | [`glm53flash-04-indexer-page.png`](../../09_inference_system/sglang/assets/glm53flash-04-indexer-page.png) | indexer 的页布局 |
| 第 6 讲 | [`glm53flash-05-host-tier.png`](../../09_inference_system/sglang/assets/glm53flash-05-host-tier.png) | KV / KDA 的 host tier |
| 第 7 讲 | [`deepseek-v4-v41-pools.svg`](../../09_inference_system/assets/deepseek-v4-v41-pools.svg) | V4 / V4.1 的池与持久层对照 |
| 第 7 讲 | [`deepseek-v41-fig1b-kv-per-token.png`](../../09_inference_system/assets/deepseek-v41-fig1b-kv-per-token.png) | global KV 逐代下降曲线 |
| 第 7 讲 | [`deepseek-v41-fig3-architecture.png`](../../09_inference_system/assets/deepseek-v41-fig3-architecture.png) | CED 的 encoder / decoder 结构 |
| 第 7 讲 | [`deepseek-v41-fig4-csa2-modes.png`](../../09_inference_system/assets/deepseek-v41-fig4-csa2-modes.png) | CSA2 的 Full / Reindex / Reuse |
| 第 7 讲 | [`deepseek-v41-fig5-hierarchical-indexer.png`](../../09_inference_system/assets/deepseek-v41-fig5-hierarchical-indexer.png) | 层级 indexer 的候选池与 top-k |

第 4 讲和第 8 讲没有现成的单页总图，直接用源码对象图、`code/parse_stats_log.py` 的输出和 `capacity_plan.py` 的算例表做板书。需要新图时，先确认是否已有文章图可复用，避免同一概念出现两套画法。
