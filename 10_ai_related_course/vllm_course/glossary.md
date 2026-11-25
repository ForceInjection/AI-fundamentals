# 术语表

按讲次首次出现的顺序排列。定义以课程基线（nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`）代码为准。

## 引擎与架构（第 1 讲）

| 术语             | 含义                                                                 |
| ---------------- | -------------------------------------------------------------------- |
| `AsyncLLM`       | V1 的 API 层入口，处理 OpenAI 兼容请求、持有输出处理器                |
| `EngineCore`     | V1 核心层独立进程，持有调度器与 KV 状态，与 API 层经 ZMQ 通信；进程内无 tokenizer（tokenizer 在前端 OutputProcessor） |
| `Executor`       | 执行层抽象；`UniProcExecutor`（单进程）与 `MultiprocExecutor`（多进程）等实现，由 `Executor.get_class` 按 `distributed_executor_backend` 选择 |
| `MessageQueue`   | `vllm/distributed/device_communicators/shm_broadcast.py` 的共享内存广播队列（含 zmq 通知），服务于 Executor↔Worker 的 SchedulerOutput 广播与输出回传 |
| `Worker`         | 一个 GPU 上的执行单元，持有 Model Runner                              |
| `Model Runner`   | 模型执行的具体实现；当前形态为 `GPUModelRunner`（`vllm/v1/worker/gpu/model_runner.py`）及其 `gpu/` 子包 |
| Model Runner V2  | 2025-11 落地的 ModelRunner 模块化重构(PR #25266);2026-11 旧 V1 实现整体删除(PR #63373)后成为唯一实现，无 V1/V2 切换开关。设计文档 `docs/design/model_runner_v2.md` |
| V0 / V1          | vLLM 两代引擎；V1 于 2025-01 发布并成为默认，V0 已删除                 |

## 调度（第 2 讲）

| 术语                  | 含义                                                             |
| --------------------- | ---------------------------------------------------------------- |
| chunked prefill       | 长 prompt 切块分步 prefill，与 decode 混合批；源自 Sarathi-Serve  |
| async scheduling      | 调度与上一步执行重叠的调度方式                                   |
| preemption            | 显存不足时回收请求 KV 的策略；V1 只剩 recompute，swap 已移出调度器（见第 9 讲） |
| `--long-prefill-token-threshold` | 单步 prefill token 预算相关配置                        |

## KV cache（第 3 讲）

| 术语                    | 含义                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| PagedAttention          | 分页 KV 管理与配套 kernel；管理思想沿用至今，kernel 已于 2026 退役   |
| `KVCacheManager`        | V1 的 KV 管理器入口（组、协调、块池三层）                          |
| `KVCacheCoordinator`    | 多 KV cache 组（如混合注意力）的协调层                              |
| `BlockPool`             | 空闲块与哈希前缀缓存的管理                                        |
| HMA                     | Hybrid Memory Attention 混合 KV 管理器，服务混合注意力模型          |
| `KVCacheGroup`（旧名 `KVCacheGroupId` 已废弃） | 同组层共享一份物理 KV 池的抽象 |

## 注意力后端与编译（第 4、5 讲）

| 术语                    | 含义                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| attention backend       | 按模型架构 × 硬件选择的注意力实现（FlashAttention / FlashInfer / Triton / FlashMLA 等） |
| `VLLM_ATTENTION_BACKEND`| **已删除**(PR #32812)；强制指定后端现行写法为 `--attention-backend` |
| piecewise CUDA Graph    | 只 capture attention 外分段、attention 走 kernel 的图捕获策略       |
| `CUDAGraphMode`         | 五值：NONE / PIECEWISE / FULL / `FULL_DECODE_ONLY` / `FULL_AND_PIECEWISE`（默认 -O2 即最后者）；`config/compilation.py:163-220` |
| `-O0/-O1/-O2/-O3`       | torch.compile 优化级别四档；默认 O2，当前 O3 == O2（`vllm.py:178-190`）|
| `--enforce-eager`       | 关闭 CUDA Graph 与编译，直接 eager 执行                             |

## 并行（第 6 讲）

| 术语        | 含义                                                                       |
| ----------- | -------------------------------------------------------------------------- |
| TP / PP / DP / EP | 张量 / 流水线 / 数据 / 专家并行                                       |
| Wide-EP     | 大规模专家并行部署形态（DeepSeek 2.2k tok/s/H200 口径）                    |
| DP attention| attention 走数据并行、MoE 走 EP 的混合并行                                |

## 投机解码与量化（第 7、8 讲）

| 术语                | 含义                                                             |
| ------------------- | ---------------------------------------------------------------- |
| acceptance length   | 投机解码每步被接受的平均 token 数（含 bonus token,= 1 + accepted/drafts），决定收益上限 |
| ngram / suffix / EAGLE / EAGLE3 / `draft_model` / MTP | HEAD 在职的投机解码方法；Medusa 已移除（历史方法） |
| LLM Compressor      | 官方量化工作流工具                                               |
| FP8 / W4A16 / AWQ / GPTQ | 量化格式与方案；gptq/awq 系收敛为 `auto_gptq`/auto_awq（别名 `gptq_marlin`/awq_marlin）；Marlin 活在 `kernels/linear` 注册表 |
| online 量化              | 加载时量化 BF16 权重的子系统（`--quantization fp8_per_tensor` 等 shorthand，`quantization/online/`）|

## KV 分载与生态（第 9、10 讲）

| 术语                       | 含义                                                               |
| -------------------------- | ------------------------------------------------------------------ |
| `vllm/v1/kv_offload/`      | 当前原生 offloading 的分层后端(base/factory/cpu/fs/tiering)       |
| `vllm/v1/simple_kv_offload/` | 简化 offloading 路径（manager/metadata/copy_backend 等）          |
| KV Connector               | 跨实例 KV 传输插件接口(mooncake / nixl / LMCache / HMA connector) |
| OffloadingSpec / OffloadingManager | v0.11-v0.2x 时代的 offloading 三件套（已重构，演进素材）       |
| disagg prefill             | prefill 与 decode 分离部署；`docs/features/disagg_prefill.md`      |
| production-stack           | vLLM 官方 K8s 部署栈                                              |
| AIBrix                     | 字节开源的 LLM 推理控制面                                         |
| llm-d                       | Red Hat 主导的基于 vLLM 的分布式推理方案                          |
