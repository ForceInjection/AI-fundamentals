# B300 上的模型部署与 KV Cache 实践

**as-of 2026-09-15** ｜ 本文只讲两件事：**怎么把模型跑起来**、**KV Cache 怎么配**。全部内容以 vLLM/SGLang 的官方 recipe、厂商 model card 与本地源码为依据，不做抽象推导。

## 口径与来源

| 标注           | 含义                                                 |
| -------------- | ---------------------------------------------------- |
| **【SGLang】** | SGLang cookbook（本地检出 `c415f977b8`，2026-09-10） |
| **【vLLM】**   | vLLM 官方文档/博客/release notes                     |
| **【NVIDIA】** | NVIDIA model card 或官方博客                         |
| **【源码】**   | 本人对本地 vLLM/SGLang 检出逐行核对                  |
| **【实测】**   | 有测量数据的第三方或独立结果                         |
| **【查不到】** | 明确留空                                             |

**一条贯穿全篇的限制**：截至写作，**没有任何第三方独立复现过 B300 上的完整推理基准**。下面所有性能数字都来自厂商或其合作方的实测。

---

## 一、模型部署

### 1.1 先过三道门槛

**① SM103 是独立 target，不是 SM100。**

`sm_100a` 的 cubin 不能直接跑在 B300 上，必须走 `sm_100f` family 目标或 `sm_103a`。这是大量 `no kernel image is available` 事故的根因。**用 CUDA 13 镜像是充分条件**——不要去找所谓「SM103 专用 tag」，SGLang 安装页里 grep `SM103`/`B300` 是零命中。

**② INT8 在这块卡上不可用。**

PTX 的 `.kind::i8` 目标列表是 `sm_100a / sm_101a / sm_110a`，**从未扩展到 sm_103a**；CUTLASS 按 arch guard 跳过 INT8 UMMA；vLLM 没有 INT8 GEMM。所以：

> **所有量化方案只能走 FP8 / NVFP4。**

⚠️ **失败时机很坑**：vLLM 是在**模型完整加载之后**才硬报错。POC 阶段先用 ~1B 模型冒烟，不要等 405B 下载完。

**③ vLLM 镜像 tag 有个反转陷阱。**

≥ v0.20.0 起**无后缀 = CUDA 13**，`-cu129` 才是 CUDA 12 退出版；v0.19.x 及更早相反。NVIDIA 自己在 Model-Optimizer PR #2042 里记了这条：`Don't trust the tag name — select a tag reporting CUDA_VERSION >= 13`。

### 1.2 引擎与镜像

| 引擎         | 版本（as-of 2026-09-15） | 镜像                                            | 备注                                                                  |
| ------------ | ------------------------ | ----------------------------------------------- | --------------------------------------------------------------------- |
| **vLLM**     | v0.29.0                  | `vllm/vllm-openai:v0.29.0`                      | 三家唯一正式 release；官方推荐 B300 用 CUDA 13                        |
| **SGLang**   | v0.5.19                  | `lmsysorg/sglang:v0.5.19-cu130-runtime`         | CUDA 12 lane 已退役                                                   |
| TensorRT-LLM | **1.3.0rc26**            | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc26` | **迭代 26 个 RC 仍无 GA**；legacy TensorRT backend 已移除。不建议主线 |

**镜像已知问题（未修复）**：

- vLLM 官方镜像的 `TORCH_CUDA_ARCH_LIST` **不含 `10.3`**（v0.29.0 实测仍只有 `10.0`，PR #44344 未合并）→ 建议自建镜像显式加入
- vLLM：DeepEP MoE all-to-all 在 SM103/GB300 **不可用**（issue #41687 仍 open），实测「通信占 decode kernel 时间 ~93%」
- SGLang：tcgen05 kernel 在 sm_103 失败 Xid 13（#34340 open）；MegaMoE 路径 `CUDA_ERROR_ILLEGAL_ADDRESS`（#37559 open）

### 1.3 按模型给配方

以下命令均来自官方 cookbook / model card，可直接作为起点。

#### Kimi-K3 on B300 1×8

节点数由硬件决定，不是独立选择：**B300 1×8 = TP8 / DCP8**（Balanced 档）。全系列节点配置：

| 硬件     | 节点    | 说明                           |
| -------- | ------- | ------------------------------ |
| B200     | 2×8     |                                |
| GB200    | 4×4     |                                |
| **B300** | **1×8** |                                |
| GB300    | 2×4     |                                |
| H200     | 2×8     | Unified High-Throughput 用 4×8 |

**大规模预设（32 GPU @ B300，每节点跑相同命令、只改 `--node-rank`）**【SGLang】：

```bash
SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=20480 \
sglang serve \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 32 --ep-size 32 \
  --enable-dp-attention --dp-size 4 --enable-dp-lm-head \
  --nnodes 4 --node-rank <rank> --dist-init-addr <node0-ip>:20000 \
  --moe-a2a-backend megamoe --moe-runner-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mem-fraction-static 0.92 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
  --host 0.0.0.0 --port 30000
```

扩展方式：**保持每 replica 的形状不变，只动 replica 数**。

| GPUs | B200/B300 节点 | `--tp-size` / `--ep-size` | `--dp-size` |
| ---- | -------------- | ------------------------- | ----------- |
| 16   | 2×8            | 16                        | 2           |
| 32   | 4×8            | 32                        | 4           |
| 64   | 8×8            | 64                        | 8           |

**两个关键 flag 的理由**：

- `--kv-cache-dtype fp8_e4m3` **是承重的**——cookbook 原文：`bf16 KV does not fit 128 requests per replica`
- `--mamba-ssm-dtype bfloat16`：KDA state 的 dtype 决定每卡账单，只有 attention-TP 宽度、SSM dtype、cache 策略三个旋钮能动它

#### DeepSeek-V4 系列

**单节点 TP=4**（B200/B300/GB200/GB300/H200 通用；RTX PRO 6000 用 TP=2，H100 用 TP=8）【SGLang】。

**Agentic 长上下文 + HiCache DRAM offload**（B200 FP4 + DSpark，TP8，并发 8–16）【SGLang】：

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Pro-0813 \
  --trust-remote-code --tp 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --enable-deepseek-v4-fp4-indexer \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.90 \
  --swa-full-tokens-ratio 0.1 \
  --chunked-prefill-size 8192 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 6 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.75 \
  --hicache-write-policy write_through \
  --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct
```

高并发档（DEP8 + DP attention，并发 64–160）换 `--dp 8 --enable-dp-attention --ep-size 8 --mem-fraction-static 0.88 --swa-full-tokens-ratio 0.02 --hicache-ratio 8`，并把 `--chunked-prefill-size` 提到 49152。

⚠️ **`--chunked-prefill-size` 是全局预算，会被 `--dp` 均分**——上面那个 49152 除以 dp8 才是每 rank 的 6144。

⚠️ **DSv4 的 HiCache host 层用 `--hicache-ratio`（host/device token 比）定容，不是 `--hicache-size`**。

### 1.4 并行策略：从实践里读出的三条

**① MLA 模型上，attention TP > 1 会复制 KV，不是切分。**

这一点在【源码】里可以直接看到：

```python
# vllm/config/model.py:1504-1509
if self.use_mla:
    # When using MLA during decode it becomes MQA
    return 1
...
return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

注意 GQA 分支：**KV heads 少于 TP 时也是「复制」**——Qwen3-235B（4 个 KV head）上开 TP8，8 卡各存一份完整副本。

**SGLang cookbook 把这条写成了 Deep PP 的依据**：Deep PP 用 `--tp-size 1 --pp-size 8`（B300/GB300），因为

> `--tp-size 1` is also what buys context: above TP1 the MLA KV is replicated across the TP ranks, so **TP2 × PP8 holds roughly half the tokens of TP1 × PP16 for the same memory**.

实测（GB200，ISL 8192 / 并发 32）【SGLang】：

| 形状           | prefill tok/s/GPU |
| -------------- | ----------------- |
| **PP16 × TP1** | **4550**          |
| PP8 × TP2      | 3596              |
| TEP16          | 2407              |
| TP16           | 1652              |

**但并发低于 ~8 时反过来**：pipeline 填不满，TEP16 领先（1947 vs 1227），此时用 `--tp-size 16 --ep-size 16`。

**② 大 MoE 的默认形态是 Attention DP + MoE EP**，不是单一 TP。

**③ GB300 跨 pod MNNVL 传输要加三个环境变量。**

cookbook 原文：某些 GB300 集群上跨 pod NVLink 传 KV 会失败于 `nvlink_transport.cpp:497 Requested address ... not found!`，解法是在 prefill 和 decode 两侧的 `sglang serve` 前都加上：

```bash
MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1
```

### 1.5 PD 分离：先看厂商自己的数据

**vLLM 官方文档开篇直写：`Disaggregated prefill DOES NOT improve throughput`。**

Dynamo 自己的 GB300 实测（Kimi-K3，agentic 负载）【NVIDIA】：

| 配置                          | 系统吞吐 tok/s/GPU | 每用户 tok/s | TTFT P50   |
| ----------------------------- | ------------------ | ------------ | ---------- |
| SGLang **Aggregated**         | **84.4**           | 51.3         | **573 ms** |
| SGLang **Disaggregated 1P1D** | 61.0               | **75.9**     | 4,809 ms   |
| vLLM **Aggregated**           | **62.2**           | 57.8         | 879 ms     |
| vLLM **Disaggregated 1P1D**   | 50.3               | **90.0**     | 6,357 ms   |

**聚合部署的系统吞吐更高、TTFT 低 5–6 倍**；PD 分离只在 per-user 吞吐上赢。

**它会减少 decode 侧的 KV 容量**（Dynamo 明文）：`Moving prefill to dedicated workers ... reserves GPUs for a pool whose KV cache is not used during decode`。而且两侧都付权重代价。

**Dynamo 明确拒绝给固定 P:D 比**：`Treat replica counts as a response to observed bottlenecks rather than as fixed ratios.`

**实际配 PD 时要处理的细节**【SGLang】：

- 混合模型（如 K3）的传输要搬**两样**：paged MLA KV **和** KDA recurrent state
- 端口：prefill `30000`、decode `30100`，派生的 ZMQ/dist 范围不能在同一台机器上撞车
- `--prefill` 后面那个位置参数 `8998` 必须和 `--disaggregation-bootstrap-port` 一致，否则只有 decode worker 会注册
- `--disaggregation-decode-extra-slots` **要显式钉住**：不钉的话，32 请求以下默认翻倍、以上默认**归零**

---

## 二、KV Cache

### 2.1 先把三个数分清

| 数字       | 含义                           | 来源                             |
| ---------- | ------------------------------ | -------------------------------- |
| **288 GB** | B300 **卡规格**                | 【NVIDIA】Blackwell Ultra 博客   |
| **279 GB** | GB300 NVL72 形态下**单卡可用** | 【NVIDIA】MIG 产品页 `1x 279 GB` |
| **270 GB** | HGX B300 形态下**单卡可用**    | 【NVIDIA】MIG 产品页 `1x 270 GB` |

MIG 页脚注标明这些是 `Preliminary specifications`。**读规格表用 288，做容量规划用 279 或 270。**

**`gpu_memory_utilization` 的分母是总显存，不是剩余显存**【源码】：

```python
# vllm/v1/worker/utils.py:446-456
requested_memory = math.ceil(init_snapshot.total_memory * cache_config.gpu_memory_utilization)
```

默认 **0.92**（`vllm/config/cache.py:68`）——网上大量二手资料写 0.9，会算错。

⚠️ **TensorRT-LLM 用的是相反约定**：`free_gpu_memory_fraction` 分母是初始化时的**空闲**显存。跨引擎混用必错。

### 2.2 精度：只到 FP8

| 档位                  | 结论                       |
| --------------------- | -------------------------- |
| **FP8（`fp8_e4m3`）** | ✅ 唯一可放心用。KV 池翻倍 |
| **NVFP4 KV**          | ❌ **不进入任何生产配置**  |

**FP8 在实践里的地位**——它不是可选项，是承重项。SGLang cookbook 写得很直接：`--kv-cache-dtype fp8_e4m3` **is load-bearing**，因为 bf16 KV 装不下每 replica 128 个请求。

**NVFP4 KV 的禁用理由**（vLLM issue #55673，2026-09-07 开，**仍 open**）【实测】：

4×B200、Qwen3.5-397B、FlashInfer TRT-LLM attention，1,319 题 GSM8K：

| KV cache / max seqs | Flexible exact match | Invalid responses |
| ------------------- | -------------------- | ----------------- |
| **FP8, 512**        | **96.664%**          | 0.682%            |
| NVFP4, 64           | 6.520%               | 93.177%           |
| NVFP4, 512          | **4.549%**           | **94.920%**       |

报告人排除了「高并发伪影」（两个并发上限都失败，FP8 对照组用同一后端）。**根因至今未定位**：PR #55670 修了一个真实的 scale 转换缺陷，打补丁后仍是 4.776%，说明该缺陷 `is not sufficient to explain this Qwen failure`。

**适用范围**：特定几何（`head_dim=256 / 8 query heads / 1 KV head`）上的问题，不等于所有模型都会挂。但**同族、同后端、对照组正常、根因未定位**——足以构成生产禁用。

SGLang 侧同一能力的官方 KV4 表也印证：GPT-OSS-120B 在 aime25 上从 KV8 的 0.7667 掉到 **KV4 的 0.3533**。

**两个记账陷阱**：

- **MLA 的 per-token KV 有两种口径，差 13.9%**：

| 口径                                            | 每 token 每层 | 61 层全模型  |
| ----------------------------------------------- | ------------- | ------------ |
| BF16 未量化                                     | 1152 B        | 70,272 B     |
| FP8 朴素口径（`kv_lora_rank 512 + qk_rope 64`） | 576 B         | 35,136 B     |
| **FP8 vLLM 实际打包 `fp8_ds_mla`**              | **656 B**     | **40,016 B** |

NVIDIA + SGLang 的 GB300 长文用的是朴素 576；vLLM 实际是 512 B FP8 NoPE + 16 B scales + **128 B BF16 RoPE**。**按 576 算容量会少算 14%。**

- **`--cpu-offload-gb` 是权重卸载，不是 KV 卸载**；**`--swap-space` 已从 vLLM 移除**。

### 2.3 分层：HiCache 的实际配置

SGLang 的 HiCache 分三档，**每档都有一套 canonical 参数**，不要自己发挥【SGLang】：

| 档位                          | 配置                                      | 说明                                                                                   |
| ----------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------- |
| **L2（GPU + CPU）**           | Storage 留 `auto`                         | 冷 KV 页只溢出到 CPU pinned memory                                                     |
| **L3（GPU + CPU + Storage）** | 选 `file` / `mooncake` / `hf3fs` / `nixl` | Playground 会生成 `page_first_direct` + `direct` IO backend + `wait_complete` 预取策略 |

**写策略**默认 `write_through`（上游默认）；存储层慢时切 `write_back` / `write_through_selective` 用持久性换写速。

**一条容易踩的坑**（K3 的 DCP recipe）：**host 层还没完全 DCP-aware**——

- L3 **总是**丢掉 DCP flag
- L1+L2 **开着 Spec Decode 时**也丢；关掉 Spec Decode 才保留
- 丢掉 DCP 之后，MLA KV 退回 TP 复制，**每请求的 KV 容量相应缩水**

**真正的原生机制**（vLLM 侧）是 `OffloadingConnector` + `TieringOffloadingSpec`。要点：

- 只有 CPU 一级 tier 能直连 GPU；二级 tier（`fs` / `obj` / `p2p`）**必须经 CPU 中转**
- 单层（纯 CPU）配置时，**`cpu_bytes_to_use` 要大于 GPU KV 总量**——卸载是即时的，CPU tier 比 GPU 小就只是镜像，不提升命中率

**卸载的收益与代价（实测数据）**：

| 层级转移        | 实测效果                                                                                    | 口径     |
| --------------- | ------------------------------------------------------------------------------------------- | -------- |
| CPU DRAM        | TTFT 降 2–22×，吞吐最高 9×                                                                  | 【vLLM】 |
| 命中 vs 重算    | 1k tokens 时 2.2×，80k 时 32.8×                                                             | 【实测】 |
| CPU DRAM on/off | 10k/40k/80k 改善 1.30/1.18/1.08×；**1k 时四种配置无差别**                                   | 【实测】 |
| SSD 盈亏平衡    | 8k prompt 需 **77.8%** 前缀复用才回本；80k 时降到 7.8%                                      | 【实测】 |
| SSD 超大规模    | 100k prompt 超 HBM 时 −79% TTFT、+264% 吞吐；但 cache 达 12.6–13.7M tokens 时最好也只有 −3% | 【实测】 |

**核心独立结论**：`a cache hit is not sufficient for caching to be beneficial` / `External KV caching should therefore be treated as a setup specific admission decision`。【实测】

**带宽现实**：GPU↔CPU 在 PCIe 5.0 ×16 上实测 54–56 GB/s（理论的 40–45%）；NVMe 单盘 6–13.5 GB/s。**不要把链路速率当吞吐。**

⚠️ **一条反直觉的实测**：GPUDirect Storage（KvikIO/cuFile）在这个负载上 `was slower than all our other implementations`。

### 2.4 复用：两项已经被实测的收益

**Prefix Caching / RadixAttention**

Agent 场景的真实命中率 **95.7%**（~4,300 个 Claude Code + Codex session，~350,000 LLM steps）【实测】：

- fresh tokens 只占 append tokens 的 **19.0%**——**约 81% 的 prefill 原则上可命中**
- miss 是**空闲驱动**的：间隔超 5 分钟开始出现低命中，1 小时后几乎全 miss
- **cache 命中占 agent 总成本 59.5%**，append 占 29.2%，output 只占 11.2%

**一个直接可用的调参**：超时从 1 分钟提到 1 小时，命中率 85.4% → 98.6%，但存储比从 R=0.74 涨到 5.07（**约 7 倍**）。**大部分收益是便宜的**——5 分钟时已达 ~94% 命中，R≈1.9。

**Radix cache 不是永远开着好**：K3 cookbook 明确——**对无前缀的流量（离线批处理、评测）关掉它**，因为一个请求占 4–5 个 state slot，关掉只占 1 个。

**cache-aware 路由**

| 调度器                 | 输出 tok/s | TTFT p90    |
| ---------------------- | ---------- | ----------- |
| **precise-scheduling** | **8730**   | **0.542 s** |
| approximate            | 6944       | 31.083 s    |
| load-based             | 4429       | 94.865 s    |
| random                 | 4429       | 92.551 s    |

（8 vLLM pod / 16×H100，Qwen3-32B，150 个 B2B 客户 × 6,000-token 共享上下文）【llm-d】

**但有反方观点**（Anyscale，作者含 NVIDIA 人员）：`balancing KV cache reuse with token load leads to better overall serving performance than maximizing KV cache reuse alone`。两个具名失效模式：**request herding** 与 **session-level imbalance**。

**私有化多租户必须配 `cache_salt`**：它注入首个 block 的 hash，保证只有同 salt 的请求能复用 KV block。**不配会跨租户泄露。**

**harness 侧的硬规则**（可直接抄进开发规范）：

- 保持 prompt 前缀稳定——**哪怕一个 token 的差异都会让从该点起的缓存全部失效**
- 系统提示开头放时间戳会直接杀掉命中率
- 上下文保持 append-only
- **序列化必须确定性**——很多库不保证 JSON key 顺序稳定，会静默破坏缓存

### 2.5 容量与并发：实际调过的数字

**同一份硬件上的真实并发**（GB300 NVL72，NVIDIA + SGLang 联署）【实测】：

| 指标                                    | GB300                              | GB200       |
| --------------------------------------- | ---------------------------------- | ----------- |
| `mem_fraction_static = 0.75` 下静态预算 | ≈216 GB                            | ≈144 GB     |
| 权重（DeepSeek-R1 NVFP4, EP16/TP16）    | ≈40 GB                             | ≈40 GB      |
| **KV 池**                               | **≈176 GB**                        | ≈104 GB     |
| 单请求 KV（136K cached tokens）         | ≈4.45 GiB                          | —           |
| 理论上限                                | **≈40 req/GPU**                    | ≈24 req/GPU |
| 按 ~85% 运维目标                        | **36 req/GPU**（DEP16 → 576 并发） | 20 req/GPU  |

⚠️ **该页自身有内部不一致**：TL;DR 按 DEP8 写「288 concurrent」，正文按 DEP16 写 576；加速比出现 1.38X–1.58X / 1.4X–1.6X / 1.4X–1.5X 三种表述。

**Kimi-K3 on 8×B300 的现成配置**【NVIDIA model card】：

```
--quantization modelopt_mixed --tensor-parallel-size 8
--moe-backend flashinfer_trtllm --kv-cache-dtype fp8
--max-model-len 196608 --max-num-seqs 32
--attention-backend FLASHINFER_MLA
```

配套限制：`flashinfer_trtllm` **是强制的**——`auto-resolution never triggers the TRT-LLM deferred-finalize path, and flashinfer_cutlass lacks a SiTU kernel for routed experts`；且 `a pip-installed SGLang cannot load this checkpoint`（SGLang 路径需专用镜像）。

**DCP 换并发**【SGLang】：`--dcp-size 8` 去重 attention-TP 组内的 MLA KV——**同等引擎吞吐下并发上限 +72%，代价是 ITL 约 1.8×**。适用于上下文 ≥ ~16K，或每 replica 并发超过 128。

### 2.6 常见误算

**那个 `GPU KV cache size` 日志行不是容量承诺**【源码】：

```python
# vllm/v1/core/kv_cache_utils.py:1871
return int(max_concurrency * max_model_len), max_concurrency
```

它字面上就是 `max_concurrency × max_model_len`，**派生的最坏情况乘积**。maintainer 原话：`That log message simply shows a theoretical upper bound ... KV cache blocks are allocated lazily and incrementally, not all at once.`

**按危害排序的误算清单**：

1. **activation / CUDA graph 显存没算进去**（vLLM 现已默认开启估算并打印等效换算）
2. **profiling 看不到的临时 buffer**——KDA 的 chunked-scan buffer 随 `max_num_batched_tokens` 线性增长、在 forward 内部瞬时分配，**启动 profiling 覆盖不到**；超出某个 chunk size 后引擎会在**服务中途**死掉
3. **把 `max_model_len × 并发` 当 KV 需求**
4. **preemption 在 OOM 之前先毁掉 p99**——读 `vllm:num_preemptions`，不要从日志推断
5. **hybrid attention 模型给滑窗层分配了全上下文 KV**（修复后 SWA 层改用 `SlidingWindowSpec`）
6. **`max_num_seqs` 一职两用，且等待队列无界**（`--max-num-queued-tokens` 默认关闭）
7. **block size 在 hybrid 模型上被逼到病态值**——GLM-5.3-Flash 上曾出现 block size 7808，一个 12-token 的 prompt 占掉 32.9% 的池
8. **按 576 B/token 算 MLA 容量**（实际打包是 656 B，少算 14%）

**一条最好的单变量对照**（RTX 4090 / Qwen3-8B bf16）【提交者自测】：`max_num_batched_tokens` 从 2048 提到 8192，KV 池缩 10%，**p99 TTFT 涨 71%**（23.9s → 40.8s），goodput 从 54.7% 掉到 46.2%，而总吞吐不变。

**`--mem-fraction-static` 的实践取值**：SGLang 在 B300 上给 VLM 的保守起点是 **0.82**（`raise it toward 0.85 if startup reports insufficient memory`）；大 MoE 吞吐档用 0.88–0.92。DSv4 的 DSpark 路径明确要 `Keep --mem-fraction-static 0.90 to leave enough headroom for the batch-256 verify graph`。

---

## 三、上线前必测项

| #   | 项                                                     | 理由                                                                                                                                                                            |
| --- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ①   | **KV 量化的 ≥100K needle retrieval**                   | vLLM #56700 原话：`Wrong interpretation passes short-prompt smoke tests and only fails at ≥100K needle retrieval — silent numerical corruption.` **短 prompt 冒烟测试等于没测** |
| ②   | 官方镜像 `TORCH_CUDA_ARCH_LIST` 无 `10.3` 的长期稳定性 | 建议自建镜像显式加入                                                                                                                                                            |
| ③   | TRT-LLM arm64/sbsa 容器可用性（若用 GB300）            | NGC API 403，无法枚举                                                                                                                                                           |
| ④   | 长上下文任务的**忠实度**回归                           | 独立论文：INT4 KV 下 `over 90% of faithfulness changes are negative, i.e., accuracy metrics are blind to this regression`                                                       |
| ⑤   | 精度验收绑定 kernel backend 版本                       | 同一份 NVFP4 权重换 backend 差约 1 分 GSM8K                                                                                                                                     |

**另外两条来自 cookbook 的提醒**：

- **B300 1×8 上只有 `Unified` 的 Low-Latency 与 Balanced 两格是 Verified**，其余全部标 `Final Verification In Progress`——「treat those as starting points to verify」
- **大规模预设没有一个在最终权重上跑过完整 serving round**：`the constants derive from measured single- and dual-node rounds plus a 64-GPU sweep. Validate throughput and accuracy on your workload before committing a fleet.`

---

## 四、查不到（不编）

1. 任何独立的、非厂商的 B300/GB300 KV 卸载基准（最好的独立工作只跑在 H100/RTX 上）
2. 任何独立的 NVFP4-KV 长上下文基准
3. 框架文档级的 FP8 vs BF16 MMLU/LongBench 对照表
4. NVIDIA 官方的 KV Cache 容量计算公式（其 TCO 博客无任何 KV 尺寸公式）
5. NVIDIA 的固定 P:D 比建议（Dynamo 明确拒绝给）
6. PD 分离对 KV 总容量的净影响量化
7. B300 显存带宽的口径：NVIDIA 博客写 8 TB/s，而仓库文档区分 HGX 7.7 / NVL72 8.0 TB/s，无法判定后者是否是有意的形态区分

---

## 附：三条收口判断

1. **KV 量化只到 FP8。** NVFP4 KV 有未定位的灾难性精度 bug，不碰。
2. **容量规划用实测 + 带 SLO 的 goodput**，别信 `GPU KV cache size` 那行日志，别用 288 GB 当分母，别按 576 B/token 算 MLA。
3. **第一杠杆是模型选型。** MLA 与 Qwen3-235B 的 GQA 差近 5 倍 per-token KV（40 KB vs 188 KB），量级大于任何量化手段。

---

## 相关阅读

- [KV Cache 技术体系](../kv_cache/README.md)——本文只讲 B300 上的配置实践，压缩、淘汰、卸载的机制原理见该目录
- [显存估算](../memory_calc/README.md)——容量测算的方法与脚本
- [vLLM 助力 DeepSeek 吞吐量飙升 5 倍](../vllm/hardware_optimization/deepseek_blackwell_wide_ep.md)——WideEP、NVFP4/FP8 与 Weight Offloading v2 的原理拆解，本文 §1.4 的并行策略是它在部署侧的另一面
- [把 KV Cache 压缩推到极限：DeepSeek-V4.1-Flash 技术报告精读](../deepseek-v41-flash-kv-compression.md)——模型架构侧的 KV 压缩（CSA2、FP4 main KV），与本文的引擎侧实践互补
- [NVIDIA GB300 NVL72 架构解析](../../01_hardware_architecture/superchips/nvidia_gb300.md)——本文用到的显存口径在那一篇有完整的拓扑与带宽背景
- [核心推理优化技术深度解析](../reference_design/03-核心推理优化技术深度解析.md)——KV Cache、Continuous Batching、量化等技术的原理层梳理
