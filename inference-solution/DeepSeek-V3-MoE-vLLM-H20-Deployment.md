# DeepSeek‑V3 MoE 基于 vLLM + NVIDIA H20 的企业级部署方案（简版）

> 目标：在 **32 张 H20**（4 台 × 8 卡）的集群上，使用 vLLM 部署 **DeepSeek‑V3（671B MoE, 37B 激活）**，在**不量化、不蒸馏**前提下，达成如下 `SLO`：
>
> * **并发**：`200` 活跃会话（`continuous batching`）
> * **上下文**：`32k tokens`（`max model len`）
> * **吞吐**：≥ `50,000 tokens/s`（系统级），优先优化 **稳定 30–40k** 并冲刺 `50k`
> * **TTFT**：< `1.2s`（常见请求，≤`4k prefill`）
>
> **硬件规格说明**：NVIDIA H20显存为96GB，按标准换算为89.4 GiB（96 × 1000³ ÷ 1024³ = 89.4 GiB）

---

## 1. 概览与关键决策（Executive Summary）

* **并行策略（建议）**：`TP=8 / EP=4 / DP=1`（共 32 卡，单副本），必要时可扩容为 `DP=2`（64 卡）以更稳妥冲击 50k tokens/s。
* **可行性**：`FP16` 权重总量约 **1.342e12 bytes ≈ 1249.8 GiB**。按 `TP×EP=32` 分片后，**单卡常驻权重≈39.06 GiB**，`H20` 可用 HBM `89.4 GiB`，**余量≈50.34 GiB**（89.4 - 39.06）用于 KV 与运行时，显存上可行。
* **性能瓶颈**：`prefill` 阶段计算主导；`decode` 阶段 **MoE All‑to‑All（A2A）** 与 **All‑Reduce** 的通信比重上升 → 需要 **EP 友好布局**、**A2A 拓扑亲和**（同岛/同 NVSwitch 优先）、以及 **通信‑计算重叠**。
* **vLLM 选型**：启用 **Expert Parallel**、**Paged KV**、**Continuous Batching**、**Chunked Prefill**；通过 **placement group** 保证 `intra‑node` 走 `NVLink`、`inter‑node` 走 `RoCEv2`/`NVLink‑Switch`。
* **SLO 达成路径**：先以 **保守 η=0.55–0.65**（实现效率）达成 **30–40k tokens/s**，再通过 **EP 路由优化、A2A 拓扑对齐、prefill chunking、KV 热页策略** 冲刺 **50k tokens/s**。

> 规格口径说明：H20 的 **FP16 峰值≈148 TFLOPS/卡**；公开“296 TFLOPS”多为 `FP8`/`INT8` 口径。NVLink **900 GB/s 为双向**，建模采用 **单向≈450 GB/s** 并乘利用率 ρ（`0.7`–`0.85`）。

---

## 2. 并行策略选型与理由

### 2.1 候选方案对比

* **方案 A（推荐）**：`TP=8 / EP=4 / DP=1`

  * **优点**：`32` 卡资源充分利用；权重分片后单卡约 `39.06 GiB`；机内 `NVLink` 充分；`EP=4` 使专家按岛/机内布置，`A2A` 通信更聚合。
  * **风险**：单副本，扩缩容以水平扩展为主；冲击 50k 需较高实现效率与通信重叠。
* **方案 A+（扩展）**：`TP=8 / EP=4 / DP=2`（需 64 卡）

  * **优点**：更稳妥达到 50k；同时提升高可用与隔离能力。
  * **代价**：资源翻倍。

### 2.2 单卡显存与权重分片

* **H20显存规格**：`96GB = 96 × 1000³ ÷ 1024³ = 89.4 GiB`
* **总权重（FP16）**：`671e9 × 2 bytes = 1.342e12 bytes ≈ 1249.8 GiB`
* **单卡常驻权重**：`1.342e12 / (TP×EP=32) ≈ 41.94e9 bytes ≈ 39.06 GiB`
* **单卡可用 HBM**：`89.4 GiB` → **剩余≈50.34 GiB**（`89.4 - 39.06 = 50.34`）

### 2.3 MoE 通信与拓扑

* **Dense 层**：`All‑Reduce`（AR）。
* **MoE 层**：`token`→`expert` 的 **All‑to‑All（A2A）** 通信为主；随 `Top‑k=8`、`capacity factor`（`CF≈1.1–1.3`）、专家跨卡分布（`EP`）强相关。
* **布局**：优先将 `4` 个 `EP` 分片 **对齐 4 台服务器**（每台 `8` 卡，内部再做 `TP=8`），确保 **A2A 主要走机间骨干，但尽量保持成对/成组映射**，减少跨机交换次数与热点。

---

## 3. 容量规划：上下文、KV 与并发

### 3.1 目标与假设

* **并发：200**；最大上下文：`32k`；生成长度：平均 `512`；`MLA` 降低 `KV` 维度。
* 假设 `MLA` 的有效 `KV` 维度 `d_kv_eff` 取情景化数值：**1024 / 2048 / 4096**（用于保守‑乐观三档）。

### 3.2 KV‑Cache 预算（全局量纲）

通用公式：`KV ≈ concurrency × context × d_kv_eff × bytes × replication_factor`。

* 以 `FP16 bytes=2`；`replication_factor` 视并行与实现（通常 < 1 的分布式分片 + 运行时开销）。
* **示例（全局）**：

  * `d_kv_eff=1024`：KV ≈ `200×32,000×1,024×2` ≈ **12.2 GiB**。
  * `d_kv_eff=2048`：KV ≈ `200×32,000×2,048×2` ≈ **24.4 GiB**。
  * `d_kv_eff=4096`：KV ≈ `200×32,000×4,096×2` ≈ **48.8 GiB**。

> 注意：`vLLM` 的 **Paged KV** 会把 `KV` 分块、热页常驻、冷页分页；**单卡占用**低于简单均分（还需考虑 `TP`/`EP` 分片与 `runtime` 缓冲）。在`H20`单卡剩余`50.34` GiB的约束下，以上三档`KV`配置均可行，但需谨慎控制 **运行时临时张量与碎片**。

---

## 4. 吞吐与延迟模型（prefill / decode 分相位）

### 4.1 FLOPs 近似

* `DeepSeek‑V3` 激活参数≈**37B** → 每 `token` 前向 `FLOPs` 近似 `≈ 2×37B = 74e9`。
* 集群峰值：`148e12 FLOPS/卡 × 32 卡 = 4.736 PFLOPS`；有效 `FLOPS` = 峰值 × 实现效率 `η`（建议按 `0.55`/`0.65`/`0.78` 三档评估）。

### 4.2 系统吞吐上限（一阶估算）

* `tokens/s ≈ (η × 4.736e15) / (74e9)`  →

  * `η=0.55` → **35.2k**；`η=0.65` → **41.6k**；`η=0.78` → **50.0k**。
* 由此可见：**50k 目标需要 η≈0.78 的综合效率**（算子、编译、通信重叠、路由/调度）。

### 4.3 通信时间（AR / A2A）与带宽口径

* `AR`（环式）单向带宽：`B_uni≈450 GB/s`，利用率 `ρ=0.7–0.85`。
* 张量并行 P=TP，单卡消息量：`N = (B × S × H / TP) × bytes`；
* `t_AR ≈ 2*(P-1)/P × N / (B_uni × ρ)`（每层 1–2 次）。
* A2A（MoE）：`N_A2A ≈ B × S × d_act × bytes × CF`，强依赖 `EP` 布局；可通过 **topology‑aware placement**、**容量因子**、**分桶/排序路由** 降低实际代价。

### 4.4 TTFT 控制

* **Chunked Prefill**：把 `prefill` 切片（如 `1–2k token chunk`）与 `decode` 并行化，提高调度粒度；
* **静态热身**：模型加载后预热若干 `batch`；
* **连续批处理**：启用 `request merging`，缩短排队；
* **QoS class**：为短上下文请求设高优先级队列，保障 `TTFT < 1.2s`。

---

## 5. vLLM 部署设计

### 5.1 进程与并行度

* 每副本（`DP`=1）使用 `32` GPU：`TP=8` × `EP=4`。
* 进程拓扑：每机 `8` 卡为一个 **TP island**；`4` 台机各承载一个 **EP shard**，机内 `AR` 优先走 `NVLink`；跨机 `A2A` 走 `RoCEv2`。

### 5.2 关键启动参数（示例）

```bash
vllm serve deepseek-v3 \
  --tensor-parallel-size 8 \
  --expert-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp16 \
  --enable-moe true \
  --moe-topk 8 \
  --moe-capacity-factor 1.25 \
  --max-num-seqs 2048 \
  --gpu-memory-utilization 0.92 \
  --enforce-eager \
  --disable-log-stats false \
  --api-key <redacted>
```

> 说明：`max-num-seqs` 与 `gpu-memory-utilization` 需结合压测逐步拉升；`--enforce-eager` 可与编译/内核优化二选一；如使用 `CUDA Graph`/`TensorRT‑LLM` 编译路径，则替换为对应开关。

### 5.3 资源与亲和

* **NCCL**：`NCCL_TOPO_FILE`/`NCCL_NET_GDR_LEVEL=PHB`/`NCCL_SOCKET_NTHREADS` 等按机型微调；
* **绑定**：`NUMA` 亲和、固定核心 `pinning`；
* **文件系统**：权重本地 `NVMe` 缓存 + 冷备对象存储；
* **容器**：`nvidia-container-toolkit` + `ulimit`/`IPC`/`shm` 调优。

### 5.4 路由与调度

* **Token Bucket / SLA‑aware scheduler**：短请求优先；
* **Chunked Prefill + Decode overlap**：减小 `TTFT`；
* **EP 路由排序**：按专家负载做分桶与 `token` 排序，降低 `A2A` 扰动；
* **连续批处理**：启用队列合并（`vLLM` 默认支持），并针对 `decode` 步长设置合理 `step`。

---

## 6. 网络与拓扑

* **机内**：`NVLink`/`NVSwitch`（单向≈`450 GB/s`）承载 `AR`；
* **机间**：`100/200 Gbps RoCEv2`；选择 **同 POD 内 4 台互联**，`EP shard` 与 `TP island` 对齐；
* **队列策略**：`ECMP` 权重、禁用大流重排序；
* **测量**：用 `NCCL tests`、`iperf`、`nvidia-smi nvlink` 基线测量，记录单向/双向带宽与延迟。

---

## 7. 观测与容量自动化

* **时延**：`TTFT`、`TBT`、`P50`/`P95`/`P99`；
* **利用率**：`GPU SM`/`DRAM`、`NVLink`/网卡带宽、`A2A`/`AR` 时间分解；
* **内存**：`KV` 热/冷页命中率、`OOM` 报警；
* **扩缩容**：`HPA`（并发/队列时延驱动）；按 `SLO` 触发 `DP` 扩容。

---

## 8. 验证与压测方法

1. **单机 8 卡基线**：测 `TP=8`（无 `EP`）`dense` 层 `AR` 上界。
2. **加入 EP=4**：观测 `A2A` 占比与瓶颈位置；
3. **chunked prefill**：扫描 `chunk=512/1024/2048` 的 `TTFT` 与吞吐折中；
4. **并发爬坡**：从 `100` → `150` → `200` 并发，记录 `tokens/s` 与 `P95` 时延；
5. **上下文阶梯**：`8k`/`16k`/`32k`；
6. **故障注入**：单机失联、单卡故障、重试与排空时间。

---

## 9. 风险与缓解

* **50k tokens/s 依赖高效率（η≈0.78）**：若未达成，先以 `35–42k` 投产，随后通过 `EP` 路由、`A2A` 亲和、内核编译（或 `TensorRT‑LLM`）拉升。
* **A2A 热点**：使用拓扑感知放置与请求分桶；必要时提高网卡速率或引入 `NVLink‑Switch` 级联。
* **显存碎片**：`Paged KV` 页大小与回收策略联调，保持 `gpu-memory-utilization ≤ 0.92`；
* **TTFT 尾延迟**：短请求优先队列 + `prefill chunking`；
* **高可用**：若业务需强 `HA`，采用 `DP=2`（64 卡）或多集群 `active‑active`。

---

## 10. 附：关键公式与示例数值

* **每 token FLOPs**（一阶）：`≈ 74e9`；
* **系统吞吐**：`tokens/s ≈ (η × 4.736e15) / 74e9`；
* **AR 时间**：`t_AR ≈ 2*(TP-1)/TP × (B×S×H/TP × bytes) / (B_uni × ρ)`；`B_uni≈450 GB/s`；
* **A2A 量纲**：`N_A2A ≈ B × S × d_act × bytes × CF`；
* **KV 预算**：`KV ≈ concurrency × context × d_kv_eff × bytes × replication`。

---

## 11. 参考实现与资料

* **DeepSeek‑V3 技术报告**：`671B` 总参、`37B` 激活、`MLA`、`MoE` 架构等；
* **vLLM Expert Parallel 文档**：`EP` 支持与部署参数；
* **NVIDIA 关于 MoE/并行的工程实践（TensorRT‑LLM / 博客）**，`NVLink` 带宽口径说明。

---

## 12. 结论

在 `32` 张 `H20` 上，采用 `TP=8 / EP=4 / DP=1` 能**稳态提供 30–40k tokens/s**，并在通信‑计算重叠、`EP` 路由与编译优化到位时，**有机会达到 50k tokens/s** 与 `TTFT < 1.2s` 的目标。若 `SLO` 必须在初期就锁定 `50k+`，建议 **扩展至 DP=2（64 卡）** 或引入更强的内核/通信优化链路。
