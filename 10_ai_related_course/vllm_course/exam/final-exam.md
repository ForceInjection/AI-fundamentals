# vLLM 推理系统课程 · 结课考试卷

> **考试时间**：120 分钟 | **总分**：100 分（另设附加题 10 分，不计入总分，用于评优）
> **命题范围**：课程十讲（基线：nano-vllm `b9dd55a` + vLLM `2f5d8a2b1`）+ 本仓库相关深读内容
> **答题须知**：闭卷；计算题需给出算式；简答按要点给分，答出参考要点之外的合理机制同样得分

---

## 一、单项选择题（每题 2 分，共 20 分。每题只有一个正确选项）

**1.** vLLM V1 中，EngineCore 进程内**不包含**以下哪个组件？

A. Scheduler　　B. KVCacheManager　　C. Tokenizer　　D. StructuredOutputManager

**2.** 关于 V1 调度器的「无阶段」设计，正确的是：

A. prefill 与 decode 分两个独立队列调度
B. 每个请求仅由 `num_computed_tokens` 等计数驱动，无 prefill/decode 阶段属性
C. chunked prefill 需要显式开启才生效
D. 抢占默认走 swap

**3.** 课程基线(HEAD `2f5d8a2b1`)上 PagedAttention 的现状，正确的是：

A. 自研 CUDA kernel 仍是主路径
B. kernel 已退役；分页管理思想由 `block_table` 与 BlockPool 承载
C. 已被 MLA 全面取代
D. 仅 ROCm 平台保留

**4.** 在 H100(SM90)上，下列哪个后端**不会**被 dense GQA 模型的自动选择命中？

A. `FLASH_ATTN`　　B. `FLASHINFER`　　C. `TRITON_ATTN`　　D. `FLASHMLA`

**5.** 关于 HEAD 上 torch.compile 优化级别，正确的是：

A. 默认 -O1;-O3 比 -O2 多若干融合项
B. 默认 -O2;-O3 与 -O2 逐字段相同
C. 默认 -O3
D. 默认 -O0，需显式开启编译

**6.** 纯 TP-EP（单机 TP=8 开 EP，无 DP）时，MoE 的 token 汇总方式是：

A. 专家间 all2all 通信
B. 靠 TP allreduce 汇总，无专家间 all2all
C. 走 deepep low latency
D. 走 NVLink peer 直推

**7.** HEAD 上投机解码方法的合法名，下列**错误**的是：

A. `ngram`　　B. `eagle3`　　C. `medusa`　　D. `draft_model`

**8.** `--quantization fp8` 传入「不含量化信息的 BF16 checkpoint」时的行为是：

A. 在线量化为 FP8
B. 直接报错并指引改用 `fp8_per_tensor`
C. 静默回退 BF16
D. 自动调用 LLM Compressor

**9.** HEAD 上 CPU offload 的「换出决策」由谁做出？

A. 主 Scheduler　　B. KVCacheManager
C. connector 侧的 manager　　D. Worker 进程

**10.** 下列哪个**不是** vLLM OpenAI 兼容 server 的通用端点？

A. `/health`　　B. `/metrics`　　C. `/ping`　　D. `/version`

---

## 二、多项选择题（每题 4 分，共 20 分。多选、错选不得分；漏选得 2 分）

**11.** 影响单请求 KV 显存占用的因素包括：

A. `num_hidden_layers`　　B. kv head 数与 `head_dim`
C. KV dtype(fp16/fp8)　　D. `max_num_seqs`

**12.** 关于 HEAD 的 CUDA Graph，正确的是：

A. 默认档为 `FULL_AND_PIECEWISE`
B. decode 走整图、prefill 走分段图
C. capture 尺寸分两族网格（prefill token 族 / decode 请求族）
D. capture 经由 `_dummy_run` 触发

**13.** HEAD 上仍在职的投机解码方法包括：

A. `ngram`　　B. `medusa`　　C. `eagle3`　　D. `mtp`

**14.** 关于 HEAD 的量化体系，正确的是：

A. `gptq_marlin` 是 `auto_gptq` 的别名
B. Marlin 推理时执行 INT8/FP8 激活量化
C. online shorthand（`fp8_per_tensor` 等）在加载时量化 BF16 权重
D. bitsandbytes 与 GGUF 以 OOT 插件形式提供

**15.** 关于 KV offloading 的两条并行实现，正确的是：

A. `OffloadingConnector` 与 `SimpleCPUOffloadConnector` 零代码共享
B. 换出决策由 connector 侧的 manager 做出
C. 主 Scheduler 的抢占只剩 recompute
D. offload 到 CPU 的块无法再被前缀查询命中

---

## 三、填空题（每空 1 分，共 10 分）

**16.** V1 中 EngineCore 进程内没有 **\_\_**，它位于 API 层的 OutputProcessor 中。

**17.** `EngineCore.step` 中以非阻塞方式下发模型执行的调用是 `self.model_executor.______`。

**18.** EngineCore 侧负责「socket 收发 ↔ 计算队列」解耦的两个方法，定义在 `core.py` 的哪个类/函数中？（答出方法名即可）

**19.** UCIe.mem 的 256 字节 FLIT 中，数据载荷占 **\_\_** 字节。

**20.** UCIe 4.0 协同优化功耗的三个前提：工艺代兼容、共享电压域、互连距离约 **\_\_** 以内。

**21.** vLLM 的 `world_size = ______ × PP × PCP`（不含 DP 与 EP）。

**22.** NVLink 5 单 GPU 双向带宽为 **\_\_**。

**23.** Marlin 推理时激活精度恒为 **\_\_** 位。

**24.** 投机解码的 Mean acceptance length 含 bonus token，公式为 1 + **\_\_**。

**25.** 容器化部署 vLLM 后开启 CPU offload，常因 Docker 默认的共享内存上限(64M)过小而启动失败；需要调整的 `docker run` 参数是 **\_\_**。

---

## 四、计算题（每题 10 分，共 20 分。要求给出算式）

**26.** Qwen3-0.6B（28 层，GQA 8 个 kv-head，`head_dim` 128,BF16）部署在单卡 H100 上，目标承载 32K 上下文 × 64 并发：

1. 计算每 token 的 KV 占用(B)与该负载下的 KV 总量(GB)；
2. 判断单卡(80GB)能否承载；若不能，给出两条可行的解决路径并说明各自代价。

**27.** 8×H100 节点，以 `--tensor-parallel-size 2 --data-parallel-size 4 --enable-expert-parallel` 部署一个 MoE 模型：

1. 列出进程拓扑（API server / EngineCore / Worker 各多少，EP 组大小）；
2. MoE 层的 token 路由是否产生 all2all 通信？说明判断依据（引用课程第 6 讲的判定条件）。

---

## 五、简答题（每题 10 分，共 30 分。每题按要点给分）

**28.** V1 的抢占为什么只剩 recompute？swap 的能力去了哪里？（10 分；要求：决策位置的变化 + 功能去向 + 与第 9 讲 offloading 的衔接）

**29.** 简述 vLLM 的 CPU offloading 两条并行实现（`OffloadingConnector` + `kv_offload/` 分层后端，与 `SimpleCPUOffloadConnector`）的差异。（10 分；要求：至少三个差异点，从决策位置、实现结构、观测方式等角度展开，机制讲清即可）

**30.** `-O2`（默认）与 `--enforce-eager` 各自关掉了哪一层？为什么生产上默认 -O2，而排障时常用 --enforce-eager？（10 分；要求：说清 `FULL_AND_PIECEWISE`、编译、capture 三者的层次关系）

---

## 附加题（10 分，不计入总分，用于评优）

**31.** 不少于 800 字：为一组 Agent 长会话负载（8K system prompt 复用率高，单请求输出 1K，尾延迟敏感，QPS 200 峰值）撰写**生产部署与观测方案**，要求覆盖：

- 并行组合与实例拓扑（第 6 讲）
- 调度参数（第 2 讲）
- 投机解码选型（第 7 讲）
- 量化与 KV dtype（第 8 讲）
- offload 与 PD 分离的取舍（第 9 讲）
- 观测面板指标与告警阈值（第 10 讲）

每项决策须附「机制依据」（引用课程讲次与章节），厂商口径与独立口径分开标注。
