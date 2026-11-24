# 第 9 讲：KV 分载与 PD 分离 — PPT 大纲

## 设计说明

- **总页数**：约 23 页
- **演讲时长**:120 分钟（nano 回顾 10 min + 三代对照 15 min + 现状走读 45 min + 实验 25 min + 答疑 10 min）
- **核心视觉**：三代演进对照表（本讲锚点，第 6 页，决策者/搬运者/淘汰策略三行）；两条并行实现关系图（第 10 页，「平行不是子集」）；请求旅程图局部放大（offload 分支，第 13 页）
- **教学策略**：从学生 fork 的 kv_swap 开场（「你写过这一代的实现」）；三代对照先立框架；两条路径用「平行实现」破除「简化版」误解；disagg 用角色配对讲
- **配色**:nano 橙色系、三代对照蓝紫渐变、Connector 全家福绿色系、易混淆红色系
- **边界**:LMCache/NIXL/Mooncake 内部不展开；租约只讲语义；tiering 二级层实现自读

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**:KV 分载与 PD 分离
- **副标题**:vLLM 十讲 · 第 9 讲
- **基线信息**:nano-vllm fork（含你的 kv_swap）+ vLLM `1acec5c009` | 核对日期 2026-11-24
- **一张图**：显存水位线图——线下 GPU 块，线上 CPU/磁盘/远端，三条上升箭头各标一代

### 第 2 页：本讲核心问题与议程

- **核心问题**：显存装不下的 KV，分载到 CPU、磁盘、另一台机器，分别怎么做?你的 swap 离生产版还有几代?
- **四段议程**:nano 回顾（10 min，第 3–4 页）| 三代演进对照（15 min，第 5–6 页）| 现状走读六环节（45 min，第 7–18 页）| 实验（25 min，第 20–21 页）

## 第一部分：nano 回顾（10 min，第 3–4 页）

### 第 3 页：nano 回顾：你写过 V0 的实现

- **复用你 fork 的锚点图**（三队列 + swap 两层决策）
- **上稿**:
  - 决策耦合调度：`Scheduler.schedule()` 内显式 swap_in/out（waiting/running/swapped 三队列）
  - 搬运：`block_manager.py:142-180`，GPU↔CPU pinned 整请求搬运
  - 约束：`can_swap_out` 要求 `ref_count==1`（`:135-140`）;CPU 块不参与前缀缓存
- **实测行**:RTX 3090/Qwen3-0.6B:SWAP ≈ RECOMPUTE（收益随模型规模增长）

### 第 4 页：你的 swap 对应哪一代

- **上稿**：三代定位卡——你的实现 ≈ V0 CacheEngine swap + 自加 swapped 队列（v0.11 三件套的前身思路）；对应 HEAD 的两代差距：决策权移出调度器 + 分层后端/多级/可插拔策略
- **口播**:「差距不是质量差距，是三年间生产约束的沉淀。这节课就是量这三年的尺子」

## 第二部分：三代演进对照（15 min，第 5–6 页）

### 第 5 页：三代演进对照表（本讲锚点）

- **一张大表**（README 同款三行，每行三列）:
  - V0 CacheEngine swap：主 Scheduler 决策（SWAPPED 队列）| CacheEngine 整请求搬运 | GPU/CPU 双池 LRU
  - v0.11-v0.2x 三件套：OffloadingManager（调度器进程）| OffloadingHandler（worker 异步）| Manager 内 LRU（后补 ARC）
  - HEAD:connector 侧 OffloadingConnectorScheduler/SimpleCPUOffloadScheduler 决策；主 Scheduler 只剩 recompute + 询问 | OffloadingWorker（每方向 CUDA stream DMA；二级层经 CPU 中转）| CachePolicy 可插拔(lru/arc)+ 二级层背压
- **口播**：每一代的决策权都在往外搬——从调度器到 connector，是「可插拔」的代价与收益

### 第 6 页：措辞修正：Handler 去哪了

- **上稿**：`OffloadingHandler` 抽象被 `OffloadingWorker`（`base.py:741`）取代；「Handler」降级为内部实现名（`SingleDirectionOffloadingHandler`，`cpu/gpu_worker.py:314`；`StoreTriggerHandler`）
- **对照**：课程旧文的「三件套」表述按此修正

## 第三部分：现状走读（45 min，第 7–18 页）

### 第 7 页：环节① 配置入口

- **一张图**：两条入口路径（`--kv-offloading-size` + backend 翻译成 KVTransferConfig，`vllm.py:1250-1280`；或直接 JSON kv_connector+kv_role）
- **红色易混框**：`vllm/config/offload.py` 是**模型权重** offload(UVA/Prefetch)，与 KV 无关——别混

### 第 8 页：环节② 两条并行实现（关系图）

- **一张大图**:KVConnectorBase_V1（`v1/base.py:182`）下挂两个 connector；左 OffloadingConnector → kv_offload/ 分层后端（Spec 工厂仅 CPU/Tiering 两个内置）；右 SimpleCPUOffloadConnector → simple_kv_offload/ 自带 manager/worker
- **红色结论框**:**平行实现，零代码共享**——不是「简化子集」;simple 多 hugepage/显式块数/磁盘后端（`disk_backend.py:49` writev/readv 后台线程）
- **端到端路径行**：调度循环问 `get_num_new_matched_tokens`（`manager.py:362`）→ `build_connector_meta`（`:557`）→ worker start_load_kv/wait_for_save（`worker.py:323/340`）→ get_finished（`:364`）

### 第 9 页：路径 A 决策者：OffloadingManager

- **上稿**（`base.py:329-546`）:14 个方法接口一览(lookup/prepare_load/complete_load/prepare_store/complete_store/on_new_request/on_schedule_end/reset_cache/get_stats…)；四态 LookupResult（MISS/HIT/HIT_PENDING/RETRY，`:155-161`）
- **CPU 层落点**：`CPUOffloadingManager`（`cpu/manager.py:55`，容量不足返回 keys_to_retry）；逐出委托 CachePolicy
- **口播**：接口为什么这么细?——决策要异步化，每步只做「下一步」，完成靠回调

### 第 10 页：路径 A 搬运者：OffloadingWorker

- **一张图**:CPU 路径搬运（CPUOffloadingWorker，`cpu/gpu_worker.py:811`；两个 SingleDirectionOffloadingHandler（`:314`）各占一条 CUDA stream；指针三元组 DMA 描述符）
- **上稿**：二级层（fs/obj/p2p/kvcr，`tiering/factory.py`）一律经 CPU 中转（`tiering/base.py:146` 注释）；背压 EMABackpressureDetector（`backpressure.py:189`）
- **孤儿警示框**（红色）：顶层 `fs/` 的 FSOffloadingWorker 未接线（仅测试引用）——生产 FS 走 tiering 二级层

### 第 11 页：淘汰策略可插拔

- **上稿**（`cpu/policies/`）：`CachePolicy` ABC（`base.py:48`：get/insert/remove/on_request_finished/evict）；内置 lru（`lru.py:26`，按请求粒度）/ **arc**（`arc.py:28`，T1/T2+ghost，活跃维护 #69110）；外部策略经 `cache_policy_module_path`（`factory.py:12`）
- **对照行**:nano 的 free/used 集合 + ref_count ≈ 单一 LRU——可插拔是 HEAD 新能力

### 第 12 页：环节③ 调度集成：谁在做 swap 决策

- **一张图**：第 1 讲请求旅程图的 offload 分支放大——主 Scheduler 每请求问 connector（get_num_new_matched_tokens，`scheduler.py:1161`）→ connector 侧 manager 决策 → worker 搬运 → get_finished 上报；抢占时 worker handle_preemptions（`worker.py:408`）善后，主 Scheduler 仍只 recompute（`scheduler.py:1830-1877`）
- **上稿**:offload_prompt_only 默认 True（`base.py:827`，只 offload prefill 块）;StoreTrigger 两类（ON_COMPUTE 算完即存 / ON_HANDOFF，`store_triggers/`）
- **口播**:「调度器不再知道 CPU 的存在」——这是三代演进的最终形态

### 第 13 页：请求旅程局部放大

- **一张图**：课程主页请求旅程图 + offload 分支（显存不足 → KV 移 CPU/磁盘；跨实例 → connector 传输）
- **呼应**：与 nano fork 的 swap 时序并排（学生实现 vs 生产两代）

### 第 14 页：环节④ KV Connector 全家福

- **一张大表**（绿色，15 个连接器，注册名/文件/一句话）:LMCacheConnectorV1+MP / NixlConnector(Pull/Push)/ MooncakeConnector+Store / MultiConnector / OffloadingConnector / SimpleCPUOffloadConnector / FlexKV / HF3FS / HiSparse / MoRIIO / UMBP / DecodeBench / Example×2
- **接口行**:worker 侧 4 抽象(start_load_kv/wait_for_layer_load/save_kv_layer/wait_for_save)+ scheduler 侧 3 抽象（`v1/base.py:182`；374/557）
- **口播**:「offloading 是 connector 家族的一员，不是独立子系统」

### 第 15 页：环节⑤ disagg prefill

- **一张图**:P 实例(kv_producer)→ connector（KV 传输）→ D 实例(kv_consumer)
- **上稿**（`disagg_prefill.md`）:
  - 动机：分实例调 TP/PP 独立优化 TTFT 与 ITL;**明确声明不提升吞吐**
  - 示例：`{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["UCX","GDS"]}}`(doc:24-50)
  - 请求级协作：`kv_transfer_params` 的 `do_remote_prefill`；decode 端复用 prefill 端 token ids 免二次分词
- **红色修正框**:XpYd 概念不在 disagg 文档——出处 `mooncake_store_connector_usage.md:93`
- **精巧细节行**:NIXL KV 租约（初始 30s，心跳续 2/3；防 stale-read，`nixl_kv_cache_lease.md`）

### 第 16 页：环节⑥ 观测口径

- **一张表**（绿色）：两套 metrics 并存（`vllm:kv_offload_*` / `vllm:simple_kv_offload_*`）；命中来源标签 `prompt_tokens_cached_by_source` 的 source=host/disk/p2p（device 合计守恒）;CPU 层 gauge(usage_perc/allocation_size)；日志确认行（"Creating offloading spec with name: ..."，`factory.py:57`）
- **口播**：两套并存是「平行实现」的直接代价——观测也要分路径

### 第 17 页：三代差距量化：你的 swap vs HEAD

- **一张对照卡**（nano fork vs HEAD，六行）:
  - 决策位置：Scheduler.schedule 内 vs connector 侧 manager
  - CPU 块元数据：不参与哈希 vs 参与命中(source=host)
  - 淘汰：GPU/CPU 双池 LRU vs CachePolicy 可插拔(lru/arc)
  - 搬运：同步整请求 vs 每方向 CUDA stream 异步
  - 层级：单 CPU 层 vs CPU+fs/obj/p2p/kvcr 多级
  - 观测：无 vs 专用 metrics 家族
- **口播**：这就是「三年」的刻度——作业 3 要你把它写成正式对照分析

### 第 18 页：V0/旧教程 vs HEAD

- **六行对照表**（README 同款）:offload.py 无关 / XpYd 位置 / fs 孤儿 / simple 平行 / metrics 两套 / Handler 词降级

## 第四部分：实验与收尾（第 19–23 页）

### 第 19 页：本讲实验

- **三行实验卡**：实验 1（核心，单卡）CPU offload 生效观察；实验 2 双路径对照（simple env 开关）；实验 3（可选，双机）NixlConnector PD
- **产出要求**：生效日志行；两路径对照表；PD 配置（实验 3）

### 第 20 页：实验 1 详解（CPU offload 生效观察）

- **上稿**：长 prompt 探针 + `--kv-offloading-size 8`；日志找 "Creating offloading spec with name: CPUOffloadingSpec"；命中来源读 `prompt_tokens_cached_by_source` 的 host 标签；重算对比（recompute-only 服务）
- 思考题：为什么小模型 SWAP≈RECOMPUTE（引用你 fork 的 3090 实测结论，设问「H100 上这个等式还成立吗」）

### 第 21 页：实验 2 详解（双路径对照）

- **上稿**：默认(native→OffloadingConnector)vs `VLLM_USE_SIMPLE_KV_OFFLOAD=1`；对照两套 metrics 名、hugepage 特性、磁盘模式(disk_backend)
- 思考题：两条路径的维护成本 vs 特性收益，vLLM 为什么保留两条?

### 第 22 页：实验 3（可选）PD 分离最小复现

- **上稿**：双机（或单机双实例端口区分）按 disagg 文档示例配 NixlConnector kv_both;P 实例与 D 实例的 kv_role 日志；KV 传输计数的 metrics 行
- 无双机替代：纸面推演 + 读 `disagg_prefill.md` 示例命令逐参数解释

### 第 23 页：结课与预告

- **本讲你应能**：复述三代对照表；区分双路径；写 PD 分离最小配置；辨认「Handler 词降级」「XpYd 出处」类修正
- **作业**：作业 3（计考核）——你的 swap 与 HEAD offloading 的正式对照分析
- **预告**：单引擎全部机制讲完。第 10 讲跳出来：生产运维与生态（metrics/压测/production-stack/控制面），以及引擎→控制面的行业演进
- **下节材料**：第 10 讲 syllabus 阅读清单

---

## 图资产清单（自制）

| 图 | 页 | 说明 |
| --- | --- | --- |
| 显存水位三代图 | 1 | 开场 |
| fork swap 锚点图 | 3 | 复用先修课/fork 设计文档 |
| 三代对照大表 | 5 | 锚点；三行三列 |
| 双路径关系图 | 8 | 「平行不是子集」 |
| OffloadingManager 接口一览 | 9 | 14 方法 |
| CPU 搬运图 | 10 | 双 stream DMA |
| offload 分支旅程放大 | 12-13 | 课程主页旅程图复用 |
| Connector 全家福表 | 14 | 15 个绿色 |
| PD 角色配对图 | 15 | producer→connector→consumer |
| 三代差距量化卡 | 17 | fork vs HEAD 六行 |
