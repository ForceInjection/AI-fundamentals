# 第 6 讲：并行 TP 到 EP — PPT 大纲

## 设计说明

- **总页数**：约 23 页
- **演讲时长**:120 分钟（nano 回顾 10 min + 轴全景 15 min + 现状走读 45 min + 实验 25 min + 答疑 10 min）
- **核心视觉**：六轴全景图（本讲锚点，第 6 页）;EP 组 rank 布局图（第 10 页，DP×PCP×TP,PP 不参与）；部署台阶图（第 15 页，单机 EP → 2 机 Wide-EP）；八条失效对照表（第 19 页）
- **教学策略**：从 nano「TP 之外什么都没有」切入；轴全景先立公式再讲语义；EP 部分用「三个反直觉」组织（整只专家/无 all2all/PP 不参与）；部署用台阶图（单卡 → 单机 TP → 单机 EP → 2 机 Wide-EP）
- **配色**:nano 橙色系、六轴图蓝色系（每轴一色）、EP 通信绿色系、失效表红色系
- **边界**:NCCL/custom allreduce kernel 实现一行带过；MoE 路由数学不展开；PD 分离只点名（第 9 讲）

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：并行：TP 到 EP
- **副标题**:vLLM 十讲 · 第 6 讲
- **基线信息**:nano-vllm `b9dd55a` + vLLM `1acec5c009` | 核对日期 2026-11-24
- **一张图**：左「nano:1 个 TP 维度」右「vLLM:6 根并行轴（蓝紫绿黄红灰六色）」

### 第 2 页：本讲核心问题与议程

- **核心问题**：一台机器装不下 MoE 的专家，把专家切到多台机器上，怎么切才算切对了?
- **四段议程**:nano 回顾（10 min，第 3 页）| 轴全景与三个公式（15 min，第 4–7 页）| 现状走读六环节（45 min，第 8–18 页）| 实验（第 20–21 页）

## 第一部分：nano 回顾（10 min，第 3 页）

### 第 3 页：nano 回顾：TP 之外什么都没有

- **复用先修课第 8 课锚点图**（SHM 广播）
- **锚点**（橙色）：`ModelRunner.__init__` 建 SHM（`:42-49`）;rank>0 `loop()` 等Event 反序列化方法调用（`:62-67`）;rank0 `write_shm`/`call` 广播（`:77-90`）
- **口播**:「nano 的并行 = 把方法调用广播给兄弟进程；没有 PP/DP/EP，没有集合通信。今天看六根轴怎么长出来」

## 第二部分：轴全景与三个公式（15 min，第 4–7 页）

### 第 4 页：六根轴

- **一张图**：六轴全景（每轴一色）:TP / PP / PCP（新，prefill 序列切分）/ DP / DCP（新，decode KV 切分）/ EP（开关）；`ParallelConfig`（`parallel.py:248`）
- **口播**：两根新轴 PCP/DCP 是 2026 年长出来的——长序列与混合注意力逼的

### 第 5 页：三个公式

- **上稿**（三个公式卡片）:
  1. `world_size = TP × PP × PCP`（`parallel.py:1182-1186`）——**不含 DP 与 EP**
  2. EP 组大小 = `DP × PCP × TP`（不是独立配置项）
  3. `EP_SIZE = TP × DP`（部署文档自动计算，`expert_parallel_deployment.md:49-55`）
- **红色框**：写 `world_size = TP×PP×DP×EP` 是错的——DP 用 `world_size_across_dp` 单独表达（`:823`）

### 第 6 页：DP vs EP 语义（代码注释原文）

- **一张图**：同一批 rank 的两种切法（DP 切法 vs EP 切法，`parallel_state.py:2200-2207` 注释）
- **上稿**:DP = attention 数据并行（独立 KV cache、独立 batch，组内须同时 forward 否则死锁）;EP = 专家在 DP×TP 展开后的专家并行组；**PP 不参与 EP**
- **口播**:「同一批卡，两种团队组织方式——这决定了后面通信原语的选择」

### 第 7 页：executor 缺省逻辑

- **上稿**（`parallel.py:1272-1321`）：默认 **mp 尽量单机**;world_size > 本机 GPU 数且未设 nnodes → 直接报错；world_size==1 → uni；有 ray placement group 才 ray
- **对照框**:「多机默认 Ray」是文档旧说法（`parallelism_scaling.md:31` 与代码有出入；nnodes>1 只允许 mp/uni/external）

## 第三部分：现状走读（45 min，第 8–18 页）

### 第 8 页：环节① 配置与校验

- **上稿**（`_validate_parallel_config`，`parallel.py:681-770`）:all2all "auto" 解析（GB200 优先 flashinfer_nvlink_one_sided）;EPLB 前置（EP 开 + TP×PCP×DP>1）;TP 被 DCP 整除（PCP=1 时）
- **口播**：这些校验就是实验 1 的素材——错误组合的报错是学习材料

### 第 9 页：环节② 进程组初始化：九种 group

- **一张大图**：`initialize_model_parallel`（`parallel_state.py:2131-2460`）切出九组，按 rank 布局图：ExternalDP×DP×PP×PCP×TP → TP(2219)→ DCP(2315)→ PCP(2346)→ PP(2351)→ DP(2379)→ **EP（2400，仅 MoE）**→ EPLB（2440，防死锁的独立组）
- **上稿**:EP 组 rank 布局 = DP×PCP×TP(transpose(1,2));**PP 不参与 EP**;EP 组仅 `model_config.is_moe` 时创建
- **口播**：九种组一次建齐；每组的通信原语不同(TP allreduce / EP all2all / DP gloo sync)

### 第 10 页：EP 组 rank 布局图（重点页）

- **一张图**:8 卡 TP=2 DP=4 的 rank 排布 → EP 组切法（每 2 卡一组，共 4 组，每组跨 DP）——对照 nano 的单一 TP 组
- **上稿**:GroupCoordinator 封装（getter `:1613-1734`）；各组通信原语速查（TP allreduce / EP all2all / DP gloo / PP 点对点）

### 第 11 页：环节③ TP 通信链

- **一张图**:RowParallelLinear.forward → all_reduce → 后端优先级链（`cuda_communicator.py:389-455`）:symm-mem → quick AR → flashinfer PCIe → flashinfer NVLink → aiter → custom AR → pynccl
- **上稿**:ColumnParallel（`:447`，all_gather `:639`）与 RowParallel（`:1722`）的经典分工；custom AR 系只对 TP 组启用（`:59-65`）;DBO overlap（`:762-772`）

### 第 12 页：环节④ EP 判定：专家整只放卡

- **上稿**（`fused_moe/config.py:1067-1296`）:
  - `use_ep = (DP×PCP×TP>1) and enable_expert_parallel`（`:1260-1264`）
  - **开 EP 后专家层 `tp_size` 置 1**（`:1292-1296`）——每只专家整只在一张卡，无张量切分
  - 放置：`determine_expert_map`（linear / round_robin;deepep_ll/nixl 强制轮转，`:137-139`）;EPLB 动态重平衡（`vllm/distributed/eplb/`）
- **口播**:「TP 切专家是切每只专家的宽度；EP 切专家是分专家的只数」——两种粒度，EP 走了后者

### 第 13 页：dispatch/combine 抽象

- **一张图**:MoE 层 forward 的 token 往返（dispatch → experts → combine，`moe_runner.py:878/914`）与抽象接口（`FusedMoEPrepareAndFinalize`，`modular_kernel.py:241`）
- **上稿**：后端实现族在 `fused_moe/prepare_finalize/`(deepep_ht/ll/v2、flashinfer_nvlink×2、moonep、mori、nixl_ep、passthrough、batched…)；通信实体 = EP 组上的 all2all manager（`all2all.py:57/257/316`）

### 第 14 页：all2all 后端族与触发条件

- **一张表**：后端 × 场景（deepep_high_throughput=多节点 prefill / deepep_low_latency=多节点 decode+CUDA graph / flashinfer_nvlink=MNNVL / mori=ROCm / passthrough=FlashInfer megakernel / allgather_reducescatter=兜底；`expert_parallel_deployment.md:32-46`）
- **红色反直觉框**:**纯 TP-EP（无 DP）不产生 all2all**——`use_all2all_kernels` 要求 dp/pcp/sp>1（`fused_moe/config.py:1088`）;token 本在 TP 组冗余，输出靠 TP allreduce 汇总
- **口播**:「all2all 的现象要在 DP>1 的部署里才看得到」——这决定实验设计

### 第 15 页：部署台阶图

- **一张台阶图**（四级）：单卡 → 单机 TP（`--tensor-parallel-size 8`）→ 单机 EP（`-t 1 -dp 8 --enable-expert-parallel`，`expert_parallel_deployment.md:85-96`）→ 2 机 Wide-EP（16 卡，`--all2all-backend deepep_low_latency --data-parallel-size 16 --headless`，`:166-193`）
- **上稿**:EP_SIZE=TP×DP 自动计算；EPLB 配置（`:210-332`）;PD 分离形态(prefill HT / decode LL)第 9 讲回收
- **性能口径行**：博客「2.2k tok/s/H200」(2025-12-17)对应 2 机形态；仓库文档无此数字，引用注明厂商口径

### 第 16 页：DP attention 与 DCP

- **上稿**:
  - DP attention **不是独立实现**:DP replica 复制 attention,MoE 层 EP 打通（`data_parallel_deployment.md:5-16`）；同步靠 `dispatch_cg_and_sync_dp` + 每 16 步 gloo（`dp_sync_interval`）
  - DCP：按 head 交错切 KV，上限 = tp/kv_heads；案例 R1(H=1)tp8→dcp8、Qwen3-235B(H=4)tp8→dcp2
  - PCP:prefill 分 chunk,「partial Q full KV」或 ring-attention
- **口播**：三根新轴各服务一类模型形态——这不是过度设计，是新模型形态倒逼的

### 第 17 页：环节⑥ 设计文档核对

- **上稿**：`arch_overview.md` 进程计数表漏 PCP 轴（Worker=DP×PP×TP 应为 ×PCP）；`fused_moe_modular_kernel.md`/`moe_kernel_features.md`/`dbo.md` 与代码一致；`docs/serving/` 四篇部署文档与 CLI 示例一致（无 yaml）
- **口播**：部署文档的质量比 design 文档高——因为部署命令直接对用户

### 第 18 页：V0/旧教程 vs HEAD

- **八行对照表**（README 同款）:FusedMoE 类没了 / world_size 公式 / 纯 TP-EP 无 all2all / naive-pplx 删 / 多机默认 Ray 失效 / ep 目录不存在 / DP attention 非独立实现 / arch_overview 缺 PCP

## 第四部分：实验与收尾（第 19–23 页）

### 第 19 页：实验 1 素材预告（课堂用）

- **上稿**：实验 1 的素材预告——错误组合报错就是校验逻辑（`_validate_parallel_config`）的输出；预告三个必试组合

### 第 20 页：本讲实验

- **三行实验卡**：实验 1（纯 CPU）配置组合校验；实验 2（单卡）TP/EP 启动日志走读；实验 3（可选，双卡）DP attention 观察
- **产出要求**：三组报错解读；日志 group 分配表；配置决策说明

### 第 21 页：实验 1 详解（配置组合校验）

- **上稿**：三个必试组合（附预期报错方向）:① `--tensor-parallel-size 8` 单机 4 卡（超卡数报错）;② `--decode-context-parallel-size 3 --tensor-parallel-size 4`（DCP 不整除）;③ `--enable-eplb`（缺 `--enable-expert-parallel` 报错）
- 思考题：为什么 EPLB 要求 TP×PCP×DP>1?（答案：单卡无重平衡意义）

### 第 22 页：实验 2 详解（日志走读）

- **上稿**：单卡起服务的 init 日志逐行标注（九组各在哪行出现；EP 组为何缺席——dense 模型）；有双卡时对照 TP=2 的 group 变化
- 思考题：dense 模型 `_EP=None`（`parallel_state.py:2402`）——EP 组不建，EP_SIZE 公式还成立吗?（答案：公式是 MoE 语境；dense 无专家轴）

### 第 23 页：结课与预告

- **本讲你应能**：写三公式；选 TP/DP/EP 组合并写部署命令；认八条失效说法
- **预告**：并行把算力摊开了，但 KV 在多实例间还是孤岛——第 9 讲：KV 分载与 PD 分离（你 fork 的 swap 与 vLLM 三代 offloading 的对照）
- **下节材料**：第 9 讲 syllabus 阅读清单（第 7、8 讲之后）

---

## 图资产清单（自制）

| 图                                   | 页  | 说明                                      |
| ------------------------------------ | --- | ----------------------------------------- |
| 六轴全景                             | 4   | 每轴一色；新旧轴标注                      |
| 公式三卡                             | 5   | 红色错误公式对照                          |
| DP/EP 同 rank 两切法                 | 6   | 代码注释图解                              |
| 九组初始化总图                       | 9   | rank 布局 + 各组通信原语                  |
| EP 组 rank 布局                      | 10  | 8 卡示例排布                              |
| allreduce 优先级链                   | 11  | 与第 3 讲风格一致                         |
| token 往返(dispatch→experts→combine) | 13  | MoE 层 forward                            |
| 部署台阶图                           | 15  | 四级台阶 + 性能口径行                     |
| 两棵决策树并排                       | 16  | 标准 + Mamba（复用第 4 讲）+ DCP/PCP 标注 |
