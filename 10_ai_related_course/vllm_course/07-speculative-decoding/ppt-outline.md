# 第 7 讲：投机解码 — PPT 大纲

## 设计说明

- **总页数**：约 22 页
- **演讲时长**:120 分钟（nano 回顾 10 min + 演进 15 min + 现状走读 45 min + 实验 25 min + 答疑 10 min）
- **核心视觉**：一步的流水时序图（本讲锚点，第 12 页：target 前向 1+K 展开 → 采样即验证 → draft K 步）；五法选型速查表（第 11 页）;acceptance 口径对照卡（第 17 页）
- **教学策略**：收益公式 (1+a)/(1+c) 先立住，全讲围绕「a 怎么变大、c 怎么变小」；验证循环用「常见错讲法 vs HEAD 事实」对比切入；Medusa 移除作为「演进素材」案例
- **配色**:nano 橙色系、Speculator 体系蓝色系、验证/回滚红色系、指标绿色系
- **边界**：训练侧只给链接（speculators 库）;rejection 数学只讲机制；draft 图细节引用第 5 讲

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：投机解码
- **副标题**:vLLM 十讲 · 第 7 讲
- **基线信息**:nano-vllm `b9dd55a` + vLLM `1acec5c009` | 核对日期 2026-11-24
- **一张图**：串行生成（一格格爬楼梯）vs 投机生成（一次跳多格 + 检查点）

### 第 2 页：本讲核心问题与议程

- **核心问题**：一次生成多个 token 还不出错，收益的上限由什么决定，代价藏在哪里?
- **四段议程**:nano 回顾——为什么没有（10 min，第 3 页）| 演进：Proposer→Speculator、Medusa 移除（15 min，第 4–6 页）| 现状走读五环节（45 min，第 7–16 页）| 实验（25 min，第 20–21 页）

## 第一部分：nano 回顾（10 min，第 3 页）

### 第 3 页：nano 回顾：一步一个 token 的世界

- **上稿**:
  - 先修课原文：nano-vllm「完全不涉及」投机解码——while 循环、一次前向一个 token
  - 为什么没有：草稿来源 / KV 预留 / 验证回滚，三套基础设施任何一套都会让教学实现翻倍
  - 为什么生产需要：decode 是 memory-bound（每步读全部权重与 KV，算力闲置）；投机 = 用闲置算力换并行验证
- **收益公式**（大字）：加速 ≈ (1 + a) / (1 + c)——a=每步接受数，c=草稿相对成本；全讲围绕「a 怎么变大、c 怎么变小」

## 第二部分：演进（15 min，第 4–6 页）

### 第 4 页：Proposer → Speculator

- **一张图**：目录迁移图（`vllm/v1/spec_decode/` → `vllm/v1/worker/gpu/spec_decode/`）+ 类层级图（BaseSpeculator:74 → DraftModelSpeculator:165 → AutoRegressiveSpeculator → TargetDependentARSpeculator → 各方法）
- **上稿**：统一接口 `propose`；GPU MODE Lecture 22 讲的 V1 机制全部有效，但**类名与路径按本讲校准**（NgramProposer→NgramGPUSpeculator 等）

### 第 5 页：Medusa 移除（演进案例）

- **上稿**（红色）:
  - method 合法值里没有 medusa（`config/speculative.py:113-123`）；类与文档全无
  - 仓库旧文 `speculative_decoding_landscape.md` 的 Medusa 条目 = 演进素材
  - 路线被 EAGLE3/MTP 取代（吃 target 隐状态的草稿头，与 MTP 同构）
- **口播**:「讲义一律以 HEAD 为准——你外面看到的『vLLM 六法含 Medusa』已经过期」

### 第 6 页：方法全集：比六种大得多

- **上稿**:method 合法值 11+ 内置(ngram/draft_model/suffix/eagle/eagle3/mtp/dflash/dspark/dfly/lilicorr/…)、~40 个 MTP 模型别名（**用即告警弃用，归一为 mtp**，`config/speculative.py:1495-1499`）；自动检测：`_CHECKPOINT_DETECTION_ORDER` 按 draft checkpoint 的 model_type 判断（`speculator_config.py:492`）
- **绿色框**：三个横切机制：adaptive verification（按置信度裁剪）、dynamic SD（按 batch 调 K）、fly（有损窗口）

## 第三部分：现状走读（45 min，第 7–16 页）

### 第 7 页：环节① 配置解析

- **上稿**：`--speculative-config` JSON；关键字段：`num_speculative_tokens`（`:469`）/ `model`（`:472`）/ `rejection_sample_method`（`:620`，默认 standard）/ `draft_sample_method`（`:701`，默认 greedy）/ `enable_adaptive_verification`（`:643`）
- **易错框**：`"method":"draft"` 不是合法值——是 `"draft_model"`

### 第 8–10 页：环节② 五法在职（速查表，每法一页）

- **第 8 页：ngram 与 suffix（无模型双雄）**
  - ngram:GPU Triton 扫全 token 历史（`ngram/speculator.py:151`；`ngram_gpu` 是别名）；零成本兜底
  - suffix：变长后缀（树深 24）+ **跨请求语料**（`suffix/speculator.py:501`；corpus 1M token，置 0 只匹配本请求）——代码编辑/agentic 场景首选
- **第 9 页：EAGLE/EAGLE3 与 draft_model（两个「要模型」）**
  - EAGLE：独立权重 + 消费 target 末层隐状态（`eagle/speculator.py:12`，继承 TargetDependentARSpeculator）；通用低延迟首选
  - draft_model：完整小模型自建 KV（`standalone_ar/speculator.py:32`）；有现成小模型时零训练成本
- **第 10 页：MTP（模型自带）**
  - 加载 target checkpoint 的 MTP 层（`mtp/speculator.py:12`）；`default_model`=target 自身并继承量化（`mtp/config.py:41-57`）；`num_speculative_tokens` 缺省从 checkpoint `n_predict` 读（`speculator_config.py:333-349`）
  - 适配模型：DeepSeek/Qwen3-Next/GLM-MoE/Kimi

### 第 11 页：五法选型速查表

- **一张大表**（README 同款）：方法 × 需额外模型 × 训练介入 × 典型场景 × 配置片段
- **教法**：当决策工具教；给定负载现场查表（代码补全→suffix；有 MTP 权重→mtp;…）
- **绿色通用键行**：`num_speculative_tokens` / `rejection_sample_method` / `draft_sample_method` / `enable_adaptive_verification` / 动态 SD

### 第 12 页：环节③ 验证循环（本讲锚点，常见错讲法对比）

- **一张大图**：一步的流水时序：
  - target 前向 1 次（query 展开 = 1 bonus + K draft，`input_batch.py:475`）
  - **采样即验证**（无独立 verify 步）：`RejectionSampler`（`rejection_sampler.py:152`；kernel `:1128`；greedy 时 draft_logits=None 退化为 argmax 匹配）
  - draft 前向 K 步（`propose`，`model_runner.py:2426-2458`）
- **红色错讲框**:「先 draft K 个、再单独 verify 一步」——旧讲法；HEAD 是 step N propose / step N+1 验证的**流水**
- **接线行**:rejection 接线 `model_runner.py:1788-1800`；构造 `:563`

### 第 13 页：变体验证器与图融合

- **上稿**:
  - `rejection_sample_method`：standard / synthetic（合成接受率测开销）/ block / fly（有损）
  - 图融合（引用第 5 讲）:draft 只支持 FULL_DECODE_ONLY；多步 draft 融合进一张 FULL 图（`speculator.py:36-52`）
- **口播**：变体验证器是「c 侧」的旋钮——接受率不变，验证开销变

### 第 14 页：环节④ 调度交互

- **上稿**（种子发芽图，引用第 2 讲）:
  - `draft_slots`（`scheduler.py:730`；基类 0 / draft_model 1 / EAGLE 按 lookahead）
  - lookahead KV 预留（1+K+lookahead，`:384-391`）
  - async 占位符（`num_output_placeholders`；拒绝回滚 `:2456-2463`）+ stale 机制
- **新增两行**:can_skip_drafts（高并发仅 target 步，`:349`）；动态 SD 查表（`:1354-1403`）

### 第 15 页：环节⑤ 观测指标（三口径）

- **一张表**（绿色）:
  - 日志：`SpecDecoding metrics: Mean acceptance length: %.2f ...`（`metrics.py:64/87/96`）
  - Prometheus：`vllm:spec_decode_num_accepted_tokens_total` 等（docstring 自带三条 PromQL，`:195`）
  - per-request：`--per-request-spec-decode-metrics summary`（`observability.py:52`）
- **红色口径框**:**日志均值含 bonus token**（= 1 + accepted/drafts，`:115-116`）;per-request 直方图不含——两套数字不要直接比

### 第 16 页：bench 口径

- **上稿**：无 `vllm bench spec` 子命令；两条路：`--dataset-name spec_bench`（`datasets.py:3011`）+ serve.py 自动抓 `vllm:spec_decode*` 写进结果（`serve.py:1169-1181`，输出 `spec_decode_acceptance_rate` 等）

## 第四部分：实验与收尾（第 17–22 页）

### 第 17 页：acceptance 口径对照卡

- **一张卡**（绿色，答疑用）：日志均值（含 bonus）/ per-request 直方图（不含）/ Prom 计数器（原始计数）——三个数字各是什么、何时可比

### 第 18 页：V0/旧教程 vs HEAD

- **六行对照表**（README 同款）:Medusa 移除 / Proposer→Speculator 迁移 / draft_model 名 / 验证时序流水 / 方法全集 11+ / bench 无 spec 子命令

### 第 19 页：选型实战（课堂演练）

- **一张流程图**：负载 → 问三问（有无现成草稿来源?任务重复性?QPS 高低?）→ 查速查表 → 配置片段
- 三个演练：代码补全助手(suffix)/ DeepSeek 部署(mtp)/ 通用客服（ngram 兜底或 EAGLE）

### 第 20 页：本讲实验

- **三行实验卡**：实验 1 ngram 开关对照（零依赖）；实验 2 三口径读数；实验 3（可选）动态 SD
- **产出要求**：开/关对照数字；三口径读数与口径说明；动态 SD 观察日志

### 第 21 页：实验 1 详解

- **上稿**：两轮起服务（默认 / `--speculative-config '{"method":"ngram","num_speculative_tokens":5}'`）× 同一探针；记录 TPOT 与日志 acceptance length；思考题：为什么提取式负载的 ngram accept 高、闲聊负载低?

### 第 22 页：结课与预告

- **本讲你应能**：按负载选方法写配置；实测三口径；辨认「六法含 Medusa」类过期说法
- **预告**：模型算得快了、调度排顺了——第 8 讲处理「模型本身太大」：量化(LLM Compressor / FP8 / Marlin)
- **下节材料**：第 8 讲 syllabus 阅读清单

---

## 图资产清单（自制）

| 图 | 页 | 说明 |
| --- | --- | --- |
| 爬楼梯 vs 跳格子 | 1 | 开场 |
| 目录迁移 + 类层级 | 4 | Proposer→Speculator |
| 一步流水时序图 | 12 | 锚点图；target 展开/采样即验证/draft K 步 |
| 错讲法 vs 事实对比框 | 12 | 红色 |
| 五法速查大表 | 11 | 绿色；配片段 |
| 调度种子发芽图 | 14 | 引用第 2 讲 |
| 三口径对照卡 | 17 | 绿色；含 bonus 与否标注 |
| 选型流程图 | 19 | 三问 → 查表 → 配置 |
