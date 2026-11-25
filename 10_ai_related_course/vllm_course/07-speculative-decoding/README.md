# 第 7 讲 · 投机解码

> 120 分钟 | 基线：nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`（v0.34.1rc0-84,2026-11-25 核对）

本讲回答一个核心问题：**一次生成多个 token 还不出错，收益的上限由什么决定，代价又藏在哪里？**

知识链：`--speculative-config`(method + `num_speculative_tokens`、`subfolder`(checkpoint 子目录,新))→ `init_speculator` 实例化 Speculator → 每步 `speculator.propose` 产 K 个 draft → 下一步 target 前向按 1+K 展开 → 采样即验证(RejectionSampler)→ 拒绝则回滚 → accept length 指标全程记账。

## 一、nano 已学形态（回顾——本讲是反向的）

先修课明确标注：nano-vllm **完全不涉及**投机解码。它的世界是第 1 课那个 while 循环：每 step 一次前向、一个 token。这不是疏忽——投机解码需要草稿来源、KV 预留、验证回滚三套基础设施，任何一套都足以让 1,400 行的教学实现翻倍。

生产 vLLM 为什么要它：decode 是 memory-bound（每步读全部权重与 KV，算力大量闲置），投机解码把闲置算力换成「一次前向验证多个 draft」。收益公式一句话：**每步接受 a 个 draft、开销是草稿成本 c，则加速 ≈ (1+a)/(1+c)**；上限由 acceptance length 决定，代价藏在 draft 前向与 KV 预留里。本讲就是把这个公式拆成代码。

## 二、演进：Proposer → Speculator

- V1 时代：proposer 体系（`NgramProposer`/`EagleProposer` 等在 `vllm/v1/spec_decode/`）——GPU MODE Lecture 22 讲的就是这一代。
- MRV2 时代(HEAD)：整体改名 **Speculator** 并迁移到 `vllm/v1/worker/gpu/spec_decode/`；统一接口 `BaseSpeculator.propose`（`speculator.py:74`，抽象签名 `:100-128`）；类层级：`DraftModelSpeculator`（`:165`）→ `AutoRegressiveSpeculator`（`autoregressive/speculator.py:24`，多步 draft 循环与图融合在此）→ `TargetDependentARSpeculator`（`target_dependent_ar/speculator.py:23`，EAGLE/MTP 这类吃 target 隐状态的）→ 各方法类。旧目录只剩 `metrics.py`。
- **Medusa 已移除**：method 合法值（`config/speculative.py:113-123`）、类、文档全无。仓库旧文 `speculative_decoding_landscape.md` 里的 Medusa 条目作为演进素材保留——「多头草稿」这条路线被 EAGLE3/MTP 取代。

## 三、现状走读

### 环节 1：配置解析

入口 `--speculative-config '{"method": "...", "num_speculative_tokens": K, ...}'`。method 合法值全集（`config/speculative.py:113-123`）：`ngram`、`draft_model`、`suffix`、`eagle`/`eagle3`、`mtp`（及 ~40 个模型专属别名如 `deepseek_mtp`/`qwen3_next_mtp`——**用别名会告警并归一为 `mtp`**，`:1495-1499`）、另有 `dflash`/`dspark`/`dfly` 等实验方法;`lilicorr` 为实验分支、尚未接入 config 注册。无显式 method 时按 `_CHECKPOINT_DETECTION_ORDER` 从 draft checkpoint 的 `model_type` 自动检测（`speculator_config.py:492`）；EAGLE 靠名字含 `eagle-`，MTP 靠 `model_type ∈ MTPModelTypes`。关键字段：`num_speculative_tokens`、`subfolder`(checkpoint 子目录,新)（`:447`）、`model`（draft 权重路径）、`rejection_sample_method`（`:604`）、`draft_sample_method`（`:685`，默认 greedy）、`enable_adaptive_verification`（`:627`）、动态 SD（`:564`）。

### 环节 2：五法在职（选型速查）

| 方法         | 需额外模型               | 需训练介入                | 典型场景                                         | 配置片段                                                                        |
| ------------ | ------------------------ | ------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------- |
| ngram        | 否                       | 否                        | 提取式/重复任务，零成本兜底                      | `{"method":"ngram","num_speculative_tokens":5}`                                 |
| suffix       | 否                       | 否                        | 代码编辑/agentic 循环（**跨请求复用语料**）      | `{"method":"suffix","num_speculative_tokens":8}`                                |
| EAGLE/EAGLE3 | 是（eagle 权重）         | 是                        | 通用低延迟首选                                   | `{"method":"eagle","model":"<eagle 权重>","num_speculative_tokens":2}`          |
| `draft_model`  | 是（小 LM）              | 否                        | 有现成小模型                                     | `{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5}` |
| MTP          | 否（target 自带 MTP 层） | 需官方发过 MTP checkpoint | DeepSeek/Qwen3-Next/GLM-MoE/Kimi 类；高 QPS 也稳 | `{"method":"mtp","num_speculative_tokens":1}`（model 可省略，draft=target）     |

各方法的实现落点：ngram 是 **GPU Triton 实现**（`ngram/speculator.py:151`，`propose` 用两个 Triton kernel 扫全 token 历史；`ngram_gpu` 是别名）；suffix 在 ngram 之上加了变长后缀匹配与**跨请求语料**（`suffix/speculator.py:501`，已完成请求的输出进 GPU corpus 按频率选续写）；EAGLE 系消费 target 最后一层隐状态（`eagle/speculator.py:12`）；MTP 加载 target checkpoint 自带的 MTP 层（`mtp/speculator.py:12`，`default_model` 直接取 target 路径并继承量化配置）。

### 环节 3：验证循环——一步到底过几次模型

这是本讲最容易讲错的地方，按 HEAD 代码说：`EngineCore.step`（`core.py:701-731`）的一次 step 内发生三件事——

1. **target 前向 1 次**：每请求的 query 长度被展开为 `1 bonus + K draft`（`input_batch.py:475 combine_sampled_and_draft_tokens`）；
2. **采样即验证**：没有独立的 verify 调用——`RejectionSampler.__call__`（`rejection_sampler.py:152`，核心 kernel `rejection_sampler_utils.py:1128`）在采样路径里完成拒绝/接受；greedy draft（默认）退化为 target-argmax 匹配；
3. **draft 前向 K 步**：`speculator.propose`（`model_runner.py:2436-2465`）驱动 draft 模型跑 K 步，产出的 draft id 写回 `req_states.draft_tokens`（`:2464-2473`），供下一步 target 前向展开。

所以是**流水**而非串行：step N 的 propose 产物在 step N+1 的 target 前向里被验证。与第 5 讲的接口：draft 的多步解码整体融合进一张 FULL 图（`FULL_DECODE_ONLY`，PIECEWISE 不支持 draft）；异步调度下用 `num_output_placeholders` 占位（第 2 讲），拒绝时回滚（`scheduler.py:2454-2462`）。

### 环节 4：调度交互

第 2 讲埋的两颗种子在这里发芽：`draft_slots`（`scheduler.py:728`）为 draft 的 KV 写入预留预算（基类 0、`draft_model` 1、EAGLE 系按 `num_lookahead_tokens`）；lookahead scheduling 按 `1 + K + lookahead` 预留块（`:385`）。新增两个优化：`can_skip_drafts`（`:348`）高并发时发「仅 target」步跳过 draft 前向；动态 SD（`num_speculative_tokens_per_batch_size`，`:564`）按 batch size 调 K。抢占时清 draft 占位与 stale 机制（第 2 讲已铺）。

### 环节 5：观测指标

- **日志口径**：`SpecDecodingLogging`（`vllm/v1/spec_decode/metrics.py:64`）周期打印 `Mean acceptance length: %.2f` 与 per-position acceptance rate——**均值含 bonus token**（= 1 + accepted/drafts，`:130`），per-request 直方图不含，两套口径别混（`acceptance_metrics.md` 有对账表）。
- **Prometheus**：`vllm:spec_decode_num_accepted_tokens_total` 等计数器（`metrics.py:195`，docstring 直接给三条 PromQL）。
- **per-request**：`--per-request-spec-decode-metrics summary` 后响应挂 `metrics.speculative_decoding`（`observability.py:52`）。
- **bench 口径**：没有 `vllm bench spec` 子命令；用 `--dataset-name spec_bench` + serve.py 自动抓 `/metrics` 的 `vllm:spec_decode*` 写进结果（`benchmarks/serve.py:1227-1236`）。

## 四、V0/旧教程 vs HEAD

| 旧说法                                        | HEAD 事实                                                                    |
| --------------------------------------------- | ---------------------------------------------------------------------------- |
| 「六法：ngram/suffix/Medusa/EAGLE/draft/MTP」 | Medusa 已移除；在职五法 + 11 个实验方法；draft 的合法名是 `draft_model`      |
| 「Proposer 体系在 `vllm/v1/spec_decode/`」    | 改名 Speculator 并迁到 `vllm/v1/worker/gpu/spec_decode/`；旧目录只剩 metrics |
| 「draft K 个 → 独立 verify 一步」             | 验证融合在 Sampler 路径；一步 = target 前向 1 次（展开 1+K）+ draft K 步     |
| 「ngram 是 Python/CPU 实现」                  | GPU Triton kernel；`ngram_gpu` 只是别名                                      |
| 「`--speculative-model` 选草稿」              | 旧旗标已废弃；用 `--speculative-config` JSON                                 |
| 「vllm bench 有 spec 子命令」                 | 没有；`spec_bench` 数据集 + Prometheus 抓取                                    |

## 五、实验与产出

见 [hands-on-exercise.md](hands-on-exercise.md)：实验 1 ngram 开关对照（零额外模型）；实验 2 accept 指标三口径读取；实验 3（可选）动态 SD/自适应验证。课后作业见 [homework.md](homework.md)。

## 六、边界

| 内容                             | 本讲                   | 去处                                                |
| -------------------------------- | ---------------------- | --------------------------------------------------- |
| 各方法的训练侧（speculators 库） | 只给链接               | `docs/features/speculative_decoding/speculators.md` |
| rejection sampling 的数学证明    | 只讲机制（概率比接受） | 论文/教材                                           |
| draft 图融合细节                 | 第 5 讲已铺，引用      | —                                                   |

## 七、前后衔接

第 2 讲的 `draft_slots` 与占位符、第 5 讲的 `FULL_DECODE_ONLY`、本讲新引入的 lookahead KV 预留，在本讲汇合。第 8 讲的量化可与 draft 模型组合（draft 小模型量化收益更直接）。
