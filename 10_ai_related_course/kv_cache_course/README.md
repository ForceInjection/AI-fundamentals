# KV Cache 八讲：从 token 账本到生产引擎

> 8 讲 × 120 分钟 | 2026-10

KV Cache 是推理系统里随上下文和并发增长的那部分显存开销。模型权重固定，KV 却按 token 数和请求数线性增长。上下文从 8K 拉到 128K，或者并发从 8 加到 64，先撞到显存上限的往往是 KV，不是权重。

这门课用八讲把这件事拆开：先算一个 token 的账，再看它在引擎里怎么放、怎么复用，最后用真实模型做容量规划。

**适合谁**：有 Python / PyTorch 基础的工程师。做推理、平台、SRE、算法都可以来。不需要 CUDA、Kubernetes 经验；第 1 讲从 QKV 讲起，不要求提前掌握注意力细节。

| 项目   | 内容                                                      |
| ------ | --------------------------------------------------------- |
| 讲次   | 8 讲，每讲 120 分钟                                       |
| 形式   | 概念讲解 + 源码/演示 + 课堂实验 + 答疑                    |
| 语言   | 中文                                                      |
| 前置   | 矩阵乘法、PyTorch 基础；第 4 讲前了解 vLLM 的基本请求流程 |
| 不需要 | CUDA、Kubernetes；第 1–2 讲实验可纯 CPU 完成              |

## 一、为什么要学

推理成本里，权重只付一次，KV 每来一个请求、每多一个 token 都要付。不会算这笔账，调参只能靠试：`max_model_len` 设多少、batch 开多大、prefix cache 值不值得开，都没有依据。

学完这门课，你能做四件事：

1. 拿到模型 config 和部署参数，算出每 token KV、单请求总量和可用并发，不用等压测。
2. 在 nano-vllm、vLLM、SGLang 里找到 KV 的分配、复用、驱逐和换出路径，看懂 `file:line` 背后的对象关系。
3. 判断混合注意力模型的瓶颈：GLM-5.3-Flash 的 KDA state 和 token 池谁先满；DeepSeek-V4.1-Flash 的 SWA KV 和 global KV 怎么分工。
4. 给 Agent 长会话写容量规划和引擎选型说明。PD 分离、offload、prefix cache 是否值得，用测到的数说话。

## 二、课程是什么

课程只围绕一个问题：KV Cache 存了什么、放在哪、怎么复用、什么时候不够用。

八讲可以分成四块：

- 第 1–2 讲，账本与布局：一个 token 占多少 KV，token 怎么落到 block、page、slot。
- 第 3–5 讲，引擎与复用：nano-vllm 的最小实现、vLLM 的调度回收、radix tree 与 HiCache。
- 第 6–7 讲，真实模型：GLM-5.3-Flash 的三本账、DeepSeek-V4.1-Flash 的压缩与 SWA。
- 第 8 讲，容量与选型：两层账本合流，给 Agent 长会话做规划。

### 八讲一览

| #   | 主题                                           | 你会学到什么                                                                                                                                                             | 材料                                                                                      |
| --- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| 1   | **KV Cache 为什么存在**                        | Q/K/V 在注意力里做什么；prefill 与 decode 的计算差异；每 token KV 公式；MHA / GQA / MQA / MLA 对缓存的影响；TTFT、TPOT 与吞吐                                            | [导读](01-kv-cache-basics/README.md) · [安排](01-kv-cache-basics/syllabus.md)             |
| 2   | **KV Cache 怎么放**                            | token、block/page、slot、block table、slot mapping；两种 KV 池形状；reservation / internal / external 三种浪费；prefix 共享与 `ref_count`；`block_size` 取舍             | [导读](02-kv-cache-layout/README.md) · [安排](02-kv-cache-layout/syllabus.md)             |
| 3   | **KV Cache 在 nano-vllm 里怎么实现**           | 沿 `LLMEngine.step()` 走完分配、写入、前缀命中、抢占、换出、释放；前缀哈希链与 `ref_count`；RECOMPUTE 与 SWAP 两条抢占路径                                               | [导读](03-nanovllm-kv-cache/README.md) · [安排](03-nanovllm-kv-cache/syllabus.md)         |
| 4   | **KV Cache 在 vLLM 里怎么调度与回收**          | `KVCacheManager`、`KVCacheCoordinator`、`SingleTypeKVCacheManager`、`BlockPool` 的分工；prefix cache 查询与驱逐；recompute 抢占与 CPU offload 的区别                     | [导读](04-vllm-kv-cache/README.md) · [安排](04-vllm-kv-cache/syllabus.md)                 |
| 5   | **KV Cache 怎么复用与分层**                    | `UnifiedRadixCache` / `UnifiedTreeCore` 对象图；`RadixKey` 命中条件与 split；三套引用语义；四状态与驱逐；HiCache 三层；`host_value` 与 `HostPoolGroup` | [导读](05-kv-cache-reuse/README.md) · [安排](05-kv-cache-reuse/syllabus.md) |
| 6   | **KV Cache 在 GLM-5.3-Flash 里怎么变成三本账** | 45 层文本塔里 34 层 KDA + 11 层 full 的构成；token KV、indexer、KDA state 三本账的单位；7,260 B/token 与 147.6 MB/state slot 的复算；先爆哪本账                          | [导读](06-glm53-flash-kv-cache/README.md) · [安排](06-glm53-flash-kv-cache/syllabus.md)   |
| 7   | **KV Cache 在 DeepSeek-V4.1-Flash 里怎么压缩** | global KV 890 B/token 的逐项算式与成立条件；SWA 2.7 MB/request；FP4 路径；prefix caching 与 bounded replay 的边界                                                        | [导读](07-deepseek-v41-kv-cache/README.md) · [安排](07-deepseek-v41-kv-cache/syllabus.md) |
| 8   | **KV Cache 怎么算容量、怎么选引擎**            | token 池与 request state 池两层账本；SGLang / vLLM / nano-vllm 的参数映射；PD 分离、offload、prefix cache 的判断依据；一页容量规划                                       | [导读](08-kv-capacity-planning/README.md) · [安排](08-kv-capacity-planning/syllabus.md)   |

### 怎么学习

按 1 → 8 的顺序学。第 1–2 讲是后面所有账本的基础；第 3–4 讲对照源码；第 5 讲把复用和分层串起来；第 6–7 讲可以按兴趣选读，但第 8 讲会同时用到；第 8 讲是结课。

每讲 120 分钟，通常这样分配：

| 讲次      | 概念   | 源码/演示 | 实验   | 答疑   |
| --------- | ------ | --------- | ------ | ------ |
| 第 1–2 讲 | 30 min | 45 min    | 35 min | 10 min |
| 第 3–4 讲 | 30 min | 50 min    | 30 min | 10 min |
| 第 5–8 讲 | 30 min | 40 min    | 40 min | 10 min |

第 5–8 讲的源码锚点建议课前预读，课堂时间留给实验和讨论。

每讲的闭环是：读本章 README → 看 PPT 提纲或 PPT → 跑课堂实验 → 做课后作业 → 用自己的模型复算一遍。只看不跑，容量数字很容易算错。

每讲目录里的材料：

| 文件                   | 用途                                |
| ---------------------- | ----------------------------------- |
| `README.md`            | 本章导读、学习目标、产出与前后衔接  |
| `syllabus.md`          | 120 分钟安排、重点难点、参考材料    |
| `ppt-outline.md`       | 页面提纲；第 6–8 讲按需生成 PPT     |
| `*.pptx`               | 第 1–5 讲本地课件（不入库），统一命名 `第NN讲-主题.pptx` |
| `hands-on-exercise.md` | 课堂实验，30–40 分钟                |
| `homework.md`          | 课后作业                            |
| `lab-environment.md`   | 本讲环境说明；共用部分指向根目录    |

## 三、实验与作业

### 实验要求

- 每讲一个课堂实验，30–40 分钟，随讲完成。
- 第 1–2 讲只依赖 Python 3.10+，纯 CPU，不需要 GPU 和模型权重。
- 第 3 讲起需要本地源码：第 3 讲用 nano-vllm，第 4 讲用 vLLM，第 5–8 讲按模型选 SGLang 或 vLLM；三个仓库各钉一个 commit，版本和安装命令见[共用实验环境](lab-environment.md)。
- GPU 按讲次选择：读源码和跑纯 Python 演示不需要 GPU；端到端推理实验才需要。第 1、2 讲不要求安装 `flash-attn`。
- 提交物：能复现的命令、脚本输出或截图，以及一句结论。每讲的 `hands-on-exercise.md` 写了具体产出。

| 讲次 | 实验内容                                                                                                                              | 产出                         |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| 1    | 用 [`kv_estimator.py`](01-kv-cache-basics/code/kv_estimator.py) 算 Qwen3-0.6B、Qwen2.5-7B、Qwen2.5-72B 的每 token KV 和总占用         | 一页 KV 成本模型             |
| 2    | 用 [`block_table_demo.py`](02-kv-cache-layout/code/block_table_demo.py) 走 10-token 映射、共享前缀、A/B 碎片对照                      | 一张 token → block → slot 图 |
| 3    | 跑 nano-vllm 的 L03 调度模拟，核对 E / E2，填一次请求从分配到抢占、换出的状态表                                                       | 一张生命周期时序表 + E/E2 对照表 |
| 4    | 用 [`parse_stats_log.py`](04-vllm-kv-cache/code/parse_stats_log.py) 读 vLLM V1 的 block table 与 prefix cache 统计，和 nano-vllm 对照 | 一张生命周期映射表           |
| 5    | 用 [`prefix_cache_sim.py`](05-kv-cache-reuse/code/prefix_cache_sim.py) 走命中、加锁、驱逐；有 GPU 时看 HiCache 三层命中               | 一张统一观察表               |
| 6    | 用 [`calc_glm53_kv.py`](06-glm53-flash-kv-cache/code/calc_glm53_kv.py) 复算三本账，判断先爆哪本                                       | 一张三本账容量表             |
| 7    | 用 [`kv_account.py`](07-deepseek-v41-kv-cache/code/kv_account.py) 复算 890 B/token，做一次 H = 32K 的 replay 算例                     | 一张分层表                   |
| 8    | 用 [`capacity_plan.py`](08-kv-capacity-planning/code/capacity_plan.py) 给一个 Agent 长会话负载做容量规划和引擎选型                    | 一页容量规划 + 一页选型理由  |

### 课后作业

每讲有 `homework.md`，以计算题、源码阅读题和开放题为主。建议做完当讲作业再进入下一讲。

三门计入考核的作业：

| 作业   | 落点    | 内容                                                                 | 计入                    |
| ------ | ------- | -------------------------------------------------------------------- | ----------------------- |
| 作业 1 | 第 1 讲 | 写一个通用 KV 估算器：Qwen3-0.6B / Qwen2.5-7B，MHA / GQA，FP16 / FP8 | 源码笔记 30% 的基础部分 |
| 作业 2 | 第 3 讲 | 给 nano-vllm 加 KV 观察点或一个简单驱逐策略                          | 源码笔记 30% 的源码部分 |
| 作业 3 | 第 8 讲 | Agent 长会话容量规划与引擎选型，附一页决策备忘                       | 结课作业 40%            |

验收方式：

| 项目     | 权重 | 形式                                      |
| -------- | ---- | ----------------------------------------- |
| 课堂实验 | 30%  | 每讲提交脚本输出、截图或结论              |
| 源码笔记 | 30%  | 作业 1 + 作业 2，写清类、池、树节点和参数 |
| 结课作业 | 40%  | 作业 3，一页决策备忘必做                  |

自学不必按权重交作业；把这三份产出写出来，能力目标就达到了。

## 四、总结

学完八讲，你应该能独立回答：

1. 给定模型 config 和部署参数，算出 token 池账、request state 池账，以及 page/slot 对齐后的容量。
2. 看懂 nano-vllm、vLLM、SGLang 里 KV Cache 的分配、复用、驱逐和换出路径。
3. 判断 GLM-5.3-Flash 的 KDA 子池与 KV 子池谁先成为瓶颈，以及 DeepSeek-V4.1-Flash 的 SWA KV 与 global KV 谁先成为瓶颈。
4. 为 Agent 长会话负载写一份容量规划与引擎选型说明。

课程边界：

- 这门课不替代 vLLM 入门。vLLM 的调度、PagedAttention 和 KV 优化基础，建议先看 `ai-infra-course` 模块 5/6；这门课把时间留给 KV 的结构、源码和容量判断。
- 不做论文串讲。每个模型只读与 KV 相关的类、池、树节点和启动参数，论文用来补背景。
- 以本地代码为准。nano-vllm、vLLM、SGLang 各钉一个 commit，课上的 `file:line` 都按这些版本讲；换版本要重新核对行号。

开始学习：

- [第 1 讲：KV Cache 为什么存在](01-kv-cache-basics/README.md)
- [课程总纲](syllabus.md)
- [术语表](glossary.md)

## 附：课程资料与参考

### 共享文件

- [课程总纲](syllabus.md)
- [共用实验环境](lab-environment.md)
- [术语表](glossary.md)
- [可复用配图](visuals.md)

### 源码与材料基线

| 材料            | 版本 / 路径                                         | 用途                                               |
| --------------- | --------------------------------------------------- | -------------------------------------------------- |
| nano-vllm       | 本地 `~/Project/study/nano-vllm`，HEAD `b9dd55a`    | 第 2–3 讲实验；第 4 讲用 `BlockManager` 做对照阅读 |
| vLLM            | 本地 `~/Project/ai-infra/vLLM`，HEAD `8b365ff949`   | 第 4、7 讲源码                                     |
| SGLang          | 本地 `~/Project/ai-infra/sglang`，HEAD `f4de6abee6` | 第 5、6 讲源码                                     |
| AI-fundamentals | 本仓库 `09_inference_system/`                       | 概念、GLM-5.3-Flash、DeepSeek-V4.1-Flash 预读      |
| ai-infra-course | 本地 `~/Project/study/ai-infra-course`              | 模块 5/6 作为前置；复用公式图、计算器和 trace 脚本 |

### 口径约定

- 容量数字要同时写清：模型变体、精度、统计范围、是否单 rank、`attn_tp` 取值。
- 课程里的 `MB` 按 `10^6 B` 计，不用 MiB；换算成 MiB 时另标单位。
- `page/slot` 是 token 池的分配粒度，不是独立的容量池；DRAM 副本属于 host cache，不参与“可用并发”上限。
- 课程中出现的 `nanovllm` 只作为 Python 包名或路径；展示名统一写 `nano-vllm`。

### 参考材料

- [AI-fundamentals KV Cache 技术体系](../../09_inference_system/kv_cache/README.md)
- [三层状态：图解 GLM-5.3-Flash 的 KV Cache 布局](../../09_inference_system/sglang/glm53-flash-kv-layout.md)
- [DeepSeek-V4.1-Flash 技术报告精读](../../09_inference_system/kv_compression/02-deepseek-v41-flash.md)
- [nano-vllm](https://github.com/ForceInjection/nano-vllm)
- [vLLM DeepSeek-V4.1-Flash](https://vllm-project.github.io/2026/10/07/deepseek-v41-flash.html)
- [SGLang GLM-5.3-Flash cookbook](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx)
