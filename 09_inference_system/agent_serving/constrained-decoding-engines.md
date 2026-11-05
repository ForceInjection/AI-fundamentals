# 约束解码的性能账单：vLLM 与 SGLang 的结构化输出实现拆解

> Agent 的每一次工具调用都要求模型输出合法 JSON。结构化输出（structured output）用「schema 编译成 FSM，每步采样前屏蔽非法 token」保证这一点，xgrammar 论文把它做到「near-zero overhead」。本文不讨论原理本身，只回答一个工程问题：这套机制在 vLLM 和 SGLang 里怎么落地，开销发生在哪几步，哪些坑在生产里会撞上。
>
> 2026-09 | 基于 vLLM（`43d691ec6b`，2026-08-07）与 SGLang（`f7101b0ae6`，2026-08-18）源码验证
>
> **性质说明**：机制与行为均经源码验证；涉及耗时/字节数的数字为量级示意，实测项见文末（拟基于 [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) 压测方法执行）。

---

## 一、从可选到必选

Chat 时代，结构化输出是个锦上添花的功能：输出格式歪了，人看得懂就行。Agent 时代它变成硬约束：一次工具调用要过 JSON 解析器，输出缺个引号，工具执行失败，Agent 重试，流量翻倍。AgentSysBench 的测量显示工具定义与系统指令最高占上下文窗口的 84.3%（见 [Agent 负载一篇](agent-workload-serving.md)），与之对应，输出侧的合法性靠的就是本文的主角：约束解码（constrained decoding）。

两引擎的默认后端都是 xgrammar（SGLang 的 `grammar_backend` 缺省 None，`_handle_grammar_backend` 将其解析为 `"xgrammar"`；vLLM 的 V1 结构化输出以 xgrammar 为首选后端，另备 outlines/guidance/lm_format_enforcer）。下文所有机制均以 xgrammar 后端为准，版本号见文首。

## 二、三笔账单

### 2.1 编译账（落在 TTFT 上）

请求到达时，引擎要把 JSON schema（或正则、EBNF）编译成可执行的语法对象。编译发生在请求生命周期内：vLLM 在 `grammar_init`（`v1/structured_output/__init__.py:114`）触发 `backend.compile_grammar`（`backend_xgrammar.py:78`）；SGLang 在 `GrammarManager.process_req_with_grammar`（`constrained/grammar_manager.py:131`）里初始化。

编译不是免费的。schema 越大越复杂，编译耗时越长（大 schema 数十到数百毫秒，量级示意）。冷启动的服务收到第一批请求时，这层延迟直接加在 TTFT 上。两引擎都用「编译缓存」缓解：同样的 schema 第二次请求直接命中。缓存键就是 SGLang 的 grammar key 与 vLLM 的 `structured_output_key`（`request.py:82`，请求类型 + 语法规格）。对 Agent 场景这是个好消息：工具 schema 在部署期基本不变，编译缓存命中率高。

### 2.2 每步账（落在吞吐上）

decode 的每一步，引擎要为当前 batch 生成 token bitmask，形状约 `[max_num_seqs, vocab_size/32]` 的 int32。以 Qwen2.5 的 15 万级词表算，每个序列约 19KB，batch 64 时一步约 1.2MB（量级示意）。开销分四段：分配、CPU 填充、H2D 拷贝、logits 屏蔽。

两引擎都做了短路：vLLM 调度器用 `has_structured_output_requests` 标志跳过无约束 batch 的整段逻辑（`v1/core/sched/scheduler.py:1347`），且 prefill chunk 不参与（`not request.is_prefill_chunk`）；SGLang 只对带 grammar 的请求填充。真正要留意的是混合场景：batch 里只要有一个约束请求，那一步就要走完整的分配-填充-拷贝链路。

### 2.3 交互账（组合税）

约束解码不单独存在，和三个热门特性叠加时各有故事：投机解码要在验证后回滚 FSM 状态（SGLang 的 `rollback(k)`，`xgrammar_backend.py:107`）；reasoning 模型要先自由思考、到输出段才施加约束（vLLM 的 `trim_reasoning_for_advance`，`__init__.py:462`；SGLang 有 `ReasonerGrammarBackend` 和 thinking budget 联动，`grammar_manager.py:124`）；PD/PP 分离下编译状态要跨进程同步（SGLang 的 `_drain_pp_sync_work`，`grammar_manager.py:72`）。

## 三、双引擎实现

### 3.1 vLLM：单例管理器 + 请求状态机

vLLM V1 的结构化输出由 `StructuredOutputManager`（`__init__.py:35`）单例驱动，每个请求挂一个 `StructuredOutputRequest`（`request.py:22`）状态机。流程：

1. 请求进入时 `grammar_init` 触发编译，状态机用 `is_grammar_ready`（`request.py:62`）暴露就绪位，编译缓存键 `structured_output_key` 命中时立即可用；
2. decode 循环中调度器调用 `get_grammar_bitmask`（`scheduler.py:1655`），Manager 的 `grammar_bitmask`（`__init__.py:212`）汇总各请求的 mask；
3. 填充是异步的：`_async_submit_fill_bitmask`（:207）把 CPU 侧填充提交出去，与 GPU 前向重叠；
4. `should_fill_bitmask`（:361）与 `should_advance`（:381）分别门控「这步要不要屏蔽」「FSM 要不要推进」。

reasoning 集成的位置很讲究：`_find_reasoning_end_index`（:442）先定位思考段结束点，`</think>` 之前的 token 不进 FSM，思考结束后约束才接管。

### 3.2 SGLang：语法队列 + 跳出重分词

SGLang 把这套逻辑做成显式的 `GrammarManager`（`grammar_manager.py:26`），最大的特点是**把「编译未完成」当作调度层的头等状态**：

- 语法没就绪的请求进 waiting grammars 队列（`has_waiting_grammars`，:69），不进 batch，避免让已排队的请求空等；
- 编译走 future 模式（`get_cached_or_future_value`，经 `process_req_with_grammar` :153 使用）：首次提交异步编译，第二次同 schema 请求直接命中缓存；编译失败的 schema 也会被缓存（:164），防止反复撞墙；
- 会话结束后 `set_cache`（:286）把语法回写缓存。

跳出优化（jump-forward）是 SGLang 的独门：`try_jump_forward`（`xgrammar_backend.py:164`）探测 FSM 当前状态下的确定性字符串段，`jump_and_retokenize`（:174）把这整段一次生成、按原文重分词后同步 FSM，跳过逐 token 的 mask 与步进。outlines 后端有对应的 `outlines_jump_forward.py`。vLLM 侧只有一处注释提到 `find_jump_forward_string` 的文档链接（`backend_xgrammar.py:141`），实现未见。

### 3.3 逐项对照

| 维度           | vLLM                                                | SGLang                                                        |
| -------------- | --------------------------------------------------- | ------------------------------------------------------------- |
| 后端集合       | xgrammar / outlines / guidance / lm_format_enforcer | xgrammar / outlines / llguidance / none                       |
| 默认           | xgrammar 优先                                       | None → xgrammar（`server_args.py` `_handle_grammar_backend`） |
| 编译缓存键     | `structured_output_key`（request 级）               | grammar key（manager 级，含无效缓存）                         |
| 未就绪处理     | 请求状态位门控（`is_grammar_ready`）                | 显式 waiting 队列，不进 batch                                 |
| bitmask 填充   | 异步提交（`_async_submit_fill_bitmask`）            | 按请求填充                                                    |
| 跳出重分词     | 无实现（仅注释）                                    | `try_jump_forward` + `jump_and_retokenize`                    |
| 投机解码       | 经 `should_advance` 状态推进                        | `rollback(k)` 显式回滚                                        |
| reasoning 集成 | `trim_reasoning_for_advance`                        | `ReasonerGrammarBackend` + thinking budget                    |
| PP/PD 同步     | —                                                   | `_drain_pp_sync_work`                                         |

## 四、三个权衡

**编译等待 vs 排队**。编译期间请求怎么办？SGLang 选择显式排队，代价是队头编译慢时后面整队等待，好处是 batch 里永远不会有一个「语法没准备好」的请求白白占位。vLLM 的状态位门控粒度更细，但应用侧要自己面对「首 token 到底什么时候来」的方差。两者没有对错，监控指标不同：前者看队列深度，后者看 ready 位翻转时间。

**跳出重分词的收益与风险**。JSON 输出里有大量确定性片段（键名、引号、括号），jump-forward 把它们一次吐出，省掉的是逐 token 的 mask 计算和 FSM 步进。代价是重分词：跳出的字符串要按分词器重新切，切分边界若与 FSM 期望不一致，需要回退处理。这正是 XGrammar 论文（arXiv:2411.15100）里 adaptive token mask 思想的工程延续，SGLang 把它接进了调度循环。

**全量 vs 惰性**。mask 的开销与 batch 大小成正比，两引擎都选择「只对约束请求填充」，但短路粒度不同：vLLM 在调度器入口用 `has_structured_output_requests` 一次短路整个 batch，SGLang 按请求逐个判断。对全无约束的纯 Chat 流量，两者开销都趋近于零；对全约束的 Agent 流量，这层优化帮不了你，账单得照付。

## 五、待实测项与生产建议

实测（拟基于 [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) 开源压测方法执行，跑完补数）：

1. 冷/热编译缓存的首 token 延迟差（同一大 schema，服务重启 vs 连续请求）；
2. 约束开/关的吞吐差（Qwen2.5-7B，同一 trace）；
3. 混布半径：batch 64 中仅 1 个约束请求时，其余 63 个请求的 TPOT 损失。

生产建议（源码现状推论，未经实测）：

- **schema 复用优先**：工具 schema 部署期固定，编译缓存命中率天然高；避免在运行期拼接动态 schema（比如把用户 ID 拼进 enum），那会打穿缓存；
- **监控编译队列**：SGLang 看 waiting grammars 深度，vLLM 看 ready 位翻转时长；队列堆积说明 schema 太复杂或并发太高；
- **大 schema 拆解**：枚举值越全、嵌套越深，编译越慢；把不稳定部分移到输出后校验，比硬塞进 schema 便宜。

值得一提的是，XGrammar 团队已经把「Agentic 场景的动态结构化生成」做成了独立方向（XGrammar-2，ACM 2026）——约束解码正在从「一个功能」长成「一个子系统」，值得持续观察。

---

## 源文件索引

| 文件                                                | 关键符号                                                                                                                 | 引用点                              |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| `vllm/v1/structured_output/__init__.py`             | `StructuredOutputManager`、`grammar_init`、`grammar_bitmask`、`_async_submit_fill_bitmask`、`trim_reasoning_for_advance` | :35、:114、:212、:207、:462         |
| `vllm/v1/structured_output/request.py`              | `StructuredOutputRequest`、`is_grammar_ready`、`structured_output_key`                                                   | :22、:62、:82                       |
| `vllm/v1/structured_output/backend_xgrammar.py`     | `compile_grammar`、`allocate_token_bitmask`                                                                              | :78、:128（jump-forward 注释 :141） |
| `vllm/sampling_params.py`                           | `StructuredOutputsParams`                                                                                                | :72                                 |
| `vllm/v1/core/sched/scheduler.py`                   | `get_grammar_bitmask`、`has_structured_output_requests`                                                                  | :1655、:1347                        |
| `python/sglang/srt/constrained/grammar_manager.py`  | `GrammarManager`、`has_waiting_grammars`、`process_req_with_grammar`、`set_cache`                                        | :26、:69、:131、:286                |
| `python/sglang/srt/constrained/xgrammar_backend.py` | `XGrammarGrammarBackend`、`try_jump_forward`、`jump_and_retokenize`、`rollback`                                          | :208、:164、:174、:107              |
| `python/sglang/srt/server_args.py`                  | `GRAMMAR_BACKEND_CHOICES`、`grammar_backend`、`_handle_grammar_backend`                                                  | :245、:1734                         |

_版本：vLLM `43d691ec6b`（2026-08-07）、SGLang `f7101b0ae6`（2026-08-18）。bitmask 字节数与编译耗时为量级示意。_

## 参考资料

- [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100), arXiv:2411.15100——本文默认后端的原理出处
- [XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs](https://dl.acm.org/doi/10.1145/3786335.3813124), ACM 2026——面向 Agent 动态 schema 的后续工作
- [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)——开源实现
