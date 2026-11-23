# 从 TTFT 到 goodput：官方基准工具的口径与六个陷阱

> **元信息**：vLLM 核对基线 `1acec5c009`（vllm/benchmarks/），SGLang 核对基线 `f4de6abee6`（python/sglang/benchmark/ 与 benchmark/agentic-rollout/）。2026-11-23 静态源码核对，本文不含实测数字；文中 `vllm:` 前缀指 vLLM 仓库的 benchmarks 目录，`sglang:` 前缀指 SGLang 仓库，引用路径相对各自仓库根。第三方压测工具（AIPerf、GuideLLM、genai-perf）不在本文范围。

一份压测报告：P99 TTFT 120 毫秒，吞吐 8000 token/秒，结论是配置可以上线。上线之后，用户投诉首字要等三秒。引擎没变，配置没变，变的是压测时的假设：那时候 prefix cache 里躺着 warmup 请求留下的前缀，客户端没有并发上限，流式一个不落。数字是工具口径的函数，口径搞错，结论跟着错。

本文拆两份官方基准工具的源码：vLLM 的 `vllm/benchmarks/`（入口 serve.py）与 SGLang 的 `python/sglang/benchmark/serving.py`，对照它们对同一组指标的定义差异；再看 SGLang 的 agentic-rollout，一个为 Agent 负载造的会话仿真器，它回应了[总纲第五节](../../agent_serving/01_agent-workload-serving.md)留下的三个度量缺口。最后把六个容易测错的地方逐一摆到源码行上。

## 一、两份官方工具的口径

### 1.1 文件结构：入口与计时是分开的

vLLM 侧，serve.py（2478 行）只做 CLI、编排与指标聚合，HTTP 请求的计时逻辑在 `vllm/benchmarks/lib/endpoint_request_func.py`。SGLang 侧，仓库根的 `python/sglang/bench_serving.py` 只是一个 22 行的兼容薄壳，`from sglang.benchmark.serving import *` 加一条弃用警告，真身在 `python/sglang/benchmark/serving.py`（2807 行）。读源码先找对文件，改口径先改对地方。

### 1.2 四个核心指标的定义

两边的核心指标同名同义，计算公式一致：

| 指标 | 定义 | 源码 |
| --- | --- | --- |
| TTFT | 请求发出到首个含 choices 的 chunk | `vllm: lib/endpoint_request_func.py:250-251`；`sglang: serving.py` 同口径 |
| TPOT | `(latency − ttft) / (output_len − 1)` | `vllm: endpoint_request_func.py:287-295`；SGLang 的 output_len 优先取服务端 usage |
| ITL | 相邻两个 chunk 的时间戳差，逐 chunk 记录 | `vllm: endpoint_request_func.py:315` |
| E2EL | 最后一个 chunk 时间戳减发送时刻 | 两边一致 |

vLLM 的 BenchmarkMetrics 聚合里还有 throughput、request_goodput、client_queue_time 等扩展项（vllm: serve.py:787-811）。SGLang 多报两组 vLLM 没有的东西：max_itl_ms，以及 retokenized 系列（用 bench 端 tokenizer 对输出重分词后再算一遍 ITL，sglang: serving.py:1147-1151，仅 sglang-oai 后端且 accept_length>0 时启用）。投机解码场景下服务端 chunk 边界与真实 token 边界不一致，这组数字就是给这个场景准备的。

### 1.3 百分位：两边默认口径对不上

vLLM 的百分位完全可配：`--percentile-metrics` 不传时，生成任务解析为 `ttft,tpot,itl`（注意没有 e2el），池化任务为 `e2el`（vllm: serve.py:2273-2281）；`--metric-percentiles` 默认只有 `99`（serve.py:1876-1884）。SGLang 相反：固定输出 mean/median/std/p90/p95/p99，没有任何百分位参数可调（sglang: serving.py:1273-1298）。抄「默认输出」做对比表必然缺列错位，比较两引擎必须显式指定口径。

### 1.4 负载形状与 warmup

`--request-rate` 两边默认 inf：所有请求在 t=0 同时发出。vLLM 用 `--burstiness` 调节压力曲线：1（默认）是泊松过程（指数间隔），非 1 走 Gamma 分布（shape=burstiness，vllm: serve.py:504-513,985）；SGLang 只有泊松指数间隔（sglang: serving.py:1121），没有对应参数，压力曲线的形状本身就不可比。vLLM 另有 `--probe-request-rate`（默认 0）与 ramp-up 策略。

warmup 的默认也不一样：vLLM `--num-warmups` 默认 0，warmup 请求与主测量分离、不计指标（serve.py:931-954）；SGLang `--warmup-requests` 默认 1（serving.py:2654-2658），且只在 `SGLANG_IS_IN_CI` 或显式 `--flush-cache` 时才清缓存（serving.py:1512-1517）。流式则是三工具的默认：vLLM 四种请求函数硬编码 `"stream": True`、没有关闭选项（lib/endpoint_request_func.py:236 等），SGLang 提供 `--disable-stream`，agentic-rollout 硬编码流式。

dataset 方面 vLLM 有十三种后端（sharegpt、burstgpt、random、random-mm、random-rerank、hf、custom、custom_audio、custom_image、prefix_repetition、spec_bench、speed_bench、timed_trace，默认 random，vllm: benchmarks/datasets/datasets.py:1613-1633）；SGLang 的数据集机制独立，选型时按各自文档核对。

## 二、SLO 与 goodput

vLLM serve.py 提供 `--goodput` 参数：格式为空格分隔的多对 `KEY:VALUE`，KEY 取 `ttft`、`tpot`、`e2el`，VALUE 是毫秒数，例如 `--goodput ttft:100 tpot:20`（serve.py:1885-1896）。判定在 bench 客户端侧：对每个请求，取它的实测指标向量与 SLO 向量逐项比较，全部满足才算一次「成功的 good 请求」，最终报 `request_goodput = good_completed / duration`（serve.py:674-697,784；校验逻辑 1532-1551）。这与 vLLM 引擎内部的 goodput 调度策略是两回事：bench 侧只做事后判定，不参与调度。

这个参数把 [Token 工厂的产能框架](../../../99_misc/token_factory_talk/cluster-capacity-measurement.md)里的 SLO Goodput 定义（每秒成功且体验达标的请求产出的 output tokens）接到了官方工具上：压测时带上与产品品质承诺一致的 SLO，测出来的就不是吞吐而是可售卖产能。前提是 SLO 数值真实反映用户承诺，以及 1.4 节的负载形状设置得像真实流量（rate=inf 的满发压测测不出排队行为下的 goodput）。

## 三、Agent 负载回放：agentic-rollout

总纲第五节指出现有 benchmark 的流量形态是 Chat 的：独立请求、稳定前缀，并留下三个缺口指标：turn-resume TTFT、会话 KV 命中率、每任务 token 成本。SGLang benchmark 目录下的 agentic-rollout 是迄今最接近的一份官方回应。

### 3.1 会话仿真器

simulate.py 对每个会话跑这样的循环（README 与 spec 的原文）：prompt → generate → tool delay → append tool result → repeat，跑满配置的轮数收尾。工具等待用 `--tool-delay MIN MAX` 控制，默认 1 到 3 秒均匀采样、首轮无延迟，等待期间释放并发槽位（simulate.py:185,385-398）。这正是 Agent 流量的关键形态：会话在工具执行期间占着 KV 但不发请求。

上下文增长有两种模式。full-history 模式每轮重发全量 history（模型输出按精确 token ID 回填，合成工具结果按 token 切块追加）；ordinary 与 streaming 模式先 `POST /open_session` 开会话，之后每轮只发新增 chunk 加 session_params（simulate.py:193-230,349-354）：前者测重算路径，后者测会话路由与服务端会话状态。输入是可读合成文本，先 tokenize 再重复修剪到精确长度；一个专门的会话前缀设计切断了跨会话的前缀复用，保证命中率数字可归因。

### 3.2 服务端配置与 DP 粘性

总纲建议的三层会话状态选型，在 README 的示例命令里就是三个服务端旗标的组合：`--enable-session-radix-cache`（引擎内 radix）、`--enable-hierarchical-cache --hicache-ratio 1.6 --hicache-write-policy write_through`（外置分层）。客户端另有一个 `--disable-dp-sticky-routing`：默认会话 i 粘到 DP rank `i % dp_size`（simulate.py:409-412），这个默认把会话粘性做进了压测拓扑，关闭它就能测「无粘性时命中率掉多少」。

### 3.3 指标两层

turn 级指标落在 requests.jsonl：每轮的 ttft_s、avg_token_time_s、context_tokens、input_tokens、client_wait_s、tool_delay_s（simulate.py:88-91,120-125,194-203）。带工具等待间隔的会话轮转 TTFT，就是总纲说的 turn-resume TTFT；client_wait_s 把客户端排队单独记账。集群命中率是另一层：metrics.py 抓引擎的 Prometheus 指标 `sglang:prefill_effective_tokens_total`，按 device_hit / host_hit / storage_hit / input 四种模式做增量，输出设备命中、主机命中、总命中（含存储层）三个百分比，按 worker token 数加权（metrics.py:69-96）；另有 L1/L2 occupancy 算各层 token 占用（metrics.py:59-66）。总纲三缺口里的前两个，在这里有了官方测法。

spec/ 目录值得单独一提：每个 workload 特性有一份规格文档（feature-00-session-hicache-latency，状态 implemented）加一份验证证据（实测命令与结果），并诚实写明边界：仅合成文本、模拟延迟、无真实工具执行、无引擎改动。基准工具自己先讲清楚「测的是什么、不是什么」，这个实践比工具本身更值得抄。

## 四、六个测错点

以下每条都有源码依据，按「现象 → 证据 → 正确做法」给出。

**一、非流式下 TTFT 根本不可比。** SGLang 非流式把 TTFT 直接赋值为总延迟（sglang: serving.py:507-509，注释自认）；vLLM 非流式收不到可计时的 chunk，请求被标记失败「Never received a valid chunk to calculate TTFT」（vllm: lib/endpoint_request_func.py:308-313），serve.py:788 的注释明言「ttfts is empty if streaming is not supported by the endpoint」。同一个「TTFT」在不同流式设置下是完全不同的量。对比测试锁定流式。

**二、默认百分位缺列。** vLLM 生成任务的默认指标集不含 e2el、百分位只有 P99；SGLang 固定 p90/95/99 且不可配。只改引擎不改参数，两张报告的列都对不齐。显式传 `--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99`。

**三、ignore_eos 两边默认相反。** SGLang 默认 `payload["ignore_eos"] = True`（sglang: serving.py:470-472，注释写着「Default to False for more realistic behavior」，实际取反）；vLLM 需要显式 `--ignore-eos`（serve.py:1849-1854）。同一个 max_output_tokens 下，实际输出长度分布一个截尾一个自然终止，TPOT 与吞吐全被扭曲。跨引擎对比时两边显式设成同一个值。

**四、缓存污染默认不设防。** SGLang 默认跑一条 warmup，flush 只在 CI 环境变量或显式 `--flush-cache` 时执行——源码注释写明 flush 的目的是「让测量不吃 warmup 留下的请求局部前缀复用」，并注明 vLLM 用另一个 development-mode 端点做同样的事（sglang: serving.py:1510-1517）；vLLM 侧 `--num-warmups` 默认 0，bench 也不主动调 flush。两边的默认都可能带着上一轮压测残留的前缀缓存测 TTFT。报告里写明 warmup 条数与 flush 状态；要干净基线就在 SGLang 侧显式 `--flush-cache`，或者接受缓存命中并把它作为口径的一部分声明出来。

**五、token 计数双轨。** vLLM 优先采信服务端 usage 的 completion_tokens，缺失时退回 bench 端重分词，源码自注「this may inflate the output token count slightly」（vllm: serve.py:643-656）；SGLang 同时报服务端计数与 retokenized 计数两套（serving.py:1264-1272）。tokenizer 与模型不匹配时输入长度也会失真；agentic-rollout 干脆把 prompt_tokens 不等当作错误抛出（simulate.py:112-113）。报告里注明 token 口径来源；压测前核对 tokenizer。

**六、客户端排队混进 TTFT。** 开了 `--max-concurrency` 之后，请求在客户端信号量上等待的时间会混进 TTFT 与 E2EL。vLLM 的对策是新增 `client_queue_time` 与 `e2el_including_client_queue` 两个口径（--percentile-metrics 的可选值，vllm: serve.py:1869-1872），把信号量等待从首字延迟里拆出来；agentic-rollout 单独记 client_wait_s（simulate.py:217-221）。拿开了并发上限的压测直接报 TTFT 而不声明口径，是最常见的测错方式。

## 五、总评：一套可执行的建议测法

六个陷阱对应六条纪律：锁流式；显式指定指标集与百分位；ignore_eos 两边设同值；声明 warmup 与 flush；声明 token 口径；报 TTFT 时声明是否含客户端排队。再加三条通用项：报 dataset 名称与 request-rate 曲线形状（rate=inf 与泊松是两种负载）；goodput 的 SLO 值取自真实的品质承诺而不是随手写个整数；Agent 场景用会话仿真器（agentic-rollout 或自建同型工具），Chat 工具的数字代表不了 Agent 负载，总纲引用的 AgentSysBench 数据（re-prefill 占 cache 相关 token 的 55.9%）说的就是这件事。

折扣清单：本文全部结论来自静态源码核对，未跑实测；两仓库都在快速演进，行号与默认值随版本漂移，引用前建议按元信息里的基线核对；第三方工具（AIPerf、GuideLLM、genai-perf）不在范围内，产能评估的完整框架见 [Token 工厂的产能怎么算](../../../99_misc/token_factory_talk/cluster-capacity-measurement.md)；vLLM 的 dataset 机制与 SGLang 独立，数据集层面的口径差异本文未展开。

## 源文件索引

| 仓库 | 文件 | 关键内容 |
| --- | --- | --- |
| vLLM | benchmarks/serve.py | BenchmarkMetrics :787-811；goodput 判定 :674-697；参数默认值 :1866-1896；warmup :931-954 |
| vLLM | benchmarks/lib/endpoint_request_func.py | TTFT/TPOT/ITL 计时 :250-315；非流式失败标记 :308-313 |
| vLLM | benchmarks/datasets/datasets.py | 十四种 dataset 后端 :1613-1633 |
| SGLang | python/sglang/benchmark/serving.py | 指标与百分位 :1273-1298；非流式 TTFT :507-509；ignore_eos :470-472；warmup/flush :1512-1517,2654-2658 |
| SGLang | python/sglang/bench_serving.py | 兼容薄壳，真身转发 |
| SGLang | benchmark/agentic-rollout/simulate.py | 会话循环与三模式 :193-230,349-354；tool-delay :185,385-398；DP 粘性 :409-412 |
| SGLang | benchmark/agentic-rollout/metrics.py | 三层命中率 :69-96；occupancy :59-66 |
| SGLang | benchmark/agentic-rollout/spec/ | 特性规格与验证证据（feature-00） |

## 参考资料

- 实践方法论姊妹篇：[TTFT 多了 15 毫秒之后](01_kv-cache-perf-methodology.md)——真机排查视角的同一套纪律
- 度量缺口的出处：[当 Agent 流量成为推理系统的主要负载](../../agent_serving/01_agent-workload-serving.md)第五节
- 产能框架：[Token 工厂的产能怎么算？](../../../99_misc/token_factory_talk/cluster-capacity-measurement.md)
- vLLM benchmarks 文档与 SGLang benchmark 目录（基线见元信息行）
- [MLPerf Agentic Inference](https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/)（多轮、上下文增长、闭环工作流的标准化尝试）
