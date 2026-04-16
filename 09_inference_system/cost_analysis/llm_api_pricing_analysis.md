# 大模型 API 定价策略定量分析框架

本文基于 OpenRouter 聚合平台的动态定价数据，提供一套针对主流大模型（如 OpenAI、Anthropic、Google、DeepSeek 等）API 的定量成本测算与商业分析框架。

## 快速使用：运行测算脚本

本文提供了一个配套的 Python 脚本 `fetch_pricing.py`，用于动态获取并计算最新价格。该脚本无需安装任何第三方依赖（仅使用 Python 3 标准库）。

```bash
# 在终端中运行脚本，使用默认参数
python fetch_pricing.py

# 也可以通过命令行参数自定义测算条件（如调整缓存命中率、汇率、输入/输出 Tokens）
python fetch_pricing.py --hit-rate 0.6 --input-tokens 30.0 --output-tokens 1.0 --exchange-rate 6.9

# 强制忽略本地缓存，从服务器拉取最新数据
python fetch_pricing.py --no-cache
```

---

## 1. 核心模型分类与数据采集机制

通过标准化分类矩阵对当前市场主流模型进行分级，结合聚合平台提供的实时 API 接口，构建动态、可靠的价格获取链路。此结构直接映射 `fetch_pricing.py` 中的 `CATEGORIES` 配置，确保文档分析与代码逻辑一致。

### 1.1 主流大模型分级矩阵

当前市场的大语言模型生态呈现明显的梯队化分布。基于模型能力与目标场景，可划分为旗舰、经济、推理等核心层级，涵盖国际与国内头部厂商。

- **OpenAI 体系**：涵盖 `gpt_flagship` (GPT 旗舰)、`gpt_mini` (GPT 经济型)，以及专注于深度逻辑的 `o_series` (高级推理) 和 `o_mini` (轻量推理)。
- **Anthropic 体系**：包含 `claude_opus` (顶级旗舰)、`claude_sonnet` (均衡旗舰) 与 `claude_haiku` (经济型)，满足不同响应速度与成本要求。
- **Google 体系**：由 `gemini_pro` (专业旗舰)、`gemini_flash` (均衡速度) 及 `gemini_flash_lite` (极致轻量) 构成多模态生态。
- **国内核心梯队**：以 `deepseek_chat` (常规交互) 和 `deepseek_reasoner` (R1 逻辑推理) 为代表的低成本高智商模型，以及阿里的 `qwen_max` (通义旗舰) 和 `qwen_plus` (通义均衡)。
- **开源旗舰生态**：Meta 发布的 `llama_70b`，为私有化部署及高并发场景提供极具性价比的基准。

### 1.2 定价数据动态采集方案

传统依赖手动查阅各厂商官网定价页面的方式难以应对频繁的价格调整。本框架采用基于 OpenRouter 统一接口的动态拉取策略。

- **数据源**：聚合接口 `https://openrouter.ai/api/v1/models` 提供全量模型最新报价。
- **版本发现引擎**：配套脚本使用正则表达式 `\d+\.\d+|\d+` 动态提取版本号元组，并结合 `include`/`exclude` 规则过滤出指定分类的最新稳定或可用预览版本，无需人工干预即可抵御版本更迭（如 GPT-4.1 至 GPT-5.4）。
- **汇率转换机制**：脚本内置示例固定汇率常数（如取 USD to CNY = 6.9，实际使用中应根据当前真实汇率动态调整），自动将美元计价放大转换为“元/百万 Token”。

---

## 2. 用户视角：API 调用成本定量分析

构建业务场景下的实际开销评估模型，将抽象的 Token 单价转化为具象的单次交互成本与月度预算。

### 2.1 主流大模型 API 定价与计费维度

下表展示了当前主流模型的基准定价。数据由配套脚本 `fetch_pricing.py` 动态生成，模型版本和价格可能随厂商策略实时波动。请运行脚本以获取最新报价。

**表 1：API 基础定价表**：

| 模型名称 (**自动匹配最新版**)     | API ID                                 | 输入单价 (¥/M) | 缓存读 (¥/M) | 缓存写 (¥/M) | 输出单价 (¥/M) | 联网搜索 (¥/次) |
| :-------------------------------- | :------------------------------------- | :------------- | :----------- | :----------- | :------------- | :-------------- |
| **GPT 旗舰 (v5.4)**               | `openai/gpt-5.4`                       | 17.2500        | 1.7250       | -            | 103.5000       | 0.0690          |
| **GPT 经济型 (v5.4)**             | `openai/gpt-5.4-mini`                  | 5.1750         | 0.5175       | -            | 31.0500        | 0.0690          |
| **OpenAI O系列 高级推理 (v3)**    | `openai/o3`                            | 13.8000        | 3.4500       | -            | 55.2000        | 0.0690          |
| **OpenAI O系列 轻量推理 (v4)**    | `openai/o4-mini`                       | 7.5900         | 1.8975       | -            | 30.3600        | 0.0690          |
| **Claude Opus 顶级旗舰 (v4.6)**   | `anthropic/claude-opus-4.6`            | 34.5000        | 3.4500       | 43.1250      | 172.5000       | 0.0690          |
| **Claude Sonnet 均衡旗舰 (v4.6)** | `anthropic/claude-sonnet-4.6`          | 20.7000        | 2.0700       | 25.8750      | 103.5000       | 0.0690          |
| **Claude Haiku 经济型 (v4.5)**    | `anthropic/claude-haiku-4.5`           | 6.9000         | 0.6900       | 8.6250       | 34.5000        | 0.0690          |
| **Gemini Pro (v3.1)**             | `google/gemini-3.1-pro-preview`        | 13.8000        | 1.3800       | 2.5875       | 82.8000        | -               |
| **Gemini Flash (v3)**             | `google/gemini-3-flash-preview`        | 3.4500         | 0.3450       | 0.5750       | 20.7000        | -               |
| **Gemini Flash-Lite (v3.1)**      | `google/gemini-3.1-flash-lite-preview` | 1.7250         | 0.1725       | 0.5750       | 10.3500        | -               |
| **DeepSeek Chat (latest)**        | `deepseek/deepseek-chat-v3.1`          | 1.0350         | -            | -            | 5.1750         | -               |
| **DeepSeek Reasoner (R1)**        | `deepseek/deepseek-r1-0528`            | 3.4500         | 2.4150       | -            | 14.8350        | -               |
| **Qwen Max 通义旗舰 (v3)**        | `qwen/qwen3-max`                       | 5.3820         | 1.0764       | -            | 26.9100        | -               |
| **Qwen Plus 通义均衡 (v3.6)**     | `qwen/qwen3.6-plus`                    | 2.2425         | -            | -            | 13.4550        | -               |
| **Llama 70B 开源旗舰 (v3.3.70)**  | `meta-llama/llama-3.3-70b-instruct`    | 0.6900         | -            | -            | 2.2080         | -               |
| **Kimi 核心模型 (v2.5)**          | `moonshotai/kimi-k2.5`                 | 2.6406         | 1.3203       | -            | 11.8680        | -               |
| **Kimi 推理模型 (v2)**            | `moonshotai/kimi-k2-thinking`          | 4.1400         | 1.0350       | -            | 17.2500        | -               |
| **GLM 旗舰 (v5.1)**               | `z-ai/glm-5.1`                         | 6.5550         | 3.2775       | -            | 21.7350        | -               |
| **GLM 经济型 (v4.32)**            | `z-ai/glm-4-32b`                       | 0.6900         | -            | -            | 0.6900         | -               |
| **MiniMax 核心模型 (v2.7)**       | `minimax/minimax-m2.7`                 | 2.0700         | 0.4071       | -            | 8.2800         | -               |

**表格字段说明（客户视角计费逻辑）**：

- **输入单价 (`Price_Input` / Prefill 阶段)**：**“阅后即焚”的普通输入费**。用户提交一次性问题或指令，模型看完（进行 Prefill 预填充计算，将输入 Token 转化为中间状态）并回答后即丢弃上下文。这是最基础的计费项。
- **输出单价 (`Price_Output` / Decoding 阶段)**：**“回答生成费”**。模型根据前置的上下文，逐字自回归生成回复（Decoding 解码计算）所收取的费用。由于 Decoding 阶段受限于显存带宽（Memory Bound），单 Token 计算成本远高于并行的 Prefill，因此输出单价通常显著高于输入费用。
- **缓存读 (`input_cache_read` / 省略 Prefill 阶段)**：**“老客户复用折扣”**。在后续的多轮对话中，若用户的输入命中了此前已存好的 KV Cache，模型将直接读取这些状态，完全免去昂贵的 Prefill 计算过程。这部分 Token 将享受大幅折扣（通常比普通输入便宜 50%-90%）。
- **缓存写 (`input_cache_write` / 首存 Prefill 阶段)**：**“首存保护费”**。当用户首次提交需要长效保留的背景文本（如系统提示词、超长文档库）并开启 Prompt Caching 时，模型必须进行完整的 Prefill 计算生成 KV Cache，并额外占用服务端存储资源将其持久化（留给后续对话复用）。因此，部分厂商（如 Claude）会对此类“首存”行为收取高于普通输入的溢价。**若厂商未定义专用 Cache Write 价格，则视为未开启 Prompt Caching 功能或等同于普通输入价。**
- **联网搜索 (`web_search`)**：**“工具调用附加费”**。当模型判断需要调用外部搜索引擎获取实时信息时，每次触发该工具所产生的固定附加费用（按次计费，与 Token 长度无关）。

**计价机制与商业策略剖析**：

透过表象的 Token 单价，各大模型厂商在缓存策略、工具调用及推理计算上暗藏了差异化的商业意图与计费逻辑：

- **上下文缓存（Prompt Caching）**：当模型支持缓存功能时，首次长文本输入按“缓存写”计费，后续重复命中的输入 token 按“缓存读”优惠价格计费（通常比标准输入低 30-90%）。以 DeepSeek R1 为例，缓存读节省 30%，在长上下文重复场景中可显著降低总成本。
- **联网搜索（Web Search）**：部分模型支持原生的网络搜索功能。该功能通常按次计费。在评估智能体（Agent）或 RAG 场景时，需将此固定费用作为附加成本计入单次调用总成本中。
- **推理模型（Reasoning Models）隐式消耗**：如 `DeepSeek Reasoner` 及 `Kimi 推理模型`，内部会生成大量思维链 Token。这部分隐式消耗会被折算进输出价格中，导致输出单价显著偏高，实际开发需预留 3-5 倍 Token 预算。
- **长文本同价下沉策略**：如 `GLM 经济型 (v4.32)` 采用输入输出同价策略，旨在鼓励开发者上传超大上下文（如长文档、代码库），以此快速进行市场渗透。

### 2.2 OpenRouter API 计费账单（Usage）解析

在探讨具体的计算公式之前，我们先来看 OpenRouter 在真实的工程运行中，每次 API 响应（如开启了 `include_reasoning: true` 或原生支持 `usage` 追踪时）直接返回的精确计费“电子回单”。这种结构极为清晰地展现了底层计费的维度。

以下是一个典型的、包含上下文缓存（Prompt Caching）与推理（Reasoning）明细的 JSON 账单结构（数据规范参考自 OpenRouter 官方文档：[Usage Accounting](https://openrouter.ai/docs/guides/administration/usage-accounting)）：

```json
"usage": {
  "prompt_tokens": 10339,
  "completion_tokens": 60,
  "total_tokens": 10399,
  "prompt_tokens_details": {
    "cached_tokens": 10318,
    "cache_write_tokens": 21
  },
  "completion_tokens_details": {
    "reasoning_tokens": 0
  },
  "cost": 0.00154,
  "cost_details": {
    "upstream_inference_cost": 0.00154
  }
}
```

**账单字段解析**：

- `prompt_tokens`：本次请求传入的总输入长度。
  - `cached_tokens`：命中缓存的 Token 数，享受专属的“老客户复用折扣”（`Price_Input_Cache_Read`）。
  - `cache_write_tokens`：未命中且被系统新写入缓存的 Token 数，需支付“首存保护费”（`Price_Input_Cache_Write`）。
  - _(隐式字段)_ `normal_tokens`：既未命中缓存，也无需写入缓存的常规尾部输入（如每次不同的简短提问），按普通输入（`Price_Input`）计费。
- `completion_tokens`：总输出 Token 数。
  - `reasoning_tokens`：大模型内部的隐式思维链消耗，合并计入总输出，按输出单价（`Price_Output`）计费。
- `cost`：即基于上述实际 Token 消耗与模型单价计算出的**单次调用总金额**（USD）。

在此回单结构下，OpenRouter 后端计算最终 `cost` 的精确逻辑如下：

$$
\begin{aligned}
\text{cost} &= (\text{cached\_tokens} \times \text{Price\_Input\_Cache\_Read}) \\
&\quad + (\text{cache\_write\_tokens} \times \text{Price\_Input\_Cache\_Write}) \\
&\quad + \big[ (\text{prompt\_tokens} - \text{cached\_tokens} - \text{cache\_write\_tokens}) \times \text{Price\_Input} \big] \\
&\quad + (\text{completion\_tokens} \times \text{Price\_Output}) \\
&\quad + (\text{Web\_Search\_Count} \times \text{Price\_Web\_Search})
\end{aligned}
$$

**验证实例（以 Claude Haiku 4.5 为例）**：

假设上述 JSON 账单是由调用 Claude Haiku 4.5 模型产生的（其 OpenRouter 基准定价为：输入 $1.0/M、输出 $5.0/M、缓存写 $1.25/M、缓存读 $0.1/M）。将账单中的 Token 明细代入上述公式：

$$
\begin{aligned}
\text{cost} &= \left( 10318 \times \frac{0.1}{10^6} \right) + \left( 21 \times \frac{1.25}{10^6} \right) + \left( [10339 - 10318 - 21] \times \frac{1.0}{10^6} \right) + \left( 60 \times \frac{5.0}{10^6} \right) + 0 \\
&= 0.0010318 + 0.00002625 + 0 + 0.0003 \\
&= 0.00135805 \, \text{USD}
\end{aligned}
$$

这完美展示了 JSON 账单中的微观字段是如何通过加权计算得出最终 `cost` 的。

### 2.3 成本预测公式

基于 2.2 节真实的账单结构，我们可以构建出极其清晰的成本测算模型。在系统架构设计期，API 的总成本预测公式如下：

$$
\begin{aligned}
\text{单次调用总成本} &= (\text{Tokens}_{cached} \times \text{Price\_Input\_Cache\_Read}) \\
&\quad + (\text{Tokens}_{cache\_write} \times \text{Price\_Input\_Cache\_Write}) \\
&\quad + \big[ (\text{Tokens}_{prompt} - \text{Tokens}_{cached} - \text{Tokens}_{cache\_write}) \times \text{Price\_Input} \big] \\
&\quad + (\text{Tokens}_{completion} \times \text{Price\_Output}) \\
&\quad + (\text{Web\_Search\_Count} \times \text{Price\_Web\_Search})
\end{aligned}
$$

**宏观测算中的合理简化（保守高估）**：

在实际业务测算中（如评估 RAG 或 Agent 场景），我们通常无法预知每一次调用的精确 `cache_write` 和 `normal` 比例，而是基于宏观的 **缓存命中率（$R_{hit}$）** 进行估算。

因此，为了简化测算并保证预算的安全性，我们（及配套脚本）采用**保守高估策略**：

1. **命中部分**：按宏观命中率 $R_{hit}$ 统一折算为缓存读，适用单价 $P_{hit}$。若 API 未单独设定缓存读价格，则回退为普通输入价。
2. **未命中部分**：由于长文本场景下未命中部分多为新写入的超长文档，我们将其全部视作需新写入缓存的 Token，适用单价 $P_{miss}$。若 API 未单独设定缓存写价格，则回退为普通输入价。

基于此策略的**宏观估算公式**如下：

$$
\begin{aligned}
\text{Input\_Cost} &= \text{Total\_Input\_Tokens} \times \big[ R_{hit} \times P_{hit} + (1 - R_{hit}) \times P_{miss} \big] \\
\text{单次调用总成本} &= \text{Input\_Cost} \\
&\quad + (\text{Total\_Output\_Tokens} \times \text{Price\_Output}) \\
&\quad + (\text{Web\_Search\_Count} \times \text{Price\_Web\_Search})
\end{aligned}
$$

其中，输入侧动态单价的自适应回退逻辑为：

$$
P_{hit} =
\begin{cases}
\text{Price\_Input\_Cache\_Read}, & \text{if } > 0 \\
\text{Price\_Input}, & \text{otherwise}
\end{cases}
$$

$$
P_{miss} =
\begin{cases}
\text{Price\_Input\_Cache\_Write}, & \text{if } > 0 \\
\text{Price\_Input}, & \text{otherwise}
\end{cases}
$$

> [!NOTE]
> 注 1：$\text{Web\_Search\_Count}$ 指 API 后台实际发起的搜索工具调用次数，可能大于用户在界面上的单次触发请求。
> 注 2：该回退策略仅在模型支持缓存功能但未单独定价时完全适用。若模型底层完全不支持上下文缓存（如 DeepSeek Chat），则 $P_{hit}$ 与 $P_{miss}$ 将自动回退为普通输入价，即不受缓存命中率 $R_{hit}$ 的影响。

### 2.4 典型业务场景下的成本测算

基于上述简化的预测公式，我们设定了三种典型的高负载交互场景对各模型进行对标测算：

- **场景 A（知识库检索增强 RAG / 长文档分析）**：输入极长，输出较短。由于每次检索拼入的文档片段不同，上下文变动大，缓存命中率较低。**（预设：20K 入 / 1K 出 / 20% 命中）**
- **场景 B（沉浸式多轮对话 / Agent 智能体）**：上下文极长，输出较短。由于 System Prompt 和前期历史对话在会话期间固定不变，缓存命中率极高。**（预设：30K 入 / 1K 出 / 80% 命中）**
- **场景 C（批量数据清洗 / 机器翻译）**：输入与输出长度接近。由于每次处理全新的独立文本段，完全无法利用上下文缓存。**（预设：5K 入 / 5K 出 / 0% 命中）**

**表 2：典型业务场景单次调用成本估算 (¥)**：

| 模型名称                          | 场景 A: RAG/长文档 (20K 入 / 1K 出 / 20% 命中) | 场景 B: 沉浸式 Agent (30K 入 / 1K 出 / 80% 命中) | 场景 C: 批量翻译 (5K 入 / 5K 出 / 0% 命中) |
| :-------------------------------- | :--------------------------------------------- | :----------------------------------------------- | :----------------------------------------- |
| **GPT 旗舰 (v5.4)**               | 0.3864                                         | 0.2484                                           | 0.6038                                     |
| **GPT 经济型 (v5.4)**             | 0.1159                                         | 0.0745                                           | 0.1811                                     |
| **OpenAI O系列 高级推理 (v3)**    | 0.2898                                         | 0.2208                                           | 0.3450                                     |
| **OpenAI O系列 轻量推理 (v4)**    | 0.1594                                         | 0.1214                                           | 0.1898                                     |
| **Claude Opus 顶级旗舰 (v4.6)**   | 0.8763                                         | 0.5140                                           | 1.0781                                     |
| **Claude Sonnet 均衡旗舰 (v4.6)** | 0.5258                                         | 0.3084                                           | 0.6469                                     |
| **Claude Haiku 经济型 (v4.5)**    | 0.1753                                         | 0.1028                                           | 0.2156                                     |
| **Gemini Pro (v3.1)**             | 0.1297                                         | 0.1314                                           | 0.4269                                     |
| **Gemini Flash (v3)**             | 0.0313                                         | 0.0324                                           | 0.1064                                     |
| **Gemini Flash-Lite (v3.1)**      | 0.0202                                         | 0.0179                                           | 0.0546                                     |
| **DeepSeek Chat (latest)**        | 0.0259                                         | 0.0362                                           | 0.0311                                     |
| **DeepSeek Reasoner (R1)**        | 0.0797                                         | 0.0935                                           | 0.0914                                     |
| **Qwen Max 通义旗舰 (v3)**        | 0.1173                                         | 0.0850                                           | 0.1615                                     |
| **Qwen Plus 通义均衡 (v3.6)**     | 0.0583                                         | 0.0807                                           | 0.0785                                     |
| **Llama 70B 开源旗舰 (v3.3.70)**  | 0.0160                                         | 0.0229                                           | 0.0145                                     |
| **Kimi 核心模型 (v2.5)**          | 0.0594                                         | 0.0594                                           | 0.0725                                     |
| **Kimi K2 Thinking**              | 0.0876                                         | 0.0669                                           | 0.1070                                     |
| **GLM 旗舰 (v5.1)**               | 0.1397                                         | 0.1397                                           | 0.1414                                     |
| **GLM 经济型 (v4.32)**            | 0.0145                                         | 0.0214                                           | 0.0069                                     |
| **MiniMax 核心模型 (v2.7)**       | 0.0430                                         | 0.0305                                           | 0.0517                                     |

> [!NOTE]
> 注：对于未原生支持上下文缓存功能（Prompt Caching）的模型（如 DeepSeek Chat），在测算中命中部分与未命中部分将自动回退为普通输入价，即不受缓存命中率 $R_{hit}$ 的影响，测算结果依然准确有效。

**成本对标结论与业务启示**：

1. **缓存命中决定高端模型的适用性**：对比场景 A 和场景 B 可见，Claude Opus/Sonnet 在高命中率的 Agent 场景中（场景 B）调用成本骤降 40% 以上，这使得在多轮沉浸式对话中使用旗舰模型变得具有商业可行性；而在低命中的 RAG 场景中（场景 A），因需支付昂贵的 `Cache_Write` 首存费用，其成本极为高昂。
2. **输出成本主导批处理任务**：在场景 C（批量翻译/数据清洗）中，输出 Token 数量大且无法命中缓存。此时，输出单价高昂的模型（如 GPT-5.4 和 Claude 系列）成本呈指数级膨胀，而像 Llama 3 70B 或 GLM 经济型这种输出价格极低（或输入输出同价）的模型则展现出绝对统治力的性价比优势。
3. **推理模型（Reasoning）的隐式开销**：DeepSeek R1 在三个场景中的成本均保持在相对稳定的区间，这说明其极具竞争力的基础定价抵消了长文本和思维链带来的 Token 膨胀，在中轻度复杂度的全场景中都具备极佳的泛用性。

---

## 3. 厂商视角：供给侧成本与商业分析

探讨算力基建的固定支出与 Token 计费模式之间的财务平衡，揭示价格战背后的成本底线。

### 3.1 GPU 算力成本基准

以业界主流的高性能 AI 加速卡为参照系，结合公有云（如 AWS、阿里云）的公开报价与 vLLM/TensorRT-LLM 的权威基准测试，评估底层基础设施的摊销压力。

| GPU 型号             | 显存 | 月租估算 (元/卡) | 单卡 FP8 吞吐参考 (Token/s)\* | 适合部署的模型与配置策略                                                 |
| :------------------- | :--- | :--------------- | :---------------------------- | :----------------------------------------------------------------------- |
| **NVIDIA H100**      | 80GB | 27,000 - 62,000  | ~4,000 (输入) / ~2,500 (输出) | 旗舰级模型 (如 Llama 3 70B 需 2 卡 TP，DeepSeek-V3 需 8 卡 EP)           |
| **NVIDIA H20**       | 96GB | 15,000 - 22,000  | ~1,200 (输入) / ~1,800 (输出) | 国产/开源大模型 (H20 显存带宽极高，生成性价比优于 H100，但 Prefill 较弱) |
| **华为 Ascend 910B** | 64GB | 20,000 - 30,000  | 依赖于 MindIE 框架优化水平    | 信创国产化替代方案 (如 Qwen-72B、GLM 等)                                 |

**\*吞吐量估算边界条件与推导模型 (以 70B/72B 级别 FP8 稠密模型为例)**：

以上吞吐量（单卡折算值）基于以下基准条件：连续批处理（Continuous Batching）、中等上下文长度（约 2K - 8K Tokens）、高 GPU 显存利用率（> 90%）。_注：生产环境实际吞吐量会随最大 Batch Size、上下文长度及服务框架的调度策略发生显著波动。_

- **1. 数据来源依据**
  - **官方性能基准**：参考 [NVIDIA NIM Llama-3.3-70b-instruct Benchmarks](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html)（8x H100 最高并发输出吞吐约 20,000 Tokens/s）。
  - **第三方评测**：参考 [ArtificialAnalysis AI Hardware Benchmarking](https://artificialanalysis.ai/benchmarks/hardware?model=llama-3-3-instruct-70b) 对 Llama 3 70B 和 DeepSeek R1 的极限并发负载测试。
  - **硬件算力规格**：基于 NVIDIA Hopper 架构白皮书（H100 SXM5 FP8 算力 1,979 TFLOPS；H20 显存带宽 4.0 TB/s，FP8 算力 296 TFLOPS）。

- **2. 输入阶段 (Prefill) 吞吐推导**
  预填充阶段受限于**计算力（Compute Bound）**。遵循第一性原理，处理 70B 模型 1 个 Token 需消耗计算量约 $2 \times 70 \times 10^9 = 140\,\text{GFLOP}$（1400 亿次浮点运算）。
  - **H100 测算**：主流推理框架下 MFU（模型算力利用率）介于 20%~35%，取 28% 中位值估算：
    $$ \text{H100 单卡输入吞吐} \approx \frac{1{,}979 \times 10^3 \times 28\%}{140} \approx 3{,}958\,\text{Tokens/s} \quad \text{(表内取整 ~4,000)} $$
  - **H20 测算**：算力较低更易饱和，MFU 取 55% 估算：
    $$ \text{H20 单卡输入吞吐} \approx \frac{296 \times 10^3 \times 55\%}{140} \approx 1{,}162\,\text{Tokens/s} \quad \text{(表内取整 ~1,200)} $$

- **3. 输出阶段 (Decode) 吞吐推导**
  大模型生成受限于**显存带宽（Memory Bound）**，需逐个 Token 自回归生成。
  - **H100 测算**：8x H100 节点部署 70B (TP=4/8)，满载并发输出极限约 20,000 Tokens/s，折算单卡为 **~2,500 Tokens/s**。
  - **H20 测算**：显存带宽（4.0 TB/s）虽高于 H100，但在大 Batch 下受算力天花板拖累。经验数据表明其高并发输出约为 H100 的 70%-75%，折算为 **~1,800 Tokens/s**。

- **4. 并发与显存限制（KV Cache 墙）**
  **最大并发请求数**极度受限于单请求 KV Cache 的显存占用。
  - **占用预估**：以 70B 模型（8K 上下文）为例，单请求理论占用上限约 **1.25GB**（1/8 GQA 比例 + FP8 精度）至 **5.0GB**（1/4 GQA 比例 + BF16 精度）显存（_实际部署中通过引入 Token 剪枝、低比特量化、跨层共享及系统级卸载等四维 KV Cache 压缩技术，实测显存占用通常远低于此理论估算_）。
  - **并发瓶颈**：若在单张 80GB H100 上极限部署 70B（权重占用约 70GB），剩余 10GB 理论仅能支撑 **4-8 路并发**。并发过低将导致算力闲置，无法充分摊薄计算开销。
  - **缓解策略 (KV Cache Offloading)**：为突破 GPU 显存墙限制，现代推理系统（如 vLLM、NVIDIA LMCache）支持将非活跃的 KV Cache 卸载（Offload）至 CPU 主存甚至高速外部存储（如 NVMe SSD，利用 GPUDirect Storage 等技术）中。这一机制用空间换取了更高的并发度，但由于跨总线（PCIe）传输带宽远低于 GPU HBM 显存，频繁换页调度会引入额外的 I/O 延迟。

### 3.2 每百万 Token 硬件处理成本模型

从硬件折旧到实际 Token 吞吐量的转化测算，明确盈亏的物理边界。本模型基于分离计算模式进行推导：

**1. 基础折旧与有效工时假设**：

- **月度折旧成本 ($C_{month}$)**：取 H100 预留实例的折中值，假设为 **¥45,000/卡/月**。
- **月度有效运行秒数 ($T_{effective}$)**：实际业务请求存在明显的高峰（如白天 70% 利用率）与低谷（夜间 20% 利用率）。此处取综合利用率中位数 55%（_注：此测算为理想线性扩展下限，未计入 Prefill-Decode 干涉导致的排队损耗及显存碎片导致的 Batch 填充不足损耗_）：
  $$ T_{effective} = 30\,\text{天} \times 24\,\text{小时} \times 3600\,\text{秒} \times 55\% \approx 1{,}425{,}600\,\text{秒} $$

**2. 输入阶段 (Prefill) 成本测算**：

- **单卡输入吞吐 ($V_{in}$)**：基于 3.1 节推导，取 **4,000 Tokens/s**。
- **月度总输入产能 ($N_{in}$)**：
  $$ N_{in} = T_{effective} \times V_{in} = 1{,}425{,}600 \times 4{,}000 = 5{,}702{,}400{,}000\,\text{Tokens} $$
- **每百万 Token 输入硬件成本 ($Price_{in\_hw}$)**：
  $$ Price_{\text{in\_hw}} = \left( \frac{C_{\text{month}}}{N_{in}} \right) \times 10^6 = \left( \frac{45{,}000}{5{,}702{,}400{,}000} \right) \times 10^6 \approx 7.89\,\text{¥/M} $$
  > [!NOTE]
  > 注：此处推导出的 7.89 ¥/M 为**理想峰值吞吐下的边际成本极限**。在实际生产环境中，随着上下文长度（Context Length）的增加，Prefill 阶段的计算量呈平方级增长，吞吐量可能急剧下降（如 8K 上下文下吞吐可能跌至 1,500 Tokens/s 以下），导致实际硬件成本通常是该理论下限的 2-3 倍。

**3. 输出阶段 (Decode) 成本测算**：

- **单卡输出吞吐 ($V_{out}$)**：基于 3.1 节推导，取 **2,500 Tokens/s**。
- **月度总输出产能 ($N_{out}$)**：
  $$ N_{out} = T_{effective} \times V_{out} = 1{,}425{,}600 \times 2{,}500 = 3{,}564{,}000{,}000\,\text{Tokens} $$
- **每百万 Token 输出硬件成本 ($Price_{out\_hw}$)**：
  $$ Price_{\text{out\_hw}} = \left( \frac{C_{\text{month}}}{N_{out}} \right) \times 10^6 = \left( \frac{45{,}000}{3{,}564{,}000{,}000} \right) \times 10^6 \approx 12.63\,\text{¥/M} $$

### 3.3 商业定价与系统级 TCO 对标分析

将物理硬件成本与市场 API 定价进行横向对标，揭示各大厂商的利润空间与技术壁垒。

- **高毛利与品牌溢价**：如 GPT-5.4、Claude 4.6 旗舰等（输出单价均超 100 ¥/M），其 API 标价高达纯稠密模型硬件成本（12.63 ¥/M）的 8-10 倍。需要指出的是，GPT-5.4 等极有可能采用 MoE 架构，实际推理激活参数远小于 70B，其真实底层计算成本会大幅低于稠密模型基线，这意味着闭源头部厂商维持着更为惊人的毛利空间与品牌溢价。与之形成鲜明对比的是 Llama 3 70B 开源模型（输出约 2.20 ¥/M），其定价已极度贴近甚至击穿纯硬件折旧成本。
- **底层架构降本（MoE 的隐性壁垒）**：DeepSeek-V3 的价格定位（输入 `1.0350 ¥/M`，输出 `5.1750 ¥/M`）远低于上述 H100 硬件成本基线估算。这得益于 MoE 架构天然的稀疏激活优势（每次仅激活约 37B 参数，将基础计算量砍半），结合集群通信与 FP8 内核算子的极致工程优化。但需注意，MoE 强依赖于**集群通信带宽的极致利用**（昂贵的 All-to-All 集合通信）。若无高速互联（如 NVLink/InfiniBand）摊薄通信开销，其实际物理成本将回升至稠密模型水平，这构成了极强的自建集群技术壁垒。
- **系统级 TCO（总体拥有成本）修正**：上述代码模型仅涵盖 GPU 硬件折旧。在实际 IDC 环境中，电力（PUE）、机架、网络设备、存储以及集群调度闲置损耗均构成额外开销。行业通用的 TCO 修正系数通常介于 **1.3 至 2.0** 之间。若取 1.5 倍中值系数，H100 输出阶段硬件成本将抬升至约 **18.95 ¥/M**。这意味着部分“看似微利”的模型，其定价仍需覆盖极高的综合运营门槛。

### 3.4 延伸探讨：API 计费与内置 RAG 检索

分析大模型原生 API 与 Web 端产品在知识检索与工具调用上的商业边界。

当前各大厂商的 Web 端（如 ChatGPT Plus、Kimi、通义千问网页版）普遍内置了免费的 RAG（检索增强生成）与联网搜索功能。然而，在标准 API 调用层面（如 OpenAI Chat Completions 接口），模型本身通常**并不包含隐式的 RAG 流程**。API 提供的是纯粹的“大脑推理”算力。若 API 开发者需要联网或 RAG 能力，一般会导致隐性成本的增加：

1. **开发者自建外挂 RAG**：开发者需自行部署向量数据库（Vector DB），将检索到的文档片段（Context）拼接到 Prompt 中传入 API。这会导致输入 Token 数量剧增，虽然丰富了上下文，但也大幅拉高了 API 侧的计费营收（这正是厂商鼓励长文本输入的核心商业动机之一）。
2. **调用厂商的专用工具（Tools/Search 接口）**：部分厂商（如 Perplexity，或开启了特定 `tools` 字段的 OpenAI 接口）支持原生联网。在此模式下，模型会在后台隐式发起搜索，抓取的回调网页内容同样会转化为巨量的隐式输入 Token 计入用户的账单；或者厂商会直接收取单次的 Tool Call 附加费（如 2.2 节表格中的 `web_search` 计费项）。

---
