# Claude 提示词缓存机制与源码实现深度分析

本文基于 Claude 官方技术文档（来源：`https://platform.claude.com/docs/en/build-with-claude/prompt-caching`）以及 Claude Code 源码实现，分析其如何在终端 Agent 环境下落地 Prompt Caching（提示词缓存）机制。该机制通过复用请求的上下文前缀，显著降低大规模或重复性任务的处理延迟与 API 调用成本。

## 1. 核心机制与优势

Claude 的提示词缓存以严格前缀匹配为基础。它将工具定义、系统提示词和消息历史串联为可复用的上下文前缀。一旦命中，服务端即可直接复用已构建的前缀计算状态，从而同时降低首 Token 延迟与输入 Token 成本，并在零数据保留（Zero Data Retention, ZDR）场景下维持企业级数据治理要求。

提示词缓存通过在服务端缓存特定的上下文前缀，优化多轮对话或包含大量重复背景信息的 API 请求，同时兼顾了企业级的数据隐私安全。

- **性能与成本优化**：在处理包含大量示例、复杂背景知识或多轮对话的请求时，系统优先检索并复用已缓存的前缀。若命中缓存，则直接跳过该部分的处理，从而减少计算耗时与 Token 消费。
- **数据隐私保护**：该功能支持 ZDR 协议。在配置了 ZDR 的企业环境中，通过缓存机制传输的数据在 API 返回响应后不会被持久化存储。
- **完整前缀匹配**：提示词缓存严格按照上下文顺序运行，依次包含工具（Tools）、系统提示词（System）和消息（Messages），直到遇到被标记为 `cache_control` 的文本块。

---

## 2. 缓存控制模式与生命周期

Claude API 将缓存控制抽象为断点选择与 TTL 管理两个正交维度。前者决定哪一段前缀进入可复用区，后者决定缓存保活窗口和读写计费方式。两者组合后，开发者可以在命中率、稳定性与调用成本之间进行工程化权衡。

### 2.1 缓存控制模式

自动缓存适合标准对话链路，由服务端自动选择最后一个可缓存文本块作为断点。显式缓存则适合长系统提示词、Few-shot 示例或 RAG 背景等固定前缀明显的场景，允许开发者精确控制缓存截断位置。

Claude API 提供自动和显式两种缓存控制策略，开发者可根据具体业务场景的上下文复杂度灵活选择接入方式。

- **自动缓存（Automatic caching）**：通过在请求体顶层配置单一的 `cache_control` 字段即可开启。系统会自动识别并将缓存断点应用于请求中最后一个可缓存的文本块。在多轮对话中，该断点会随着新消息的追加自动向前推进。
- **显式缓存（Explicit cache breakpoints）**：开发者将 `cache_control` 标记直接注入到特定的内容块上。该模式适用于需要对缓存内容边界进行细粒度控制的复杂编排场景。

### 2.2 生命周期与定价策略

Claude 将缓存生命周期设计为带刷新语义的 TTL（存活时间）合约。5 分钟缓存适合高频连续对话，1 小时缓存适合长时间挂起的高价值上下文。写入加价、命中低价读取的定价模型，则把缓存优化直接转化为可量化的成本收益。

提示词缓存引入了基于存活时间（TTL）的差异化计费模型，通过提高初次写入成本来换取极低的缓存命中读取成本。

- **默认生命周期与刷新机制**：缓存默认的存活时间为 5 分钟。每次成功命中缓存时，该 5 分钟的存活周期会自动刷新。
- **长周期选项**：对于需要更长上下文保持时间的业务，系统额外提供 1 小时 TTL 的缓存选项，对应更高的写入费率。
- **计费倍率映射**：不同的缓存操作对应不同的计费乘数，此倍率规则可与 Batch API 折扣等其他计费策略叠加生效。

| 计费操作类型   | 费率倍数（相对于基础输入 Token 单价） |
| -------------- | ------------------------------------- |
| 5 分钟缓存写入 | 1.25x                                 |
| 1 小时缓存写入 | 2.00x                                 |
| 缓存命中读取   | 0.10x                                 |

### 2.3 API 显式缓存示例

下面的示例展示的是 Anthropic API 层面的显式缓存用法，用于帮助理解第 2 节所述的断点控制模式。Claude Code 在此基础上又自动化了系统提示词切分、消息断点推进、TTL 选择和状态诊断，因此其工程实现比这个最小示例更复杂。

```python
# 以下示例展示如何在 Anthropic API 中显式设置缓存断点，
# 以便复用长系统提示词对应的前缀计算结果。
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "这是一段非常长的背景故事或知识库文档内容...",
            # 将缓存断点显式放在长文本块上，便于后续请求复用此前缀
            "cache_control": {"type": "ephemeral"},
        }
    ],
    messages=[
        {
            "role": "user",
            "content": "根据上述背景，主人公的最终选择是什么？",
        }
    ],
)
```

---

## 3. 技术约束与底层引擎对比

提示词缓存并非在任意请求上都能稳定生效。它的工程可用性同时受最小 Token 门槛、断点数量上限和模型支持范围约束。它与底层 Prefix Caching（前缀缓存）的关系，则决定了开发者获得的是“尽力而为”的命中概率，还是具备 TTL 保证的可控缓存契约。

### 3.1 约束条件与支持模型

不同模型对可缓存前缀长度设置了明确下限。过短请求即使携带 `cache_control`，也不会形成有效缓存。同时，单次请求最多仅允许 4 个缓存断点。这意味着复杂提示词编排必须在命中收益与断点预算之间做取舍。

在工程接入提示词缓存时，需注意以下底层机制的约束条件：

- **最小 Token 阈值**：不同模型有最低缓存长度要求。若内容低于阈值，缓存请求会静默失效。例如，Claude Sonnet 3.7 要求 1024 tokens；某些高阈值模型或测试版本在实际文档与测试样例中可见 4096 tokens 的门槛，接入时应以目标模型的最新官方页面或实测结果为准。
- **最大缓存断点数量**：每次请求最多允许设置 4 个缓存断点。若已存在 4 个显式断点，API 将返回 400 错误。
- **支持模型矩阵**：目前所有活跃的 Claude 模型（包括最新的 Claude 4 全系列、Claude 3.7 Sonnet 等）均原生支持提示词缓存功能。

### 3.2 顶层控制与底层前缀缓存对比

前缀缓存关注的是推理引擎层的 KV Cache 复用效率，其行为受显存容量和淘汰策略支配。提示词缓存则把这一能力提升为 API 契约，并额外提供断点显式控制、TTL 保活与差异化计费。因此，它更适合需要稳定 SLA 和可预测成本的生产系统。

提示词缓存与底层前缀缓存共享相同的 KV 复用机制，但前者在控制权与产品化层面上做了进一步封装和增强：

- **底层被动优化 vs 顶层主动控制**：前缀缓存（如 vLLM APC）是对用户完全透明的底层被动优化机制，依赖显存容量的 LRU 淘汰；提示词缓存（如 Anthropic API）则是构建在之上的产品化功能，开发者通过 `cache_control` 显式标记，提供明确的 TTL 保障。
- **纯前缀缓存的局限性**：在多租户高并发场景下，长文本缓存极易被挤出显存导致延迟波动；且任何头部动态变量的微小变化都会导致后续数十万 Token 缓存失效。
- **提示词缓存的商业价值**：通过明确的 TTL 保障机制和重构的 API 计费经济模型，倒逼开发者将提示词解耦为“静态冻结区”和“动态变化区”，最大化契合前缀匹配机制。

---

## 4. Claude Code 缓存架构源码解析

Claude Code 没有停留在 API 层面的简单开关，而是围绕“前缀稳定性”构建了一套缓存工程体系。系统提示词先做动静切分，工具定义做哈希维稳，消息序列再按对话形态动态布置断点。诊断链路还会追踪 `globalCacheStrategy`、`cache_read_input_tokens` 与 `cache_creation_input_tokens` 等关键状态，从而持续提高长上下文 Agent 工作流的命中率。

### 4.1 系统提示词的分层动静隔离

系统提示词同时承载长期稳定的工作流规则，以及高频变化的时间、环境和会话态字段。如果两者混排在前缀头部，任何动态字段抖动都可能击穿整段缓存。因此，Claude Code 通过边界标记把静态区与动态区分离，并只让稳定区承担更强的缓存职责。

在 [api.ts](./src/utils/api.ts) 中，`splitSysPromptPrefix()` 并非始终执行同一套切分策略，而是根据 `shouldUseGlobalCacheScope()`、`skipGlobalCacheForSystemPrompt` 和边界标记是否存在选择 3 种模式。启用 `global` 作用域缓存且命中边界时，系统提示词会被拆成 attribution header、prefix、静态区和动态区，其中仅静态区携带内部 `cacheScope: 'global'`。若存在 MCP 工具，则退化为 `tool_based` 路径，对应诊断与日志中的 `globalCacheStrategy = 'tool_based'`。若未命中边界或未启用 `global` 作用域缓存，则退化为 `system_prompt` 或 `none` 路径下的 `org` 级拼接策略。这里的 `org` 只是一个不影响 API 请求的内部缓存作用域抽象。最终 API 请求中，只有 `scope: 'global'` 会被显式序列化到 `cache_control` 字段。

```typescript
// 示意化伪代码：刻意省略 attribution header、prefix 和 fallback 分支，
// 只表达“命中 boundary 后静态区与动态区分离”的核心思路，不对应真实源码。
export function splitSysPromptPrefix(
  systemPrompt: SystemPrompt,
  options?: { skipGlobalCacheForSystemPrompt?: boolean },
): SystemPromptBlock[] {
  // 识别边界，将系统提示词分为静态与动态两部分
  const boundaryIndex = systemPrompt.findIndex(
    s => s === SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
  )

  // 核心切分逻辑，拆分出静态字符串与动态字符串
  // ... (省略部分细节) ...

  const result: SystemPromptBlock[] = []

  // 静态内容使用 global 缓存，动态内容取消 scope 以防止污染全局前缀
  if (staticJoined) {
    result.push({ text: staticJoined, cacheScope: 'global' })
  }
  if (dynamicJoined) {
    // 动态内容不附加 global 作用域
    result.push({ text: dynamicJoined, cacheScope: null })
  }

  return result
}
```

### 4.2 工具定义的内存哈希维稳

工具定义在 Agent 请求中通常占据大量稳定 Token，但其 JSON Schema 极易因键顺序、可选字段或功能开关波动而产生字节级变化。Claude Code 因而通过名称与序列化 Schema 的联合缓存键，锁定工具层的哈希签名与输出结果，从而降低工具层前缀漂移带来的缓存失效风险。

在 [api.ts](./src/utils/api.ts) 中的实现里，系统使用工具名称和序列化后的 `inputJSONSchema` 作为联合缓存键，并把名称、描述、`input_schema`、`strict`、`eager_input_streaming` 等 `session-stable` 字段缓存下来。这样可以避免 GrowthBook 开关或 `tool.prompt()` 漂移导致工具数组序列化结果抖动。`toolToAPISchema()` 仅在调用方显式传入 `options.cacheControl` 时，才会把 `cache_control` 透传到工具定义。主请求链路中的缓存断点，主要仍由系统提示词块和消息块承担，而不是默认给每个工具定义附加 `{ type: 'ephemeral' }`。原因也很直接：每次请求最多只有 4 个缓存断点预算，而工具数量往往远超这一上限。若把断点预算分散到工具定义层，反而会挤压系统提示词与消息前缀的缓存收益。

### 4.3 消息列表的动态游标推进

多轮对话会持续追加用户消息、助手回复和工具结果。缓存断点如果固定不动，很快就无法覆盖最新稳定前缀。Claude Code 因此把断点位置设计为可随请求语义前移或后移的动态游标，以便在常规对话与无痕探测分支之间维持缓存可复用性。

在 [claude.ts](./src/services/api/claude.ts) 中，`addCacheBreakpoints()` 负责在消息数组中只保留一个消息级 `cache_control` 标记。默认情况下，标记位于最新消息；当系统执行 `skipCacheWrite = true` 的无痕探测分支时，标记会前移到倒数第二条消息。这种游标平移策略使分支请求能够复用主干共享前缀，同时避免把分支尾部内容写入新的缓存尾点。

```typescript
// 示意化伪代码：刻意省略 querySource、cached microcompact 和 cache_edits 参数，
// 仅展示“单一断点随 skipCacheWrite 前移”的核心行为，不对应真实源码。
export function addCacheBreakpoints(
  messages: (UserMessage | AssistantMessage)[],
  enablePromptCaching: boolean,
  skipCacheWrite = false,
): MessageParam[] {
  // 根据是否需要写入缓存，决定将标记置于最后一条还是倒数第二条消息
  // skipCacheWrite 为 true 时，游标前移避免分支污染主干
  const markerIndex = skipCacheWrite
    ? messages.length - 2
    : messages.length - 1

  // 遍历所有消息，寻找断点标记位置
  const result = messages.map((msg, index) => {
    // 判定当前消息索引是否命中游标
    const addCache = index === markerIndex
    // 针对目标消息注入 cache_control: { type: 'ephemeral' }
    // ... 此处省略构造细节 ...
    return msg // 返回处理后的消息对象
  })

  return result
}
```

---

## 5. 长周期缓存与细粒度控制实践

在长时间运行的终端 Agent 中，缓存优化不仅是“是否命中”的问题，还涉及缓存保活时长、历史垃圾内容清理，以及删除操作对缓存命中统计的影响。Claude Code 因而同时实现了按请求来源升级 TTL 的策略，以及基于引用 ID 的局部缓存编辑机制。

### 5.1 长周期缓存的条件触发

1 小时 TTL 并非默认选项，而是面向主线程等高价值、可能长时间挂起的会话按条件启用。这一选择本质上是在更高写入成本与更低重建代价之间做交换，以避免大体量上下文因短 TTL 过期而被反复重算。

长周期缓存通常意味着更高的写入费用，因此系统不会全局开启该选项。源码中的 `should1hCacheTTL()` 会同时检查计费环境、用户资格和请求来源匹配关系。在一方 API 下，需要会话锁存后的用户资格成立，并命中 GrowthBook 下发的 `allowlist`。在 Bedrock 场景下，则还支持通过环境变量显式启用 1 小时 TTL。

根据 [claude.ts](./src/services/api/claude.ts) 中的实现，`getCacheControl()` 只有在 `should1hCacheTTL(querySource)` 返回 true 时才会附加 `ttl: '1h'`。`querySource` 只是匹配条件之一，而非唯一条件。例如主线程 `repl_main_thread*` 常被纳入 `allowlist`。这是因为这类会话更容易出现长时间思考、代码审查或暂离终端的高价值挂机场景。系统还通过 bootstrap state 锁存资格与 `allowlist`，避免会话中途因 overage 状态或远端配置刷新而引入 TTL 抖动。

### 5.2 结合缓存引用的精准驱逐

追加式（Append-only）缓存天然擅长复用历史前缀，却不擅长删除已经失去价值的工具输出。为此，Claude Code 引入 `cached microcompact`。在满足主线程来源、模型支持 `cache editing`、功能开关开启且请求本身启用缓存等条件时，系统不会改写本地消息历史。相反，它会先为可缓存 `tool_result` 注入 `cache_reference`，再在 API 层追加 `cache_edits`，以尽量保留原有可缓存前缀并定点删除冗余块。

在处理消息列表时，系统会先把 `cache_edits` 块插回既定的 user message 位置。随后，它只对“最后一个 `cache_control` 标记之前”的 `tool_result` 块注入引用 ID（`cache_reference: block.tool_use_id`）。当内部上下文压缩服务判定某些工具结果需要被清理时，它会向 API 发送包含这些引用 ID 的删除指令，也就是 `cache_edits` 中的 `delete` edits。这样一来，服务端就能在不破坏整体 Prefix Hash 连续性的前提下定点抹除对应的 KV Cache，也能避免对不在可缓存前缀内的块错误打标。

---

## 6. 缓存监控诊断与状态漂移

缓存系统最难处理的问题不是命中本身，而是命中率突然下降时如何快速定位漂移源。Claude Code 因此维护系统提示词、工具定义、特性开关以及 `cache_read_input_tokens` / `cache_creation_input_tokens` 的联合诊断视图。状态漂移检测模块会把每次请求前的系统哈希、工具哈希和相关配置快照保存为可比对状态，并结合 `cache_read_input_tokens` 的跌幅阈值识别缓存失效。这样可以区分 TTL 到期、Schema 漂移、缓存策略切换或请求参数变化导致的缓存重建，也就是文中所说的缓存断裂（cache break）。

由于底层前缀匹配机制具有脆弱性，模型切换、系统提示词变化、工具 Schema 漂移、beta 头变动或额外请求体参数变化都可能导致缓存重建。为此，系统构建了针对提示词状态与缓存读入 Token 的联合观测模块，而不是仅凭单次输入规模波动做粗粒度判断。

在 [promptCacheBreakDetection.ts](./src/services/api/promptCacheBreakDetection.ts) 诊断模块中，系统维护了涵盖系统哈希、工具哈希、`cacheControlHash`、beta 列表、`globalCacheStrategy`、`effortValue` 和 `extraBodyHash` 在内的全状态快照。这里的 `globalCacheStrategy` 明确区分 `tool_based`、`system_prompt` 和 `none` 三种缓存策略来源。它们分别对应“因 MCP 工具存在而退化为工具侧策略”“系统提示词拆分路径生效”以及“未启用或未命中 `global` 作用域缓存”。`cacheControlHash` 则用于捕捉文本内容不变但 `scope` 或 TTL 改变的情况。每次请求发出前，模块会先记录这些快照。待响应返回后，再比较 `cache_read_input_tokens` 是否相较上次下降超过 5%，且绝对下降值超过阈值 `MIN_CACHE_MISS_TOKENS = 2_000`。只有同时满足这两个条件时，诊断器才会将其视为缓存断裂（cache break）。随后，它会继续区分是模型切换、TTL 到期、Tool Schema 漂移、beta 变化、缓存策略切换，还是更可能的服务端侧路由或驱逐导致的命中下降。与之相对，`cache_creation_input_tokens` 更适合观测本轮新写入缓存的成本规模，而不是直接用于判定是否发生缓存断裂。
