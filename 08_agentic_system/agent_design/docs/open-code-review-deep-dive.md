# 确定性工程驯服不确定的 Agent：OpenCodeReview 三阶段架构深度解析

> 本文基于阿里巴巴开源项目 [open-code-review](https://github.com/alibaba/open-code-review)（Apache 2.0）源码（main 分支，2026.08 基线）与论文 [OpenCodeReview: Determinism over Non-Determinism for Cost-Effective Agent-Based Code Review](https://arxiv.org/html/2608.09290v1)（arXiv:2608.09290，2026）撰写。所有源码引用以 `文件:行号` 格式标注，机制描述均经源码验证。本文定位为**原理分析**——探讨这套系统如何设计、为什么这样设计，以及哪些设计原则可以迁移到其他 Agent 任务。

---

> [!NOTE] 与使用向文章的分工
> 另有从**使用视角**撰写的系列介绍：[OpenCodeReview：让 AI 代码评审从"能用"到"好用"！](https://mp.weixin.qq.com/s/ODIr8kmgSRckrUwoVR4j4Q) 和 [OpenCodeReview 新功能：让 AI 代码评审从"能审"到"审得准、接得住"](https://mp.weixin.qq.com/s/k6y1ohwHQJQmVpwMLRwS4g) ，覆盖安装、配置、CLI 用法与集成方式。本文不再重复这些内容，专注回答三个问题：**它把确定性注入在哪里？为什么注入在这些位置？注入确定性为什么反而超越了"更自由"的 Agent？**

---

## 一、背景：通用 Agent 做代码评审，差在哪

让 LLM 评审代码早已不是"把 diff 扔给模型"这么简单——但多数的工程实践恰恰停留在这个层面。当团队让 Claude Code、Codex 这类通用 Agent 通过自定义 Skill 执行代码审查时，实测暴露出的问题并非模型能力不足，而是**系统设计缺陷**。

### 1.1 两个结构性弱点

论文将通用 Agent 评审的失败归因于两个相互交织的弱点：

**非确定性（Non-Determinism）**。通用 Agent 拥有无界的工具集（终端执行、文件写入、任意搜索）和庞大的动作空间。同一份 diff，运行两次可能得到完全不同的结果——工具调用路径不同、上下文摄入不同、最终评论不同。代码评审是需要可复现性的质量活动，非确定性直接摧毁信任。

**上下文局部性（Context Locality）**。评审者的有效访问范围被限制在 diff 本身。但真实缺陷往往藏在调用方、被调方、依赖关系、并发模型之中——只看到 diff，无法发现深层问题。然而让 Agent 自由检索，又会出现上下文污染。

### 1.2 实证：通用 Agent 的基准表现

论文构建了 AACR-Bench 基准：200 个真实 PR、50 个流行开源仓库、10 种编程语言、1,505 条经 80+ 高级工程师三轮交叉验证的标注问题。对比 OpenCodeReview 与 Claude Code（`/code-review`）、Codex（`/review`）在 6 种 LLM 后端下的表现：

| 系统               | SEM-F1（最优配置）            | 精度         | 召回             | 评论量/PR       |
| ------------------ | ----------------------------- | ------------ | ---------------- | --------------- |
| **OpenCodeReview** | **25.10%**（Claude-4.6-Opus） | 33.90%       | 20.00%           | 克制            |
| Claude Code        | 14.13%（最强配置）            | 7.23%–15.93% | 高（靠海量评论） | 最高达 4,580 条 |
| Codex              | 8.36%（GPT-5.5）              | ~28%         | 极低（5%）       | —               |

关键读数不是 OpenCodeReview 的绝对分数高，而是**质量的结构差异**：Claude Code 通过生成海量低置信度评论拉高召回（4,580 条评论、精度仅 7.23%），OpenCodeReview 则以少而准取胜。且 OpenCodeReview **最弱配置**（DeepSeek-V4-Pro，17.90%）也超过了 Claude Code 的最强配置（14.13%）——系统设计对质量的贡献，大于模型选择。

---

## 二、问题：三个具体挑战

论文将上述弱点拆解为三个可工程化的问题。这三个问题贯穿全文，三阶段架构逐一回应。

### 2.1 上下文检索错位

给 Agent 的检索自由度与上下文质量之间存在"双输"区间：**检索过少**，Agent 缺乏调用方/被调方信息，只能臆断；**检索过多**，无关上下文稀释注意力，模型反而在噪声中丢失关键信号。通用 Agent 无法控制检索的时机、粒度和边界——它既可能污染上下文，也可能供给不足。

### 2.2 多文件 PR 的连贯性-效率权衡

一个 20 文件的大 PR，单 Agent 一次性审查全部文件（monolithic）：上下文连贯但成本爆炸、长上下文退化；过度细分则文件间关联断裂，跨文件缺陷漏检。通用 Agent 没有结构化的方式在这个权衡中取点。

### 2.3 幻觉评论

Agent 会捏造不存在的事实（幻觉的行号、不存在的 API、错误的语义归因）。在评审场景，每条幻觉评论都消耗人工验证成本，且侵蚀对系统的信任——高召回低精度的系统最终会被团队弃用。

---

## 三、方案总览：三阶段确定性注入

### 3.1 设计哲学：Determinism over Non-Determinism

OpenCodeReview 的核心主张反直觉：**不给 Agent 最大自由，而是在审查流水线的三个刻意选择的节点注入确定性，反而在所有模型后端上超越了约束更少的系统。**

"确定性"不是指用规则取代 LLM——语义判断（发现缺陷、归纳风险）仍然完全交给模型。确定性约束的是**模型之外的边界**：看什么文件、按什么规则、能调用哪些工具、评论落在哪一行、哪些评论可以被推翻。论文称之为 "deterministic engineering for uncertain agents"。

### 3.2 三阶段流水线

```text
┌─────────────────────────────────────────────────────────────────┐
│  阶段一：Rule-Guided Dispatch（规则引导调度）                       │
│  规则解析 → 文件过滤 → 并发分发 → 每文件独立 SubAgent                 │
├─────────────────────────────────────────────────────────────────┤
│  阶段二：Grounded File Review（接地文件审查）                       │
│  每文件：Plan 阶段 → Main 阶段（ReAct 循环 + 6 个有界工具）           │
│          → 评论收集（行号解析 → 重定位）                             │
├─────────────────────────────────────────────────────────────────┤
│  阶段三：Independent Reflection（独立反思）                        │
│  REVIEW_FILTER_TASK：只看 diff 的事实核查 → 删除可证伪的评论         │
└─────────────────────────────────────────────────────────────────┘
```

主流程入口在 `internal/agent/agent.go:229` 的 `Run()`：解析 diff → 注入 DiffMap → 过滤 → `dispatchSubtasks`（并发分发）→ 汇总评论。接下来按三个阶段逐一深入。

---

## 四、阶段一：Rule-Guided Dispatch——让文件分诊可复现

### 4.1 规则系统：四层优先级链

同一份 PR，每次运行都应该把相同的文件分配给相同的审查标准。OpenCodeReview 用**规则驱动**而非 Agent 驱动实现这一点。规则是自然语言文档（Markdown），通过 glob 路径模式决定哪些文件在范围内、按什么检查清单审查。

四层优先级（`internal/config/rules/system_rules.go:254-305`）：

| 层级      | 来源                               | 作用域             |
| --------- | ---------------------------------- | ------------------ |
| 1（最高） | `--rule` 参数指定文件              | 单次调用           |
| 2         | 仓库内 `.opencodereview/rule.json` | 项目级，随版本控制 |
| 3         | `~/.opencodereview/rule.json`      | 用户全局，跨项目   |
| 4（最低） | 内置系统规则（embedded）           | 语言默认           |

规则解析的核心是 `composedResolver`：按优先级逐层查找，**first-match-wins**。用户规则与系统规则有两种交互模式：**Replace 模式**（默认，用户规则完全取代系统规则）与 **Merge 模式**（`merge_system_rule: true` 时与系统规则拼接合并，`system_rules.go:199,391-392`）。内置规则按文件类型路由审查重点（`system_rules.json`）：

```json
"**/*.java":           "java.md",           // NPE、线程安全、N+1 查询
"**/pom.xml":          "pom_xml.md",        // 禁止 SNAPSHOT 版本依赖
"**/package.json":     "package_json.md",   // 禁止 latest/wildcard 依赖
"**/*.{ts,js,tsx,jsx}": "ts_js_tsx_jsx.md",  // XSS、竞态、async 陷阱
"**/Cargo.toml":       "cargo_toml.md"
```

30+ 种语言/文件类型各有专属规则文件。规则是纯文本 Markdown，团队可以 fork 内置规则或编写自己的规则——确定性内容可以版本控制、评审、演化。

### 4.2 文件过滤：六阶段漏斗

被纳入评审的文件经过六道过滤（`preview.go:34-57` 的 `whyExcluded` 实现）：

1. **排除二进制文件**（`ExcludeBinary`，`preview.go:35-36`）
2. **用户排除模式**（`FileFilter.IsUserExcluded` → `ExcludeUserRule`）
3. **用户包含模式**（显式 include 时反向白名单，命中即放行）
4. **内置扩展名白名单**（`.py`、`.go`、`.java`…，不认识的扩展名不审 → `ExcludeExtension`）
5. **排除测试与生成代码**（`IsExcludedPath`，`internal/config/allowlist/allowed_ext.go:91`）——默认排除模式清单（`default_exclude_patterns.json`）覆盖 `**/*_test.go`、`**/testdata/**`、`**/fixtures/**`、`**/*.generated.*`、`**/*.gen.go`、`**/*.pb.go` 等
6. **过滤超大 diff**——diff 内容单独超过模型上下文窗口 80% 的文件直接跳过（`agent.go:1375` 的 `filterLargeDiffs`）

第六道过滤值得注意：它防的不是"模型看不下"，而是**预算失控**——一个大文件可以吃掉整个上下文预算，挤压其余文件。

### 4.3 每文件并发 SubAgent

调度器 `dispatchSubtasks`（`agent.go:490`）为每个文件启动一个独立 goroutine：

```go
sem := make(chan struct{}, concurrency)  // 默认并发 8
// ...
go func(d model.Diff) {
    completed, stop, err := a.executeSubtask(fileCtx, d)
    // panic 隔离：单文件崩溃不影响其余文件
    // 超时控制：ConcurrentTaskTimeout 分钟
} (toDispatch[i])
```

每个 SubAgent 的输入包括：文件 diff、解析后的规则、其他变更文件列表、用户背景信息——模板占位符（`{{current_file_path}}`、`{{change_files}}`、`{{system_rule}}`、`{{diff}}` 等）在 `executeSubtask` 中逐一替换（`agent.go:1146-1163`）。文件间依赖不预判——某文件审查中发现需要看其他文件时，通过 `file_read_diff` 工具按需获取。这与"单 Agent 一次看完所有文件"形成对照：

> **注**：README 宣传的 "Smart file bundling"（将相关文件捆绑为一个审查单元，如 `message_en.properties` 与 `message_zh.properties`）实现在委派模式（delegate）路径中——按解析后的规则内容分组文件（`internal/delegate/rulegroup.go:21-30`）；本文聚焦的论文三阶段主路径是**逐文件分发**，与代码一致。

| 维度         | Monolithic 单 Agent | OCR 文件级并行         |
| ------------ | ------------------- | ---------------------- |
| 上下文连贯性 | 完整                | 文件内完整，跨文件按需 |
| 成本         | 随文件数线性爆炸    | 每文件独立，并发摊薄   |
| 单文件故障   | 全 PR 失败          | 隔离，其余文件继续     |
| 可复现性     | 工具路径随机        | 同一 PR 恒定分诊       |

**"给定"在此处是**：既然规则和过滤是确定性的，**"新"** 是：分诊结果每次运行完全一致——这是可复现性的第一块基石。

---

## 五、阶段二：Grounded File Review——有界的探索才有深度

Grounded（接地）指评论必须**锚定在 diff 的证据上**：模型可以用工具自由探索上下文，但最终产出的每条评论都要落到当前文件 diff 中的真实代码片段上，不允许悬空的泛泛之谈。这是"有界探索"与"精确输出"之间的桥梁。

### 5.1 Plan + Main 双阶段

每个文件先跑 **Plan 阶段**（`executePlanPhase`，`agent.go:1462`）：把 diff、规则、变更文件列表发给模型，让它输出审查计划（关注点、路线）。Plan 的输出**不会**直接决定 Main 的执行，而是作为 Main 的上下文。当文件 diff 行数低于 `PLAN_MODE_LINE_THRESHOLD = 50`（`task_template.json`）时跳过 Plan——小改动不值得规划。

Main 阶段（`MAIN_TASK`）才是真正的审查循环：模型在提示词约束下，调用工具取证、最终通过 `code_comment` 提交评论。

### 5.2 六个有界工具：收束动作空间

通用 Agent 挂载几十个工具（终端、文件写、浏览器…）。OpenCodeReview 只给审查者六个工具，每个对应人类审查者的一种信息需求（`internal/tool/definitions.go:14-22`）：

| 工具             | 用途                    | 输出上限（源码验证）                               |
| ---------------- | ----------------------- | -------------------------------------------------- |
| `file_read`      | 按路径+行范围读完整文件 | 最多 500 行/次（`file_read.go:12`）                |
| `file_find`      | 按文件名关键字定位文件  | 最多 100 结果，10 秒超时（`file_find.go:19-20`）   |
| `code_search`    | 跨仓库文本搜索          | 最多 100 匹配，10 秒超时（`code_search.go:18-19`） |
| `file_read_diff` | 查看其他变更文件的 diff | 返回预计算 diff                                    |
| `code_comment`   | 提交审查评论            | 结构化 JSON，必须含 `existing_code`                |
| `task_done`      | 终止循环                | state=DONE/FAILED                                  |

**有界输出的设计意图**：每个工具的输出都有硬上限（行数、结果数、超时），从根上防止 "token snowball"（工具返回海量结果挤爆上下文）。论文指出，这六个工具是从大规模生产数据中**工具调用轨迹分析**提炼出的最小充分集——审查者真正需要的信息需求就这六种。

工具注册表在初始化后 `Freeze()`（`agent.go:262`），运行期间不可增删——动作空间的边界是编译期常量，不是 prompt 里的软性建议。

**严格聚焦规则（Strict Focus Rules）**。主任务提示词（`main_task_system.md`）中还有一条与工具边界同等重要的约束：

> Context tools are for understanding purposes only. Findings from other files must NOT become the subject of your comments.

工具只能用于**理解**，其他文件中发现的任何问题都**不得**成为评论主题——评论只能落在当前文件的 diff 上。这防止了一个典型的 Agent 行为漂移：取证过程中发现"顺手问题"，从而把审查范围悄悄扩大，破坏文件级隔离的边界。

### 5.3 ReAct 循环与三类终止

`RunPerFile`（`internal/llmloop/loop.go:177-308`）驱动主循环：

```go
toolReqCount := r.deps.Template.MaxToolRequestTimes  // 默认 30
const maxConsecutiveEmptyRounds = 3
for toolReqCount > 0 {
    resp, err := r.deps.LLMClient.CompletionsWithCtx(...)
    // 解析工具调用 → 执行 → 把结果追加回 messages
    if !hasValidResult {
        consecutiveEmptyRounds++
        if consecutiveEmptyRounds >= 3 { stop = StopEmptyRounds; break }
    }
}
```

三类终止条件，且每个终止都有**精确分类**（`loop.go:146-167`）：`StopMaxRounds`（30 轮预算耗尽）、`StopEmptyRounds`（连续 3 轮无有效工具结果）、`StopCompression`（压缩后仍超阈值）。不用自由文本猜测循环为什么停了——这对可观测性和预算审计至关重要。

### 5.4 三区域上下文压缩

长审查对话的上下文管理采用**三区域策略**（`compression.go:21-25`）：

```text
tokenSoftThreshold = 0.60  // 60%：触发异步后台压缩
PromptTokenLimit  = 0.80  // 80%：触发同步强制压缩
```

- **< 60%**：不干预
- **60%–80%**：异步压缩——后台把早期轮次的消息折叠为摘要，不阻塞主循环（`triggerAsyncCompression`）
- **> 80%**：同步压缩——必须等压缩完成才能继续；压缩后仍超阈值则终止（`addNextMessage` 返回 false → `StopCompression`）

压缩分区本身是三区域结构（`partitionMessages`，`compression.go:124-126`）：**frozen**（`messages[0:2]`，系统提示与初始用户消息永不压缩）、**compress**（中间轮次折叠为摘要）、**active**（最近的 K 个完整轮次按 token 预算保留原样，`computeActiveZoneSize` 从尾部倒推）。压缩按"轮次"分组（`groupIntoRounds`）——审查者最近发现的信息密度最高，旧结论只需保留摘要。

### 5.5 行号三级回退：告别"行号幻觉"

这是全系统最巧妙的设计。模型**不直接输出行号**——它输出 `existing_code`（diff 中存在的代码片段），行号由确定性算法计算。工具定义中明确说明（`tools.json`）：

> "This tool uses a dynamic sliding window algorithm to match corresponding consecutive lines in diff text based on your provided 'existing_code' parameter."

三级回退（`internal/diff/resolver.go:62-73`）：

1. **Hunk 精确匹配**（`resolveFromHunk`）：将 `existing_code` 与 diff 的 hunk 逐行匹配，先新侧（context+added 行），再旧侧（context+deleted 行）——`resolver.go:85-115`
2. **全文滑动窗口**（`resolveFromFileContent`）：hunk 匹配失败时，在新文件全文中做连续行匹配，跳过空白行——`resolver.go:172-217`
3. **LLM 辅助重定位**（`ReLocateComment`）：前两级都失败时，把 diff 和评论交给一个专门的定位模型提取正确代码片段——`loop.go:422-439`，提示词见 `re_location_task_system.md`

关键约束（`loop.go:384-388`）：模型可能幻觉路径，**`code_comment` 的 path 参数被强制注入**为当前文件路径：

```go
// Always inject the current file path for code_comment.
// The model sometimes hallucinates a path, so we override it.
if t == tool.CodeComment && newPath != "" {
    args["path"] = newPath
}
```

评论收集是异步的（`CommentWorkerPool`，`pool.go`）：行号解析、重定位、后续处理从主循环剥离到 worker 池，主循环不被 IO 阻塞。

---

## 六、阶段三：Independent Reflection——用信息边界做证伪

### 6.1 非对称信息边界

所有评审 Agent 的问题在于**自我强化偏差**：同一个模型既探索上下文又产评论，它不会推翻自己基于探索形成的结论。Reflexion、Self-Refine 等方案用"同上下文自批判"解决，但模型面对相同上下文时往往重复同一判断。

OpenCodeReview 的做法是**不对称信息**（`review_filter_task_system.md`）：

> You are a fact-checker for code review comments. These review comments come from an Agent that can invoke tools to obtain the full code context. **You can currently only see the code diff.**

反思者（ReviewFilter）只看到 diff + 评论列表，**看不到** SubAgent 通过工具获得的仓库上下文。它无法验证评论的正确性（因为没有探索上下文），但可以——也只需要——**证伪**。

### 6.2 证伪而非验证

反思提示词的核心原则（`review_filter_task_user.md`）：

> **Core principle: You need to falsify, not verify.**

两步评估：

1. **事实核查（Veto Rule）**：只有 diff 中存在直接反证时才标记评论为错误；评论引用了 diff 外的上下文（可能是 Agent 通过工具获得的），**不**标记——因为反思者无法证伪它。
2. **问题分类**：事实核查通过后，判断描述是否有 diff 可证明的显著偏差（把正常代码误判为缺陷、对可见行为的归因与代码矛盾）。

### 6.3 纯过滤设计 + fail-open

反思阶段**只能删评论，不能生成评论**（`agent.go:1293`）：

```go
indices := parseFilterResponse(resp.Content(), len(comments))
a.args.CommentCollector.RemoveByPathAndIndices(newPath, indices)
```

两个工程细节体现设计哲学：

- **fail-open**：反思 LLM 调用失败时，"Errors are logged and silently ignored"（`agent.go:1230`）——保留所有评论，优先保召回。删错比留错更危险（漏报真实问题），所以失败时倾向保守。
- **解析失败即放行**：反思输出无法解析时返回空集，不删任何评论。

论文指出，反思的独立性来自**信息边界而非模型身份**——不需要专门的小模型，同一个 LLM 在更少的信息下天然具备不同的判断视角。这是对"用更强模型解决一切"的最有力反驳：**判断的独立性，取决于它看到什么，而不是它是什么。**

---

## 七、权衡：精度与成本的交换

### 7.1 基准结果

| 指标                 | OpenCodeReview | Claude Code    | Codex |
| -------------------- | -------------- | -------------- | ----- |
| SEM-F1（最优）       | **25.10%**     | 14.13%         | 8.36% |
| 精度范围             | 25.20%–37.80%  | 7.23%–15.93%   | ~28%  |
| Token 消耗（同模型） | 385K           | 5,664K（≈15×） | —     |
| 每 PR 评论量         | 克制           | 可达 4,580 条  | —     |

**成本-质量双优**：OpenCodeReview 在 token 消耗仅为 Claude Code 1/5–1/15 的情况下，SEM-F1 反而高 1.3–2.2×；对 Codex 则在**相近成本**下 SEM-F1 高 2.5×。省钱的来源正是确定性：文件级并行避免重复扫描、六工具避免无界探索、反思阶段只删不增避免二次生成。

### 7.2 取舍的本质：精度优先，召回让位

这是刻意的产品决策。论文承认 Recall 较低——**这是"少而准"对"多而糙"的有意选择**。在一个评论会消耗人工验证成本的场景，精度是系统的生存指标：低精度高召回的系统最终会被工程师静默忽略（并可能引入新的缺陷——盲目信任 AI 评论）。

同时需指出评估的外部有效性边界：基准仅覆盖 200 个 PR、10 种语言、50 个仓库，不能覆盖所有工业代码库；对比基线只有 Claude Code 与 Codex 两个（虽然都是当时行业最新）。此外 SEM-F1 不衡量评论的可操作性、清晰度与严重性——一条"准"但无用的评论与一条"准"且关键的评论得分相同。这些局限不削弱三阶段设计本身的结论，但决定了结果的适用范围。

### 7.3 三个反直觉设计回顾

1. **约束动作空间，而非扩大模型**：最弱的 OCR 配置（DeepSeek-V4-Pro）超过最强的 Claude Code 配置——因为前者约束了模型能做的事，后者依赖模型自己判断。
2. **信息边界，而非模型边界**：反思不需要更强的模型，需要更少的上下文。
3. **确定性最便宜，却最有效**：六阶段文件过滤、三级行号回退、四层规则——这些都是几十行代码的确定性逻辑，贡献却超过任何一次模型升级。

---

## 八、总结：可迁移的设计原则

OpenCodeReview 的价值不止于一个评审工具——它是**"确定性工程 × Agent"混合架构**的完整范例，其设计原则可以迁移到任何"输出部分可验证"的 Agent 任务（测试生成、安全审计、文档审查）：

**原则一：把不确定性隔离到最小语义单元。** LLM 只负责"发现缺陷、归纳风险"，其余一切（看什么、怎么找、落在哪、能否存活）都是确定性代码。边界越清晰，模型越容易在其内发挥。

**原则二：有界工具 > 自由工具。** 六个有界工具经过真实调用轨迹验证，覆盖审查者的全部信息需求。无界探索不是能力，是随机性。工具的"界"应该是编译期常量，而非 prompt 里的建议。

**原则三：证伪比验证便宜。** 验证需要全部上下文，证伪只需要反证。反思者只给 diff——信息不对称反而制造了真正的独立视角。对输出可部分验证的 Agent 任务，"独立核查者 + 少信息"是通用模式。

**原则四：失败时偏向安全侧。** fail-open（反思失败保留全部评论）、fail-safe（定位失败降级全文扫描）——确定性系统的容错策略本身就是设计的一部分。

最后回到论文的核心结论：**系统设计对审查质量的贡献大于模型选择**。当所有人都追逐更强的模型时，OpenCodeReview 证明了另一条路——把 Agent 的边界设计好，比把 Agent 变大更重要。

---

> **参考与延伸阅读**
>
> - 项目源码：[github.com/alibaba/open-code-review](https://github.com/alibaba/open-code-review)（Apache 2.0，Go 实现；仓库 README 为官方文档入口）
> - 论文：[OpenCodeReview: Determinism over Non-Determinism for Cost-Effective Agent-Based Code Review](https://arxiv.org/html/2608.09290v1)（arXiv:2608.09290，阿里 + 南京大学 + 北大）
> - 相关：[ReAct Agent 模式详解](./react-agent.md) — 本文 §5 的循环机制源于 ReAct 模式
