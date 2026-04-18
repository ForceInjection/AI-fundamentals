# Claude Code 上下文压缩机制深度解析

大语言模型由于自身架构的限制，具备严格的 Token 输入上限。在经历几十轮源码分析和工具调用后，会话极易触及内存天花板。为了系统性地解决长程任务中的记忆暴涨问题，Claude Code 实现了一套非常完善的上下文压缩机制（Context Compression），在不丢失核心逻辑的前提下主动丢弃冗余信息，从而保证了复杂任务的长时稳定运行。

## 目录 (TOC)

- [1. 全局架构：5 层防御体系](#1-全局架构5-层防御体系)
- [2. L1 层：上下文窗口限制 (Context Window)](#2-l1-层上下文窗口限制-context-window)
- [3. L2 层：宏观压缩 (Macro Compaction)](#3-l2-层宏观压缩-macro-compaction)
  - [3.1 阈值预警与安全边界](#31-阈值预警与安全边界)
  - [3.2 主动压缩：会话记忆引擎 (SM Compact)](#32-主动压缩会话记忆引擎-sm-compact)
  - [3.3 主动压缩：传统摘要降级 (Legacy Compact)](#33-主动压缩传统摘要降级-legacy-compact)
  - [3.4 响应式压缩与上下文折叠 (Reactive Compact and Context Collapse)](#34-响应式压缩与上下文折叠-reactive-compact-and-context-collapse)
- [4. L3 与 L4 层：细粒度微压缩与 L5 收尾 (Micro Compaction & Cleanup)](#4-l3-与-l4-层细粒度微压缩与-l5-收尾-micro-compaction--cleanup)
  - [4.1 L3：缓存编辑微压缩 (Cached MC)](#41-l3缓存编辑微压缩-cached-mc)
  - [4.2 L4：时间触发微压缩 (Time-Based MC)](#42-l4时间触发微压缩-time-based-mc)
  - [4.3 L5：压缩后状态清理 (Post-Compact Cleanup)](#43-l5压缩后状态清理-post-compact-cleanup)
- [5. 核心技术特性与隔离机制](#5-核心技术特性与隔离机制)
  - [5.1 动态与静态的缓存边界隔离](#51-动态与静态的缓存边界隔离)
  - [5.2 强制保留关键决策与代码片段](#52-强制保留关键决策与代码片段)
  - [5.3 面向多场景的隔离压缩策略](#53-面向多场景的隔离压缩策略)
- [6. 实战案例：基于真实源码的压缩推演](#6-实战案例基于真实源码的压缩推演)
  - [6.1 第一阶段：高频检索与微压缩 (L3/L4)](#61-第一阶段高频检索与微压缩-l3l4)
  - [6.2 第二阶段：知识沉淀与会话记忆 (L2 SM)](#62-第二阶段知识沉淀与会话记忆-l2-sm)
  - [6.3 第三阶段：触及红线与宏观压缩 (L2 Legacy)](#63-第三阶段触及红线与宏观压缩-l2-legacy)
  - [6.4 第四阶段：异常触发与极速恢复 (L2 Reactive)](#64-第四阶段异常触发与极速恢复-l2-reactive)
- [7. 总结（各层对比与设计哲学）](#7-总结各层对比与设计哲学)
  - [7.1 压缩策略横向对比](#71-压缩策略横向对比)
  - [7.2 核心设计哲学](#72-核心设计哲学)

---

## 1. 全局架构：5 层防御体系

Claude Code 的上下文压缩并非单一操作，而是由多维策略组合而成的记忆管理引擎，可抽象为 5 个独立但又互相配合的层次（L1 ~ L5）。以下是系统架构的全局流程图：

```text
Claude Code Compaction Architecture
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ L1 Context Window     effectiveCW = CW - min(maxOut, 20K)                                       │
└───────────────────────────────────────────────────────────────┬─────────────────────────────────┘
                                                                │
┌───────────────────────────────────────────────────────────────▼─────────────────────────────────┐
│ L2 Macro Compaction                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ warn/err ~82%  |  autoCompact ~93%  |  blocking ~98%                                        │ │
│ └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
│ ┌───────────────────────────────────────────────────┐ ┌───────────────────────────────────────┐ │
│ │                  Proactive                        │ │        Reactive / Experimental        │ │
│ │ ┌───────────────────────────────────────────────┐ │ │ ┌───────────────────────────────────┐ │ │
│ │ │ Auto Compact  SM first -> Legacy fallback...  │ │ │ │ Reactive Compact                  │ │ │
│ │ └───────────────────────────────────────────────┘ │ │ │ 413 error recovery                │ │ │
│ │ ┌────────────┐  ┌────────────────┐  ┌───────────┐ │ │ └───────────────────────────────────┘ │ │
│ │ │ SM Compact │─>│ Legacy Compact │─>│ /compact  │ │ │ ┌───────────────────────────────────┐ │ │
│ │ │ Zero-LLM   │  │ Full-LLM sum.  │  │ Manual CLI│ │ │ │ Context Collapse                  │ │ │
│ │ └────────────┘  └────────────────┘  └───────────┘ │ │ │ Read-time projection              │ │ │
│ └───────────────────────────────────────────────────┘ │ └───────────────────────────────────┘ │ │
│                                                       └───────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┬─────────────────────────────────┘
                                                                │
┌───────────────────────────────────────────────────────────────▼─────────────────────────────────┐
│ L3 Micro Compaction     Clear old tool results  |  Cached MC (cache_edits)                      │
└───────────────────────────────────────┬───────────────────────────────────────┬─────────────────┘
                                        │                                       │
┌───────────────────────────────────────▼─────────────┐ ┌───────────────────────▼─────────────────┐
│ L4 Time-Based MC        gap > 60 min                │ │ L5 Post-Compact Cleanup Reset stale...  │
└─────────────────────────────────────────────────────┘ └─────────────────────────────────────────┘
```

这 5 层机制从最底层的 API 报错恢复，到日常静默的缓存清理，共同维持了 Token 消耗的平衡，实现了“无缝自救”。

---

## 2. L1 层：上下文窗口限制 (Context Window)

为了确保系统在发生 Token 溢出前拥有足够余量去执行摘要生成，Claude Code 设定了严格的可用窗口硬性计算标准。

根据代码库的定义，模型真正的有效上下文窗口（Effective Context Window）并非官方公布的绝对上限，而是需要扣除摘要生成时的预留空间（`MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000`）。在此基础上，系统会继续预留 13,000 个 Token 作为触发后续 L2 宏观压缩的安全垫（`AUTOCOMPACT_BUFFER_TOKENS`）。

```typescript
// 来源于 src/services/compact/autoCompact.ts
// 计算自动压缩的触发阈值
export function getAutoCompactThreshold(model: string): number {
  // 获取扣除了摘要输出预留空间（20,000 Tokens）后的有效上下文窗口
  const effectiveContextWindow = getEffectiveContextWindowSize(model);

  // 触发阈值为有效窗口再减去 13,000 Tokens 的安全缓冲
  const autocompactThreshold =
    effectiveContextWindow - AUTOCOMPACT_BUFFER_TOKENS;

  return autocompactThreshold;
}
```

---

## 3. L2 层：宏观压缩 (Macro Compaction)

L2 层是应对 Token 暴涨的核心防线，涵盖了从阈值预警、主动提炼到异常恢复的完整状态机闭环，能够在静默状态下用浓缩摘要替换早期对话记录。

### 3.1 阈值预警与安全边界

系统定义了多个梯度的触发水位线，以实现平滑降级处理，避免突然中断正在执行的操作链。

以 Opus 模型的 200K 上下文为例，扣除 20K 预留后，有效窗口为 180K。源码中定义了三个关键拦截点：

- **warn/err (~82%)**：达到 147K 时，UI 会提示用户 Context Low。
- **autoCompact (~93%)**：达到 167K 时，触发系统的主动压缩流。
- **blocking (~98%)**：达到 177K 时，强制阻塞后续操作直到用户手动处理。

### 3.2 主动压缩：会话记忆引擎 (SM Compact)

主动压缩是 L2 层的核心动作，其内部具有优先级路由：系统会优先尝试零 LLM 消耗的会话记忆压缩（SM Compact）。

会话记忆引擎是一个完全独立于主对话流的后台服务。它在用户与模型交互的同时，以 Markdown 文件的形式持续记录“学习”到的关键信息。为了防止记忆提取过程污染当前主线程的 Token 状态，Claude Code 利用了子智能体（Forked Agent）技术来实现底层隔离，并通过双重阈值控制其更新频率。

```typescript
// 来源于 src/services/SessionMemory/sessionMemory.ts
// 启动后台隔离的智能体来提取记忆
await runForkedAgent({
  // 将当前用户的提示词传入隔离环境
  promptMessages: [createUserMessage({ content: userPrompt })],
  // 创建隔离的运行环境缓存，避免状态突变
  cacheSafeParams: createCacheSafeParams(context),
  // 仅赋予读写特定记忆文件的权限，保障主系统安全
  canUseTool: createMemoryFileCanUseTool(memoryPath),
  querySource: "session_memory",
  forkLabel: "session_memory",
  overrides: { readFileState: setupContext.readFileState },
});
```

### 3.3 主动压缩：传统摘要降级 (Legacy Compact)

当会话记忆不足以释放足够空间，或提取失败时，系统将降级为传统摘要压缩（Legacy Compact）。

传统压缩的本质是引入了一个 `compact_boundary`（压缩边界）消息。系统会将早于边界的旧消息发送给 Claude API 进行摘要生成（Summarize），最终拼接的结构为：`[历史摘要] + [压缩边界] + [近期最新消息]`。

> [!NOTE]
> **防死循环断路器**：在长程任务中，大模型在生成摘要时有可能会遭遇失败。如果允许无限重试，将导致 API 调用的严重浪费。源码 `autoCompact.ts` 中设定了 `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3`。一旦连续失败达到 3 次，系统将放弃当前的自动压缩尝试。

### 3.4 响应式压缩与上下文折叠 (Reactive Compact and Context Collapse)

响应式压缩与上下文折叠是两个正交维度的机制：前者是被动触发的异常恢复流程，后者是可选的内容组织与结构化摘要策略。

- **响应式压缩 (Reactive Compact)**：作为触发时机机制，当发生 API `413 Prompt-too-long` 错误时激活。系统被动捕获该错误后，会启动紧急的上下文丢弃与恢复流程，以最快速度让会话重回可用状态。
- **上下文折叠 (Context Collapse)**：作为内容组织算法，聚焦于提高大模型对历史信息的注意力机制（Attention）效率。激活后，系统会重构历史记录的组织形式，将高度相关但散落各处的信息进行逻辑折叠或重组，从而优化摘要质量并提升模型的推理能力。

---

## 4. L3 与 L4 层：细粒度微压缩与 L5 收尾 (Micro Compaction & Cleanup)

微压缩是每次向大模型发送请求前执行的最轻量操作（耗时 < 1ms），它不调用大模型，而是通过正则表达式与白名单直接清理旧工具输出结果（如 `Read`、`Bash`），从而保留语义配对的同时释放 Token 空间。在架构分工上，L3 主要负责基于 API 缓存特性的缓存编辑微压缩，而 L4 则负责处理缓存失效后的时间触发微压缩。

### 4.1 L3：缓存编辑微压缩 (Cached MC)

在 API 缓存仍有效时触发，系统不修改本地消息记录，而是利用特殊的 `cache_edits` API 字段通知服务端定点清除特定工具缓存。这不仅清空了无用数据，还完美保护了剩余内容的 Prompt Cache 命中率。

```typescript
// 来源于 src/services/compact/apiMicrocompact.ts
// 只有这些工具的结果内容会被微压缩清理
const TOOLS_CLEARABLE_RESULTS = [
  ...SHELL_TOOL_NAMES,
  GLOB_TOOL_NAME,
  GREP_TOOL_NAME,
  FILE_READ_TOOL_NAME,
  WEB_FETCH_TOOL_NAME,
  WEB_SEARCH_TOOL_NAME,
];
```

### 4.2 L4：时间触发微压缩 (Time-Based MC)

当系统判定距离上次交互时间过久（默认 `gap > 60 min`），API 缓存已过期时触发。由于不再需要保护缓存，系统会直接修改本地消息历史，将旧的工具输出内容替换为 `[Old tool result content cleared]` 占位符，从而在重传时硬性节省 Token。

### 4.3 L5：压缩后状态清理 (Post-Compact Cleanup)

本层负责在压缩流程结束后执行收尾工作，确保会话环境的绝对一致性。

在宏观压缩或微压缩修改了历史消息流之后，系统需要清理残留的脏标记。例如，重置 `readFileState` 中的过期文件读取记录，或者复位 UI 层的进度指示器状态。L5 层的状态清理虽然对 Token 计算没有直接影响，但它是防止上下文逻辑断裂和保持系统内部状态同步的必要兜底机制。

---

## 5. 核心技术特性与隔离机制

为了保障上下文压缩的高效执行与数据安全性，Claude Code 在底层架构上设计了严格的边界隔离与特征保留逻辑。

### 5.1 动态与静态的缓存边界隔离

系统在构建全局提示词时，强制引入了 `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__` 常量。通过该标记，框架将具备复用价值的静态规则（如 `CLAUDE.md`）与会话独有的动态状态（如当前工作区目录）实施了物理切分，最大化 Prompt Caching 成本效益的同时避免核心约束被稀释。

### 5.2 强制保留关键决策与代码片段

上下文压缩的目标是提取“接下来要做什么”，但这并不意味着系统会丢弃过去的实现细节。

根据压缩指令模板，系统会明确指示大模型在摘要过程中必须分析并重点保留：关键决策与技术概念（`Key decisions, technical concepts and code patterns`）以及完整的代码片段（`full code snippets`）。为了在极少的 Token 限制内提炼出最高质量的摘要，Claude Code 设计了两阶段的压缩提取算法：

```typescript
// 来源于 src/services/compact/prompt.ts
// 剥离思维链推理过程，仅保留最终结论
export function formatCompactSummary(summary: string): string {
  // 1. 删除 <analysis> 块（草稿阶段，已无价值）
  let formattedSummary = summary.replace(/<analysis>[\s\S]*?<\/analysis>/, "");

  // 2. 提取 <summary> 内容，替换为可读标题
  const match = formattedSummary.match(/<summary>([\s\S]*?)<\/summary>/);
  if (match) {
    formattedSummary = formattedSummary.replace(
      /<summary>[\s\S]*?<\/summary>/,
      `Summary:\n${match[1].trim()}`,
    );
  }

  return formattedSummary.trim();
}
```

### 5.3 面向多场景的隔离压缩策略

除了标准的对话流压缩，系统还针对复杂的 Agent 编排模式设计了多场景独立的策略，以确保状态不被相互污染：

- **子 Agent 与 工作树（worktree）沙箱**：系统支持通过配置隔离属性（`isolation: 'worktree' | 'remote'`），赋予子 Agent 独立的上下文环境，其压缩过程不会影响主对话流。
- **ULTRAPLAN 远程规划会话**：通过专属标记（如 `ULTRAPLAN_TELEPORT_SENTINEL`），系统能将重度的规划任务分流到远程环境，并在完成后将最终的计划提取回本地上下文。

---

## 6. 实战案例：基于真实源码的压缩推演

本章结合 Claude Code 的实际源码逻辑与事件埋点（Telemetry Events），推演一个长程任务（如“大型遗留 React 项目迁移”）在开发过程中是如何触发这 5 层压缩机制的。

### 6.1 第一阶段：高频检索与微压缩 (L3/L4)

开发者首先让 Claude Code 了解项目结构。系统频繁调用 `Grep` 和 `Read` 工具扫描了几十个文件。

- **L3 缓存编辑微压缩**：每次 API 请求前，源码中的 `src/services/compact/microCompact.ts`（该模块协调了缓存编辑与时间触发两种策略）会通过底层的 `apiMicrocompact.ts` 生成 `cache_edits` 块，静默删除早期 `Read` 工具的大量返回内容，并触发 `tengu_cached_microcompact` 埋点。这一步在清理数百行代码输出的同时，完美保住了 Prompt Cache 命中率。
- **L4 时间触发微压缩**：当开发者吃午饭休息超过 1 小时后继续提问，由于 `gapMinutes > config.gapThresholdMinutes`，缓存已自然失效。此时 `tengu_time_based_microcompact` 介入，将旧工具结果直接替换为占位符以硬性节省 Token。

### 6.2 第二阶段：知识沉淀与会话记忆 (L2 SM)

在重构过程中，开发者与系统讨论并确定了“所有新组件必须使用 Tailwind CSS”这一核心规范。

- 此时，主窗口 Token 逐渐上涨，后台（`src/services/SessionMemory/sessionMemory.ts`）的 `runForkedAgent` 悄悄启动，将规范持久化到 `.claude/memory` 中。
- 当系统检测到 Token 压力并尝试压缩时，会优先进入 `trySessionMemoryCompaction` 分支（触发 `tengu_sm_compact` 相关埋点）。由于它直接读取外部生成的 Markdown 记忆文件，完全不调用大模型生成摘要，成功在主线程实现了零 Token 消耗的上下文瘦身。

### 6.3 第三阶段：触及红线与宏观压缩 (L2 Legacy)

随着不断修改文件，上下文 Token 飙升至 `autoCompactThreshold`（有效窗口扣除 13,000 Token 缓冲后的红线）。

- 此时由于 Session Memory 已无法释放更多空间，系统回退到 `src/services/compact/autoCompact.ts` 中的 `compactConversation` 传统摘要降级。
- 系统截断早期的对话轮次，调用 LLM 生成摘要并插入 `compact_boundary` 标记。随后触发 `tengu_compact` 事件（记录 `preCompactTokenCount` 等指标），成功将上下文暴降至安全水位，重构任务得以继续。

### 6.4 第四阶段：异常触发与极速恢复 (L2 Reactive)

在读取一个极其庞大的遗留配置文件时，单次 Token 激增，API 突然返回 `413 Prompt-too-long` 错误。

- `src/services/compact/compact.ts` 源码中的 `MAX_PTL_RETRIES`（最大 413 重试次数）机制被激活，系统触发**响应式压缩**（Reactive Compact），自动截断最旧的 API 轮次并重试，开发者几乎无感知。
- 任务完成后，系统调用 `runPostCompactCleanup()`（位于 `src/services/compact/autoCompact.ts`，即 **L5 压缩后状态清理**），重置底层的 `readFileState` 和其他 UI 状态，防止压缩后的状态发生脏读，系统恢复到最初的干净状态。

---

## 7. 总结（各层对比与设计哲学）

通过上述的深度解析，Claude Code 的复杂上下文管理系统可提炼为一个层次分明、防御递进的架构，其在资源受限环境下的工程权衡极具参考价值。

### 7.1 压缩策略横向对比

不同的压缩策略在调用成本、信息损失度和触发时机上各有侧重。以下表格直观展示了各层的核心差异：

| 属性             | 微压缩 (Microcompact)      | Session Memory             | 传统压缩 (Legacy Compact)   | PTL Recovery (故障恢复) |
| ---------------- | -------------------------- | -------------------------- | --------------------------- | ----------------------- |
| **调用 LLM**     | 否                         | 是（后台隔离，不占主窗口） | 是（通过 Fork Agent）       | 否                      |
| **信息损失**     | 最小（仅删工具输出）       | 中等（依赖已有摘要）       | 中等（生成新摘要）          | 高（丢弃最旧消息）      |
| **延迟**         | < 1 ms                     | < 10 ms                    | 5~30 s                      | ~0 s                    |
| **触发时机**     | 每轮请求前自动触发         | 满足双重阈值时优先触发     | Token 容量见底自动/手动触发 | 压缩自身遭遇 Token 超限 |
| **Prompt Cache** | 保留（使用 `cache_edits`） | 破坏（改变了历史状态）     | 共享（Fork 环境继承缓存）   | 破坏                    |
| **最大输出**     | —                          | —                          | 20,000 Tokens               | —                       |
| **断路器**       | 无                         | 无                         | 有（最大重试 3 次）         | 有（最大重试 3 次）     |

### 7.2 核心设计哲学

Claude Code 并没有简单粗暴地清空历史，而是构建了一个闭环系统。其核心设计哲学可以归纳为以下四点：

1. **渐进式降级 (Progressive Degradation)**：从无损且极低成本的操作（如微压缩）开始，逐渐过渡到有损操作（传统摘要压缩），最后才是直接丢弃信息的保底操作（PTL Recovery）。每一层的防御都比上一层代价更高，确保不到万不得已不丢失信息。
2. **缓存优先 (Cache-First)**：系统高度重视 Prompt Cache 的命中率。例如，微压缩的“缓存编辑”路径专门设计为不破坏已有缓存；而传统压缩通过 Fork Agent 机制，使其能共享主对话流的缓存，大幅降低了生成摘要的计算成本。
3. **严格的安全性与配对保证**：在所有压缩操作中，系统永远不切断 `tool_use` 与 `tool_result` 的逻辑配对，永远不分离共享 `message.id` 的消息块；同时，断路器机制与递归保护逻辑，防止了压缩代理因上下文超限而触发“压缩自身的压缩”的死循环。
4. **高度的可观测性与可配置性**：几乎所有触发阈值都支持环境变量覆盖或通过 GrowthBook 远程动态调整。此外，每一个压缩动作（如 `tengu_compact`）都会被埋点记录，包含压缩前后的 Token 变化、触发原因与重试次数，为后续的持续优化提供了坚实的数据支撑。

透过这五层精密设计的防御工事，Claude Code 成功将大模型的原生限制转化为一个可控、可观测、可自愈的系统变量，为长程智能体任务的稳定运行提供了坚实的底层保障。

---

> *注：本文所有核心机制解析与变量名、事件名（如 `tengu_` 系列埋点）、函数逻辑均严格基于 Claude Code 内部实际代码库，文件路径与执行流程完全与源码保持一致。后续版本更新可能存在部分内部实现调整，请以最新源码为准。*
