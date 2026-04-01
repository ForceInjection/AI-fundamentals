# Claude Code 源码解析：基于 Markdown 文件的持久化记忆机制

与依赖向量数据库的复杂记忆框架不同，Claude Code 采用了一条极简且内聚的纯文本工程化路线。它完全基于本地的 Markdown 文件来实现对话上下文与规则知识的长效留存。该系统通过巧妙的双层目录架构（个人与团队），在保障用户数据隐私的同时促进了项目级知识的共享；并结合智能的异步预取与防劣化机制，构建了一个能随项目共同成长的闭环记忆生态。

## 目录 (TOC)

- [1. 记忆功能全景概述](#1-记忆功能全景概述)
- [2. 记忆系统分层架构设计](#2-记忆系统分层架构设计)
  - [2.1 存储边界划分](#21-存储边界划分)
  - [2.2 索引与实体文件的解耦设计](#22-索引与实体文件的解耦设计)
- [3. 核心代码模块剖析](#3-核心代码模块剖析)
  - [3.1 记忆生命周期管理](#31-记忆生命周期管理)
  - [3.2 自动化信息提取服务](#32-自动化信息提取服务)
  - [3.3 记忆关联与检索机制](#33-记忆关联与检索机制)
  - [3.4 内置能力扩展 (Bundled Skills)](#34-内置能力扩展-bundled-skills)
- [4. 记忆管理中的防劣化机制](#4-记忆管理中的防劣化机制)
  - [4.1 记忆漂移防御](#41-记忆漂移防御)
  - [4.2 跨层级一致性校验](#42-跨层级一致性校验)
- [5. 总结](#5-总结)

---

## 1. 记忆功能全景概述

Claude Code 拥有一个基于本地文件系统的持久化记忆系统，旨在解决多轮对话与跨会话过程中的上下文丢失问题。根据作用域、生命周期与存储位置的不同，系统的记忆主要划分为三种类型：个人记忆、团队记忆与临时（自动化）记忆。

| 记忆类型               | 作用范围       | 默认存储位置                                | 核心作用与特点                                                                                                           |
| ---------------------- | -------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **个人记忆 (Private)** | 仅当前用户     | `~/.claude/projects/<project>/memory/`      | 记录特定用户的交互习惯、工作流偏好与私人指令。跨会话持久化，但不随代码库共享。                                           |
| **团队记忆 (Team)**    | 项目全体协作者 | `~/.claude/projects/<project>/memory/team/` | 承载项目架构约定、代码规范、测试命令等全局知识。可随会话同步，供所有协作者读取。                                         |
| **临时记忆 (Auto)**    | 单次会话沉淀   | `~/.claude/projects/<project>/memory/logs/` | 自动化提取的短期上下文、临时探索的中间笔记。生命周期较短，可经由 `remember` 技能提炼固化为前两种记忆或写入项目配置文件。 |

> [!NOTE]
>
> 1. **目录隔离**：`<project>` 占位符实际上是当前项目 Git 根目录的转义名称，用于隔离不同项目的记忆。
> 2. **自定义覆盖**：用户可通过环境变量 `CLAUDE_COWORK_MEMORY_PATH_OVERRIDE` 或 `settings.json`（`autoMemoryDirectory`）重定向私有记忆路径。为防御路径穿越攻击，系统会强制忽略项目代码库内的路径配置。
> 3. **远程模式**：当处于云端或远程开发环境时（存在 `CLAUDE_CODE_REMOTE_MEMORY_DIR` 变量），基准目录 `~/.claude` 会自动切换至指定的远程目录。

---

## 2. 记忆系统分层架构设计

记忆系统在物理存储层面采用了严格的隔离策略，以适应不同协作场景下的数据隔离诉求。双层架构设计既保证了个人偏好设定的独立性，也支撑了项目级业务知识的有效流转。

### 2.1 存储边界划分

私有记忆与团队记忆在目录结构上各自独立。私有数据仅存储于当前用户的本地环境，侧重记录交互习惯与个人开发偏好；而团队数据则随项目代码库同步，主要承载全局性业务规则与架构约定。具体划分依据源自 `teamMemPrompts.ts` 中的指令定义。

```typescript
// src/memdir/teamMemPrompts.ts
// 明确告知大语言模型两级记忆的不同作用域与物理存储路径
const lines = [
  "## Memory scope",
  "",
  "There are two scope levels:",
  "",
  // 私有记忆：仅当前用户可见，跨会话持久化
  `- private: memories that are private between you and the current user. They persist across conversations with only this specific user and are stored at the root \`${autoDir}\`.`,
  // 团队记忆：项目中所有协作者共享，随会话同步
  `- team: memories that are shared with and contributed by all of the users who work within this project directory. Team memories are synced at the beginning of every session and they are stored at \`${teamDir}\`.`,
];
```

### 2.2 索引与实体文件的解耦设计

为了避免每次会话都加载全量的记忆文件导致 Token 消耗过大，系统采用了“轻量级索引 + 结构化实体文件”的解耦模式。大语言模型在初始化时仅读取索引，当判定某一记忆条目与当前任务相关时，再主动检索具体的实体文件内容。以下是 `memdir.ts` 中定义的两步保存法：

```typescript
// src/memdir/memdir.ts
// 两步保存法：第一步写实体文件，第二步更新索引
const howToSave = [
  "Saving a memory is a two-step process:",
  "",
  "**Step 1** — write the memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) using this frontmatter format:",
  "",
  ...MEMORY_FRONTMATTER_EXAMPLE,
  "",
  `**Step 2** — add a pointer to that file in \`${ENTRYPOINT_NAME}\`. \`${ENTRYPOINT_NAME}\` is an index, not a memory — each entry should be one line, under ~150 characters: \`- [Title](file.md) — one-line hook\`. It has no frontmatter. Never write memory content directly into \`${ENTRYPOINT_NAME}\`.`,
];
```

---

## 3. 核心代码模块剖析

记忆功能的完整生命周期（创建、检索、更新、淘汰）由底层的核心库与上层的自动化服务共同支撑。底层模块保障了数据结构的统一，而服务模块赋予了系统主动学习与归纳的能力。

### 3.1 记忆生命周期管理

`src/memdir/` 目录构成了记忆系统的底座，直接定义了文件的读写规范、不同记忆类型的数据结构，以及防止信息过时的老化检测机制。该模块是记忆系统能够长久、稳定运行的基础。

- **数据结构定义**：`memoryTypes.ts` 中声明了诸如 `project` 和 `reference` 等多种记忆类型。这些分类不仅帮助模型更精准地理解上下文，还设定了严密的规则以处理记忆漂移问题。

````typescript
// src/memdir/memoryTypes.ts
// 实体文件的标准 Frontmatter 格式
export const MEMORY_FRONTMATTER_EXAMPLE: readonly string[] = [
  "```markdown",
  "---",
  "name: {{memory name}}",
  "description: {{one-line description — used to decide relevance in future conversations, so be specific}}",
  `type: {{${MEMORY_TYPES.join(", ")}}}`,
  "---",
  "",
  "{{memory content — for feedback/project types, structure as: rule/fact, then **Why:** and **How to apply:** lines}}",
  "```",
];
````

- **入口组装与提示词构建**：`memdir.ts` 负责将系统状态封装为大语言模型可理解的提示词，并指导模型通过两步流程保存新记忆。
- **新鲜度检测**：`memoryAge.ts` 包含时间维度的评估工具。当系统提取过往记忆时，会根据最后修改时间计算其“新鲜度”，为大语言模型采信该信息提供参考依据。

```typescript
// src/memdir/memoryAge.ts
// 生成系统提示以告知大语言模型记忆文件的陈旧程度
export function memoryFreshnessNote(mtimeMs: number): string {
  const text = memoryFreshnessText(mtimeMs);
  // 若文件在一日内更新，则不显示提示；否则附加系统级提醒
  if (!text) return "";
  return `<system-reminder>${text}</system-reminder>\n`;
}
```

### 3.2 自动化信息提取服务

`src/services/extractMemories/` 目录将零散的对话上下文转换为结构化的长期记忆，是系统实现自动学习的关键链路。这些文件负责将具有长期复用价值的对话信息持久化。

`extractMemories.ts` 与配套的 `prompts.ts` 指导模型对对话历史进行复盘，将高价值信息按照语义分类（而非时间线）写入记忆系统。为了避免信息冗余，系统会要求模型在写入新记忆前，优先检查并更新现存的相关条目。

```typescript
// src/services/extractMemories/prompts.ts
// 指导大语言模型提取记忆的核心规则
const howToSave = [
  "- Organize memory semantically by topic, not chronologically",
  "- Update or remove memories that turn out to be wrong or outdated",
  "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
];
```

### 3.3 记忆关联与检索机制

在模型读取侧，为了避免加载无关记忆污染上下文，系统设计了基于用户提示词（Prompt）的异步预取与相关性打分机制。

当用户输入指令后，系统会触发 `startRelevantMemoryPrefetch`，并利用一个轻量级的 Side Query（通常是快速模型）执行 `findRelevantMemories` 函数。该模块会扫描所有记忆文件的元数据，通过大模型判断哪些文件对当前请求真正有用，从而只将相关的实体文件挂载（Attachments）到主对话上下文中。

```typescript
// src/memdir/findRelevantMemories.ts
// 使用 Side Query 提取最多 5 个相关记忆文件，并规避已经加载过的重复文件
export async function findRelevantMemories(
  query: string,
  memoryDir: string,
  signal: AbortSignal,
  recentTools: readonly string[] = [],
  alreadySurfaced: ReadonlySet<string> = new Set(),
): Promise<RelevantMemory[]> {
  const memories = (await scanMemoryFiles(memoryDir, signal)).filter(
    (m) => !alreadySurfaced.has(m.filePath),
  );
  // ... 触发大模型筛选并返回结果 ...
}
```

### 3.4 内置能力扩展 (Bundled Skills)

`src/skills/bundled/` 目录提供了用户主动介入记忆生命周期管理的入口，确保自动化生成的临时记忆能够沉淀为正式的项目规范。这里集成了系统原生附带的“内置技能（Bundled Skills）”，它们并不是传统意义上通过网络调用的 Agent MCP Tools，而是一套预置的专项任务指令集。

其中，`remember.ts` 注册了内置的 `/remember` 技能。该技能作为记忆的“垃圾回收与整理器”，其核心执行逻辑如下：

1. **收集所有记忆层级 (Gather)**：主动读取当前项目的 `CLAUDE.md`（项目级规范）、`CLAUDE.local.md`（个人本地规范）以及存储在上下文中自动提取的临时记忆。
2. **智能分类与提拔 (Classify & Promote)**：对每一条临时记忆进行评估，判断其是否具有长期价值。如果有，则建议提拔至对应的固化文件中：
   - 具有通用性的项目规范建议移动至 `CLAUDE.md`。
   - 仅限当前用户的个人习惯建议移动至 `CLAUDE.local.md`。
   - 组织级跨库知识建议移动至 `Team memory`。
3. **识别清理机会 (Cleanup)**：跨层级扫描数据，寻找重复项（如临时记忆中已有配置）、过时项（如临时记忆与历史配置冲突）并提出清理建议。
4. **生成交互报告 (Report)**：将上述发现汇总为一份结构化的报告（分为提拔、清理、需要人工决策的模糊项等），并**强制要求在修改文件前必须获得用户的显式批准**，绝不静默篡改配置文件。

```typescript
// src/skills/bundled/remember.ts
// remember 技能的注册定义与触发机制
registerBundledSkill({
  name: "remember",
  description:
    "Review auto-memory entries and propose promotions to CLAUDE.md, CLAUDE.local.md, or shared memory. Also detects outdated, conflicting, and duplicate entries across memory layers.",
  whenToUse:
    "Use when the user wants to review, organize, or promote their auto-memory entries. Also useful for cleaning up outdated or conflicting entries across CLAUDE.md, CLAUDE.local.md, and auto-memory.",
  userInvocable: true,
  isEnabled: () => isAutoMemoryEnabled(), // 仅在记忆功能开启时可用
  async getPromptForCommand(args) {
    let prompt = SKILL_PROMPT; // 注入详尽的 4 步整理指南提示词
    if (args) prompt += `\n## Additional context from user\n\n${args}`;
    return [{ type: "text", text: prompt }];
  },
});
```

---

## 4. 记忆管理中的防劣化机制

长期运行的记忆系统极易产生信息过载与数据过期问题。为此，系统在读取与更新环节设计了多重防御机制，确保提供的上下文始终可靠。

### 4.1 记忆漂移防御

系统强制要求模型在采信记忆前进行验证。`memoryTypes.ts` 中的 `TRUSTING_RECALL_SECTION` 明确指示模型：记忆仅代表过去的切片，在执行关键操作前，必须通过读取文件或执行命令来核实现状。

```typescript
// src/memdir/memoryTypes.ts
// 防止大语言模型过度依赖过期记忆的核心提示词
export const TRUSTING_RECALL_SECTION: readonly string[] = [
  "## Before recommending from memory",
  "",
  "A memory that names a specific function, file, or flag is a claim that it existed *when the memory was written*. It may have been renamed, removed, or never merged. Before recommending it:",
  "",
  // 强制验证策略：文件需检查存在性，函数需执行 grep 搜索
  "- If the memory names a file path: check the file exists.",
  "- If the memory names a function or flag: grep for it.",
];
```

### 4.2 跨层级一致性校验

由于记忆可能散落于自动化记忆目录、项目级配置（`CLAUDE.md`）与用户级配置（`CLAUDE.local.md`）中，`remember` 技能充当了“垃圾回收与整理器”的角色。它通过大语言模型比对各个层级的数据，识别重复或矛盾的内容，并生成合并或删除建议，交由用户决策，以此维持记忆体系的整洁。

| 层级名称   | 适用范围         | 典型应用场景                           |
| ---------- | ---------------- | -------------------------------------- |
| 团队记忆   | 项目全体成员     | API 路由规范、测试命令、发版流程       |
| 私有记忆   | 单一用户本地环境 | 沟通风格偏好、提交代码前的特定检查习惯 |
| 自动化记忆 | 临时会话沉淀     | 开发过程中的中间笔记、待确认的代码模式 |

---

## 5. 总结

Claude Code 的记忆系统远不止于单纯的对话上下文留存，它更是一个具备自我学习、检索优化和自动防劣化的闭环生态系统。与 Mem0 等依赖向量数据库和图数据库的通用型外部记忆框架不同，Claude Code 选择了一条**极简且内聚的纯文本（Markdown）工程化路线**，这使得记忆数据能够完美融入 Git 版本控制与开发者现有的文件协作流中。

1. **结构清晰**：通过双层架构设计（个人与团队），在保障隐私的同时实现了项目级知识的共享。
2. **性能优化**：通过轻量级索引与异步 Side Query 的相关性打分机制，解决了全量加载导致 Token 消耗过大的痛点。
3. **长期可维护**：内置的 `/remember` 技能与严格的防漂移提示词策略，使得大模型不仅能记住知识，还能在信息过时时进行自我修正与清理。

这套记忆机制使得 Agent 能够随着使用时长的增加，逐渐演变为一个“越用越懂你、越用越懂项目”的资深结对编程助手。
