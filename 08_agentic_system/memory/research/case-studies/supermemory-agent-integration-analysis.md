# Supermemory 架构与 Agent 集成解析

## 1. 简介

`Supermemory` 是一个为 AI 赋予持久化记忆的底层数据服务系统。当前代码库采用以 `Cloudflare Workers` 为核心的 Serverless 架构，对外提供标准化的 `REST API`，并在之上构建了完整的生态工具链。

在底层存储设计上，系统采用了双重架构：使用轻量级的 `Cloudflare D1` 处理基础的关系型数据，同时通过 `Cloudflare Hyperdrive`（连接池与加速层）连接外部带有 `pgvector` 扩展的 `PostgreSQL` 数据库，以支撑复杂的大规模向量存储和检索。

其核心目的是解决大语言模型在长线任务中上下文易丢失、缺乏个性化认知的问题，为各类 `Agent` 提供高可用、强隔离的记忆读写底座。

> **部署模式说明**：目前 `Supermemory` 核心是一个基于 `Cloudflare` 生态强绑定的云端服务（SaaS）。官方虽然提供了供企业客户使用的部署包（基于 `Cloudflare Workers` 的自托管方案），但并不支持完全脱离云环境的纯本地化（Local-only）部署。

---

## 2. 功能说明

`Supermemory` 的核心功能紧密围绕数据摄入、记忆检索以及外部连接展开，旨在为 AI 构建一个全方位、可溯源的记忆知识库。

### 2.1 知识内容摄入与异步处理

任何记忆网络的前提是多模态高质量的数据输入。`Supermemory` 通过内部的异步流水线，将非结构化的外部内容转化为可检索的语义块与特征向量。核心数据的增删改查收敛于 `/v3/documents` 端点。系统在后台会自动执行提取、分块、向量嵌入和索引构建。

`Supermemory` 能够自动检测并处理极为丰富的文档类型，无需用户手动解析。以下是支持摄入的核心文档类型说明：

| 分类           | 支持格式 / 标识符                                        | 摄入与处理机制                                                                                                           |
| :------------- | :------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **文本文档**   | `text`, `pdf`, `docx`, `txt`, `md`                       | 原生支持普通文本。对于 PDF 与扫描件，内置 OCR 识别提取文本；长文档会按语义段落自动进行块切割（Chunking）。               |
| **办公文档**   | `google_doc`, `google_slide`, `google_sheet`, `onedrive` | 通过云盘 API 提取文档、表格与幻灯片内容，提取时尽量保留其原始数据结构与元数据。                                          |
| **网页与社交** | `webpage`, `tweet`                                       | 直接传入 URL 即可，系统会自动抓取网页主干内容（剥离广告与导航），或提取 Twitter 帖子及推文元数据。                       |
| **多媒体**     | `image`, `video`                                         | 图片支持 JPG/PNG/WebP 等，利用 OCR 提取画面文字；视频支持 MP4 及 YouTube 链接，系统会自动执行音频转录（Transcription）。 |
| **工作区内容** | `notion_doc`                                             | 能够读取 Notion 页面，并在提取内容的同时保留 Block 层级的结构关联。                                                      |

```json
// 端点: POST /v3/documents

// 输入示例 (Request Body)
{
  "content": "https://example.com/article", // 支持传入文本、URL，或通过 /v3/documents/file 上传文件
  "containerTag": "user_123",               // 用于数据隔离的标签
  "metadata": {
    "source": "blog"
  }
}

// 输出示例 (Response)
{
  "id": "doc_abc123xyz",
  "status": "queued" // 处理状态：queued, extracting, chunking 等
}
```

### 2.2 数据检索双轨机制

针对不同粒度的大语言模型上下文需求，系统从架构上将数据召回分流为物理文档切片与抽象语义记忆两条链路，从而在客观知识与主观偏好之间取得平衡。该机制具体分为针对客观事实的物理文档检索，以及针对主观偏好的语义画像检索。

#### 2.2.1 高质量物理文档检索

`/v3/search` 接口专注于标准的检索增强生成（`RAG`）召回。开发者可以细粒度地调整文档匹配阈值、切片阈值，并开启查询重写与重排功能，确保回答的溯源准确性，适用于严谨事实要求的业务场景。

```json
// 端点: POST /v3/search

// 输入示例 (Request Body)
{
  "q": "machine learning neural networks",
  "limit": 5,
  "documentThreshold": 0.7, // 文档匹配阈值
  "chunkThreshold": 0.8,    // 切片匹配阈值
  "rerank": true,           // 开启重排优化
  "rewriteQuery": true      // 开启查询重写
}

// 输出示例 (Response)
{
  "results": [
    {
      "documentId": "doc_ml_guide_2024",
      "title": "Machine Learning Guide",
      "score": 0.89,
      "chunks": [
        {
          "content": "Neural networks are computational models...",
          "score": 0.92,
          "isRelevant": true
        }
      ]
    }
  ],
  "total": 12,
  "timing": 156
}
```

#### 2.2.2 语义记忆与用户画像检索

`/v4/search` 接口聚焦于搜索用户的抽象记忆。它采用混合搜索模式，能够自动返回用户的偏好配置与经验总结，主要面向聊天机器人、编程助手等需要注入个性化画像的场景。

```json
// 端点: POST /v4/search

// 输入示例 (Request Body)
{
  "q": "machine learning",
  "containerTag": "user_123", // 必填，指定用户或项目的画像域
  "searchMode": "hybrid",     // 推荐模式，混合搜索文档和抽象记忆
  "limit": 5
}

// 输出示例 (Response)
{
  "results": [
    {
      "id": "mem_abc123",
      "memory": "User prefers concise explanations about neural networks.", // 提炼出的用户画像/记忆
      "similarity": 0.88,
      "metadata": {
        "source": "conversation"
      }
    }
  ]
}
```

### 2.3 外部生态互联与同步

现代知识工作者的资产散落于多个平台，系统引入了针对第三方效率工具的集成方案以维持记忆的新鲜度。通过 `/v3/connections` 接口，系统能够无缝对接 `Google Drive`、`Notion`、`OneDrive` 等主流工作区。

该模块提供了多种同步策略来保障外部数据与内部记忆库的一致性：

- **实时同步**：利用 `Webhook` 机制监听外部资产（如 `Google Drive`、`Notion`）的变化，实现秒级同步更新。
- **定时同步**：对于不支持 `Webhook` 的平台，系统默认提供每四小时一次的轮询同步（Cron 任务）。
- **手动同步**：开发者也可以通过调用 `/v3/connections/{provider}/sync` 接口强制触发特定工作区的手动同步。

在同步过程中，系统会自动将云端文档（如 `Google Docs`、`Google Slides`、`Google Sheets`）转换为 `Markdown` 格式并执行向量化，确保大模型始终能够检索到最新的工作资料。

---

## 3. 在 Coding Agent 中的应用说明

在类似 `OpenCode` 这样的 AI 编程助手场景中，`Supermemory` 扮演了 IDE 运行时与云端记忆库之间的桥梁，使得大模型能够具备跨项目、跨会话的学习与记忆能力。

### 3.1 核心价值与应用场景

通过接入持久化记忆，`Coding Agent` 能够实现上下文的隐式管理与高价值代码经验的自动沉淀。它解决了长生命周期编程任务中，大模型频繁遗忘架构约束、代码规范以及用户个人编码习惯的痛点，从而显著提升连续对话的连贯性与代码生成的准确率。

### 3.2 面向 Agent 的工具封装与端点映射

大语言模型需要结构化的元数据才能稳定执行外部动作。系统不仅提供了 `API`，还通过封装的 SDK 与 MCP 规范，将底层能力转化为开箱即用的工具。在 `opencode-supermemory` 插件中，系统为 `Agent` 暴露了统一的 `supermemory` 工具，并提供了以下核心操作模式（Mode）。这些操作通过多态路由分发，并在底层映射到核心的 `REST API` 端点，具体模式如下：

- **`add`（主动记忆）**
  - **功能**：当用户提出需要记住的规范或配置时，将信息持久化存储到指定作用域（`user` 或 `project`）。
  - **底层映射**：调用 `POST /v3/documents`（见 2.1 节），将文本片段作为实体文档进行特征提取与向量化。
- **`search`（记忆召回）**
  - **功能**：在执行复杂任务前，根据查询词检索相关的项目上下文或个人偏好。
  - **底层映射**：对应混合检索端点 `POST /v4/search`（见 2.2.2 节），以 `hybrid` 模式召回相关文档片段和抽象记忆。
- **`profile`（获取画像）**
  - **功能**：用于获取用户的完整静态与动态记忆画像，作为全局上下文前置注入。
  - **底层映射**：本质上是对 `/v4/search` 接口的特殊封装，自动提取高权重的长期习惯。
- **`list`（记忆列表）**
  - **功能**：列出最近存储的记忆项，支持指定作用域和数量限制。
  - **底层映射**：调用 `GET /v3/documents` 或专门的记忆列表端点。
- **`forget`（记忆遗忘）**
  - **功能**：允许用户通过 `memoryId` 移除不再适用的旧偏好，保持记忆的准确性。
  - **底层映射**：调用 `DELETE /v3/documents/{id}`，软删除指定的记忆数据。

---

## 4. 整体架构

为了支撑海量数据的高效流转与多端工具的无缝接入，`Supermemory` 采用了高度模块化的 `Turbo Monorepo` 设计与 Serverless 架构。从逻辑上，整体架构遵循典型的“总-分”三层结构设计，自上而下划分为：**客户端与交互层**、**网关与协议层** 以及 **核心处理与数据层**。前端交互与底层计算完全解耦，确保了系统的高可用性与可扩展性。

### 4.1 客户端与交互层

客户端层直面终端用户或业务系统，主要负责可视化管理、数据的主动采集以及用户交互请求的下发。

- **Next.js Web 控制台 (`apps/web`)**：提供全功能的用户交互界面，允许用户直观地管理个人记忆库、配置第三方工作区集成，并监控 API 使用情况。
- **Browser Extension (`apps/browser-extension`)**：运行于浏览器环境下的插件，赋予用户随时随地截取网页上下文、保存社交媒体内容的抓取能力。

### 4.2 网关与协议层

网关层起到了承上启下的作用，它是我们在 **第 2 章** 提到的核心检索（`/v3/search`、`/v4/search`）与文档摄入（`/v3/documents`）等功能的直接暴露层，向各种形态的 AI Agent 和开发者提供标准化的接入途径。

- **核心 `REST API` (`Hono`)**：基于 `Cloudflare Workers` 部署的无服务器高并发接口。它是第 2 章所有核心功能（知识摄入、双轨检索、生态互联）的入口，通过轻量级的 `Hono` 框架提供路由服务，支持各类 SDK 的直接调用。
- **MCP Server (`apps/mcp`)**：为本地大语言模型（如 `Claude Desktop`、`Cursor`）提供的标准化服务协议层。底层依托 `Cloudflare Durable Objects`，通过 Server-Sent Events (SSE) 维持与客户端的持久化长连接，实现低延迟的记忆读取。
- **SDK & Tools 包 (`packages/tools` 等)**：专门为 Vercel AI SDK、OpenAI、Mastra 等开发框架封装的中间件。将底层的 REST 请求转化为开箱即用的函数调用（如 `searchMemories` 工具）。

### 4.3 核心处理与数据层

数据层是系统的“大脑”，它是我们在 **第 2 章** 提到的各种抽象功能的底层执行者，负责非结构化文档的深度解析、向量生成、语义关系构建以及租户间的安全隔离。

- **异步摄入流水线**：接收到前端（`/v3/documents`）的文档后，通过队列触发后端的工作流。负责执行多模态数据的 OCR 识别、智能文本切片（Chunking）以及向量特征提取（对应 2.1 节）。
- **混合存储系统**：采用双重存储策略。轻量级的关系型元数据和任务状态存放在 `Cloudflare D1` 中；而复杂的大规模向量检索（支撑 2.2 节的双轨检索机制），则通过 `Cloudflare Hyperdrive`（连接池加速层）直接连接外部带有 `pgvector` 扩展的 `PostgreSQL` 数据库。
- **Container Tags 隔离机制**：为了防止多项目、多用户之间代码规范和记忆的交叉污染，系统通过哈希签名的容器标签对物理数据进行强制切分，确保查询时的绝对数据隔离。

---

## 5. 关键实现流程（以 opencode-supermemory 为例）

`opencode-supermemory` 插件展示了记忆系统如何与本地 IDE 深度整合，其内部通过五项核心机制，实现了一系列精巧的钩子拦截与提示词（`Prompt`）工程策略。

### 5.1 上下文无感注入机制

对话初始化的状态设定决定了大模型的表现下限。插件在会话启动（Session Start）时，会自动获取跨项目级别的用户偏好（User Profile）与当前代码库的专属记忆（Project Memories）。这些数据随后被格式化为特定结构的文本，作为不可见的 `Prompt` 节点（对用户不可见）推入大模型的上下文，使得 Agent 在第一条消息时就已经掌握了完整的背景约束。

### 5.2 基于意图的强制记忆诱导

为避免大模型忽略用户的记忆诉求，系统引入了正则表达式拦截机制。系统会监听用户的自然语言输入，一旦命中触发词（如 "remember", "save this", "don't forget" 等），就会向大模型下发强制记忆的指令提示。

```typescript
// 示例：监听用户的保存意图，忽略被代码块包裹的关键词
const MEMORY_KEYWORD_PATTERN = new RegExp(
  `\\b(${CONFIG.keywordPatterns.join("|")})\\b`,
  "i",
);

// 命中后注入的隐式 Prompt
const MEMORY_NUDGE_MESSAGE = `[MEMORY TRIGGER DETECTED]
The user wants you to remember something. You MUST use the \`supermemory\` tool with \`mode: "add"\` to save this information.
...`;
```

### 5.3 上下文防遗忘与压缩（Preemptive Compaction）

针对长生命周期编程任务易导致 `Token` 溢出的问题，插件接管了 IDE 原生的上下文截断逻辑。当上下文容量达到 80% 阈值时，系统会触发自动总结（Summarization），在总结的上下文中强行注入最初的项目核心记忆，并将当前的会话总结保存为一条新的记忆。这确保了在压缩历史记录的同时，核心规范和长线对话逻辑不会丢失。

### 5.4 统一工具分发总线

如 3.2 节所述，插件向大模型暴露了唯一的 `supermemory` 工具声明。其内部通过多态路由分发到 `add`、`search`、`profile`、`list`、`forget` 五个具体的处理函数，降低了 `Agent` 的认知与调用成本。

### 5.5 代码隐私安全脱敏

在将本地代码片段上报至云端记忆库前，系统内置了轻量级的静态文本审查策略。API 密钥或私密信息可以通过 `<private>` 标签包裹（例如 `<private>sk-abc123</private>`），系统在持久化前会自动过滤剥离这些标签内的内容，从源头切断核心资产的泄露风险。
