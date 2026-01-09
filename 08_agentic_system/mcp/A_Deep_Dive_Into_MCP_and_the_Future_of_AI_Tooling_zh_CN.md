# 深度解析 MCP 与 AI 工具化的未来

> 作者：`Yoko Li`  
> 原文：`https://a16z.com/a-deep-dive-into-mcp-and-the-future-of-ai-tooling/`

自 OpenAI 于 2023 年发布函数调用功能以来，我一直在思考如何构建一个开放的智能体与工具使用生态系统。随着基础模型愈发智能化，智能体与外部工具、数据和 API 的交互能力却日益碎片化：开发者需要为智能体运行的每个系统和集成对象，实现具有特殊业务逻辑的智能体程序。 

显然，我们需要一个标准化的执行接口来实现数据获取和工具调用。**API 曾是互联网首个伟大的统一者——它创造了软件间通信的通用语言——但 AI 模型至今缺乏类似的标准化协议。**

2024 年 11 月推出的模型上下文协议（MCP）已在开发者与 AI 社区中引发强烈关注，被视为潜在的解决方案。本文我们将深入探讨**MCP 的核心原理、它如何改变 AI 与工具的交互范式、开发者基于该协议已构建的应用场景，以及仍需攻克的挑战。** 

让我们深入探讨。

## 什么是 MCP？

**MCP 是一种开放协议，使系统能够以跨集成通用化的方式为 AI 模型提供上下文。** 该协议定义了 AI 模型如何调用外部工具、获取数据以及与服务交互。具体示例如下，展示了 Resend MCP 服务器如何与多个 MCP 客户端协同工作。 

[![MCP 插图](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/250319-Example-MCP-x2000-1024x455.png)](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/250319-Example-MCP-x2000.png)

这个理念并不新鲜；MCP[从 LSP（语言服务器协议）中汲取了灵感](https://spec.modelcontextprotocol.io/specification/2024-11-05/#:~:text=MCP%20takes%20some%20inspiration%20from,the%20ecosystem%20of%20AI%20applications) 。在 LSP 中，当用户在编辑器中输入时，客户端会查询语言服务器以获取自动补全建议或诊断信息。

[![MCP 插图](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/250314-LSP-x2000-1024x422.png)](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/250314-LSP-x2000.png)

MCP 的突破性在于其以代理为中心的执行模型：LSP 主要是反应式的（基于用户输入响应 IDE 的请求），而 MCP 旨在支持自主 AI 工作流。基于上下文，**AI 代理可以自主决定使用哪些工具、使用顺序以及如何串联工具链来完成任务。** MCP 还引入了人机协同能力，允许人类提供额外数据并批准执行流程。 

[![MCP 插图](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/250319-MCP-x2000-1024x541.png)](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/250319-MCP-x2000.png)

## 当前热门的应用场景 

借助适当的 MCP 服务器组合，用户可将每个 MCP 客户端转变为"万能应用"。通过[Slack MCP 服务器](https://github.com/modelcontextprotocol/servers/tree/main/src/slack) 实现 Slack 通讯功能，利用[Resend MCP 服务器](https://github.com/resend/mcp-send-email/tree/main) 完成邮件发送，或通过[Replicate MCP 服务器](https://github.com/deepfates/mcp-replicate) 进行图像生成。 

以代码编辑器光标为例：虽然其本质是代码编辑工具，但作为优质 MCP 客户端，终端用户既可通过[Slack MCP 服务器](https://github.com/modelcontextprotocol/servers/tree/main/src/slack) 将其改造为 Slack 客户端，也能利用[Resend MCP 服务器](https://github.com/resend/mcp-send-email/tree/main) 实现邮件发送功能，或通过[Replicate MCP 服务器](https://github.com/deepfates/mcp-replicate) 进行图像生成。更强大的应用方式是在单个客户端集成多个服务器以解锁新流程：用户可安装[前端 UI 生成服务器](https://github.com/21st-dev/magic-mcp) 从光标直接生成界面原型，同时调用图像生成 MCP 服务器为网站创建主视觉图。 

除光标外，当前大多数用例可归纳为两类：要么是以开发者为中心、本地优先的工作流，要么是使用大语言模型客户端构建的全新体验。

### 以开发者为中心的工作流 

对于每日与代码为伴的开发者而言，一个普遍共识是"我不希望为了执行_x_而离开我的 IDE"。MCP 服务器正是实现这一理想的绝佳方式。 

开发者无需切换到 Supabase 查看数据库状态，现在可以直接在 IDE 中使用[Postgres MCP 服务器](https://github.com/modelcontextprotocol/servers/tree/main/src/postgres) 执行只读 SQL 命令，通过[Upstash MCP 服务器](https://github.com/upstash/mcp-server) 创建和管理缓存索引。在进行代码迭代时，还可利用[Browsertools MCP 协议](https://github.com/AgentDeskAI/browser-tools-mcp) 为编码代理提供实时环境以获取反馈并进行调试。 

[![MCP 插图](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/image1-1024x989.png)](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/image1.png)

_示例展示了光标代理如何通过 Browsertools 获取控制台日志和实时数据，从而更高效地进行调试。_

在与开发工具交互的工作流之外，MCP 服务器解锁的新应用场景是通过[网页爬取](https://github.com/mendableai/firecrawl-mcp-server) 或基于文档[自动生成 MCP 服务器](https://mintlify.com/blog/generate-mcp-servers-for-your-docs) ，为编程 AI 代理注入高精度上下文。开发者无需手动配置集成，可以直接从现有文档或 API 启动 MCP 服务器，使 AI 代理能够即时访问工具。这意味着开发者可以减少样板代码的时间消耗，将更多时间投入实际工具应用——无论是获取实时上下文、执行命令，还是动态扩展 AI 助手的能力。

### 全新的体验

尽管像光标（Cursor）这样的 IDE 因 MCP 对技术用户的强大吸引力而备受关注，但它们并非唯一的 MCP 客户端。对于非技术用户，Claude Desktop 作为优秀的入口级产品，使基于 MCP 的工具更易于大众接触和使用。很快我们将看到面向商业场景的专用 MCP 客户端涌现，包括客户支持、营销文案、设计及图像编辑等领域——这些领域与 AI 在模式识别和创造性任务上的优势高度契合。 

MCP 客户端的设计及其支持的特定交互在塑造其功能方面起着关键作用。例如，聊天应用程序不太可能包含矢量渲染画布，正如设计工具不太可能提供在远程机器上执行代码的功能。最终，**MCP 客户端体验定义了整体 MCP 用户体验**——在 MCP 客户端体验领域，我们还有巨大的探索空间。  

典型案例是 Highlight 如何通过在其客户端实施[@命令](https://x.com/PimDeWitte/status/1899829221813334449) 来调用任何 MCP 服务器。这催生了一种新的 UX 模式：MCP 客户端可将生成内容管道传输至任意下游应用。 

[![MCP 插图](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/Notion-screenshot.png)](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/Notion-screenshot.png)

_Highlight 在 Notion MCP（插件）中的实施案例_

另一个典型案例是[Blender MCP 服务器](https://x.com/sidahuj/status/1901632110395265452) 应用场景：现在，几乎不了解 Blender 的业余用户可以通过自然语言描述想要构建的模型。随着社区为 Unity 和 Unreal 引擎等其他工具部署服务器，我们正在见证文本到 3D 的工作流程实时展开。 

尽管我们主要关注服务器与客户端，但随着协议的发展，MCP 生态系统正逐步成型。当前市场版图覆盖了最活跃的领域，但仍存在诸多空白。虽然我们深知 MCP 仍处于早期阶段，_但随着市场的发展和成熟，我们期待在地图上添加更多参与者_（下一章节我们将探讨其中部分未来可能性）。

[![MCP 插图](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/250319-MCP-Market-Map-v2-x2000-1024x866.png)](https://d1lamhf6l6yk6d.cloudfront.net/uploads/2025/03/250319-MCP-Market-Map-v2-x2000.png)

在 MCP 客户端领域，**目前所见的高质量客户端大多以编码为核心**。这并不令人意外，因为开发者通常是新技术的早期采用者，但随着协议的成熟，我们预期将涌现更多以业务为核心的客户端。 

**当前大多数 MCP 服务器采用本地优先策略且聚焦单机场景，这反映出目前 MCP 仅支持基于 SSE（服务器发送事件）和命令的连接方式**。然而我们预计，随着生态体系将远程 MCP 支持提升为一级功能，以及 MCP 采用[可流式 HTTP 传输协议](https://github.com/modelcontextprotocol/specification/pull/206) ，MCP 服务器的采用率将显著提升。 

为支持 MCP 服务器发现机制的实现，新一代 MCP 市场平台及服务器托管解决方案正在涌现。[Mintlify](https://mintlify.com/) 旗下的[mcpt](https://www.mcpt.com/) 、[Smithery](https://smithery.ai/) 和[OpenTools](https://opentools.com/) 等平台正在简化开发者发现、共享及贡献新 MCP 服务器的流程——这种变革类似于 npm 对 JavaScript 包管理的革新，亦如 RapidAPI 对 API 发现机制的扩展。该层级对于标准化访问高质量 MCP 服务器具有决定性意义，使得 AI 代理能够按需动态选择并集成工具。

随着 MCP 协议集的普及，**基础设施与工具链将在提升生态系统可扩展性、可靠性与可访问性方面发挥关键作用**。Mintlify、[Stainless](https://www.stainless.com/) 和[Speakeasy](https://www.speakeasy.com/) 等服务器生成工具正在减少创建兼容 MCP 协议集服务的阻力，而 Cloudflare 和 Smithery 等托管方案正着力解决部署与扩展难题。与此同时，**[Toolbase](https://gettoolbase.ai/) 等连接管理平台**已开始优化本地优先的 MCP 密钥管理与代理服务。 

## 未来发展前景 

然而，我们目前仍处于智能体原生架构演进的早期阶段。尽管当前业界对 MCP 协议集充满热情，但在构建和部署 MCP 时仍面临诸多未解决的挑战。 

协议下一版本需要解锁的关键能力包括： 

### 托管与多租户支持 

MCP 协议支持 AI 智能体与其工具间的一对多关系，但多租户架构（如 SaaS 产品）需要支持多用户同时访问共享的 MCP 服务器。默认采用远程服务器可能是短期内提升 MCP 服务器可访问性的解决方案，但许多企业仍希望自主托管 MCP 服务器，并实现数据平面与控制平面的分离。 

建立简化的工具链以支持规模化 MCP 服务器的部署与维护，将是推动更广泛采用的下一个关键突破点。

### 身份验证机制 

当前 MCP 尚未定义标准的身份验证机制来规范客户端与服务器的认证流程，也未提供 MCP 服务器与第三方 API 交互时安全管理和委派认证的框架。目前身份验证的实现方式取决于具体部署场景和开发实践。在实际应用中，MCP 当前的采用主要集中在无需显式身份验证的本地集成场景。 

更优的身份验证范式可能成为远程 MCP 采用的关键突破口之一。从开发者的角度来看，统一方案应涵盖： 

*   **客户端身份验证:** 使用 OAuth 或 API 令牌等标准方法处理客户端-服务端交互
*   **工具身份验证:** 提供辅助函数或封装器来实现第三方 API 的身份验证
*   **多用户身份验证:** 支持租户感知的企业级部署身份验证

### 授权机制 

即使工具已完成身份验证，仍需界定使用权限的边界与粒度。MCP 目前缺乏内置的权限模型，访问控制停留在会话层面——工具要么完全开放访问，要么完全受限。虽然未来可能形成更精细的权限控制机制，但当前方案依赖[基于 OAuth 2.1 的授权流程](https://github.com/modelcontextprotocol/specification/blob/5c35d6dda5bf04b5c8c76352c9f7ee18d22b7a08/docs/specification/draft/basic/authorization.md) ，通过身份验证即授予整个会话的访问权限。随着更多智能体和工具的接入，这种机制会带来额外的复杂性——每个智能体通常需要携带独立授权凭证的专属会话，导致基于会话的访问管理网络日趋复杂。 

### 网关

随着 MCP 协议集的规模化应用，网关可作为集中式层来处理身份验证、授权、流量管理和工具选择。类似于 API 网关，它将执行访问控制、将请求路由至正确的 MCP 服务器、处理负载均衡，并通过缓存响应提升效率。这对于多租户环境尤为重要，不同用户和 AI 代理需要差异化权限。标准化网关能简化客户端-服务器交互，增强安全性，并提供更好的可观测性，从而使 MCP 部署更具扩展性和可管理性。

### MCP 服务器的可发现性与可用性

当前发现和配置 MCP 服务器仍是手动流程，开发者需要定位终端节点或脚本、配置身份验证，并确保服务端与客户端的兼容性。新服务器的集成耗时耗力，且 AI 代理无法动态发现或适配可用服务器。 

根据[Anthropic 在 AI 工程师大会](https://youtu.be/kQmXtrmQ5Zg?t=4927) 上月发表的演讲，**MCP 服务器注册与发现协议即将问世**。这将为 MCP 服务器的下一阶段普及铺平道路。

### 执行环境 

大多数 AI 工作流需要连续进行多个工具调用，但 MCP 协议缺乏内置的工作流概念来管理这些步骤。要求每个客户端都自行实现可恢复性和重试机制并非理想方案。尽管当前我们看到开发者正在探索[Inngest](https://www.inngest.com/) 等解决方案来实现这一目标，但将状态化执行提升为一级概念将能为大多数开发者厘清执行模型。 

### 标准化客户端体验 

开发者社区经常提出的一个共同问题是：在构建 MCP 客户端时应如何进行工具选择？是否需要每个团队都自行实现工具的 RAG（检索增强生成）方案，还是存在待标准化的中间层？

除工具选择外，目前也没有统一的 UI/UX 范式来调用工具（我们观察到的实践从斜杠命令到纯自然语言调用应有尽有）。建立标准化的客户端层用于工具发现、排序和执行，将有助于创建更具可预测性的开发者和用户体验。 

### 调试

MCP 服务器开发者常常发现，要让同一个 MCP 服务器轻松适配不同客户端十分困难。大多数情况下，每个 MCP 客户端都有其独特特性，而客户端追踪日志要么缺失要么难以获取，这使得调试 MCP 服务器成为极具挑战性的任务。随着业界开始构建更多远程优先的 MCP 服务器，我们需要新一代工具链来统一本地和远程环境的开发体验。 

## AI 工具化的深远影响

MCP 的开发体验让我想起 2010 年代的 API 开发。这个范式崭新而令人兴奋，但工具链尚处于早期阶段。如果我们快进到数年后，当 MCP 成为 AI 驱动工作流程的事实标准会怎样？以下是若干预测：

*   **开发者优先企业的竞争优势将发生演变**，从交付最佳 API 设计扩展到同时交付最优的智能体工具集合。如果 MCP 协议集具备自主发现工具的能力，API 和 SDK 提供商需要确保其工具在搜索中易于被发现，并在功能差异度上足够突出以使智能体能为特定任务选择它们。这种选择标准相比人类开发者的需求可能更加细化和具体。 
*   **新型定价模式可能出现**：如果每个应用都成为 MCP 客户端，每个 API 都成为 MCP 服务端，智能体可能根据速度、成本和相关性动态选择工具。这可能导致更加市场驱动的工具采用过程，选择性能最优且模块化程度最高的工具，而非当前最广泛采用的那些。 
*   **文档将成为 MCP 基础设施的关键组成部分**，因为企业需要设计具有清晰机器可读格式（例如[llms.txt](https://mintlify.com/blog/simplifying-docs-with-llms-txt) ）的工具和 API，并基于现有文档将 MCP 服务器打造成实际的标准产物。 
*   **仅凭 API 已不足以满足需求，但仍是优秀的起点。** 开发者将发现从 API 到工具的映射很少是 1:1 的。工具是更高层次的抽象，对执行任务时的代理更具实际意义——代理可能不会简单调用 `send_email()`，而会选择包含多个 API 调用的 `draft_email_and_send()`函数以降低延迟。MCP 服务器的设计将以场景和使用案例为中心，而非以 API 为中心。 
*   **新的托管模式将应运而生**，如果默认情况下所有软件都成为 MCP 客户端，其工作负载特征将与传统网站托管存在本质差异。每个客户端本质上都是多步骤操作，需要可恢复性、重试机制和长时任务管理等执行保证。托管服务商还需在不同 MCP 服务器间实现实时负载均衡，以优化成本、延迟和性能，使 AI 代理能在任意时刻选择最高效的工具。

MCP 协议集已经在重塑 AI 代理生态系统，但下一阶段的进步将取决于我们如何应对基础性挑战。如果实施得当，MCP 协议集可能成为 AI 与工具交互的默认接口，并开启新一代自主、多模态、深度集成的人工智能体验。

若 MCP 协议集能获得广泛采用，它可能从根本上改变工具的构建方式、消费模式和货币化路径。我们期待见证市场对其的演进方向。今年将是关键转折点：我们是否会见证统一 MCP 市场的崛起？AI 代理的身份验证能否实现无缝对接？多步骤执行流程能否被正式纳入协议标准？