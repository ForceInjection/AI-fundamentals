# 一切皆插件：DeepSeek Harness 是怎么把 Agent 装起来的

> 让 AI 写一段代码，它写得挺好；让它在一个真实项目里连着干两小时，它就开始迷路：忘了改过什么，重复做同一件事，或者把不该动的文件动了。问题往往不在模型。今天的模型足够聪明，缺的是它和真实世界之间那层「工程外壳」——读文件、跑命令、管上下文、失败重试、断点续跑、权限审批。行业把这一整套东西叫 Harness。
>
> 2026 年 8 月 13 日，DeepSeek 把自己的 Harness 开源了。本文拆它的设计，最后顺带推荐一本同主题的新书。
>
> 2026-09 | 基于 `deepseek-ai/deepseek-harness`（`5dda764e`，`0.1.5-alpha.1`，2026-09-08）源码与官方文档核对；仓库仍处于开发者预览阶段，接口会变。

---

## 一、Harness 是什么

给编程 Agent 一个任务：「把这个仓库里的日志从 `print` 改成结构化输出」。它要做的事包括：找到所有相关文件、读它们、改它们、跑测试、看报错、再改、再跑，中途你可能还要打断它说「先别动那个模块」。

这串动作里，模型只负责「想」。剩下的全是外壳的活：

- **工具**：读写文件、执行命令、搜索、调用外部服务；
- **上下文管理**：对话长了要压缩，工具返回太长要裁剪，否则还没干几件事就把窗口塞满了；
- **循环**：想一步、做一步、看结果、再想一步，直到任务完成；
- **恢复**：进程崩了、网络断了、你手动停了，重新打开要能接着干；
- **边界**：哪些操作要问你一声，命令在什么沙箱里跑。

Harness 原意是马具。马有力气，但要靠马具才能拉车、才不会跑偏。在 Agent 语境里，这个词这两年被反复提起，各家给它的边界并不完全一样，但指的都是同一件事：模型之外，让它能真正干活的那一整套工程结构。

dsh 的官方定位很直接（`README.zh.md:5`）：

> DeepSeek Harness（`dsh`）是由 [DeepSeek AI](https://deepseek.com) 开发的开源 agent harness（智能体框架）。

开源之后的动静不小：仓库创建于 2026-08-13，到 2026-09-09 已经 21.7 万星标；GitHub 上 `dsh-plugin` 话题下聚起了 14000 多个插件仓库。（星标数会继续变，这里只是给个量级。）

---

## 二、为什么要再造一个 Agent 框架

市面上不缺 Agent 框架。但如果你真的想改点什么，往往会撞到同一堵墙：内核是特权的，插件在外围。

想换掉那个「想一步做一步」的循环？改内核。想换掉会话怎么存？改内核。想换掉模型适配层？还是改内核。改不动就只能 fork，fork 之后就再也跟不上上游了。

dsh 的赌注更激进：连循环本身都是插件。

官方文档里写得很清楚（`docs/architecture.zh.md:11`）：

> 产品的每一部分都是插件，包括模型适配器、工具注册表、会话日志，以及 agent loop（智能体循环）本身，因此每个都可以从配置替换。

以及（`docs/architecture.zh.md:13`）：

> 不存在需要打补丁的特权内核：扩展 dsh 的方式是把插件挂载到其他插件旁边。

---

## 三、一切皆插件

### 3.1 底座是 Cordis

dsh 的插件能力不是自己造的，而是构建在 [Cordis](https://github.com/cordiverse/cordis) 之上（以 vendor 方式引入，`vendor/cordis`）。Cordis 的论文把它要解决的问题拆成两个维度（[arXiv:2608.25512](https://arxiv.org/abs/2608.25512)）：时间可组合性，组件被移除时它的副作用要能被完整撤销；空间可组合性，组件之间能声明依赖，并让依赖关系被响应式地管理。

翻译成插件作者的日常，就是五件事（`docs/cordis-primer.zh.md:9-13`）：

| 概念       | 说人话                                                                            |
| ---------- | --------------------------------------------------------------------------------- |
| 插件       | 一个带 `apply(ctx)` 的函数，或一个 `Service` 子类                                 |
| 上下文     | 服务的容器。`ctx.tools`、`ctx.llm`、`ctx.sessions` 各占一个稳定名字               |
| `inject`   | 声明「我需要哪些服务」，加载顺序由依赖决定，不用手写启动序列                      |
| 类型化事件 | 插件之间通过事件通信，有 emit / waterfall / parallel / serial / bail 五种分发模式 |
| 可逆副作用 | 注册的东西（提示词片段、工具、适配器、监听器）在插件卸载时自动撤销                |

最后一条是热插拔的关键。注册不是往全局表里塞一个条目，它带「回收钩子」。插件被卸载，它注册的一切跟着消失，不会留下半个残骸。

### 3.2 没有特权内核

agent loop 自己就是个例子。它是这么声明的（`packages/core/agent-loop/src/index.ts:359-360`）：

```ts
export class AgentLoop extends Service implements AgentFactory {
  static inject = ['agents', 'sessions', 'llm', 'tools', 'systemPrompt', 'sessionProjections']
```

整个循环只依赖六个服务：活跃 agent 注册表、会话、模型适配层、工具注册表、系统提示词组装、会话投影。它自己不特殊，和其他插件一样挂在树上，一样可以被替换。

文档里有一句话说得很直接：**「具体实现为 `dsh-agent-loop` 包内部细节；循环外没有任何组件依赖它。」**（`docs/subsystems/core.zh.md:59`）循环可以被整体换掉，挂在它周围的插件一行不用改。

### 3.3 组装是分层的

一个跑起来的 dsh 是一棵插件树，由启动时按序叠加的若干层拼出来（`docs/architecture.zh.md:19-27`）：

- **profile**：一套具名组装。随发行版提供 `web`、`headless`、`sdk`、`sdk-minimal`、`acp` 五个模板；
- **bundle**：可分发的配置层。`dsh-base` 是共享底座（模型适配器、工具、持久化、沙箱与审批策略、设置、凭据、遥测），在此之上 `dsh-web-app` 加浏览器应用、`dsh-headless` 加一次性运行器、`dsh-sdk-app` 加 SDK 服务端；
- **patch**：按 id 定位某个条目、替换它的整个配置，或插入新条目。叠加顺序是「profile 列出的 bundle → profile 的 `cordis.patch.yml` → home 级 patch → 命令行 `--patch`」。

想看自己机器上到底挂了什么，一条命令：

```sh
dsh --profile web --dump-config
```

它打印出来的任何一条，都可以被你自己的 patch 换掉。这种「按层叠加、上层可覆盖下层」的路子，在容器镜像和 Linux 发行版的软件包组里都见过。

### 3.4 一个 Agent 就是一份 YAML

最能说明「一切皆插件」的，是预设文件。dsh 内置四个预设，一个预设就是一个目录，核心是 `agent.cordis.yml` 和 `preset.yml` 两份文件。以标准模式为例（`packages/preset/agent-presets/presets/standard/agent.cordis.yml`，节选）：

```yaml
- id: persona
  name: "@deepseek-ai/dsh-persona"
  config:
    suffix: Your working directory is {{cwd}}.

- id: tool-bash
  name: "@deepseek-ai/dsh-tool-bash"
  disabled: !!js process.platform === 'win32'

- id: tool-fs
  name: "@deepseek-ai/dsh-tool-fs"

- id: tool-skill
  name: "@deepseek-ai/dsh-skill-filesystem"

- id: tool-goal
  name: "@deepseek-ai/dsh-tool-goal"

- id: tool-subagent
  name: "@deepseek-ai/dsh-tool-subagent"
  config:
    provider: spawn
    backgroundMode: continuable
```

人设、bash 工具、文件工具、技能、目标、子代理……每一项都是一行插件声明。一个 Agent 的能力集合，就是一份可读、可 diff、可覆盖的配置。甚至能在配置里直接写 JS 表达式（`!!js process.platform === 'win32'`）按平台开关插件。

到这一步，「一切皆插件」就不再抽象了：加一项能力，是在配置里加一行。

---

## 四、三个看得见的设计

### 4.1 会话是一份仅追加的日志

大多数框架里，「会话」是内存里的一个对象：消息列表、状态、工具调用记录。进程一挂，它就没了；想审计「模型到底看到了什么」，得另外想办法。

dsh 换了个做法：会话是一份**仅追加的事件日志**，日志才是唯一真源（`docs/architecture.zh.md:121`）：

> 会话日志是模型所见上下文的来源。`deriveMessages()` 从中投影出模型历史。

配套的是一条硬约束（`docs/architecture.zh.md:125`）：

> **模型可见即已记录。** 抵达模型请求的一切都必须能从日志重建，并由一项运行时不变量断言这一点。

这条不变量的好处是连锁的：恢复、分叉、回放、遥测、审计，全都从同一条事件流派生，不需要各自维护一套状态。想查这个 Agent 为什么走了弯路，把日志摊开就行。

### 4.2 四个预设

四个内置预设的定位，官方描述如下（`packages/preset/agent-presets/presets/*/preset.yml`）：

| 预设     | 定位                                                                                                                       |
| -------- | -------------------------------------------------------------------------------------------------------------------------- |
| 标准模式 | 功能完整的编码 Agent，支持文件编辑、Shell、文件与网页检索、Skills、计划、目标、子代理和工作流                              |
| PTC 模式 | 功能完整的编码 Agent，但默认不提供 workflow 工具；其他工具通过 PTC 模式 SDK 呈现，让模型用一个 TypeScript 程序组合多步操作 |
| 极简模式 | 仅提供持久 bash 与 `str_replace_editor` 的双工具编码 Agent                                                                 |
| 创造模式 | 用于创建自定义 Agent preset：具备标准模式的全部能力，并提供运行时检查、插件实验和 preset 创作指导                          |

PTC 模式和创造模式值得单独说。

**PTC 模式**解决的是一个很实际的浪费：如果模型需要连续调五个工具、每次都要把结果塞回上下文再决定下一步，那往返的开销和上下文的膨胀都很可观。PTC 的做法是把工具注册表以一份生成的 TypeScript SDK 呈现给模型，模型写一个程序把这些调用组合起来，中间数据不进上下文，把「多轮往返」压成「一次编程」。仓库里给它的定义是「模型针对工具注册表编写 TypeScript」，灵感来自 Cloudflare 的 Code Mode。

**创造模式**则是这套架构最自洽的产物：Agent 被赋予了 `cordis_inspect_list`、`cordis_inspect_query`、`cordis_inspect_self`、`cordis_define`、`cordis_run`、`cordis_stop`、`cordis_undefine` 这组工具，可以检查自己运行时的插件树、现场写一个插件、挂载、试跑、再卸载（`docs/tool-catalog.zh.md`）。Agent 能改装自己，这在「内核特权」的框架里做不到，因为改内核需要重启。

### 4.3 能力 seam：换一个提供方，一串工具跟着搬

dsh 把「一项能力」拆成三个角色（`docs/glossary.zh.md:9`）：**Service Definition**（声明接口）、**Service Provider**（实现它）、**Consumer**（使用它，通常是面向模型的工具）。

规范范例是 shell 能力：`dsh-shell` 定义接口，`dsh-bash-local` 和 `dsh-bash-sandbox` 是两个提供方，`dsh-tool-bash` 是消费方。

这个拆法带来的效果很实在（`docs/architecture.zh.md:133`）：文件系统和进程提供方共享同一个执行世界，所以把它们指向远程沙箱，Bash、PTY、LSP 就一起搬了过去，不需要给每个工具写一套专用分支。

---

## 五、上手与边界

### 5.1 跑起来

```sh
npx @deepseek-ai/dsh web
```

默认在 `http://127.0.0.1:3080` 起 Web UI，本机启动会自动打开浏览器（`README.zh.md:29`）。首次进入后到「设置 → 模型」填 API Key，模型路由立即生效，不用重启服务。

几个容易忽略的点：

- **不绑定自家模型**。`packages/llm/llm-pi-ai` 提供 OpenAI、Anthropic 等兼容端点的接入，也支持自托管 Chat Completions 服务；
- **能读 `AGENTS.md` 和 `CLAUDE.md`**。`packages/context/agent-instructions` 默认把这两个文件名作为候选（`instructionFileCandidates`），从项目根目录一路加载到当前工作目录，内容相同的同级文件只渲染一次，文件被编辑后还会刷新（`packages/context/agent-instructions/README.zh.md:32-36`、`:64`）；
- **有 MCP 客户端插件**（`packages/mcp/mcp-client`，需要显式挂载，不在默认 bundle 里）与 **ACP 支持**（`packages/acp/acp`）；
- **可以把任务委托出去**。标准预设里预留了 `subagent_codex` 和 `subagent_claude_code` 两个工具，默认 `disabled: true`，装上对应 bundle、在自己的预设里去掉 `disabled` 就能启用，让 dsh 把子任务交给 Claude Code 或 Codex 干（`presets/standard/agent.cordis.yml`）。

沙箱方面，本地后端按平台走不同的隔离机制（`packages/sandbox/sandbox-local/README.zh.md:71`）：

| 平台    | runner 链          |
| ------- | ------------------ |
| Linux   | `bwrap` → Landlock |
| macOS   | Seatbelt           |
| Windows | ACL 受限令牌       |

### 5.2 四个边界

看一个开源项目，比看它能做什么更重要的，是看它的边界在哪。dsh 自己把话说得很直白：

**一、这是开发者预览。** README 里有一行加粗的话：「**未来将出现破坏兼容性的变更。**」（`README.zh.md:13`）v0.1.5-alpha.1 这个版本号本身就说明了阶段。

**二、沙箱不是安全边界。** 安全说明里写着（`SAFETY.zh.md:13`）：

> 沙箱、审批提示与权限控制可以降低风险，但不保证隔离，也不能保证防止损害。

换句话说，它约束的是「守规矩的代码」，不是恶意代码。真要跑不可信的内容，还得靠虚拟机或容器级别的隔离。

**三、有些隔离依赖平台的历史包袱。** macOS 的 Seatbelt 后端依赖已弃用的 `sandbox-exec`：macOS 目前仍然提供它，但如果 Apple 哪天移除这个私有策略引擎，这个提供方既无法替换也无法探测（`packages/sandbox/sandbox-local/README.zh.md:130`）。

**四、抽象是有成本的。** Cordis 的五个概念、profile / bundle / patch 的三层叠加、事件域的选择，这些都要学。换来的是「换掉任何一块都不必重写」，付出的是上手时的心智负担。这笔交易划不划算，取决于你是要快速搭一个能跑的 Agent，还是要长期维护一套能被别人扩展的 Agent 平台。

---

## 六、从运行时到工程：为什么推荐《Harness工程实战》

拆到这里，dsh 回答的问题已经很清楚了：Agent 的运行时该怎么搭，能力怎么组合、状态怎么持久、循环怎么替换、边界怎么划。

但还有一个问题它不负责回答：团队怎么在这套运行时之上把工程做好？

这恰恰是另一件事。比如：

- 项目里的 `AGENTS.md` 该写什么？随着 AI 犯的错越来越多，这份文件该怎么演进？
- AI 生成的代码越堆越多，那些看不见的债——你不再理解自己的代码库、架构被无序生成慢慢侵蚀——怎么防？
- 一个十年前的遗留系统，怎么让 AI 参与改造而不把地基挖塌？

这些问题，dsh 的文档不负责回答，它只提供底座。最近出的一本书补的正是这一块。

### 关于这本书

**《Harness工程实战：从零开始驾驭AI软件工程》**，张建飞 著，人民邮电出版社 · 异步图书，2026 年 9 月出版。

作者张建飞是华为软件教练、阿里巴巴前高级技术专家，开源架构框架 [COLA](https://github.com/alibaba/COLA)（1.3 万星）的创始人，此前还写过《代码精进之路：从码农到工匠》和《程序员的底层思维》。COLA 的出发点一直是同一个问题：怎么控制软件复杂度。这本书把这个问题搬到了 AI 时代。作者的判断是，AI 没有改变这个使命，改变的只是方法：从自己写代码，变成通过 Harness 调度 AI。

全书 4 部分 11 章：

| 部分                 | 内容                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| 一、Agent 基础认知   | Function Calling、MCP 标准、Coding Agent，以及从零手写一个 Agent Loop（书中带「让 AI 写贪吃蛇」的实战） |
| 二、Harness 工程理论 | 从 Prompt 工程到 Context 工程再到 Harness 工程的演进脉络与方法论                                        |
| 三、Harness 工程实践 | 经典微服务、绿地项目、棕地项目三类场景，覆盖测试、环境管理、持续交付                                    |
| 四、未来展望         | AI 时代人类工程师的新定位与软件工程的新形态                                                             |

第三部分里那个「棕地」场景值得留意：遗留系统的绞杀式重构，四步走，先用 AI 辅助提取领域知识，再用单元测试织安全网，然后特性开关加防腐隔离，最后才是受约束的重构。这类内容在讲「Agent 能做什么」的文章里几乎见不到，但真到企业里落地，决定成败的往往是它。

**为什么和 dsh 一起看？** 打个比方：dsh 是发动机和底盘，书是驾驶手册和交通规则。前者决定这台车能不能跑、能跑多快；后者决定你能不能安全地开到目的地。如果你正在用编程 Agent 干活，或者正带着团队评估 AI 开发的真实回报，两样都需要。

---

## 结语

dsh 做的事情，可以概括成一句话：它把 Agent 的骨架拆成了插件，让「换掉哪一块」不再等于「重写一遍」。连循环本身都能替换，这在 Agent 框架里是相当激进的取舍，代价是抽象层和学习曲线，收益是扩展不再需要 fork。

而 Harness 工程要回答的，是这套骨架之上怎么把活干对。工具迭代很快，方法论攒得慢，也更要紧。

---

## 相关阅读

- [DeepSeek Harness：一本书讲透 Agent 怎么组织](deepseek-harness-book-charts.md)——同主题新书（亦凯，异步图书 2026-09）的书介与概念图解，与本文第六节推荐的《Harness 工程实战》分工互补
- [OpenHarness 深入浅出：解密开源智能体基础设施](openharness-deep-dive.md)——另一个开源 Harness 的架构拆解，可对照看「Harness 该包含什么」
- [Agent Sandbox 的演进与设计范式](agent-sandbox-design.md)——沙箱从「硬件级隔离」向「策略优先」的演进
- [Claude Code Sandbox 安全隔离机制解析](claude-code-sandbox.md)——Bubblewrap 隔离架构的工程实现
- [扩展托管智能体：让决策与执行解耦](scaling-managed-agents.md)——Agent 基础设施的「POSIX 时刻」

---

## 源文件索引

| 文件                                                              | 关键内容                                                            | 引用点                             |
| ----------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------- |
| `README.zh.md`                                                    | 项目定位、启动命令、开发者预览警告                                  | :5、:13、:29                       |
| `SAFETY.zh.md`                                                    | 沙箱限制声明                                                        | :11-13                             |
| `docs/architecture.zh.md`                                         | 一切皆插件、无特权内核、profile 与 bundle 分层、能力 seam、会话日志 | :11、:13、:19-27、:121、:125、:133 |
| `docs/cordis-primer.zh.md`                                        | Cordis 五个核心概念、五种事件分发模式                               | :9-13                              |
| `docs/glossary.zh.md`                                             | 能力 seam 三角色定义                                                | :9                                 |
| `docs/subsystems/core.zh.md`                                      | `Agent` 接口与循环实现的解耦说明                                    | :59                                |
| `docs/tool-catalog.zh.md`                                         | 模型可见工具清单（含 `cordis_*` 创造模式工具组）                    | 全文                               |
| `packages/core/agent-loop/src/index.ts`                           | `AgentLoop` 类与 `inject` 声明                                      | :359-360                           |
| `packages/core/tools/README.zh.md`                                | PTC 呈现模式与 `run_code` 通道                                      | :125                               |
| `packages/preset/agent-presets/presets/*/preset.yml`              | 四个预设的名称与定位                                                | 全文                               |
| `packages/preset/agent-presets/presets/standard/agent.cordis.yml` | 标准预设的插件树（含预留的 codex / claude-code 子代理）             | 全文                               |
| `packages/sandbox/sandbox-local/README.zh.md`                     | 三平台 runner 链、Seatbelt 依赖弃用接口                             | :71、:130                          |
| `packages/llm/llm-pi-ai/README.zh.md`                             | 模型提供方接入                                                      | :42-60                             |
| `packages/context/agent-instructions/README.zh.md`                | `AGENTS.md` / `CLAUDE.md` 的加载、去重与刷新                        | :32-36、:64                        |
| `packages/mcp/mcp-client`、`packages/acp/acp`                     | MCP 客户端与 ACP 集成                                               | —                                  |

_版本：`deepseek-ai/deepseek-harness` `5dda764e`（`0.1.5-alpha.1`，2026-09-08）。星标数、插件数为 2026-09-09 口径。_

---

## 参考资料

- DeepSeek AI, [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)，MIT 协议，2026-08-13 开源——本文所有源码与文档引用的出处
- DeepSeek AI, [DeepSeek Harness 官方文档](https://deepseek-harness.github.io/deepseek-harness/)——架构、子系统与工具目录
- Cordis, [cordiverse/cordis](https://github.com/cordiverse/cordis)——插件内核，dsh 以 vendor 方式引入
- [A Programming Paradigm for Spatiotemporal Composability](https://arxiv.org/abs/2608.25512), arXiv:2608.25512, 2026——Cordis 的设计论文，时间/空间可组合性两个维度
- 张建飞, 《Harness工程实战：从零开始驾驭AI软件工程》, 人民邮电出版社 · 异步图书, 2026-09, ISBN 9787115701800
