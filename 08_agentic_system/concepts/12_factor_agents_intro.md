# 12-Factor Agents - 构建可靠 LLM 应用的原则

> 项目地址：<https://github.com/humanlayer/12-factor-agents/>

## 问题/需求：为什么我们需要新的方法论？

在当前的 AI 应用开发中，我们面临着一个普遍的问题：**大多数标榜为"AI 智能体"的产品实际上并没有那么智能**。根据 Dex 的研究和与 100 多位 SaaS 构建者的交流，发现了以下典型问题：

1. **质量瓶颈**：使用现有框架可以快速达到 70-80% 的质量标准，但要超过 80% 往往需要逆向工程整个框架
2. **框架限制**：现有框架的黑盒特性限制了开发者对提示词、控制流和上下文的精细控制
3. **生产就绪性**：大多数框架无法直接满足生产环境的严格要求
4. **维护困难**：随着需求变化，框架的抽象层反而成为阻碍

正如 Dex 所说："**我看到构建者让客户用上高质量 AI 软件的最快方式是，从智能体构建中获取小型、模块化的概念，并将它们整合到他们现有的产品中**"。

## Why 12-Factor Agents：为什么是 12 要素方法论？

12-Factor Agents 借鉴了著名的 [12 Factor Apps](https://12factor.net/) 方法论，为构建可靠的 LLM 应用提供了一套经过验证的原则。这套方法论的核心理念是：

> **将智能体视为由确定性代码和 LLM 步骤组成的软件系统，而非黑盒框架**

### 核心优势

1. **渐进式采用**：可以逐步引入，不需要重写整个系统
2. **完全控制**：开发者拥有对提示词、上下文、控制流的完全控制权
3. **生产就绪**：每个要素都考虑了生产环境的实际需求
4. **技术栈无关**：适用于任何编程语言或框架
5. **可测试性**：每个要素都可以独立测试和验证

## What 12-Factor Agents：12 个要素详解

以下是基于官方文档的 12 个要素的详细中文介绍，每个要素都包含核心思想、实现细节、最佳实践和实际应用场景。

### 要素 1：自然语言到工具调用 (Natural Language to Tool Calls)

**📁 详细文档**：[factor-01-natural-language-to-tool-calls.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-01-natural-language-to-tool-calls.md)

**可视化示例**：

![自然语言到工具调用](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/110-natural-language-tool-calls.png)

**核心思想**：将用户的自然语言输入原子化地转换为结构化的工具调用，这是构建智能体的最基础模式。

**深入解析**：

这个要素解决的是智能体如何理解用户意图并转化为可执行操作的问题。关键在于"原子化"转换 - 不是试图理解整个复杂工作流，而是将单个自然语言指令精确映射到具体的 API 调用。

**实际案例**：

考虑一个真实的付款链接创建场景：

```text
用户输入："为 Jeff 创建 750 美元的付款链接，用于赞助二月的 AI 技术聚会"
↓
结构化输出：
{
  "function": "create_payment_link",
  "parameters": {
    "amount": 75000,  // 以分为单位
    "currency": "usd",
    "customer_email": "jeff@example.com",
    "description": "二月 AI 技术聚会赞助费用",
    "metadata": {
      "event_type": "sponsorship",
      "month": "february",
      "purpose": "ai_meetup"
    }
  }
}
```

**实现模式**：

```python
# 使用 LLM 进行意图识别和参数提取
async def determine_next_step(natural_language_input: str) -> ToolCall:
    response = await llm.create_completion(
        prompt=f"""
        将以下用户请求转换为结构化的工具调用：
        用户说：{natural_language_input}
        
        可用的工具：create_payment_link, list_customers, list_products
        
        请输出 JSON 格式的工具调用。
        """,
        response_format=ToolCall
    )
    return response

# 处理工具调用
async def handle_tool_call(tool_call: ToolCall):
    if tool_call.function == 'create_payment_link':
        return await stripe.payment_links.create(**tool_call.parameters)
    elif tool_call.function == 'list_customers':
        return await stripe.customers.list(**tool_call.parameters)
    # ... 其他工具处理
```

**最佳实践**：

1. **保持原子性**：每个工具调用应该只执行一个具体操作
2. **明确参数**：为每个参数提供清晰的类型定义和验证
3. **错误处理**：当模型无法匹配到合适工具时，提供优雅的降级方案
4. **上下文增强**：在实际调用前，可能需要额外的信息收集步骤

### 要素 2：拥有自己的提示词 (Own Your Prompts)

**📁 详细文档**：[factor-02-own-your-prompts.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-02-own-your-prompts.md)

**可视化示例**：

![拥有自己的提示词](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/120-own-your-prompts.png)

**核心思想**：提示词是代码的一部分，应该像代码一样被版本控制、测试和迭代优化，而不是依赖框架的黑盒提示词。

**深入解析**：

许多框架提供"黑盒"提示词方法，如 `Agent(role="...", goal="...")`，这在初期很方便，但会限制我们对系统的精细控制。拥有自己的提示词意味着我们可以：

- 精确控制每个 token 的使用
- 进行 A/B 测试比较不同提示词效果
- 针对特定用例优化提示词
- 避免框架升级导致的意外行为变化

**实际对比**：

**❌ 黑盒方式**：

```python
# 框架提供的黑盒方法
agent = Agent(
    role="部署助手",
    goal="安全地部署应用到生产环境",
    personality="谨慎且专业",
    tools=[deploy_backend, deploy_frontend]
)
result = agent.run("部署最新后端到生产环境")
```

**✅ 自有提示词方式**：

```rust
function DetermineNextStep(thread: string) -> 
    DoneForNow | ListGitTags | DeployBackend | DeployFrontend | RequestMoreInformation {
  prompt #"
    {{ _.role("system") }}
    
    你是一个专业的部署助手，负责管理前端和后端系统的部署。
    你遵循最佳实践，确保部署的安全性和成功率。
    
    部署前的检查清单：
    - 确认部署环境（staging vs production）
    - 验证部署标签/版本是否正确
    - 检查当前系统状态
    - 高风险部署需要人工审批
    
    可用工具：deploy_backend, deploy_frontend, check_deployment_status
    敏感操作使用 request_approval 获取人工验证。
    
    思考步骤：
    1. 先检查当前部署状态
    2. 验证部署标签是否存在
    3. 如需要则请求审批
    4. 先部署到 staging 测试
    5. 监控部署进度
    
    {{ _.role("user") }}
    {{ thread }}
    
    下一步应该做什么？
  "#
}
```

**提示词工程最佳实践**：

1. **版本控制**：将提示词存储在单独的`.prompt`文件中
2. **模板化**：使用 Jinja2 或类似模板引擎管理动态内容
3. **测试覆盖**：为提示词编写单元测试和集成测试
4. **性能监控**：追踪不同提示词的 token 使用情况和响应质量
5. **角色分离**：为不同任务类型创建专门的提示词模板

### 要素 3：拥有自己的上下文窗口 (Own Your Context Window)

**📁 详细文档**：[factor-03-own-your-context-window.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md)

**可视化示例**：

![拥有自己的上下文窗口](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/130-own-your-context-building.png)

**核心思想**：完全控制如何构建和格式化传递给 LLM 的上下文，突破标准消息格式的限制，最大化 token 效率和模型理解能力。

**深入解析**：

传统的 LLM 交互使用标准的消息格式（system/user/assistant/tool），但这可能不是最优的信息传递方式。通过自定义上下文格式，我们可以：

- 优化信息密度，减少不必要的 token
- 以模型最理解的格式呈现信息
- 灵活控制哪些信息保留、哪些信息隐藏
- 实现更复杂的上下文管理策略

**标准格式 vs 自定义格式对比**：

**标准消息格式**（低效）：

```yaml
[
  {
    "role": "system",
    "content": "你是一个部署助手..."
  },
  {
    "role": "user", 
    "content": "能否部署后端？"
  },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [{"name": "list_git_tags", "arguments": {}}]
  },
  {
    "role": "tool",
    "name": "list_git_tags",
    "content": "{\"tags\": [...长 JSON...]}"
  }
]
```

**自定义XML格式**（高效）：

```xml
<slack_message>
  来自：@alex
  频道：#deployments  
  内容：能否部署最新后端到生产环境？
</slack_message>

<list_git_tags>
  意图：列出git标签
</list_git_tags>

<list_git_tags_result>
标签：
  - v1.2.3 (最新，2024-03-15)
  - v1.2.2 (稳定版，2024-03-14)
  - v1.2.1 (旧版本，2024-03-13)
</list_git_tags_result>

下一步应该做什么？
```

**上下文工程实现**：

```python
class Thread:
    events: List[Event]

class Event:
    type: Literal["slack_message", "tool_call", "tool_result", "human_input", "error"]
    data: Union[str, ToolCall, ToolResult, HumanInput, ErrorInfo]

def event_to_prompt(event: Event) -> str:
    """将事件转换为优化的提示词格式"""
    if isinstance(event.data, str):
        return f"<{event.type}>\n{event.data}\n</{event.type}>"
    else:
        # 使用YAML格式提高可读性
        yaml_data = yaml.dump(event.data, default_flow_style=False)
        return f"<{event.type}>\n{yaml_data}\n</{event.type}>"

def thread_to_prompt(thread: Thread) -> str:
    """将线程转换为完整的提示词"""
    return '\n\n'.join(event_to_prompt(event) for event in thread.events)
```

**高级上下文管理策略**：

1. **信息压缩**：对于长文本，使用摘要或关键信息提取
2. **选择性遗忘**：移除已解决错误的详细信息
3. **上下文分层**：将信息按重要性分层，确保关键信息始终保留
4. **token预算管理**：动态调整上下文长度以适应模型限制

### 要素 4：工具只是结构化输出 (Tools Are Structured Outputs)

**📁 详细文档**：[factor-04-tools-are-structured-outputs.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-04-tools-are-structured-outputs.md)

**可视化示例**：

![工具只是结构化输出](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/140-tools-are-just-structured-outputs.png)

**核心思想**：工具调用本质上是 LLM 的结构化 JSON 输出，不需要复杂的抽象层。通过简单的数据类定义，就能实现强大的工具调用功能。

**深入解析**：

这个要素强调工具调用的简洁性。我们不需要复杂的框架来处理工具调用，只需要：

1. 定义清晰的数据结构（Pydantic 模型、TypeScript 接口等）
2. LLM 输出符合该结构的 JSON
3. 用简单的 switch 语句处理不同的工具类型

**实现模式**：

```python
from typing import Literal
from pydantic import BaseModel

# 工具定义 - 简单而清晰
class Issue(BaseModel):
    title: str
    description: str
    team_id: str
    assignee_id: str

class CreateIssue(BaseModel):
    intent: Literal["create_issue"]
    issue: Issue

class SearchIssues(BaseModel):
    intent: Literal["search_issues"] 
    query: str
    what_youre_looking_for: str

# 统一的工具响应类型
ToolCall = CreateIssue | SearchIssues

# 工具处理 - 简洁的switch模式
async def handle_tool_call(tool_call: ToolCall) -> dict:
    if tool_call.intent == "create_issue":
        return await linear_client.create_issue(tool_call.issue.dict())
    elif tool_call.intent == "search_issues":
        return await linear_client.search_issues(
            query=tool_call.query,
            description=tool_call.what_youre_looking_for
        )
```

**高级技巧**：

1. **工具组合**：通过组合简单工具实现复杂功能
2. **工具验证**：在 LLM 输出后立即验证参数
3. **工具回退**：当工具调用失败时提供备用方案
4. **工具文档**：为每个工具提供清晰的描述和使用示例

### 要素 5：统一执行状态和业务状态 (Unify Execution State)

**📁 详细文档**：[factor-05-unify-execution-state.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-05-unify-execution-state.md)

**可视化示例**：

![统一执行状态](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/150-unify-state.png)

**核心思想**：将执行状态（当前步骤、等待状态、重试计数等）和业务状态（工具调用结果、对话历史等）统一存储在上下文窗口中，避免复杂的状态分离。

**深入解析**：

传统系统常常将执行状态和业务状态分开管理，但这会带来复杂性。12-Factor Agents 方法论建议将两者统一：

- **执行状态**：只是业务状态的元数据
- **业务状态**：包含所有必要的信息来恢复执行
- **单一真相源**：整个状态可以序列化为一个可恢复的 "线程"

**统一状态的优势**：

1. **简化架构**：不需要复杂的状态管理系统
2. **轻松序列化**：整个状态可以序列化为 JSON
3. **断点续传**：可以从任何状态点恢复执行
4. **调试友好**：所有状态都在一个地方，便于调试
5. **分叉能力**：可以复制状态创建新的执行分支

**实现示例**：

```python
class Thread:
    """统一的状态容器"""
    events: List[Event]           # 业务状态：所有发生的事情
    current_step: Optional[str]   # 执行状态：从事件中推断
    
    def get_execution_state(self) -> dict:
        """从业务状态推断执行状态"""
        last_event = self.events[-1] if self.events else None
        return {
            "current_step": last_event.type if last_event else None,
            "waiting_for": self._infer_waiting_for(),
            "retry_count": self._count_retries()
        }
    
    def serialize(self) -> str:
        """序列化整个状态用于存储"""
        return json.dumps({
            "events": [e.dict() for e in self.events],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        })
```

### 要素 6：使用简单 API 启动/暂停/恢复 (Launch/Pause/Resume)

**📁 详细文档**：[factor-06-launch-pause-resume.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-06-launch-pause-resume.md)

**可视化示例**：

![简单API启动暂停恢复](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/160-pause-resume-with-simple-apis.png)

**核心思想**：智能体应该像普通程序一样，支持标准的生命周期管理 API，包括启动、查询状态、暂停、恢复和停止。

**深入解析**：

这个要素解决的是智能体的工程化问题。在生产环境中，我们需要：

- **简单启动**：通过 API 调用启动智能体
- **暂停等待**：在需要人类输入或长时间操作时暂停
- **外部触发**：通过 webhooks 或其他事件恢复执行
- **状态查询**：随时检查智能体的执行状态

**生命周期管理 API 设计**：

```python
class AgentAPI:
    async def launch(self, initial_input: str) -> str:
        """启动新的智能体执行"""
        thread = Thread(initial_message=initial_input)
        thread_id = await self.store.save(thread)
        await self._process_next_step(thread_id)
        return thread_id
    
    async def pause(self, thread_id: str, reason: str):
        """暂停执行，等待外部输入"""
        thread = await self.store.load(thread_id)
        thread.events.append(PauseEvent(reason=reason))
        await self.store.save(thread)
    
    async def resume(self, thread_id: str, external_input: dict):
        """通过外部输入恢复执行"""
        thread = await self.store.load(thread_id)
        thread.events.append(ResumeEvent(data=external_input))
        await self._process_next_step(thread_id)
    
    async def get_status(self, thread_id: str) -> dict:
        """查询当前执行状态"""
        thread = await self.store.load(thread_id)
        return {
            "status": thread.get_status(),
            "last_activity": thread.last_activity,
            "waiting_for": thread.waiting_for
        }
```

**实际应用场景**：

1. **人工审批流程**：在关键操作前暂停，等待人工确认
2. **长时间任务**：处理可能需要几分钟或几小时的任务
3. **外部依赖**：等待第三方系统响应或 webhook 回调
4. **用户交互**：在 Slack、邮件等渠道中与用户交互

### 要素 7：通过工具调用联系人类 (Contact Humans With Tools)

**📁 详细文档**：[factor-07-contact-humans-with-tools.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-07-contact-humans-with-tools.md)

**可视化示例**：

![通过工具与人类联系](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/170-contact-humans-with-tools.png)

**核心思想**：将人类交互视为一种特殊的工具调用，使智能体能够以结构化方式请求人类输入、审批或协助。

**深入解析**：

这个要素突破了传统人机交互的模式。不是让智能体直接输出文本给人类，而是：

- **结构化请求**：定义清晰的请求格式
- **多渠道支持**：支持 Slack、邮件、短信等多种渠道
- **异步处理**：人类可以在任何时间响应
- **状态追踪**：记录人类响应并继续执行流程

**人类交互工具定义**：

```python
class RequestHumanInput(BaseModel):
    intent: Literal["request_human_input"]
    question: str
    context: str
    options: Options
    urgency: Literal["low", "medium", "high"]
    
class Options(BaseModel):
    format: Literal["free_text", "yes_no", "multiple_choice"]
    choices: List[str] = []
    timeout_hours: int = 24

# 使用示例
async def handle_human_request(request: RequestHumanInput):
    # 格式化消息
    message = format_human_message(request)
    
    # 通过多渠道发送
    await send_to_slack(
        channel="#deployments",
        message=message,
        thread_ts=get_thread_id()
    )
    
    # 记录等待状态
    await save_waiting_state(
        waiting_for="human_response",
        request_id=generate_request_id()
    )
```

**实际应用示例**：

```xml
<!-- 上下文中的完整交互流程 -->
<slack_message>
  来自：@alex
  频道：#deployments
  内容：能否部署后端 v1.2.3 到生产环境？
</slack_message>

<request_human_input>
  意图：请求人工确认
  问题：是否确认部署 v1.2.3 到生产环境？
  上下文：这是生产环境部署，将影响在线用户
  紧急程度：high
  选项：
    - 格式：yes_no
    - 超时：30分钟
</request_human_input>

<human_response>
  响应："是的，请继续"
  已批准：true
  用户：alex@company.com
  时间：2024-03-15T10:30:00Z
</human_response>

<deploy_backend>
  意图：部署后端
  标签：v1.2.3
  环境：production
</deploy_backend>
```

### 要素 8：拥有自己的控制流 (Own Your Control Flow)

**📁 详细文档**：[factor-08-own-your-control-flow.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-08-own-your-control-flow.md)

**可视化示例**：

![拥有自己的控制流](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/180-control-flow.png)

**核心思想**：不要依赖框架的"魔法"控制流，而是显式地设计和实现自己的控制逻辑，包括循环、条件判断、重试机制等。

**深入解析**：

这个要素让我们能够：

- **自定义中断点**：决定在哪些步骤暂停等待外部输入
- **灵活的重试逻辑**：针对不同类型的失败实施不同的重试策略
- **结果汇总**：在继续执行前汇总和分析多个工具调用的结果
- **内存管理**：实现智能的上下文窗口压缩和清理

**控制流模式示例**：

```python
async def handle_next_step(thread: Thread) -> None:
    """自定义控制流处理"""
    
    while True:
        next_step = await determine_next_step(thread_to_prompt(thread))
        
        # 模式1：需要人工澄清 - 中断并等待
        if next_step.intent == 'request_clarification':
            await handle_clarification_request(thread, next_step)
            break  # 等待人工响应
            
        # 模式2：简单查询 - 同步执行并继续
        elif next_step.intent == 'fetch_git_tags':
            result = await git_client.list_tags()
            thread.add_event('git_tags', result)
            continue  # 直接继续下一步
            
        # 模式3：高风险操作 - 需要审批
        elif next_step.intent == 'deploy_production':
            await request_approval(thread, next_step)
            break  # 等待审批
            
        # 模式4：批量处理 - 汇总多个结果
        elif next_step.intent == 'batch_process':
            results = await execute_batch(next_step.items)
            summary = await llm_summarize(results)
            thread.add_event('batch_summary', summary)
            continue
```

**高级控制流功能**：

1. **工具调用审批**：在工具执行前插入审批步骤
2. **结果验证**：用 LLM 验证工具输出是否符合预期
3. **错误恢复**：实现智能的错误处理和恢复策略
4. **性能监控**：记录每个步骤的执行时间和成功率

### 要素 9：将错误压缩到上下文窗口中 (Compact Errors)

**📁 详细文档**：[factor-09-compact-errors.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-09-compact-errors.md)

**可视化示例**：

![压缩错误到上下文窗口](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/190-factor-9-errors-static.png)

**核心思想**：将错误信息以紧凑、结构化的格式传递给 LLM，使智能体能够自我修复，同时避免错误循环。

**深入解析**：

这个要素实现了智能体的"自愈"能力。关键在于：

- **结构化错误**：将错误转换为 LLM 能理解的格式
- **紧凑表示**：避免冗长的堆栈跟踪污染上下文
- **重试限制**：防止无限重试导致的资源浪费
- **升级机制**：在重试失败后升级到人工处理

**错误处理模式**：

```python
class CompactError(BaseModel):
    type: Literal["api_error", "validation_error", "timeout", "unknown"]
    tool: str
    message: str
    retryable: bool
    suggestion: Optional[str] = None
    
class ErrorHandler:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.consecutive_errors = defaultdict(int)
    
    async def handle_error(
        self, 
        error: Exception, 
        tool_name: str,
        thread: Thread
    ) -> str:
        """处理错误并决定下一步"""
        
        # 转换为紧凑错误格式
        compact_error = self._create_compact_error(error, tool_name)
        
        # 记录错误到上下文
        thread.add_event('error', compact_error.dict())
        
        # 检查重试次数
        self.consecutive_errors[tool_name] += 1
        
        if (compact_error.retryable and 
            self.consecutive_errors[tool_name] < self.max_retries):
            return "retry"
        elif self.consecutive_errors[tool_name] >= self.max_retries:
            return "escalate_to_human"
        else:
            return "skip_tool"
    
    def _create_compact_error(self, error: Exception, tool: str) -> CompactError:
        """将异常转换为紧凑错误格式"""
        if isinstance(error, APIError):
            return CompactError(
                type="api_error",
                tool=tool,
                message=str(error),
                retryable=error.status_code >= 500,
                suggestion="检查 API 密钥和网络连接"
            )
        elif isinstance(error, ValidationError):
            return CompactError(
                type="validation_error",
                tool=tool,
                message=str(error),
                retryable=False,
                suggestion="检查输入参数格式"
            )
        # ... 其他错误类型
```

**实际应用示例**：

```python
async def execute_with_recovery(thread: Thread, tool_call: ToolCall):
    """执行工具调用并处理错误"""
    
    try:
        result = await execute_tool(tool_call)
        thread.add_event('success', result)
        return result
        
    except Exception as e:
        handler = ErrorHandler()
        action = await handler.handle_error(e, tool_call.intent, thread)
        
        if action == "retry":
            # LLM会看到错误信息并决定如何调整
            return await execute_with_recovery(thread, tool_call)
        elif action == "escalate_to_human":
            await request_human_help(thread, e)
            return None
        else:
            # 跳过这个工具，继续执行
            thread.add_event('skipped', {"tool": tool_call.intent})
            return None
```

### 要素 10：小型、专注的智能体 (Small, Focused Agents)

**📁 详细文档**：[factor-10-small-focused-agents.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-10-small-focused-agents.md)

**可视化示例**：

![小型专注的智能体](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/1a0-small-focused-agents.png)

**核心思想**：构建专注于特定任务的小型智能体（3-10 个步骤），而不是试图处理所有事情的大型智能体，以保持上下文窗口的可管理性和 LLM 性能。

**深入解析**：

这个要素基于 LLM 的固有限制：随着上下文增长，LLM 更容易迷失或失去焦点。通过限制智能体的范围，我们可以：

- **保持高性能**：小上下文窗口意味着更好的 LLM 理解
- **明确边界**：每个智能体有清晰的职责范围
- **可组合性**：通过组合多个小智能体构建复杂工作流
- **易于测试**：小范围更容易测试和验证

**智能体规模指导原则**：

| 智能体类型 | 步骤范围 | 示例场景 |
|-----------|----------|----------|
| 原子操作 | 1-3步 | 创建付款链接、查询用户信息 |
| 简单工作流 | 3-10步 | 处理客户退款、部署单个服务 |
| 复杂工作流 | 10-20步 | 完整的发布流程、多系统协调 |

**智能体组合模式**：

```python
class SmallAgent:
    """小型专注智能体基类"""
    
    def __init__(self, name: str, max_steps: int = 10):
        self.name = name
        self.max_steps = max_steps
        
    async def execute(self, input_data: dict) -> dict:
        """执行智能体任务"""
        thread = Thread()
        thread.add_event('start', input_data)
        
        for step in range(self.max_steps):
            next_action = await self.determine_next_step(thread)
            
            if next_action.intent == 'complete':
                return self.format_result(thread)
                
            result = await self.execute_action(next_action)
            thread.add_event('step_result', result)
            
        # 如果达到步骤限制，升级到更大的智能体
        return await self.escalate_to_larger_agent(thread)

# 实际使用：组合多个小智能体
class DeploymentOrchestrator:
    """通过组合小智能体实现复杂部署"""
    
    def __init__(self):
        self.agents = {
            'code_check': CodeCheckAgent(),
            'build': BuildAgent(), 
            'test': TestAgent(),
            'deploy': DeployAgent()
        }
    
    async def deploy_application(self, config: dict):
        """协调多个小智能体完成部署"""
        
        # 1. 代码检查
        check_result = await self.agents['code_check'].execute(config)
        if not check_result['success']:
            return check_result
            
        # 2. 构建
        build_result = await self.agents['build'].execute(check_result)
        if not build_result['success']:
            return build_result
            
        # 3. 测试
        test_result = await self.agents['test'].execute(build_result)
        if not test_result['success']:
            return test_result
            
        # 4. 部署
        return await self.agents['deploy'].execute(test_result)
```

**智能体演进策略**：

1. **从小开始**：先构建最小可行智能体
2. **逐步扩展**：随着 LLM 能力提升慢慢扩大范围
3. **质量监控**：始终监控性能指标，确保质量不下降
4. **分解重构**：当智能体变得过于复杂时，考虑分解为更小单元

### 要素 11：从任何地方触发 (Trigger From Anywhere)

**📁 详细文档**：[factor-11-trigger-from-anywhere.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-11-trigger-from-anywhere.md)

**可视化示例**：

![从任何地方触发](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/1b0-trigger-from-anywhere.png)

**核心思想**：支持从任何渠道触发智能体（Slack、邮件、短信、Webhook、定时任务等），并让智能体能够在相同渠道响应用户。

**深入解析**：

这个要素解决了智能体的可用性问题。用户不应该需要学习新的界面或工作流，智能体应该融入现有的工作环境：

- **多渠道触发**：支持用户习惯的沟通方式
- **上下文保持**：在不同渠道间保持对话上下文
- **响应一致性**：无论通过哪个渠道触发，响应质量一致
- **权限管理**：不同渠道可能有不同的权限要求

**渠道适配器模式**：

```python
class ChannelAdapter:
    """渠道适配器基类"""
    
    async def receive_message(self, raw_message: dict) -> dict:
        """将原始消息转换为标准格式"""
        raise NotImplementedError
    
    async def send_response(self, response: dict, context: dict):
        """将响应发送到对应渠道"""
        raise NotImplementedError

class SlackAdapter(ChannelAdapter):
    """Slack渠道适配器"""
    
    async def receive_message(self, slack_event: dict) -> dict:
        return {
            "user": slack_event["user"],
            "text": slack_event["text"],
            "channel": slack_event["channel"],
            "timestamp": slack_event["ts"],
            "thread_ts": slack_event.get("thread_ts")
        }
    
    async def send_response(self, response: dict, context: dict):
        await slack_client.chat_postMessage(
            channel=context["channel"],
            text=response["text"],
            thread_ts=context.get("thread_ts")
        )

class EmailAdapter(ChannelAdapter):
    """邮件渠道适配器"""
    
    async def receive_message(self, email_data: dict) -> dict:
        return {
            "user": email_data["from"],
            "text": email_data["body"],
            "subject": email_data["subject"],
            "message_id": email_data["message_id"]
        }
    
    async def send_response(self, response: dict, context: dict):
        await email_client.send_mail(
            to=context["user"],
            subject=f"Re: {context['subject']}",
            body=response["text"]
        )
```

**统一触发器实现**：

```python
class UniversalTrigger:
    """统一触发器管理"""
    
    def __init__(self):
        self.adapters = {
            'slack': SlackAdapter(),
            'email': EmailAdapter(),
            'webhook': WebhookAdapter(),
            'cron': CronAdapter()
        }
    
    async def handle_incoming(self, channel: str, raw_data: dict):
        """处理来自任何渠道的请求"""
        
        # 1. 使用对应适配器转换消息
        adapter = self.adapters[channel]
        message = await adapter.receive_message(raw_data)
        
        # 2. 创建或恢复线程
        thread = await self.get_or_create_thread(
            user=message["user"],
            channel=channel
        )
        
        # 3. 执行智能体逻辑
        response = await self.agent.execute({
            "message": message,
            "thread": thread
        })
        
        # 4. 通过相同渠道响应
        await adapter.send_response(response, message)
        
        # 5. 保存线程状态
        await self.save_thread(thread)
```

### 要素 12：让智能体成为无状态归约器 (Stateless Reducer)

**📁 详细文档**：[factor-12-stateless-reducer.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-12-stateless-reducer.md)

**可视化示例**：

![无状态归约器](https://raw.githubusercontent.com/humanlayer/12-factor-agents/main/img/1c0-stateless-reducer.png)

**核心思想**：将智能体设计为纯函数（归约器），输入当前状态和事件，输出新状态，实现函数式编程的无副作用特性。

**深入解析**：

这个要素将智能体提升到函数式编程的高度。通过将智能体视为纯函数：

- **可预测性**：相同的输入总是产生相同的输出
- **可测试性**：易于单元测试和验证
- **可组合性**：可以像乐高积木一样组合多个归约器
- **并发安全**：无状态设计天然支持并发执行

**纯函数智能体模式**：

```python
from typing import List, Callable, TypeVar
from functools import reduce

T = TypeVar('T')
E = TypeVar('E')

class AgentState(BaseModel):
    """智能体状态"""
    events: List[dict]
    current_step: Optional[str] = None
    waiting_for: Optional[str] = None
    
class AgentEvent(BaseModel):
    """智能体事件"""
    type: str
    data: dict
    timestamp: datetime

class StatelessReducer:
    """无状态归约器实现"""
    
    @staticmethod
    def reducer(state: AgentState, event: AgentEvent) -> AgentState:
        """纯函数归约器"""
        
        # 创建新状态（不修改原状态）
        new_state = state.copy()
        new_state.events = state.events + [event.dict()]
        
        # 根据事件类型更新状态
        if event.type == "tool_call":
            new_state.current_step = event.data["tool"]
            new_state.waiting_for = "tool_result"
            
        elif event.type == "tool_result":
            new_state.current_step = None
            new_state.waiting_for = None
            
        elif event.type == "human_request":
            new_state.current_step = "waiting_for_human"
            new_state.waiting_for = "human_response"
            
        elif event.type == "human_response":
            new_state.current_step = "processing"
            new_state.waiting_for = None
            
        return new_state
    
    @staticmethod
    def process_events(events: List[AgentEvent]) -> AgentState:
        """处理事件序列"""
        initial_state = AgentState(events=[])
        return reduce(StatelessReducer.reducer, events, initial_state)

# 使用示例
events = [
    AgentEvent(type="start", data={"input": "deploy backend"}),
    AgentEvent(type="tool_call", data={"tool": "check_git"}),
    AgentEvent(type="tool_result", data={"tags": ["v1.2.3"]}),
    AgentEvent(type="human_request", data={"question": "deploy v1.2.3?"}),
    AgentEvent(type="human_response", data={"approved": True}),
    AgentEvent(type="tool_call", data={"tool": "deploy"}),
    AgentEvent(type="tool_result", data={"status": "success"})
]

final_state = StatelessReducer.process_events(events)
```

**函数式组合**：

```python
class AgentComposer:
    """智能体组合器"""
    
    @staticmethod
    def compose(*reducers: Callable) -> Callable:
        """组合多个归约器"""
        def combined_reducer(state, event):
            for reducer in reducers:
                state = reducer(state, event)
            return state
        return combined_reducer
    
    @staticmethod
    def with_logging(reducer: Callable) -> Callable:
        """添加日志功能的装饰器"""
        def logging_reducer(state, event):
            print(f"Processing {event.type} with data: {event.data}")
            new_state = reducer(state, event)
            print(f"New state: {new_state.current_step}")
            return new_state
        return logging_reducer

# 创建带日志的组合归约器
logged_reducer = AgentComposer.with_logging(
    AgentComposer.compose(
        StatelessReducer.reducer,
        ValidationReducer.reducer,
        AuditReducer.reducer
    )
)
```

**无状态智能体的优势**：

1. **可重现性**：相同的输入总是产生相同的结果
2. **时间旅行调试**：可以回放任何历史状态
3. **并发安全**：多个实例可以安全并行执行
4. **缓存友好**：纯函数结果可以安全缓存
5. **测试简单**：不需要复杂的测试设置和清理

## 总结

12-Factor Agents 不是另一个框架，而是一套经过验证的原则和方法论。它帮助开发者构建真正可靠、可维护的 LLM 应用，而不被框架的抽象所限制。正如方法论本身所强调的：**"最好的智能体是由确定性代码和恰到好处的 LLM 步骤组成的软件系统"**。

通过采用这套方法论，开发者可以：

- 保持对系统的完全控制
- 逐步改进现有系统
- 构建真正生产就绪的 LLM 应用
- 避免常见的框架陷阱
