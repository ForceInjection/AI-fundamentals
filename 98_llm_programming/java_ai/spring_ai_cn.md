# 使用 Spring AI 构建高效 LLM 代理（第一部分）

> **原文地址**：`https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns`

在最近的一篇研究论文中：[**构建高效代理**](https://www.anthropic.com/research/building-effective-agents)，`Anthropic` 分享了关于构建高效大型语言模型（`LLM`）代理的宝贵见解。这项研究特别有趣的地方在于它强调简单性和可组合性，而非复杂框架。让我们探讨如何利用 [`Spring AI`](https://docs.spring.io/spring-ai/reference/index.html) 将这些原则转化为实际实现。

![代理系统](https://raw.githubusercontent.com/spring-io/spring-io-static/refs/heads/main/blog/tzolov/spring-ai-agentic-systems.jpg)

虽然模式描述和图解源自 `Anthropic` 的原始论文，但我们将重点放在如何使用 `Spring AI` 的模型可移植性和结构化输出功能来实现这些模式。我们建议先阅读原始论文。

[`agentic-patterns`](https://github.com/spring-projects/spring-ai-examples/tree/main/agentic-patterns) 项目实现了下文讨论的模式。

## 1. 代理系统

该研究论文在架构上区分了两种类型的代理系统：

1. **工作流**：通过预定义代码路径编排 `LLM` 和工具的系统（例如规定性系统）
2. **代理**：`LLM` 动态指导自身过程和工具使用的系统

关键洞见是，虽然完全自主的代理看似诱人，但对于明确定义的任务，工作流通常能提供更好的可预测性和一致性。这与企业需求完美契合，因为可靠性和可维护性至关重要。

让我们通过五种基本模式来研究 `Spring AI` 如何实现这些概念，每种模式服务于特定用例：

### 1.1 [链式工作流](https://github.com/spring-projects/spring-ai-examples/tree/main/agentic-patterns/chain-workflow)

链式工作流模式体现了将复杂任务分解为更简单、更易管理的步骤的原则。

![提示链式工作流](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F7418719e3dab222dccb379b8879e1dc08ad34c78-2401x1000.png&w=3840&q=75)

**适用场景：**

* 具有明确顺序步骤的任务
* 当您愿意以延迟换取更高准确性时
* 当每个步骤都基于前一步骤的输出构建时

以下是 `Spring AI` 实现的一个实际示例：

```java
public class ChainWorkflow {
    private final ChatClient chatClient;
    private final String[] systemPrompts;

    // 通过一系列提示处理输入，其中每个步骤的输出
    // 成为链中下一步骤的输入。     
    public String chain(String userInput) {
        String response = userInput;
        for (String prompt : systemPrompts) {
            // 将系统提示与先前响应结合
            String input = String.format("{%s}\n {%s}", prompt, response);
            // 通过 LLM 处理并捕获输出
            response = chatClient.prompt(input).call().content();
        }
        return response;
    }
}
```

此实现展示了几个关键原则：

* 每个步骤都有明确的职责
* 一个步骤的输出成为下一个步骤的输入
* 链易于扩展和维护

### 1.2 [并行化工作流](https://github.com/spring-projects/spring-ai-examples/tree/main/agentic-patterns/parallelization-worflow)

`LLM` 可以同时处理任务，并通过程序化方式聚合其输出。并行化工作流体现为两种关键变体：

* **分段**：将任务分解为独立的子任务进行并行处理
* **投票**：运行同一任务的多个实例以达成共识

![并行化工作流](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840&q=75)

**适用场景：**

* 处理大量相似但独立的项目
* 需要多个独立视角的任务
* 当处理时间关键且任务可并行时

并行化工作流模式展示了多个大型语言模型（`LLM`）操作的高效并发处理。此模式特别适用于需要并行执行 LLM 调用并自动聚合输出的场景。

以下是使用并行化工作流的基本示例：

```java
List<String> parallelResponse = new ParallelizationWorkflow(chatClient)
    .parallel(
        "Analyze how market changes will impact this stakeholder group.",
        List.of(
            "Customers: ...",
            "Employees: ...",
            "Investors: ...",
            "Suppliers: ..."
        ),
        4
    );
```

此示例演示了利益相关者分析的并行处理，其中每个利益相关者群体被并发分析。

### 1.3 [路由工作流](https://github.com/spring-projects/spring-ai-examples/tree/main/agentic-patterns/routing-workflow)

路由模式实现了智能任务分发，支持对不同类型输入进行专门处理。

![路由工作流](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840&q=75)

此模式专为复杂任务设计，其中不同类型的输入更适合由专门流程处理。它使用 `LLM` 分析输入内容，并将其路由到最合适的专门提示或处理程序。

**适用场景：**

* 具有不同输入类别的复杂任务
* 当不同输入需要专门处理时
* 当分类可以准确处理时

以下是使用路由工作流的基本示例：

```java
@Autowired
private ChatClient chatClient;

// Create the workflow
RoutingWorkflow workflow = new RoutingWorkflow(chatClient);

// Define specialized prompts for different types of input
Map<String, String> routes = Map.of(
    "billing", "You are a billing specialist. Help resolve billing issues...",
    "technical", "You are a technical support engineer. Help solve technical problems...",
    "general", "You are a customer service representative. Help with general inquiries..."
);

// Process input
String input = "My account was charged twice last week";
String response = workflow.route(input, routes);
```

### 1.4 [协调器-工作者](https://github.com/spring-projects/spring-ai-examples/tree/main/agentic-patterns/orchestrator-workers-workflow)

此模式展示了如何在保持控制的同时实现更复杂的类代理行为：

* 中央 `LLM` 协调任务分解
* 专门工作者处理特定子任务
* 清晰的边界保持系统可靠性

![协调工作流](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840&q=75)

**适用场景：**

* 子任务无法预先预测的复杂任务
* 需要不同方法或视角的任务
* 需要自适应问题解决的情况

实现使用 `Spring AI` 的 `ChatClient` 进行 `LLM` 交互，包含：

```java
public class OrchestratorWorkersWorkflow {
    public WorkerResponse process(String taskDescription) {
        // 1. 协调器分析任务并确定子任务
        OrchestratorResponse orchestratorResponse = // ...

        // 2. 工作者并行处理子任务
        List<String> workerResponses = // ...

        // 3. 结果合并为最终响应
        return new WorkerResponse(/*...*/);
    }
}
```

#### 1.4.1 使用示例

```java
ChatClient chatClient = // ... initialize chat client
OrchestratorWorkersWorkflow workflow = new OrchestratorWorkersWorkflow(chatClient);

// Process a task
WorkerResponse response = workflow.process(
    "Generate both technical and user-friendly documentation for a REST API endpoint"
);

// Access results
System.out.println("Analysis: " + response.analysis());
System.out.println("Worker Outputs: " + response.workerResponses());
```

### 1.5 [评估器-优化器](https://github.com/spring-projects/spring-ai-examples/tree/main/agentic-patterns/evaluator-optimizer-workflow)

评估器-优化器模式实现了一个双 `LLM` 过程，其中一个模型生成响应，另一个在迭代循环中提供评估和反馈，类似于人类作者的改进过程。该模式包含两个主要组件：

* **生成器 LLM**：生成初始响应并根据反馈改进
* **评估器 LLM**：分析响应并提供改进的详细反馈

![评估器-优化器工作流](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75)

**适用场景：**

* 存在明确的评估标准
* 迭代改进提供可衡量的价值
* 任务受益于多轮批评

实现使用 `Spring AI` 的 `ChatClient` 进行 `LLM` 交互，包含：

```java
public class EvaluatorOptimizerWorkflow {
    public RefinedResponse loop(String task) {
        // 1. Generate initial solution
        Generation generation = generate(task, context);
        
        // 2. Evaluate the solution
        EvaluationResponse evaluation = evaluate(generation.response(), task);
        
        // 3. If PASS, return solution
        // 4. If NEEDS_IMPROVEMENT, incorporate feedback and generate new solution
        // 5. Repeat until satisfactory
        return new RefinedResponse(finalSolution, chainOfThought);
    }
}
```

#### 1.5.1 使用示例

```java
ChatClient chatClient = // ... initialize chat client
EvaluatorOptimizerWorkflow workflow = new EvaluatorOptimizerWorkflow(chatClient);

// Process a task
RefinedResponse response = workflow.loop(
    "Create a Java class implementing a thread-safe counter"
);

// Access results
System.out.println("Final Solution: " + response.solution());
System.out.println("Evolution: " + response.chainOfThought());
```

## 2. Spring AI 的实现优势

`Spring AI` 对这些模式的实现提供了与 `Anthropic` 建议一致的多个优势：

### 2.1 [模型可移植性](https://docs.spring.io/spring-ai/reference/api/chat/comparison.html)

```xml
<!-- 通过依赖项轻松切换模型 -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
```

### 2.2 [结构化输出](https://docs.spring.io/spring-ai/reference/api/structured-output-converter.html)

```java
// LLM 响应的类型安全处理
EvaluationResponse response = chatClient.prompt(prompt)
    .call()
    .entity(EvaluationResponse.class);
```

### 2.3 [一致 API](https://docs.spring.io/spring-ai/reference/api/chatclient.html)

* 跨不同 `LLM` 提供商的统一接口
* 内置错误处理和重试
* 灵活的提示管理

## 3. 最佳实践与建议

基于 `Anthropic` 的研究和 `Spring AI` 的实现，以下是构建高效基于 `LLM` 系统的关键建议：

### 3.1 从简单开始

* 在添加复杂性之前从基本工作流入手
* 使用满足需求的最简单模式
* 仅在需要时增加复杂度

### 3.2 为可靠性设计

* 实现清晰的错误处理
* 尽可能使用类型安全响应
* 在每个步骤中构建验证

### 3.3 权衡考虑

* 平衡延迟与准确性
* 评估何时使用并行处理
* 在固定工作流与动态代理之间选择

## 4. 未来工作

在本系列的第二部分中，我们将探索如何通过结合这些基础模式与高级功能来构建更先进的代理：

### 4.1 模式组合

* 组合多个模式以创建更强大的工作流
* 构建利用每种模式优势的混合系统
* 创建适应不断变化需求的灵活架构

### 4.2 高级代理内存管理

* 实现跨对话的持久内存
* 高效管理上下文窗口
* 开发长期知识保留策略

### 4.3 工具与模型上下文协议（MCP）集成

* 通过标准化接口利用外部工具
* 实现 `MCP` 以增强模型交互
* 构建可扩展的代理架构

敬请期待这些高级功能的详细实现和最佳实践。

## 结论

`Anthropic` 的研究洞见与 `Spring AI` 的实践实现相结合，为构建高效的基于 `LLM` 的系统提供了强大框架。通过遵循这些模式和原则，开发者可以创建健壮、可维护且高效的 `AI` 应用，在避免不必要复杂性的同时提供实际价值。

关键是记住，有时最简单的解决方案最有效。从基本模式开始，彻底理解您的用例，并仅在复杂性明确提升系统性能或能力时才增加它。
