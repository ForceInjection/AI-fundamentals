# KAG：基于知识增强生成的大语言模型逻辑推理与问答框架

## 1. 论文解读

### 1.1 背景与挑战

随着大语言模型（LLM）的快速发展，其在通用领域的表现已令人瞩目。然而，在垂直领域（如金融、法律、医疗等）的应用中，LLM 仍面临着严重的“幻觉”问题、知识时效性滞后以及缺乏严谨逻辑推理能力等挑战。

特别是传统 RAG 系统，往往存在对**知识逻辑（如数值、时间关系、专家规则等）不敏感**以及**检索相关性不足**的问题：

- **检索增强生成 (RAG)**：主要依赖向量相似度，难以处理涉及数学计算或多步逻辑推理的复杂问题。
- **GraphRAG**：虽然引入了结构化知识，但 OpenIE 往往会引入大量噪声，且难以有效对齐领域专家的 Schema。

**KAG (Knowledge Augmented Generation)** 框架应运而生。它由 OpenSPG 团队提出，旨在通过将**知识图谱（KG）的结构化逻辑**与**矢量检索的语义覆盖能力**深度融合，显著提升 LLM 在专业领域的逻辑推理和问答能力。

### 1.2 核心理念：LLM 友好的知识表示与双向索引

KAG 借鉴了 **DIKW（数据、信息、知识、智慧）金字塔模型**，提出了 **LLMFriSPG** 框架，将知识划分为不同的层次，解决了传统 RAG 中文本块之间缺乏逻辑关联的问题。

- **数据层 (Data)**：对应原始文档的分块 (Chunks)。
- **信息层 (Information)**：对应从文本中抽取的实体和关系 (OpenIE)。
- **知识层 (Knowledge)**：对应经过专家治理、符合 Schema 约束的严谨知识。

**互索引 (Mutual-indexing) 机制**是其核心：

- **图结构与文本块互索引**：在构建索引时，KAG 不仅提取实体和关系，还建立图结构与原始文本块（RC 层）之间的双向索引。
- **上下文增强**：这种结构让图节点拥有了可追溯的文本上下文，为 LLM 提供了更丰富的语义描述（如 `description` 和 `summary` 属性），从而增强了模型对特定实例精确含义的理解。

### 1.3 关键创新：逻辑符号引导的混合推理

为了解决复杂多跳问题，KAG 引入了**逻辑形式（Logical Form）**作为中间表示，构建了逻辑符号引导的混合求解引擎。

- **符号化表达**：将自然语言问题分解为可执行的逻辑表达式，包含 `Retrieval`（检索）、`Math`（数学运算）、`Sort`（排序）、`Deduce`（推理）等函数。
- **确定性推理**：利用 LLM 的规划能力，将检索到的知识作为变量代入函数，完成诸如数值计算、集合运算或逻辑判定等确定性推理任务。
- **多轮反射机制**：求解器支持基于全局记忆的多轮反射。如果当前检索内容无法回答问题，系统会生成补充问题进入下一轮迭代，直至得出答案。

### 1.4 关键创新：语义对齐与混合检索

KAG 不仅仅依赖结构化图谱，还通过语义推理增强了系统的鲁棒性。

**基于语义推理的知识对齐 (Knowledge Alignment)**：

- **语义关系补全**：在索引阶段，利用 LLM 预测实体间的语义关系（如同义词、上下位关系 `isA`、包含关系 `contains`、因果关系 `causes` 等）。
- **逻辑路径检索**：在检索阶段，如果用户提到的术语与索引不直接匹配，系统会通过语义推理（如“白内障患者”属于“视力障碍者”）来寻找逻辑相关的正确答案。

**混合检索与图推理的协同**
KAG-Solver 在执行时会优先尝试**图推理（Graph Retrieval）**以确保逻辑严谨性；如果结构化知识不足，则转向**混合检索（Hybrid Retrieval）**。它结合了关键词精确匹配、矢量模糊搜索以及图结构的多跳关联，显著提升了在多跳问答任务中的表现。

---

## 2. KAG 项目介绍与核心实现

本节将基于 KAG 开源项目，深入解析其代码架构与关键实现。

### 2.1 整体架构

KAG 项目遵循“离线构建、在线推理”的范式，通过三大核心组件的协同工作，实现了从知识处理到逻辑推理的全流程闭环：

- **KG Builder（知识构建器）**：负责离线阶段的知识提取与索引构建，将海量数据转化为 LLM 友好的知识表示。
- **KG Solver（知识求解器）**：负责在线阶段的问答推理，利用逻辑形式引导模型进行确定性与概率性的混合推理。
- **KAG Model（模型能力增强）**：作为底层支撑，通过特定的 Prompt 工程与微调策略，全面增强 LLM 在理解、推理与生成环节的原子能力。

### 2.2 知识构建 (KG Builder)

KG Builder 负责将海量的非结构化数据和结构化数据转化为 LLMFriSPG 知识表示。

#### 2.2.1 流水线配置

构建过程通过 `BuilderChain` 管理。以下是 `kag_config.yaml` 片段，展示了 Reader -> Splitter -> Extractor -> Writer 的链路：

```yaml
kag_builder_pipeline:
  chain:
    type: unstructured_builder_chain
    extractor:
      type: knowledge_unit_extractor
      llm: *openie_llm
    reader:
      type: txt_reader
    splitter:
      type: length_splitter
      split_length: 300
    writer:
      type: kg_writer
```

#### 2.2.2 核心数据结构：Chunk 与 SPGRecord

代码中定义了 `Chunk` 和 `SPGRecord` 来实现互索引。

**Chunk (文本块)**：存储原始文本及其向量表示。

```python
# kag/interface/common/model/chunk.py
class Chunk(KAGObject):
    def __init__(self, id: str, content: str, **kwargs):
        self.id = id           # Chunk 的唯一标识
        self.content = content # 文本内容
        self.vector = kwargs.get("vector") # 向量表示
```

**SPGRecord (图记录)**：存储结构化实体，并通过属性关联回 `Chunk`。

```python
# kag/interface/common/model/spg_record.py
class SPGRecord:
    def __init__(self, spg_type_name: SPGTypeName):
        self._spg_type_name = spg_type_name
        self._properties = {}

    # 通过属性关联 Chunk ID，实现 Graph -> Chunk 索引
    def set_property(self, key, value):
        self._properties[key] = value
```

### 2.3 知识推理 (KG Solver)

KG Solver 是 KAG 的推理中枢。

#### 2.3.1 求解器初始化

在 `kag/solver/pipeline/kag_static_pipeline.py` 中，Solver Pipeline 初始化了 Planner、Executor 等组件，并编排执行流程：

```python
# kag/solver/pipeline/kag_static_pipeline.py
@SolverPipelineABC.register("kag_static_pipeline")
class KAGStaticPipeline(SolverPipelineABC):
    def __init__(self, planner: PlannerABC, executors: List[ExecutorABC], generator: GeneratorABC, ...):
        # 1. Planner: 自然语言 -> 逻辑形式 (Logic Form)
        # 2. Executors: 执行算子 (Retrieval, Math, Deduce)
        # 3. Generator: 生成最终答案
        super().__init__()
        self.planner = planner
        self.executors = executors
        self.generator = generator

    async def ainvoke(self, query, **kwargs):
        # 编排执行流程：规划 -> 执行 -> 生成
        # 1. Planning: 生成任务 DAG
        tasks = await self.planning(query, context, **kwargs)
        # 2. Execution: 并行执行任务
        ...
        # 3. Generation: 生成最终答案
        answer = await self.generator.ainvoke(query, context, **kwargs)
        return answer
```

#### 2.3.2 逻辑形式定义 (Logic Form)

Planner 利用 LLM 生成逻辑计划。

```python
# kag/solver/prompt/logic_form_plan.py
class LogicFormPlanPrompt(PromptABC):
    instruct_zh = """
    "function": [
      {
          "function_declaration": "Retrieval(s=..., p=..., o=...)",
          "description": "根据SPO结构检索信息..."
      },
      {
          "function_declaration": "Math(content=[...], target=...)",
          "description": "执行数值计算、排序或计数..."
      },
      {
          "function_declaration": "Deduce(op=judgement|entailment|..., content=[...], target=...)",
          "description": "利用LLM对检索结果进行语义推断..."
      }
    ]
    """
```

#### 2.3.3 推理执行 (Executor)

`Executor` 解析并执行算子。以 `Deduce` 为例：

```python
# kag/solver/executor/deduce/kag_deduce_executor.py
class KagDeduceExecutor(ExecutorABC):
    def call_op(self, sub_query, contents, op, **kwargs):
        if op not in self.prompt_mapping:
            op = "entailment"
        prompt = self.prompt_mapping.get(op, self.deduce_entail_prompt)

        # 调用 LLM 进行推理，传入子问题和上下文
        return self.llm_module.invoke(
            {"instruction": sub_query, "memory": contents},
            prompt,
            **kwargs,
        )
```

### 2.4 模型能力增强 (KAG-Model)

> **说明**：KAG 框架由 **KG Builder**、**KG Solver** 和 **KAG Model** 三大核心组件构成。当前开源的版本主要涵盖前两部分，独立的 KAG Model 专用模型计划在后续逐步发布。当前版本通过先进的 **Prompt Engineering**（提示工程）技术，在通用 LLM 上实现了 KAG Model 的关键能力，确保了框架功能的完整性。

**KAG Model** 是一套旨在提升大模型在专业领域表现的增强机制。它通过特定的策略（如思维链、任务分解等）深度激发模型在 **自然语言理解 (NLU)**、**逻辑推理 (NLI)** 和 **内容生成 (NLG)** 方面的潜能，使其能够精准适配 KAG 的知识构建与混合推理流程。

#### 2.4.1 显式思考机制 (Thinking Process)

为了提升推理的深度与鲁棒性，KAG 在系统 Prompt 中强制要求模型输出 `<think>` 标签包裹的思考过程，这类似于 Chain-of-Thought (CoT) 的显式化。

```python
# kag/solver/prompt/kag_model/kag_system_prompt.py
@PromptABC.register("kag_system")
class KagSystemPrompt(PromptABC):
    template_zh = """当你回答每一个问题时，你必须提供一个思考过程，并将其插入到<think>和</think>之间"""
```

#### 2.4.2 任务分解与澄清 (Clarification & Decomposition)

KAG-Model 通过 `KagClarificationPrompt` 将复杂的自然语言问题转化为结构化的函数调用链 (`Retrieval`, `Math`, `Deduce`)，这是 Solver 能够执行逻辑规划的基础。

```python
# kag/solver/prompt/kag_model/kag_clarification_prompt.py
class KagClarificationPrompt(PromptABC):
    template_zh = """您是一位函数调用专家，能够准确理解函数定义，并精准地将用户查询分解为适当的函数以解决问题。
    ...
    函数用法：Retrieval(s=s_alias:类型[`名称`], p=p_alias:边, o=o_alias:类型[`名称`], ...)
    函数用法：Deduce(op=judgement|entailment|..., content=[...], target=...)
    """
```

#### 2.4.3 答案生成与自我验证 (Generation)

在多步推理结束后，`KagGenerateAnswerPrompt` 负责汇总子问题的结果，生成最终答案，并要求按特定格式输出以供系统解析。

```python
# kag/solver/prompt/kag_model/kag_generate_answer_prompt.py
class KagGenerateAnswerPrompt(PromptABC):
    template_zh = """根据问题、前n个子问题及其答案（由#n表示）来回答最后一个子问题。请用<answer>\\boxed{你的答案}</answer>的格式包裹你的答案。"""
```

### 2.5 小结

通过上述三大核心模块的紧密协作，KAG 实现了对专业领域知识的深层次利用：**KG Builder** 夯实了数据基础，解决了“知识从何而来”的问题；**KG Solver** 提供了推理引擎，解决了“知识如何使用”的问题；而 **KAG Model** 则通过原子能力的增强，解决了“模型如何适配”的问题。三者共同构成了一个高效、严谨且可扩展的知识增强生成系统。

---

## 3. 动手指南

本节演示如何基于 KAG 快速构建一个“人物百科”知识库。我们将直接使用项目中的 `kag/examples/baike` 示例，该示例包含了周星驰、周杰伦等人物的百科数据。

### 3.1 环境准备

确保安装 Python 3.10+ 及 Docker (用于启动 OpenSPG 服务)。

```bash
# 1. 克隆项目并安装
git clone https://github.com/OpenSPG/KAG.git
cd KAG
pip install -e .

# 2. 启动 OpenSPG 服务 (参考 OpenSPG 文档)
# 确保本地 OpenSPG 服务已启动，默认端口 8887
```

### 3.2 初始化项目 (Project Initialization)

进入示例目录并初始化项目，这将连接到 OpenSPG 服务并注册 Schema。

```bash
cd kag/examples/baike

# 初始化项目
knext project restore --host_addr http://127.0.0.1:8887 --proj_path .

# 提交 Schema (定义人物、电影等类型及其属性)
knext schema commit
```

### 3.3 知识构建 (Builder)

使用 `BuilderChainRunner` 读取 `data/` 目录下的百科文本（如 `周星驰百科.txt`），构建知识图谱。

```python
import logging
from kag.builder.runner import BuilderChainRunner
from kag.common.conf import KAG_CONFIG
from kag.common.registry import import_modules_from_path

# 加载当前目录下的自定义组件
import_modules_from_path(".")

# 初始化 Builder Runner
runner = BuilderChainRunner.from_config(
    KAG_CONFIG.all_config["kag_builder_pipeline"]
)

# 执行构建：读取 ./data/ 目录下的数据
runner.invoke("./data/")
print("知识构建完成！")
```

### 3.4 问答推理 (Solver)

初始化 Solver Pipeline，对“周星驰的姓名有何含义？”等问题进行回答。

```python
import asyncio
from kag.solver.main_solver import get_pipeline_conf
from kag.interface import SolverPipelineABC
from kag.common.conf import KAG_CONFIG

# 加载配置
pipeline_config = KAG_CONFIG.all_config["kag_solver_pipeline"]

# 初始化 Solver
solver = SolverPipelineABC.from_config(pipeline_config)

# 执行问答
question = "周星驰的姓名有何含义？"
answer = asyncio.run(solver.ainvoke(question))

print(f"Question: {question}")
print(f"Answer: {answer}")
```

---

## 4. 总结

如果把传统的 RAG 比作一个**只靠直觉联想**来翻阅图书馆资料的学生，那么 KAG 就更像是一个**带着逻辑思维导图**的学生。它不仅知道哪些书可能相关（矢量检索），还能将书中的内容抽构成清晰的逻辑公式（知识图谱与逻辑形式），通过严谨的计算和推导来得出最终答案。

KAG 通过 **LLM 友好的互索引知识表示**与**逻辑符号引导的混合推理引擎**，有效解决了专业领域应用中对逻辑敏感性不足的问题，为构建高精度企业级 AI 应用提供了全新范式。

在社区活动[《大模型背景下私域知识库的构建和可信问答 Meetup》](https://github.com/orgs/OpenSPG/discussions/45)，蚂蚁集团的**桂正科**分享了 KAG 框架的设计理念与实现细节，同时邀请了行业专家参与讨论，探讨了 KAG 在实际场景中的应用：[KAG-知识增强⽣成的垂域知识库知识服务框架-桂正科](https://github.com/user-attachments/files/17741194/KAG-.-.pdf)。

## 5. 参考文献

1. Liang, Lei, et al. "KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation." _arXiv preprint arXiv:2409.13731_ (2024).
2. OpenSPG Team. "KAG Documentation." [https://github.com/OpenSPG/KAG](https://github.com/OpenSPG/KAG)
