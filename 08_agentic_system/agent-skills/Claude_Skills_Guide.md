# 给 Claude 写本“标准操作手册”：Agent Skills 实战与深度解析

想象一下，你招聘了一位极其聪明的实习生（Claude）。你给了他计算器、字典和浏览器（Tools），他能完美地执行“计算这个数字”或“查找那个单词”这样的原子任务。

但是，当你要求他“完成这份年度市场分析报告”时，问题出现了。尽管他有工具，但他不知道公司的报告格式标准是什么，不知道数据该去哪里抓取，也不知道分析的逻辑顺序。

这时候，你有两种选择：

1. **口头传授**：每次都在对话中把几十条规则絮絮叨叨说一遍（费时费力，还容易遗漏）。
2. **给他一本标准操作手册（SOP）**：写好一份标准的作业流程文档，告诉他：“遇到这类任务，就按这个手册做。”

**Claude Agent Skills，就是这本“标准操作手册”。**

本文将带你从零开始，通过亲手构建一个 **PDF 翻译助手**，来直观感受什么是 Agent Skills，并深入探讨为什么它是构建高级 AI Agent 的关键拼图。

---

## 1. 动手实战：构建你的第一个 Skill

`Show me the code`：让我们通过动手构建一个 `pdf-translator` Skill 来直观地理解这些概念。

### 1.1 我们的目标

我们希望 Claude 具备一项新技能：**当用户扔给它一个 PDF 文件并要求翻译时，它能自动提取文字、翻译成目标语言，并整整齐齐地存为一个 Markdown 文件。**

### 1.2 准备工作

首先，我们需要为 Claude 准备好“工作台”（目录结构）和“工具箱”（Python 环境）。

**目录结构推荐**：

```text
skills/
└── pdf-translator/
    ├── SKILL.md              # 这就是那本“标准操作手册”
    ├── requirements.txt      # 依赖清单
    └── scripts/              # 具体的工具脚本
        ├── extract_text.py
        └── generate_md.py
```

**环境安装**：

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装必要的库 (PyPDF2 用于读 PDF, reportlab 用于生成测试文件)
pip install PyPDF2 reportlab
```

### 1.3 第一步：编写“标准操作手册” (SKILL.md)

这是最关键的一步。`SKILL.md` 文件由两部分组成：

1. **Frontmatter (头部配置)**：用 YAML 格式定义，告诉系统这个 Skill 叫什么、干什么用。
2. **Instruction (正文指令)**：用 Markdown 格式定义，告诉 Claude 具体的操作步骤。

**文件路径**：`skills/pdf-translator/SKILL.md`

```markdown
---
name: pdf-translator
description: Extract text from PDF files, translate it to a target language, and save the result as a Markdown file. Use this skill when the user wants to translate a PDF document.
---

# PDF Translator Skill

## Instructions

Follow these steps to translate a PDF file:

1.  **Identify the PDF File**: Confirm the path to the PDF file the user wants to translate. If the path is relative, resolve it to an absolute path.
2.  **Extract Text**: Use the `extract_text.py` script located in the `scripts` directory of this skill to extract text from the PDF.
    - Command: `python3 skills/pdf-translator/scripts/extract_text.py <path_to_pdf>`
    - Note: Ensure you are using the correct python environment.
3.  **Translate Content**:
    - Read the output from the extraction step.
    - Translate the extracted text into the target language requested by the user.
    - Maintain the original structure (headings, paragraphs) as much as possible using Markdown formatting.
4.  **Save Output**:
    - Create a new Markdown file with the translated content.
    - Filename format: `<original_filename>_translated.md`.
    - Notify the user of the output file location.

## Examples

**User**: "Translate the file papers/deep_learning.pdf to Chinese."

**Claude**:

1.  Locates `papers/deep_learning.pdf`.
2.  Runs: `python3 skills/pdf-translator/scripts/extract_text.py papers/deep_learning.pdf`
3.  Translates the extracted text to Chinese.
4.  Saves the result to `papers/deep_learning_translated.md`.
5.  Responds: "I have translated the PDF and saved it to `papers/deep_learning_translated.md`."
```

### 1.4 第二步：打造工具 (Python 脚本)

“标准操作手册”里提到了要使用工具来提取文本和保存文件。Claude 自己没有手，我们得给它造一双。

**1. 提取文本的工具 (`scripts/extract_text.py`)**：

```python
import sys
import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    try:
        reader = PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        # 用双换行符连接，保留段落感
        print("\n\n".join(text))
    except Exception as e:
        print(f"Error extracting text: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py <path_to_pdf>", file=sys.stderr)
        sys.exit(1)
    extract_text_from_pdf(sys.argv[1])
```

**2. 生成文件的工具 (`scripts/generate_md.py`)**：

```python
import sys
import os
import datetime

def generate_markdown(content, output_path, source_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 加上一点元数据，让文件看起来更专业
    header = f"""---
title: Translated Document
source: {source_file}
date: {timestamp}
generated_by: Claude Agent Skill (pdf-translator)
---

"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + content)
        print(f"Successfully generated Markdown file at: {output_path}")
    except Exception as e:
        print(f"Error writing file: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_md.py <output_path> <source_filename> [input_text_file]", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[1]
    source_filename = sys.argv[2]

    # 支持从文件读取或直接从管道(stdin)读取
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = sys.stdin.read()

    generate_markdown(content, output_path, source_filename)
```

### 1.5 第三步：上岗实操 (注册与验证)

一切准备就绪，现在我们需要告诉 Claude：“嘿，你学会了一项新技能。”

**注册技能**：
在终端中运行：

```bash
/plugin add ./skills/pdf-translator
```

**开始验证**：
你可以像对待一位同事一样对 Claude 说：

> "请帮我把 `skills/pdf-translator/test_sample.pdf` 翻译成中文。"

此时，你会看到一个关键变化：Claude 不再只是“会聊天”，而是开始按你定义的流程执行：调用脚本 -> 提取文字 -> 翻译 -> 保存文件。

---

## 2. 深度思考：为什么我们需要 Agent Skills？

通过刚才的实战，你可能已经感觉到了 Skill 的威力。但你可能会问：_“我直接用 Function Calling 给 Claude 一个 `extract_pdf` 函数不就行了吗？为什么要搞这么复杂的 Skill 结构？”_

### 2.1 从“原子工具”到“工作流”

传统的 **Function Calling (工具调用)** 赋予了 LLM 执行**原子操作**的能力（例如：查询天气、执行 SQL）。这就像是给了厨师一把刀。

但是，现实世界的任务往往是**复杂的工作流**。比如“翻译 PDF”这个任务，它包含了：

1. 检查文件是否存在。
2. 提取文本。
3. 处理文本（翻译）。
4. 格式化输出。
5. 保存文件。

如果没有 Skill，你需要：

- **要么**：在 System Prompt 里写一大堆规则（容易导致 Context 爆炸，且容易被遗忘）。
- **要么**：在你的应用程序代码里写死这个逻辑（失去了 LLM 的灵活性）。

**Agent Skills 填补了这一空白**。它允许你定义**“如何组合使用工具来完成特定任务”**的知识（Procedural Knowledge）。

### 2.2 核心概念辨析：Skills vs. Function Calling vs. MCP

为了更清晰地理解，我们来看一下这三个容易混淆的概念：

| 特性             | Agent Skills               | Function Calling (Tool Use) | MCP (Model Context Protocol) |
| :--------------- | :------------------------- | :-------------------------- | :--------------------------- |
| **核心定位**     | **流程编排与知识注入**     | **原子能力执行**            | **标准化连接协议**           |
| **它解决什么？** | "我该按什么步骤做这件事？" | "我现在需要执行什么动作？"  | "我能连接哪些数据和工具？"   |
| **定义方式**     | Markdown (`SKILL.md`)      | API Schema (JSON)           | 协议标准 (Server/Client)     |
| **抽象层级**     | 高层 (High-level)          | 低层 (Low-level)            | 接口层 (Interface layer)     |

### 2.3 一个生动的比喻：烹饪

如果把构建智能助手比作**烹饪一道大餐**：

> 在本文中，“菜谱”就是“标准操作手册（SOP）”的通俗说法。

- **Function Calling** 是**厨师的手**：它具备切菜、开火、撒盐等基本动作能力。
- **MCP** 是**标准化厨房接口**：它规定了燃气灶怎么接、冰箱怎么开，让厨师可以连接并使用各种品牌的厨具（工具）和食材库（数据源）。
- **Agent Skills** 是**菜谱 (SOP)**：它指导厨师“先切菜，再热油，最后炒制”，通过编排一系列动作来完成特定的烹饪任务。

在我们的 `pdf-translator` 案例中：

1. Python 脚本是具体的**厨具**。
2. Claude 的 Function Calling 能力是**厨师的手**。
3. `SKILL.md` 就是那份**菜谱**，告诉 Claude 先用提取器（切菜），再翻译（烹饪），最后保存（装盘）。

---

## 3. 进阶解析：它是如何工作的？

既然 Skill 不是可执行代码（它只是 Markdown），那它到底是怎么“运行”的呢？让我们揭开它的面纱。

### 3.1 架构解密：Meta-tool 与 Prompt 注入

根据 First Principles Deep Dive [3] 的深度解析，Claude Agent Skills 的架构其实包含两个层面：

1. **Skill Meta-tool (大写的 Skill 工具)**：这是一个“元工具”，它像一个总管，管理着所有的具体技能。
2. **Individual Skills (具体的技能)**：如我们的 `pdf-translator`，它们是“指令包”。

当 Claude 决定使用 `pdf-translator` 时，系统实际上执行了 **Prompt Expansion (提示词展开)**。它读取 `SKILL.md` 的内容，将其作为一段新的 System Prompt 或 Context 注入到当前的对话中。

**关键点**：Skill 不仅仅注入文本，它还能修改 **Execution Context (执行上下文)**，比如改变当前可用的工具集合，甚至切换底层模型。这使得 Skill 是在**运行时 (Runtime)** 动态地重塑 Claude 的行为。

### 3.2 决策机制：没有魔法，只有推理

Claude 是如何知道该用哪个 Skill 的？

你可能会认为系统里写了复杂的正则表达式或分类器来匹配用户意图。**事实并非如此。**

这一切完全依赖于 **Pure LLM Reasoning (纯 LLM 推理)**：

1. 系统将所有可用 Skill 的 `name` 和 `description`（即 Frontmatter 里的信息）展示给 Claude。
2. Claude 运用其强大的语言理解能力，阅读这些描述。
3. 如果用户说“帮我翻译这个 PDF”，Claude 看到 `pdf-translator` 的描述是 "Extract text from PDF files..."，通过逻辑推理判定匹配。

没有硬编码的路由，没有机器学习分类器，完全是 Claude 自己在做判断。

### 3.3 幕后机制：渐进式披露 (Progressive Disclosure)

你可能会担心：_“如果我有 100 个 Skill，全部注入到 Prompt 里，Context Window 岂不是瞬间爆炸？”_

Claude 采用了一种聪明的**渐进式披露**机制，这与人类学习新知识的过程非常相似：

1. **目录阶段 (Discovery)**：Claude 初始只看到所有 Skill 的元数据（Frontmatter）。系统会扫描 `~/.config/claude/skills/`、项目下的 `.claude/skills/` 等路径来构建这个目录。
2. **调取阶段 (Loading)**：只有当 Claude 决定调用某个 Skill 后，系统才会加载该 Skill 完整的 `SKILL.md` 内容。
3. **执行阶段 (Execution)**：在执行过程中，按需加载辅助脚本和资源。

这种机制确保了 Claude 既能拥有“广博”的技能树，又能保持“专注”的上下文环境。

### 3.4 特性对比：并发与状态

最后，一个容易被忽视的技术细节：

- **Function Calling (Tools)** 通常是**无状态**且**并发安全**的。你可以同时让 Claude 查询 10 个城市的天气，它会并行执行。
- **Agent Skills** 是**有状态**且**非并发安全**的。因为 Skill 本质上是修改了对话的上下文（Context），它改变了“当前的对话状态”。你不能在同一个对话线程里“并行”地运行两个 Skill，因为它们会争夺对上下文的控制权。

工程上，一段对话线程建议一次只激活一个 Skill；需要并行时，用多个会话或任务队列拆分。

---

## 4. 总结

Claude Agent Skills 代表了 AI Agent 开发的一种新范式：**通过自然语言编程**。

我们不再仅仅是编写 Python 函数供 LLM 调用，而是开始编写 **Markdown 文档** 来传授 LLM **流程性知识**。通过构建 `pdf-translator`，我们看到了这种范式的强大之处：它结合了代码的精确性（工具脚本）和自然语言的灵活性（SKILL.md），让构建复杂的智能助手变得前所未有的简单。

---

## 参考文献

[1] Anthropic. "Agent Skills - Claude Code Docs." _Claude Code Documentation_. 2026-01-02. [Online]. Available: https://code.claude.com/docs/en/skills

[2] Travisvn. "Awesome Claude Skills." _GitHub_. 2026-01-02. [Online]. Available: https://github.com/travisvn/awesome-claude-skills

[3] Lee Hanchung. "First Principles Deep Dive: Claude Skills." _Blog_. 2025-10-26. [Online]. Available: https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/
