# LangChain 记忆功能演示（LangChain 0.3+）

大模型本身无状态，每一轮请求都要把「历史会话 + 用户画像 + 业务知识」重新组装到 prompt 里。本项目以 **LangGraph 短期记忆（MessagesState + checkpointer + thread_id）** 为主线，同时保留一套经典 `ConversationXxxMemory + ConversationChain` 的 demo 作为迁移参考，覆盖从完整缓冲到摘要+窗口的几种常见记忆策略，帮助开发者直观理解不同模式的取舍与迁移路径。

## 1. LangChain 0.3+ 迁移说明

从 **LangChain 0.3** 起，`ConversationBufferMemory` / `ConversationSummaryMemory` / `ConversationBufferWindowMemory` / `ConversationSummaryBufferMemory` 以及 `ConversationChain` 均已被标记为 **deprecated**；官方推荐迁移到 **LangGraph 持久化**。本项目的代码已完成迁移，每个经典 API 都能在 LangGraph 中找到等价实现：

| Legacy（已弃用）                  | Modern LangGraph 方案                                 | 适用场景                          |
| --------------------------------- | ----------------------------------------------------- | --------------------------------- |
| `ConversationBufferMemory`        | `StateGraph(MessagesState)` + `MemorySaver`           | 短对话、调试                      |
| `ConversationBufferWindowMemory`  | 上面 + `trim_messages(strategy="last", max_tokens=k)` | 任务型多轮，只关注最近 N 条       |
| `ConversationSummaryMemory`       | 上面 + `summarize_node` + `RemoveMessage(...)`        | 长对话、成本敏感                  |
| `ConversationSummaryBufferMemory` | `trim_messages` + `summarize_node`（两者组合）        | 生产默认：窗口控成本 + 摘要保记忆 |

> 会话隔离不再需要自建 `memories: Dict[session_id, Memory]`；把 `session_id` 直接作为 LangGraph 的 `thread_id` 即可。
>
> **LangChain 1.x 安装说明**：从 LangChain **1.0** 起，上表 Legacy 列的类已从主包 `langchain` 迁出，转移至独立的 `langchain-classic` 包。本仓库的 `basic_memory_examples.py` 已将两种布局都兼容：优先导 `langchain_classic.memory`，失败后再回退到 `langchain.memory`。如果 pip 解析出的是 langchain 1.x，需额外安装 `pip install langchain-classic` 才能运行 Legacy 迁移参考 demo；仅运行 LangGraph demo 的话无需。

## 2. 项目结构

```bash
code/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python 依赖（LangChain 0.3+ / LangGraph 0.2+）
├── config.example.py            # 配置文件模板
├── config.py                    # 实际配置文件（需要创建）
├── llm_factory.py               # LLM 工厂类
├── basic_memory_examples.py     # Legacy 四种记忆 + 四种 LangGraph 等价实现
├── smart_customer_service.py    # 智能客服机器人（LangGraph 短期记忆）
├── langgraph_memory_example.py  # LangGraph MessagesState + checkpointer
├── main.py                      # 主运行脚本
└── sessions/                    # 会话元数据 JSON / SqliteSaver 数据库
```

## 3. 快速开始

### 3.1 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate   # Linux / macOS
# 或
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

关键依赖版本：

- `langchain>=0.3.0`，`langchain-core>=0.3.0`，`langchain-community>=0.3.0`
- `langchain-openai>=0.2.0`，`openai>=1.40.0`
- `langgraph>=0.2.50`，`langgraph-checkpoint-sqlite>=2.0.0`（启用 `--persistent` 时需要）
- `langchain-classic>=1.0.0`（可选：当 pip 解析出 langchain 1.x 时，Legacy 迁移参考 demo 需要此包）

### 3.2 配置 LLM

#### 3.2.1 方法一：使用配置文件（推荐）

```bash
cp config.example.py config.py
vim config.py   # 或使用其他编辑器
```

在 `config.py` 中配置：

```python
# OpenAI
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"

# DeepSeek
DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_MODEL = "deepseek-chat"

# MiniMax
MINIMAX_API_KEY = "your-minimax-api-key"
MINIMAX_MODEL = "MiniMax-M2.7"

# 本地模型（Ollama / vLLM / LocalAI 等）
LOCAL_BASE_URL = "http://localhost:11434/v1"
LOCAL_MODEL = "qwen2.5"
```

#### 3.2.2 方法二：使用环境变量

```bash
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export MINIMAX_API_KEY="your-minimax-api-key"
export LOCAL_BASE_URL="http://localhost:11434/v1"
```

### 3.3 检查配置

```bash
python main.py --config-check
```

### 3.4 运行演示

```bash
# 基础记忆演示（Legacy + Modern 对照）
python main.py --demo basic

# 智能客服演示（LangGraph 短期记忆）
python main.py --demo customer

# LangGraph 记忆演示（MessagesState + checkpointer）
python main.py --demo langgraph

# 运行全部
python main.py --demo all

# 交互式聊天
python main.py --interactive

# 启用 SqliteSaver（跨进程短期记忆；可验证重启后仍能续上对话）
python main.py --demo customer  --persistent
python main.py --demo langgraph --persistent
python main.py --interactive    --persistent
```

## 4. 功能说明

### 4.1 基础记忆类型演示（`basic_memory_examples.py`）

文件分为两部分：

- **Part 1 / Legacy API —— 已弃用，仅作迁移参考**
  - `demo_conversation_buffer_memory` — `ConversationBufferMemory`
  - `demo_conversation_summary_memory` — `ConversationSummaryMemory`
  - `demo_conversation_buffer_window_memory` — `ConversationBufferWindowMemory`
  - `demo_conversation_summary_buffer_memory` — `ConversationSummaryBufferMemory`
- **Part 2 / Modern LangGraph API —— 官方推荐**
  - `demo_langgraph_full_history` — `MessagesState + MemorySaver`
  - `demo_langgraph_windowed_history` — 上面 + `trim_messages(strategy="last")`
  - `demo_langgraph_summary` — 上面 + `summarize_node` + `RemoveMessage`
  - `demo_langgraph_summary_buffer` — `trim_messages` + `summarize_node`

```bash
python main.py --demo basic
```

### 4.2 智能客服机器人（`smart_customer_service.py`）

已完全迁移到 LangGraph：

- `SessionManager` 不再持有自己的 `memory` 字典；对话正文全部写进 checkpointer，键为 `thread_id = session_id`；
- `build_graph(memory_type)` 按 `buffer / window / summary_buffer` 三种策略编译三张图，公用一个 `MemorySaver`（默认）或 `SqliteSaver`（`--persistent`）；
- `save_session` 只负责把 **会话元数据 + 性能指标** 写到 `sessions/*.json`，对话正文由 checkpointer 持久化；
- 业务层 `CustomerServiceBot` 接口兼容旧版，`main.py` 不需要改动。

```bash
python main.py --demo customer
python main.py --demo customer --persistent   # 启用 SqliteSaver
```

### 4.3 LangGraph 记忆管理（`langgraph_memory_example.py`）

- `ConversationState(MessagesState)` 扩展 `summary / user_id / session_id` 字段；
- `chat_node` 直接把 `state["messages"]`（`HumanMessage` / `AIMessage` / ...）喂给 LLM；
- `summarize_node` 在消息数超阈值时生成摘要、用 `RemoveMessage(id=...)` 精准删除旧消息；
- `--persistent` 开启 `SqliteSaver`，进程重启后 demo 会继续此前的对话。

```bash
python main.py --demo langgraph
python main.py --demo langgraph --persistent
```

### 4.4 交互式聊天

```bash
python main.py --interactive
python main.py --interactive --persistent   # 聊天历史写入 sessions/checkpoints.sqlite
```

支持的命令：

- `quit` / `exit`：退出聊天
- `clear`：重新开始对话（新建一个 thread_id）
- `summary`：查看当前会话的统计信息

## 5. 配置选项

### 5.1 LLM 配置

| 参数               | 说明                | 默认值                          |
| ------------------ | ------------------- | ------------------------------- |
| `OPENAI_API_KEY`   | OpenAI API 密钥     | 无                              |
| `OPENAI_BASE_URL`  | OpenAI API 基础 URL | `https://api.openai.com/v1`     |
| `OPENAI_MODEL`     | OpenAI 模型名称     | `gpt-4o-mini` / `gpt-3.5-turbo` |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥   | 无                              |
| `DEEPSEEK_MODEL`   | DeepSeek 模型名称   | `deepseek-chat`                 |
| `MINIMAX_API_KEY`  | MiniMax API 密钥    | 无                              |
| `MINIMAX_MODEL`    | MiniMax 模型名称    | `MiniMax-M2.7`                  |
| `LOCAL_BASE_URL`   | 本地模型 API URL    | `http://localhost:11434/v1`     |
| `LOCAL_MODEL`      | 本地模型名称        | `qwen2.5`                       |

### 5.2 记忆配置

| 参数                 | 说明                                  | 默认值 |
| -------------------- | ------------------------------------- | ------ |
| `MAX_HISTORY_LENGTH` | window 策略保留的最大消息数           | 20     |
| `SUMMARY_THRESHOLD`  | 触发摘要节点的消息阈值（×2 后使用）   | 10     |
| `MAX_TOKEN_LIMIT`    | 预留给 token-based 裁剪的上限（可选） | 2000   |

### 5.3 性能配置

| 参数              | 说明               | 默认值 |
| ----------------- | ------------------ | ------ |
| `TEMPERATURE`     | 模型温度           | 0.7    |
| `MAX_TOKENS`      | 最大生成 token 数  | 2048   |
| `REQUEST_TIMEOUT` | 请求超时时间（秒） | 60     |

## 6. 支持的 LLM 提供商

### 6.1 OpenAI

```python
OPENAI_API_KEY = "sk-..."
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
```

### 6.2 DeepSeek

```python
DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_MODEL = "deepseek-chat"
```

### 6.3 MiniMax

```python
MINIMAX_API_KEY = "your-minimax-api-key"
MINIMAX_MODEL = "MiniMax-M2.7"
```

MiniMax 提供 OpenAI 兼容接口，支持 `MiniMax-M2.7`（1M 上下文）/ `MiniMax-M2.7-highspeed`。

### 6.4 本地模型（Ollama）

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5
ollama serve
```

```python
LOCAL_BASE_URL = "http://localhost:11434/v1"
LOCAL_MODEL = "qwen2.5"
```

### 6.5 其他 OpenAI 兼容服务

```python
# 例如：vLLM, FastChat, LocalAI 等
OPENAI_BASE_URL = "http://your-server:8000/v1"
OPENAI_MODEL = "your-model-name"
```

## 7. 故障排除

### 7.1 常见问题

1. **导入错误**：`pip install -r requirements.txt`，注意 `langchain>=0.3.0` 与 `langgraph>=0.2.50`。
2. **配置验证失败**：`python main.py --config-check`。
3. **LLM 连接失败**：检查 API Key、网络以及（本地模型的）服务是否启动。
4. **启用 `--persistent` 后报 `No module named 'langgraph.checkpoint.sqlite'`**：执行 `pip install langgraph-checkpoint-sqlite`。
5. **`sessions/` 下出现 `checkpoints.sqlite` / `langgraph_memory.sqlite`**：这是 LangGraph SqliteSaver 的数据库文件，删除即可重置持久化记忆。

### 7.2 调试模式

在代码中启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 8. 参考资源

- [LangChain Memory Migration Guide](https://python.langchain.com/docs/versions/migrating_memory/)
- [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph MessagesState / add_messages](https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate)
- [trim_messages](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.trim_messages.html)
- [How to add memory to chatbots](https://python.langchain.com/docs/how_to/chatbots_memory/)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [Ollama 文档](https://ollama.ai/)
