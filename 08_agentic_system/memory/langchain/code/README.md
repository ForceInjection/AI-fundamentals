# LangChain 记忆功能演示

这个项目演示了如何使用 LangChain 实现智能对话机器人的记忆功能，包括基础记忆类型、智能客服应用和现代 LangGraph 记忆管理。

## 1. 项目结构

```bash
code/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python 依赖包
├── config.example.py           # 配置文件模板
├── config.py                   # 实际配置文件（需要创建）
├── llm_factory.py              # LLM 工厂类
├── basic_memory_examples.py     # 基础记忆类型演示
├── smart_customer_service.py    # 智能客服机器人
├── langgraph_memory_example.py  # LangGraph 记忆管理
└── main.py                     # 主运行脚本˝
```

---

## 2. 快速开始

### 2.1 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2.2 配置 LLM

#### 2.2.1 方法一：使用配置文件（推荐）

```bash
# 复制配置模板
cp config.example.py config.py

# 编辑配置文件
vim config.py  # 或使用其他编辑器
```

在 `config.py` 中配置你的 LLM 设置：

```python
# OpenAI 配置
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"  # 或其他兼容的 API
OPENAI_MODEL = "gpt-3.5-turbo"

# 或本地模型配置
LOCAL_BASE_URL = "http://localhost:11434/v1"  # Ollama
LOCAL_MODEL = "llama2"
```

#### 2.2.2 方法二：使用环境变量

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-3.5-turbo"

# 或本地模型
export LOCAL_BASE_URL="http://localhost:11434/v1"
export LOCAL_MODEL="llama2"
```

### 2.3 检查配置

```bash
python main.py --config-check
```

### 2.4 运行演示

```bash
# 基础记忆演示
python main.py --demo basic

# 智能客服演示
python main.py --demo customer

# LangGraph 记忆演示
python main.py --demo langgraph

# 运行所有演示
python main.py --demo all

# 交互式聊天
python main.py --interactive
```

## 3 功能说明

### 3.1 基础记忆类型演示 (`basic_memory_examples.py`)

演示四种核心记忆类型：

- **ConversationBufferMemory**: 保存完整对话历史
- **ConversationSummaryMemory**: 自动总结对话内容
- **ConversationBufferWindowMemory**: 保持固定窗口大小的对话历史
- **ConversationSummaryBufferMemory**: 结合缓冲和总结的混合策略

```bash
python main.py --demo basic
```

### 3.2 智能客服机器人 (`smart_customer_service.py`)

功能特性：

- 多用户会话管理
- 智能记忆类型选择
- 性能监控和统计
- 会话持久化存储
- 错误处理和恢复

```bash
python main.py --demo customer
```

### 3.3 LangGraph 记忆管理 (`langgraph_memory_example.py`)

现代记忆管理方案：

- 状态图管理
- 持久化记忆存储
- 自动记忆总结
- 跨会话记忆保持

```bash
python main.py --demo langgraph
```

### 3.4 交互式聊天

提供实时聊天界面：

```bash
python main.py --interactive
```

支持的命令：

- `quit` / `exit`: 退出聊天
- `clear`: 清除对话历史
- `summary`: 查看对话摘要

## 4. 配置选项

### 4.1 LLM 配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | None |
| `OPENAI_BASE_URL` | OpenAI API 基础URL | "<https://api.openai.com/v1>" |
| `OPENAI_MODEL` | OpenAI 模型名称 | "gpt-3.5-turbo" |
| `LOCAL_BASE_URL` | 本地模型 API URL | "<http://localhost:11434/v1>" |
| `LOCAL_MODEL` | 本地模型名称 | "llama2" |

### 4.2 记忆配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MAX_HISTORY_LENGTH` | 最大历史记录长度 | 20 |
| `SUMMARY_THRESHOLD` | 触发总结的消息数量 | 10 |
| `WINDOW_SIZE` | 窗口记忆大小 | 5 |
| `MAX_TOKEN_LIMIT` | 最大token限制 | 2000 |

### 4.3 性能配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `TEMPERATURE` | 模型温度 | 0.7 |
| `MAX_TOKENS` | 最大生成token数 | 1000 |
| `REQUEST_TIMEOUT` | 请求超时时间（秒） | 30 |

## 5. 支持的 LLM 提供商

### 5.1 OpenAI

```python
OPENAI_API_KEY = "sk-..."
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-3.5-turbo"  # 或 gpt-4
```

### 5.2 本地模型 (Ollama)

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载模型
ollama pull llama2
ollama pull qwen:7b

# 启动服务
ollama serve
```

```python
LOCAL_BASE_URL = "http://localhost:11434/v1"
LOCAL_MODEL = "llama2"  # 或 qwen:7b
```

### 5.3 其他兼容 OpenAI API 的服务

```python
# 例如：vLLM, FastChat, LocalAI 等
OPENAI_BASE_URL = "http://your-server:8000/v1"
OPENAI_MODEL = "your-model-name"
```

## 6. 故障排除

### 6.1 常见问题

1. **导入错误**

   ```bash
   pip install -r requirements.txt
   ```

2. **配置验证失败**

   ```bash
   python main.py --config-check
   ```

3. **LLM 连接失败**
   - 检查 API 密钥是否正确
   - 检查网络连接
   - 检查服务是否运行（本地模型）

4. **记忆存储问题**
   - 检查文件权限
   - 确保有足够的磁盘空间

### 6.2 调试模式

在代码中启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 7. 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [Ollama 文档](https://ollama.ai/)
- [Conversational Memory for LLMs with Langchain](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)
- [How to add memory to chatbots](https://python.langchain.com/docs/how_to/chatbots_memory/)

---
