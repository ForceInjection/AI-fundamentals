#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础记忆示例 - 展示 LangChain 记忆能力的"旧"与"新"两种写法

⚠️ 重要更新（LangChain 0.3+）
    ConversationBufferMemory / ConversationSummaryMemory /
    ConversationBufferWindowMemory / ConversationSummaryBufferMemory
    以及 ConversationChain 已被官方标记为 **deprecated**。
    官方推荐迁移到 **LangGraph 持久化**（MessagesState + checkpointer + thread_id）。

本文件保留四种经典记忆类型的 demo 作为"迁移参考"，并新增四个
基于 LangGraph 的等价实现，展示现代推荐写法。
"""

import sys
import uuid
import warnings
from typing import Any, Dict, List

# --- 经典（已弃用）API ---
# LangChain 0.3.x 里这些类位于 `langchain.memory` / `langchain.chains`；
# LangChain 1.x 起被迁移到独立的 `langchain_classic` 包。两种布局都兼容。
try:
    from langchain_classic.memory import (
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryBufferMemory,
        ConversationSummaryMemory,
    )
    from langchain_classic.chains import ConversationChain
    LEGACY_MEMORY_AVAILABLE = True
except ImportError:
    try:
        from langchain.memory import (
            ConversationBufferMemory,
            ConversationBufferWindowMemory,
            ConversationSummaryBufferMemory,
            ConversationSummaryMemory,
        )
        from langchain.chains import ConversationChain
        LEGACY_MEMORY_AVAILABLE = True
    except ImportError:
        ConversationBufferMemory = None
        ConversationBufferWindowMemory = None
        ConversationSummaryBufferMemory = None
        ConversationSummaryMemory = None
        ConversationChain = None
        LEGACY_MEMORY_AVAILABLE = False

# --- 现代 API ---
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

try:
    from .config import config
    from .llm_factory import get_llm
except ImportError:
    from config import config
    from llm_factory import get_llm


# 关闭 LangChain 对旧 Memory 类的 DeprecationWarning 噪音（我们已在文档里明确标注）
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.*")


# ============================================================================
# 扩展状态：在 MessagesState 基础上加一个 summary 字段
# ============================================================================
class SummaryState(MessagesState):
    """MessagesState + summary，便于演示"摘要记忆"模式。"""
    summary: str


class MemoryExamples:
    """LangChain 记忆能力演示（经典 API 与 LangGraph 现代 API 双版本）"""

    def __init__(self) -> None:
        try:
            self.llm = get_llm()
            print(f"✅ 成功初始化 LLM: {type(self.llm).__name__}")
        except Exception as e:
            print(f"❌ LLM 初始化失败: {e}")
            sys.exit(1)

    # ========================================================================
    # Part 1：经典（已弃用）API —— 保留作为迁移参考
    # ========================================================================

    def demo_conversation_buffer_memory(self):
        """[DEPRECATED] ConversationBufferMemory：保存完整对话历史。"""
        print("\n" + "=" * 60)
        print("📝 [DEPRECATED] ConversationBufferMemory 演示")
        print("   迁移目标：demo_langgraph_full_history()")
        print("=" * 60)

        memory = ConversationBufferMemory(return_messages=True, memory_key="history")
        conversation = ConversationChain(llm=self.llm, memory=memory, verbose=False)

        for i, user_input in enumerate(
            [
                "你好，我叫张三，是一名软件工程师",
                "我正在学习 LangChain 的记忆功能",
                "你还记得我的名字和职业吗？",
            ],
            1,
        ):
            print(f"\n👤 用户 {i}: {user_input}")
            print(f"🤖 助手 {i}: {conversation.predict(input=user_input)}")

        print("\n📋 记忆内容:")
        print(memory.buffer)
        return memory

    def demo_conversation_summary_memory(self):
        """[DEPRECATED] ConversationSummaryMemory：自动总结对话历史。"""
        print("\n" + "=" * 60)
        print("📝 [DEPRECATED] ConversationSummaryMemory 演示")
        print("   迁移目标：demo_langgraph_summary()")
        print("=" * 60)

        memory = ConversationSummaryMemory(
            llm=self.llm, return_messages=True, memory_key="history"
        )
        conversation = ConversationChain(llm=self.llm, memory=memory, verbose=False)

        for i, user_input in enumerate(
            [
                "你好，我是李四，今年 25 岁，住在北京",
                "我是一名数据科学家，专门研究机器学习",
                "我最近在做一个关于自然语言处理的项目",
                "这个项目使用了 BERT 和 GPT 模型",
                "你能总结一下我们刚才聊的内容吗？",
            ],
            1,
        ):
            print(f"\n👤 用户 {i}: {user_input}")
            print(f"🤖 助手 {i}: {conversation.predict(input=user_input)}")

        print("\n📋 记忆摘要:")
        print(memory.buffer)
        return memory

    def demo_conversation_buffer_window_memory(self):
        """[DEPRECATED] ConversationBufferWindowMemory：只保留最近 k 轮对话。"""
        print("\n" + "=" * 60)
        print("📝 [DEPRECATED] ConversationBufferWindowMemory 演示")
        print("   迁移目标：demo_langgraph_windowed_history()")
        print("=" * 60)

        memory = ConversationBufferWindowMemory(
            k=2, return_messages=True, memory_key="history"
        )
        conversation = ConversationChain(llm=self.llm, memory=memory, verbose=False)

        for i, user_input in enumerate(
            [
                "我叫王五，是一名教师",
                "我教数学，已经工作 5 年了",
                "我喜欢看书和旅游",
                "我最近去了日本旅游",
                "你还记得我的名字吗？",
            ],
            1,
        ):
            print(f"\n👤 用户 {i}: {user_input}")
            print(f"🤖 助手 {i}: {conversation.predict(input=user_input)}")
            print(f"📋 当前窗口 (k={memory.k}): {memory.buffer}")
        return memory

    def demo_conversation_summary_buffer_memory(self):
        """[DEPRECATED] ConversationSummaryBufferMemory：摘要 + 缓冲混合策略。"""
        print("\n" + "=" * 60)
        print("📝 [DEPRECATED] ConversationSummaryBufferMemory 演示")
        print("   迁移目标：demo_langgraph_summary_buffer()")
        print("=" * 60)

        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=100,
            return_messages=True,
            memory_key="history",
        )
        conversation = ConversationChain(llm=self.llm, memory=memory, verbose=False)

        for i, user_input in enumerate(
            [
                "你好，我是赵六，是一名产品经理",
                "我在一家互联网公司工作，负责移动应用产品",
                "我们公司主要做电商平台，用户量超过 1000 万",
                "我最近在负责一个新的推荐系统项目",
                "这个推荐系统使用了深度学习和协同过滤算法",
                "我们希望通过这个系统提高用户的购买转化率",
                "你能帮我分析一下推荐系统的优化方向吗？",
            ],
            1,
        ):
            print(f"\n👤 用户 {i}: {user_input}")
            print(f"🤖 助手 {i}: {conversation.predict(input=user_input)}")
            if getattr(memory, "moving_summary_buffer", ""):
                print(f"🧠 摘要: {memory.moving_summary_buffer}")
        return memory

    # ========================================================================
    # Part 2：LangGraph 现代推荐写法
    #   - MessagesState + add_messages（自动合并消息）
    #   - MemorySaver 作为 checkpointer
    #   - thread_id 实现会话隔离
    # ========================================================================

    def _run_langgraph_demo(self, graph, turns: List[str], thread_id: str) -> None:
        """统一的交互循环，避免重复样板。"""
        cfg = {"configurable": {"thread_id": thread_id}}
        for i, text in enumerate(turns, 1):
            print(f"\n👤 用户 {i}: {text}")
            result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=cfg)
            last_ai = next(
                (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
                None,
            )
            if last_ai:
                print(f"🤖 助手 {i}: {last_ai.content}")

    def demo_langgraph_full_history(self):
        """[MODERN] LangGraph 完整历史：ConversationBufferMemory 的替代。"""
        print("\n" + "=" * 60)
        print("✨ [MODERN] LangGraph 完整历史 (替代 ConversationBufferMemory)")
        print("=" * 60)

        llm = self.llm

        def chat_node(state: MessagesState) -> Dict[str, Any]:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        graph = (
            StateGraph(MessagesState)
            .add_node("chat", chat_node)
            .add_edge(START, "chat")
            .add_edge("chat", END)
            .compile(checkpointer=MemorySaver())
        )

        self._run_langgraph_demo(
            graph,
            [
                "你好，我叫张三，是一名软件工程师",
                "我正在学习 LangGraph 的记忆功能",
                "你还记得我的名字和职业吗？",
            ],
            thread_id=f"full-history-{uuid.uuid4().hex[:8]}",
        )

    def demo_langgraph_windowed_history(self, window: int = 4):
        """[MODERN] LangGraph 窗口：ConversationBufferWindowMemory 的替代。

        使用 ``trim_messages(strategy="last", token_counter=len, max_tokens=window)``
        仅保留最近 N 条消息作为 LLM 上下文；checkpointer 里仍然保存完整历史，
        只是不再全部喂给模型，从而复现"窗口"语义。
        """
        print("\n" + "=" * 60)
        print("✨ [MODERN] LangGraph 窗口记忆 (替代 ConversationBufferWindowMemory)")
        print(f"   window = {window} 条消息（含 system）")
        print("=" * 60)

        llm = self.llm

        def chat_node(state: MessagesState) -> Dict[str, Any]:
            trimmed = trim_messages(
                state["messages"],
                strategy="last",
                token_counter=len,  # 按"消息条数"计数，简单可控
                max_tokens=window,
                start_on="human",
                include_system=True,
                allow_partial=False,
            )
            response = llm.invoke(trimmed)
            return {"messages": [response]}

        graph = (
            StateGraph(MessagesState)
            .add_node("chat", chat_node)
            .add_edge(START, "chat")
            .add_edge("chat", END)
            .compile(checkpointer=MemorySaver())
        )

        self._run_langgraph_demo(
            graph,
            [
                "我叫王五，是一名教师",
                "我教数学，已经工作 5 年了",
                "我喜欢看书和旅游",
                "我最近去了日本旅游",
                "你还记得我的名字吗？",  # 此时因窗口裁剪，模型大概率已记不住
            ],
            thread_id=f"window-{uuid.uuid4().hex[:8]}",
        )

    def demo_langgraph_summary(self, summarize_after: int = 6):
        """[MODERN] LangGraph 摘要：ConversationSummaryMemory 的替代。

        ``summarize_node`` 在消息数超过阈值时，调用 LLM 生成摘要并通过
        ``RemoveMessage(id=...)`` 删除旧消息，只保留最新一轮 + 摘要。
        """
        print("\n" + "=" * 60)
        print("✨ [MODERN] LangGraph 摘要记忆 (替代 ConversationSummaryMemory)")
        print(f"   触发阈值 = {summarize_after} 条消息")
        print("=" * 60)

        llm = self.llm

        def chat_node(state: SummaryState) -> Dict[str, Any]:
            system_msgs: List[Any] = []
            if state.get("summary"):
                system_msgs.append(
                    SystemMessage(content=f"以下是此前对话的摘要：{state['summary']}")
                )
            response = llm.invoke(system_msgs + state["messages"])
            return {"messages": [response]}

        def summarize_node(state: SummaryState) -> Dict[str, Any]:
            if len(state["messages"]) <= summarize_after:
                return {}
            prev = state.get("summary", "")
            prompt = (
                (f"此前摘要：{prev}\n\n" if prev else "")
                + "请把下面这段对话压缩成一段简洁的中文摘要，保留关键事实与用户偏好：\n"
                + "\n".join(
                    f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                    for m in state["messages"]
                    if isinstance(m, (HumanMessage, AIMessage))
                )
            )
            summary = llm.invoke([HumanMessage(content=prompt)]).content
            # 只保留最新一轮（user + assistant）
            to_remove = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
            return {"summary": summary, "messages": to_remove}

        graph = (
            StateGraph(SummaryState)
            .add_node("chat", chat_node)
            .add_node("summarize", summarize_node)
            .add_edge(START, "chat")
            .add_edge("chat", "summarize")
            .add_edge("summarize", END)
            .compile(checkpointer=MemorySaver())
        )

        thread_id = f"summary-{uuid.uuid4().hex[:8]}"
        self._run_langgraph_demo(
            graph,
            [
                "你好，我是李四，今年 25 岁，住在北京",
                "我是一名数据科学家，专门研究机器学习",
                "我最近在做一个关于 NLP 的项目",
                "这个项目使用了 BERT 和 GPT 模型",
                "你能总结一下我们刚才聊的内容吗？",
            ],
            thread_id=thread_id,
        )

        final_state = graph.get_state({"configurable": {"thread_id": thread_id}}).values
        print(f"\n🧠 最终摘要: {final_state.get('summary', '(尚未生成)')}")
        print(f"📦 checkpoint 里保留的消息数: {len(final_state.get('messages', []))}")

    def demo_langgraph_summary_buffer(self, window: int = 6, summarize_after: int = 8):
        """[MODERN] LangGraph 摘要 + 窗口：ConversationSummaryBufferMemory 的替代。

        同时应用 ``trim_messages`` 控制单次 LLM 上下文长度，并在总量超阈值时
        调用 ``summarize_node`` 归档旧消息。这是生产环境最常用的组合。
        """
        print("\n" + "=" * 60)
        print("✨ [MODERN] LangGraph 摘要 + 窗口 (替代 ConversationSummaryBufferMemory)")
        print(f"   window = {window}, summarize_after = {summarize_after}")
        print("=" * 60)

        llm = self.llm

        def chat_node(state: SummaryState) -> Dict[str, Any]:
            trimmed = trim_messages(
                state["messages"],
                strategy="last",
                token_counter=len,
                max_tokens=window,
                start_on="human",
                include_system=True,
                allow_partial=False,
            )
            system_msgs: List[Any] = []
            if state.get("summary"):
                system_msgs.append(
                    SystemMessage(content=f"以下是此前对话的摘要：{state['summary']}")
                )
            response = llm.invoke(system_msgs + trimmed)
            return {"messages": [response]}

        def summarize_node(state: SummaryState) -> Dict[str, Any]:
            if len(state["messages"]) <= summarize_after:
                return {}
            prev = state.get("summary", "")
            prompt = (
                (f"此前摘要：{prev}\n\n" if prev else "")
                + "请在已有摘要的基础上，把新增对话整合进来，输出一段更新后的中文摘要：\n"
                + "\n".join(
                    f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                    for m in state["messages"]
                    if isinstance(m, (HumanMessage, AIMessage))
                )
            )
            summary = llm.invoke([HumanMessage(content=prompt)]).content
            to_remove = [RemoveMessage(id=m.id) for m in state["messages"][:-window]]
            return {"summary": summary, "messages": to_remove}

        graph = (
            StateGraph(SummaryState)
            .add_node("chat", chat_node)
            .add_node("summarize", summarize_node)
            .add_edge(START, "chat")
            .add_edge("chat", "summarize")
            .add_edge("summarize", END)
            .compile(checkpointer=MemorySaver())
        )

        thread_id = f"summary-buffer-{uuid.uuid4().hex[:8]}"
        self._run_langgraph_demo(
            graph,
            [
                "你好，我是赵六，是一名产品经理",
                "我在互联网公司负责移动应用产品",
                "我们公司主要做电商平台，用户量超过 1000 万",
                "我最近在负责一个推荐系统项目",
                "系统使用了深度学习和协同过滤算法",
                "我们希望提高用户的购买转化率",
                "你能帮我分析一下推荐系统的优化方向吗？",
            ],
            thread_id=thread_id,
        )

        final_state = graph.get_state({"configurable": {"thread_id": thread_id}}).values
        print(f"\n🧠 最终摘要: {final_state.get('summary', '(尚未生成)')}")
        print(f"📦 checkpoint 里保留的消息数: {len(final_state.get('messages', []))}")

    # ========================================================================
    # 对比表
    # ========================================================================

    def compare_memory_types(self) -> None:
        print("\n" + "=" * 72)
        print("📊 Legacy ConversationXxxMemory → LangGraph 现代方案 迁移对照")
        print("=" * 72)

        rows = [
            (
                "ConversationBufferMemory",
                "StateGraph(MessagesState) + MemorySaver",
                "短对话、调试：完整上下文最简单",
            ),
            (
                "ConversationBufferWindowMemory",
                "上面 + trim_messages(strategy='last', max_tokens=k)",
                "任务型多轮对话：只关注最近 N 条",
            ),
            (
                "ConversationSummaryMemory",
                "上面 + summarize_node + RemoveMessage(...)",
                "长对话、成本敏感：关键信息压缩",
            ),
            (
                "ConversationSummaryBufferMemory",
                "trim_messages + summarize_node（两者组合）",
                "生产环境默认选择：窗口控成本 + 摘要保记忆",
            ),
        ]
        print(f"{'Legacy (deprecated)':<35}{'Modern LangGraph 模式':<45}适用场景")
        print("-" * 72)
        for legacy, modern, scene in rows:
            print(f"{legacy:<35}{modern:<45}{scene}")

    # ========================================================================
    # 汇总运行
    # ========================================================================

    def run_all_demos(self) -> None:
        print("🚀 开始 LangChain 记忆功能演示")
        print(f"使用模型: {type(self.llm).__name__}")

        try:
            print("\n\n" + "#" * 72)
            print("# Part 1 / Legacy API —— 已弃用，仅作迁移参考")
            print("#" * 72)
            if not LEGACY_MEMORY_AVAILABLE:
                print(
                    "\n⚠️  当前环境未安装 `langchain-classic`（或者用的是仅限 LangGraph 的最小安装），"
                    "\n   整个迁移参考章节将被跳过。若需完整运行 Legacy 演示，请先 `pip install langchain-classic`。"
                )
            else:
                self.demo_conversation_buffer_memory()
                self.demo_conversation_summary_memory()
                self.demo_conversation_buffer_window_memory()
                try:
                    self.demo_conversation_summary_buffer_memory()
                except Exception as e:
                    # 某些模型不支持旧版 token 计数，跳过即可
                    print(f"\n⚠️  跳过 ConversationSummaryBufferMemory 演示: {e}")

            print("\n\n" + "#" * 72)
            print("# Part 2 / Modern LangGraph API —— 官方推荐")
            print("#" * 72)
            self.demo_langgraph_full_history()
            self.demo_langgraph_windowed_history()
            self.demo_langgraph_summary()
            self.demo_langgraph_summary_buffer()

            self.compare_memory_types()
            print("\n✅ 所有演示完成！")
        except Exception as e:
            print(f"❌ 演示过程中出现错误: {e}")
            import traceback

            traceback.print_exc()


def main() -> None:
    print("LangChain 记忆功能基础演示")
    print("=" * 40)

    if not config.validate_config():
        print("❌ 配置验证失败！请检查配置文件或环境变量")
        print("\n请参考 config.example.py 文件进行配置")
        return

    MemoryExamples().run_all_demos()


if __name__ == "__main__":
    main()
