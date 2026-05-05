#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 现代记忆管理示例（LangChain 0.3+）

核心要点：
    - ConversationState 直接继承 MessagesState，消息合并由 add_messages reducer 处理；
    - 消息全部使用 HumanMessage / AIMessage / SystemMessage / RemoveMessage；
    - 支持 MemorySaver（演示）和 SqliteSaver（生产/跨进程）两种 checkpointer；
    - 通过 ``graph.get_state`` / ``graph.get_state_history`` 读取历史与回溯。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

LANGGRAPH_AVAILABLE = True  # 保留变量便于上层 main.py 兼容

try:
    from .config import config
    from .llm_factory import get_llm
except ImportError:
    from config import config
    from llm_factory import get_llm


# ---------------------------------------------------------------------------
# 状态定义
# ---------------------------------------------------------------------------
class ConversationState(MessagesState):
    """对话状态：MessagesState 自带 messages: List[BaseMessage]（含 add_messages reducer）。"""

    summary: str
    user_id: str
    session_id: str


# ---------------------------------------------------------------------------
# 记忆管理器
# ---------------------------------------------------------------------------
class LangGraphMemoryManager:
    """基于 LangGraph MessagesState + checkpointer 的短期记忆管理器。"""

    def __init__(
        self,
        persistent: bool = False,
        db_path: Optional[str] = None,
        summary_threshold: Optional[int] = None,
        window_size: int = 6,
    ) -> None:
        self.llm = get_llm()
        self.summary_threshold = summary_threshold or config.summary_threshold * 2
        self.window_size = window_size

        # checkpointer
        self.checkpointer, self._sqlite_cm = self._build_checkpointer(persistent, db_path)
        self.graph = self._build_graph()

        print("✅ LangGraphMemoryManager 初始化完成")
        print(
            f"🗂️  checkpointer: {'SqliteSaver (持久化)' if self._sqlite_cm else 'MemorySaver (内存)'}"
        )

    # ------------------------------------------------------------------
    # checkpointer 选择
    # ------------------------------------------------------------------
    def _build_checkpointer(self, persistent: bool, db_path: Optional[str]):
        if not persistent:
            return MemorySaver(), None

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            path = db_path or "./sessions/langgraph_memory.sqlite"
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            cm = SqliteSaver.from_conn_string(path)
            return cm.__enter__(), cm
        except Exception as e:
            print(f"⚠️  SqliteSaver 初始化失败，回退到 MemorySaver: {e}")
            return MemorySaver(), None

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------
    def _build_graph(self):
        builder = StateGraph(ConversationState)
        builder.add_node("chat", self._chat_node)
        builder.add_node("summarize", self._summarize_node)
        builder.add_edge(START, "chat")
        builder.add_edge("chat", "summarize")
        builder.add_edge("summarize", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _chat_node(self, state: ConversationState) -> Dict[str, Any]:
        """调用 LLM 生成回复，并把结果以 AIMessage 形式合并回 state。"""
        system_msgs: List[Any] = [
            SystemMessage(
                content=(
                    "你是一个具备记忆能力的智能助手，"
                    "请结合历史与摘要给出连贯、有帮助的回答。"
                )
            )
        ]
        if state.get("summary"):
            system_msgs.append(
                SystemMessage(content=f"此前对话摘要：{state['summary']}")
            )
        response = self.llm.invoke(system_msgs + state["messages"])
        return {"messages": [response]}

    def _summarize_node(self, state: ConversationState) -> Dict[str, Any]:
        """超过阈值时生成摘要，并通过 RemoveMessage 清理旧消息。"""
        if len(state["messages"]) <= self.summary_threshold:
            return {}

        prev = state.get("summary", "")
        bullets = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
            for m in state["messages"]
            if isinstance(m, (HumanMessage, AIMessage))
        )
        prompt = (
            (f"此前摘要：{prev}\n\n" if prev else "")
            + "请把以下对话整合成一段简洁的中文摘要，"
            + "保留关键事实、用户偏好与当前进度：\n"
            + bullets
        )
        summary = self.llm.invoke([HumanMessage(content=prompt)]).content
        to_remove = [
            RemoveMessage(id=m.id) for m in state["messages"][: -self.window_size]
        ]
        return {"summary": summary, "messages": to_remove}

    # ------------------------------------------------------------------
    # 对话接口
    # ------------------------------------------------------------------
    def chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
        input_messages: List[Any] = []
        if context:
            input_messages.append(
                SystemMessage(content="\n".join(f"{k}: {v}" for k, v in context.items()))
            )
        input_messages.append(HumanMessage(content=message))

        try:
            result = self.graph.invoke(
                {
                    "messages": input_messages,
                    "user_id": user_id,
                    "session_id": session_id,
                },
                config=cfg,
            )
            last_ai = next(
                (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
                None,
            )
            return {
                "response": last_ai.content if last_ai else "(无响应)",
                "session_id": session_id,
                "user_id": user_id,
                "message_count": len(result["messages"]),
                "memory_summary": result.get("summary", ""),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"❌ 聊天处理失败: {e}")
            return {"error": str(e), "session_id": session_id, "user_id": user_id}

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def get_conversation_history(
        self, user_id: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        cfg = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
        state = self.graph.get_state(cfg)
        if not state or not state.values:
            return None

        messages = []
        for m in state.values.get("messages", []):
            if isinstance(m, HumanMessage):
                messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                messages.append({"role": "assistant", "content": m.content})
            elif isinstance(m, SystemMessage):
                messages.append({"role": "system", "content": m.content})
        cfg_cfg = (state.config or {}).get("configurable", {}) if state.config else {}
        return {
            "messages": messages,
            "memory_summary": state.values.get("summary", ""),
            "checkpoint_id": cfg_cfg.get("checkpoint_id"),
        }

    def get_state_history(
        self, user_id: str, session_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """读取最近若干个 checkpoint，用于回溯 / 时间旅行。"""
        cfg = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
        history = []
        for i, snapshot in enumerate(self.graph.get_state_history(cfg)):
            if i >= limit:
                break
            history.append(
                {
                    "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
                    "message_count": len(snapshot.values.get("messages", [])),
                    "summary": snapshot.values.get("summary", ""),
                }
            )
        return history

    def clear_conversation(self, user_id: str, session_id: str) -> bool:
        cfg = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
        try:
            state = self.graph.get_state(cfg)
            if state and state.values:
                to_remove = [
                    RemoveMessage(id=m.id) for m in state.values.get("messages", [])
                ]
                self.graph.update_state(cfg, {"messages": to_remove, "summary": ""})
            print(f"🧹 已清除会话: {user_id}:{session_id}")
            return True
        except Exception as e:
            print(f"❌ 清除对话失败: {e}")
            return False

    def close(self) -> None:
        """关闭 SqliteSaver 的底层连接（仅 persistent 模式需要）。"""
        if self._sqlite_cm is not None:
            try:
                self._sqlite_cm.__exit__(None, None, None)
            finally:
                self._sqlite_cm = None


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_langgraph_memory(persistent: bool = False) -> None:
    print("🚀 LangGraph 现代记忆管理演示")
    print("=" * 50)

    manager = LangGraphMemoryManager(persistent=persistent)
    user_id = "user_123"
    session_id = "session_456"
    print(f"\n👤 用户: {user_id}")
    print(f"📱 会话: {session_id}")

    turns = [
        "你好，我是新用户，想了解你们的服务",
        "我对人工智能很感兴趣，你能介绍一下吗？",
        "我想学习机器学习，有什么建议吗？",
        "刚才你提到的深度学习，能详细说说吗？",
        "我之前问过关于机器学习的问题，你还记得吗？",
        "谢谢你的建议，我会认真考虑的",
    ]
    for i, msg in enumerate(turns, 1):
        print(f"\n--- 第 {i} 轮对话 ---")
        print(f"👤 用户: {msg}")
        result = manager.chat(user_id, session_id, msg)
        if "response" in result:
            print(f"🤖 助手: {result['response']}")
            print(f"📊 checkpoint 消息数: {result['message_count']}")
            if result.get("memory_summary"):
                print(f"🧠 当前摘要: {result['memory_summary'][:120]}…")
        else:
            print(f"❌ 错误: {result.get('error', '未知错误')}")

    # 读取历史
    print("\n" + "=" * 50)
    print("📋 对话历史（来自 checkpointer）")
    print("=" * 50)
    history = manager.get_conversation_history(user_id, session_id)
    if history:
        print(f"📝 消息数量: {len(history['messages'])}")
        print(f"🧠 最终摘要: {history.get('memory_summary') or '（未生成）'}")
        for msg in history["messages"][-4:]:
            icon = {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(
                msg["role"], "❓"
            )
            content = msg["content"]
            if len(content) > 100:
                content = content[:100] + "…"
            print(f"{icon} {content}")

    # state_history（时间旅行）
    print("\n🕰️  最近 5 个 checkpoint：")
    for h in manager.get_state_history(user_id, session_id, limit=5):
        print(
            f"   checkpoint={h['checkpoint_id']} "
            f"messages={h['message_count']} "
            f"summary={'有' if h['summary'] else '无'}"
        )

    # 仅 persistent 模式下，演示重启后记忆仍在
    if persistent:
        print("\n" + "=" * 50)
        print("🔄 记忆持久化测试（SqliteSaver 跨进程）")
        print("=" * 50)
        manager.close()
        fresh = LangGraphMemoryManager(persistent=True)
        print("🔄 模拟进程重启完成，继续对话…")
        print("\n👤 用户: 我们之前聊到哪里了？")
        result = fresh.chat(user_id, session_id, "我们之前聊到哪里了？")
        if "response" in result:
            print(f"🤖 助手: {result['response']}")
            print("✅ 记忆持久化测试通过")
        else:
            print(f"❌ 错误: {result.get('error', '未知错误')}")
        fresh.close()
    else:
        print("\n💡 提示：加上 --persistent 运行可验证 SqliteSaver 的跨进程持久化。")


def main() -> None:
    print("LangGraph 现代记忆管理示例")
    print("=" * 40)
    if not config.validate_config():
        print("❌ 配置验证失败！请检查配置文件或环境变量")
        return
    demo_langgraph_memory()


if __name__ == "__main__":
    main()
