#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能客服机器人 —— LangGraph 短期记忆版本（LangChain 0.3+）

本实现将原先基于 ConversationChain + ConversationBufferMemory 的写法
全部迁移到 LangGraph：
    - StateGraph(MessagesState) 定义对话状态
    - MemorySaver / SqliteSaver 作为 checkpointer 实现短期记忆
    - thread_id = session_id 实现多用户会话隔离
    - trim_messages 实现"窗口"策略
    - summarize_node + RemoveMessage 实现"摘要+窗口"策略

会话元数据（用户信息、统计指标）仍以 JSON 形式落盘；对话内容本身由
checkpointer 负责持久化，开启 ``persistent=True`` 后可跨进程恢复。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

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


MemoryType = Literal["buffer", "window", "summary_buffer", "auto"]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class SessionInfo:
    """会话元数据（不包含对话正文）。"""

    session_id: str
    user_id: str
    created_at: datetime
    last_active: datetime
    message_count: int
    memory_type: MemoryType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """单次 chat 调用的性能指标。"""

    response_time: float
    token_usage: int
    memory_size: int
    timestamp: datetime


class CustomerServiceState(MessagesState):
    """客服对话状态：MessagesState + summary。"""

    summary: str


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------
class SessionManager:
    """基于 LangGraph 的多用户会话管理器。

    线程安全性：当前实现 **不是线程安全的**。`sessions` / `performance_metrics` / `_graphs`
    等可变字典没有加锁保护，适合 CLI 、单进程演示和测试。如需在多线程 Web 框架
    （FastAPI workers、Flask 多线程等）中复用，请在外层自行加 `threading.Lock`，
    或改为每个请求/会话单独实例化。
    """

    def __init__(
        self,
        storage_dir: str = "./sessions",
        persistent: bool = False,
        sqlite_path: Optional[str] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # 元数据（正文由 checkpointer 管理）
        self.sessions: Dict[str, SessionInfo] = {}
        self.performance_metrics: Dict[str, List[PerformanceMetrics]] = {}

        # 持久化 checkpointer 的 context manager 引用（SqliteSaver 时非空）
        self._sqlite_cm: Optional[Any] = None

        try:
            self.llm = get_llm()
            print(f"✅ 会话管理器初始化成功，使用模型: {type(self.llm).__name__}")
        except Exception as e:
            print(f"❌ LLM 初始化失败: {e}")
            raise

        # 按需创建 checkpointer（短期记忆）
        self.checkpointer = self._build_checkpointer(persistent, sqlite_path)

        # 三种记忆策略各编译一张图；所有图共享同一个 checkpointer，
        # 因此相同的 thread_id 可被多次读写，会话内容不会丢失。
        self._graphs: Dict[MemoryType, Any] = {
            "buffer": self._build_graph("buffer"),
            "window": self._build_graph("window"),
            "summary_buffer": self._build_graph("summary_buffer"),
        }

    # ------------------------------------------------------------------
    # checkpointer
    # ------------------------------------------------------------------
    def _build_checkpointer(self, persistent: bool, sqlite_path: Optional[str]):
        if not persistent:
            print("🗂️  使用 MemorySaver（进程内 checkpointer）")
            return MemorySaver()

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = sqlite_path or str(self.storage_dir / "checkpoints.sqlite")
            print(f"🗂️  使用 SqliteSaver（持久化）: {db_path}")
            # SqliteSaver.from_conn_string 返回 context manager；
            # 这里交给调用方长期持有，显式 __enter__ 保证 connection 存活。
            cm = SqliteSaver.from_conn_string(db_path)
            self._sqlite_cm = cm
            return cm.__enter__()
        except Exception as e:
            print(f"⚠️  SqliteSaver 初始化失败，回退到 MemorySaver: {e}")
            return MemorySaver()

    def close(self) -> None:
        """释放 SqliteSaver 的 context manager（MemorySaver 时为空操作）。

        建议在长生命周期的 Web/服务场景中，在进程或 SessionManager 实例
        退出前显式调用，避免 SQLite 连接泄漏。
        """
        if self._sqlite_cm is not None:
            try:
                self._sqlite_cm.__exit__(None, None, None)
            finally:
                self._sqlite_cm = None

    # ------------------------------------------------------------------
    # 图构建（三种记忆策略）
    # ------------------------------------------------------------------
    def _build_graph(self, memory_type: MemoryType):
        llm = self.llm
        window_size = max(4, config.max_history_length)
        summarize_after = max(window_size + 2, config.summary_threshold * 2)

        def _select_context(state: CustomerServiceState) -> List[Any]:
            """按记忆策略挑选喂给 LLM 的消息列表。"""
            messages = state["messages"]
            if memory_type == "window":
                messages = trim_messages(
                    messages,
                    strategy="last",
                    token_counter=len,
                    max_tokens=window_size,
                    start_on="human",
                    include_system=True,
                    allow_partial=False,
                )
            elif memory_type == "summary_buffer":
                messages = trim_messages(
                    messages,
                    strategy="last",
                    token_counter=len,
                    max_tokens=window_size,
                    start_on="human",
                    include_system=True,
                    allow_partial=False,
                )

            system_msgs: List[Any] = []
            if state.get("summary"):
                system_msgs.append(
                    SystemMessage(content=f"此前对话摘要：{state['summary']}")
                )
            return system_msgs + list(messages)

        def chat_node(state: CustomerServiceState) -> Dict[str, Any]:
            response = llm.invoke(_select_context(state))
            return {"messages": [response]}

        def summarize_node(state: CustomerServiceState) -> Dict[str, Any]:
            if len(state["messages"]) <= summarize_after:
                return {}
            prev = state.get("summary", "")
            prompt = (
                (f"此前摘要：{prev}\n\n" if prev else "")
                + "请把以下对话整合成一段简洁中文摘要，重点保留用户诉求、订单/商品信息与上一轮进展：\n"
                + "\n".join(
                    f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                    for m in state["messages"]
                    if isinstance(m, (HumanMessage, AIMessage))
                )
            )
            summary = llm.invoke([HumanMessage(content=prompt)]).content
            to_remove = [RemoveMessage(id=m.id) for m in state["messages"][:-window_size]]
            return {"summary": summary, "messages": to_remove}

        builder = StateGraph(CustomerServiceState)
        builder.add_node("chat", chat_node)
        builder.add_edge(START, "chat")

        if memory_type == "summary_buffer":
            builder.add_node("summarize", summarize_node)
            builder.add_edge("chat", "summarize")
            builder.add_edge("summarize", END)
        else:
            builder.add_edge("chat", END)

        return builder.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # 会话生命周期
    # ------------------------------------------------------------------
    def create_session(
        self,
        user_id: str,
        memory_type: MemoryType = "auto",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now()

        if memory_type == "auto":
            memory_type = self._auto_select_memory_type(user_id)

        self.sessions[session_id] = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_active=now,
            message_count=0,
            memory_type=memory_type,
            metadata=metadata or {},
        )
        self.performance_metrics[session_id] = []

        print(
            f"📝 创建会话: {session_id[:8]}… "
            f"(用户: {user_id}, 记忆类型: {memory_type})"
        )
        return session_id

    def _auto_select_memory_type(self, user_id: str) -> MemoryType:
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id]
        if not user_sessions:
            return "buffer"
        avg_messages = sum(s.message_count for s in user_sessions) / len(user_sessions)
        if avg_messages < 10:
            return "buffer"
        elif avg_messages < 30:
            return "window"
        return "summary_buffer"

    # ------------------------------------------------------------------
    # 对话接口
    # ------------------------------------------------------------------
    def chat(
        self,
        session_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if session_id not in self.sessions:
            raise ValueError(f"会话不存在: {session_id}")

        start = time.time()
        session_info = self.sessions[session_id]
        graph = self._graphs[session_info.memory_type]

        try:
            input_messages: List[Any] = []
            if context:
                # 将业务上下文（如系统提示词、订单信息）以 SystemMessage 注入
                input_messages.append(
                    SystemMessage(
                        content="\n".join(f"{k}: {v}" for k, v in context.items())
                    )
                )
            input_messages.append(HumanMessage(content=message))

            result = graph.invoke(
                {"messages": input_messages},
                config={"configurable": {"thread_id": session_id}},
            )
            last_ai = next(
                (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
                None,
            )
            response = last_ai.content if last_ai else "(无响应)"

            response_time = time.time() - start
            memory_size = sum(len(getattr(m, "content", "")) for m in result["messages"])
            token_usage = len(message) + len(response)

            session_info.last_active = datetime.now()
            session_info.message_count += 1
            self._record_performance(session_id, response_time, token_usage, memory_size)

            return {
                "response": response,
                "session_id": session_id,
                "message_count": session_info.message_count,
                "response_time": response_time,
                "memory_type": session_info.memory_type,
                "memory_size": memory_size,
                "summary": result.get("summary", ""),
            }
        except Exception as e:
            print(f"❌ 聊天处理失败: {e}")
            return {"error": str(e), "session_id": session_id}

    # ------------------------------------------------------------------
    # 指标 / 信息查询
    # ------------------------------------------------------------------
    def _record_performance(
        self,
        session_id: str,
        response_time: float,
        token_usage: int,
        memory_size: int,
    ) -> None:
        metrics = PerformanceMetrics(
            response_time=response_time,
            token_usage=token_usage,
            memory_size=memory_size,
            timestamp=datetime.now(),
        )
        buf = self.performance_metrics.setdefault(session_id, [])
        buf.append(metrics)
        if len(buf) > 100:
            del buf[:-100]

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id not in self.sessions:
            return None
        session_info = self.sessions[session_id]
        metrics = self.performance_metrics.get(session_id, [])
        avg_rt = sum(m.response_time for m in metrics) / len(metrics) if metrics else 0
        avg_tk = sum(m.token_usage for m in metrics) / len(metrics) if metrics else 0
        return {
            "session_info": asdict(session_info),
            "performance": {
                "avg_response_time": avg_rt,
                "avg_token_usage": avg_tk,
                "total_interactions": len(metrics),
            },
        }

    def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        items = [
            {
                "session_id": sid,
                "created_at": info.created_at.isoformat(),
                "last_active": info.last_active.isoformat(),
                "message_count": info.message_count,
                "memory_type": info.memory_type,
            }
            for sid, info in self.sessions.items()
            if info.user_id == user_id
        ]
        return sorted(items, key=lambda x: x["last_active"], reverse=True)

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """从 checkpointer 读取会话历史（不依赖 self.sessions 是否存在）。"""
        memory_type = (
            self.sessions[session_id].memory_type if session_id in self.sessions else "buffer"
        )
        graph = self._graphs[memory_type]
        state = graph.get_state({"configurable": {"thread_id": session_id}})
        messages = (state.values or {}).get("messages", []) if state else []
        result = []
        for m in messages:
            if isinstance(m, HumanMessage):
                result.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                result.append({"role": "assistant", "content": m.content})
            elif isinstance(m, SystemMessage):
                result.append({"role": "system", "content": m.content})
        return result

    # ------------------------------------------------------------------
    # 元数据持久化（不再负责对话正文）
    # ------------------------------------------------------------------
    def save_session(self, session_id: str) -> bool:
        if session_id not in self.sessions:
            return False
        try:
            path = self.storage_dir / f"{session_id}.json"
            payload = {
                "session_info": asdict(self.sessions[session_id]),
                "performance_metrics": [
                    asdict(m) for m in self.performance_metrics.get(session_id, [])
                ],
                "_note": (
                    "对话正文由 LangGraph checkpointer 持久化；"
                    "本文件仅保存会话元数据与统计指标。"
                ),
            }

            def _default(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=_default)
            return True
        except Exception as e:
            print(f"❌ 保存会话失败: {e}")
            return False

    def load_session(self, session_id: str) -> bool:
        try:
            path = self.storage_dir / f"{session_id}.json"
            if not path.exists():
                return False
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            info = payload["session_info"]
            info["created_at"] = datetime.fromisoformat(info["created_at"])
            info["last_active"] = datetime.fromisoformat(info["last_active"])
            self.sessions[session_id] = SessionInfo(**info)

            self.performance_metrics[session_id] = [
                PerformanceMetrics(
                    response_time=m["response_time"],
                    token_usage=m["token_usage"],
                    memory_size=m["memory_size"],
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                )
                for m in payload.get("performance_metrics", [])
            ]
            print(f"✅ 已加载会话元数据: {session_id[:8]}…")
            return True
        except Exception as e:
            print(f"❌ 加载会话失败: {e}")
            return False

    def cleanup_inactive_sessions(self, hours: int = 24) -> None:
        cutoff = datetime.now() - timedelta(hours=hours)
        inactive = [sid for sid, info in self.sessions.items() if info.last_active < cutoff]
        for sid in inactive:
            self.save_session(sid)
            self.sessions.pop(sid, None)
            self.performance_metrics.pop(sid, None)
        print(f"🧹 清理了 {len(inactive)} 个不活跃会话")


# ---------------------------------------------------------------------------
# 客服门面
# ---------------------------------------------------------------------------
class CustomerServiceBot:
    """智能客服机器人（在 SessionManager 之上的业务层封装）。

    实例持有一个 `SessionManager`；如果启用了 `persistent=True`（SqliteSaver），
    在服务退出前应调用 `close()` 释放底层连接。
    """

    def __init__(self, persistent: bool = False) -> None:
        self.session_manager = SessionManager(persistent=persistent)
        self.system_prompt = (
            "你是一个专业的客服助手，具备以下特点：\n"
            "1. 友好、耐心、专业；\n"
            "2. 能够结合对话历史给出连贯回复；\n"
            "3. 提供准确的帮助与建议；\n"
            "4. 无法解决时引导用户联系人工客服。"
        )

    def close(self) -> None:
        """释放底层 SessionManager 持有的 checkpointer 资源。"""
        self.session_manager.close()

    def start_conversation(
        self, user_id: str, user_name: Optional[str] = None
    ) -> tuple[str, str]:
        metadata = {"user_name": user_name} if user_name else {}
        session_id = self.session_manager.create_session(
            user_id=user_id, memory_type="auto", metadata=metadata
        )
        welcome = (
            f"您好{user_name or ''}！我是智能客服助手，很高兴为您服务。请问有什么可以帮助您的吗？"
        )
        return session_id, welcome

    def chat(self, session_id: str, message: str) -> Dict[str, Any]:
        return self.session_manager.chat(
            session_id, message, context={"系统角色": self.system_prompt}
        )

    def get_conversation_summary(self, session_id: str) -> str:
        info = self.session_manager.get_session_info(session_id)
        if not info:
            return "会话不存在"
        s = info["session_info"]
        p = info["performance"]
        return (
            "📊 对话摘要\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 用户ID: {s['user_id']}\n"
            f"🕐 创建时间: {s['created_at']}\n"
            f"💬 消息数量: {s['message_count']}\n"
            f"🧠 记忆类型: {s['memory_type']}\n"
            f"⚡ 平均响应时间: {p['avg_response_time']:.2f}秒\n"
            f"📝 平均Token使用: {p['avg_token_usage']:.0f}"
        )


# ---------------------------------------------------------------------------
# Demo 入口
# ---------------------------------------------------------------------------
def demo_customer_service(persistent: bool = False) -> None:
    print("🤖 智能客服机器人演示（LangGraph 短期记忆）")
    print("=" * 50)

    bot = CustomerServiceBot(persistent=persistent)

    users = [
        {"user_id": "user_001", "name": "张三"},
        {"user_id": "user_002", "name": "李四"},
    ]
    sessions: Dict[str, str] = {}
    for user in users:
        sid, welcome = bot.start_conversation(user["user_id"], user["name"])
        sessions[user["user_id"]] = sid
        print(f"\n👤 {user['name']} 开始对话")
        print(f"🤖 {welcome}")

    conversations = {
        "user_001": [
            "我想查询我的订单状态",
            "我的订单号是 ORD123456",
            "什么时候能发货？",
            "好的，谢谢你的帮助",
        ],
        "user_002": [
            "我收到的商品有质量问题",
            "是一件衣服，颜色和描述不符",
            "我想申请退货",
            "需要什么手续吗？",
        ],
    }

    max_rounds = max(len(c) for c in conversations.values())
    for r in range(max_rounds):
        for user_id, turns in conversations.items():
            if r >= len(turns):
                continue
            user_name = next(u["name"] for u in users if u["user_id"] == user_id)
            sid = sessions[user_id]
            print(f"\n👤 {user_name}: {turns[r]}")
            result = bot.chat(sid, turns[r])
            if "response" in result:
                print(f"🤖 客服: {result['response']}")
                print(f"📊 响应时间: {result['response_time']:.2f}秒")
            else:
                print(f"❌ 错误: {result.get('error', '未知错误')}")

    print("\n" + "=" * 50)
    print("📋 会话摘要")
    print("=" * 50)
    for user in users:
        print("\n" + bot.get_conversation_summary(sessions[user["user_id"]]))

    print("\n💾 保存会话元数据…")
    for sid in sessions.values():
        ok = bot.session_manager.save_session(sid)
        print(f"{'✅' if ok else '❌'} 会话 {sid[:8]}… 元数据保存{'成功' if ok else '失败'}")


def main() -> None:
    print("智能客服机器人 - LangGraph 版")
    print("=" * 40)
    if not config.validate_config():
        print("❌ 配置验证失败！请检查配置文件或环境变量")
        return
    demo_customer_service()


if __name__ == "__main__":
    main()
