import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import weakref

class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

class MessagePriority(Enum):
    """消息优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class Message:
    """消息数据结构"""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None表示广播消息
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None  # 用于关联请求和响应
    reply_to: Optional[str] = None  # 回复地址
    headers: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息对象"""
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

@dataclass
class Subscription:
    """订阅信息"""
    subscriber_id: str
    message_types: Set[MessageType]
    callback: Callable[[Message], None]
    filters: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.filters is None:
            self.filters = {}
    
    def matches(self, message: Message) -> bool:
        """检查消息是否匹配订阅条件"""
        # 检查消息类型
        if message.message_type not in self.message_types:
            return False
        
        # 检查过滤条件
        for key, value in self.filters.items():
            if key == 'sender_id' and message.sender_id != value:
                return False
            elif key == 'priority' and message.priority.value < value:
                return False
            elif key in message.headers and message.headers[key] != value:
                return False
        
        return True

class MessageQueue:
    """消息队列"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queues = {
            priority: deque() for priority in MessagePriority
        }
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._size = 0
    
    async def put(self, message: Message) -> bool:
        """添加消息到队列"""
        async with self._lock:
            if self._size >= self.max_size:
                # 队列满时，移除最低优先级的旧消息
                await self._remove_lowest_priority_message()
            
            self._queues[message.priority].append(message)
            self._size += 1
            
            async with self._not_empty:
                self._not_empty.notify()
            
            return True
    
    async def get(self, timeout: Optional[float] = None) -> Optional[Message]:
        """从队列获取消息（按优先级）"""
        async with self._not_empty:
            # 等待消息可用
            if self._size == 0:
                try:
                    await asyncio.wait_for(self._not_empty.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    return None
            
            # 按优先级获取消息
            for priority in sorted(MessagePriority, key=lambda p: p.value, reverse=True):
                queue = self._queues[priority]
                if queue:
                    message = queue.popleft()
                    self._size -= 1
                    return message
            
            return None
    
    async def _remove_lowest_priority_message(self):
        """移除最低优先级的消息"""
        for priority in sorted(MessagePriority, key=lambda p: p.value):
            queue = self._queues[priority]
            if queue:
                queue.popleft()
                self._size -= 1
                break
    
    def size(self) -> int:
        """获取队列大小"""
        return self._size
    
    def clear(self):
        """清空队列"""
        for queue in self._queues.values():
            queue.clear()
        self._size = 0

class MessageBus:
    """消息总线 - 智能体间通信的核心组件"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 消息队列管理
        self._agent_queues: Dict[str, MessageQueue] = {}
        self._global_queue = MessageQueue(max_size=self.config.get("global_queue_size", 10000))
        
        # 订阅管理
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._subscription_lock = asyncio.Lock()
        
        # 消息路由
        self._routing_table: Dict[str, str] = {}  # agent_id -> queue_name
        self._topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # 消息历史和统计
        self._message_history: deque = deque(maxlen=self.config.get("history_size", 1000))
        self._message_stats = {
            "total_sent": 0,
            "total_received": 0,
            "total_dropped": 0,
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int)
        }
        
        # 性能监控
        self._performance_metrics = {
            "avg_delivery_time": 0.0,
            "peak_queue_size": 0,
            "active_subscriptions": 0
        }
        
        # 错误处理
        self._error_handlers: List[Callable[[Exception, Message], None]] = []
        self._dead_letter_queue = MessageQueue(max_size=self.config.get("dlq_size", 100))
        
        # 运行状态
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        
        # 消息过期清理
        self._cleanup_interval = self.config.get("cleanup_interval", 60)  # 秒
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动消息总线"""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Starting message bus")
        
        # 启动消息处理工作器
        num_workers = self.config.get("num_workers", 3)
        for i in range(num_workers):
            task = asyncio.create_task(self._message_worker(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
        
        self.logger.info(f"Message bus started with {num_workers} workers")
    
    async def stop(self):
        """停止消息总线"""
        if not self._running:
            return
        
        self._running = False
        self.logger.info("Stopping message bus")
        
        # 取消所有工作器任务
        for task in self._worker_tasks:
            task.cancel()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self._worker_tasks, self._cleanup_task, return_exceptions=True)
        
        self.logger.info("Message bus stopped")
    
    async def register_agent(self, agent_id: str, queue_size: int = 100) -> bool:
        """注册智能体"""
        if agent_id in self._agent_queues:
            self.logger.warning(f"Agent {agent_id} already registered")
            return False
        
        self._agent_queues[agent_id] = MessageQueue(max_size=queue_size)
        self._routing_table[agent_id] = agent_id
        
        self.logger.info(f"Agent {agent_id} registered with queue size {queue_size}")
        return True
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体"""
        if agent_id not in self._agent_queues:
            return False
        
        # 清理队列
        del self._agent_queues[agent_id]
        del self._routing_table[agent_id]
        
        # 清理订阅
        async with self._subscription_lock:
            if agent_id in self._subscriptions:
                del self._subscriptions[agent_id]
        
        # 清理主题订阅
        for subscribers in self._topic_subscribers.values():
            subscribers.discard(agent_id)
        
        self.logger.info(f"Agent {agent_id} unregistered")
        return True
    
    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        try:
            # 验证消息
            if not self._validate_message(message):
                return False
            
            # 记录统计
            self._update_send_stats(message)
            
            # 添加到历史
            self._message_history.append(message)
            
            # 路由消息
            if message.receiver_id:
                # 点对点消息
                await self._route_to_agent(message)
            else:
                # 广播消息
                await self._broadcast_message(message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message {message.message_id}: {str(e)}")
            await self._handle_error(e, message)
            return False
    
    async def _route_to_agent(self, message: Message):
        """路由消息到特定智能体"""
        agent_id = message.receiver_id
        
        if agent_id not in self._agent_queues:
            self.logger.warning(f"Agent {agent_id} not found for message {message.message_id}")
            await self._dead_letter_queue.put(message)
            return
        
        queue = self._agent_queues[agent_id]
        success = await queue.put(message)
        
        if not success:
            self.logger.warning(f"Failed to queue message {message.message_id} for agent {agent_id}")
            await self._dead_letter_queue.put(message)
    
    async def _broadcast_message(self, message: Message):
        """广播消息到所有订阅者"""
        # 添加到全局队列
        await self._global_queue.put(message)
        
        # 直接发送给订阅者
        async with self._subscription_lock:
            for agent_id, subscriptions in self._subscriptions.items():
                for subscription in subscriptions:
                    if subscription.matches(message):
                        try:
                            # 创建消息副本
                            agent_message = Message(
                                message_id=str(uuid.uuid4()),
                                sender_id=message.sender_id,
                                receiver_id=agent_id,
                                message_type=message.message_type,
                                priority=message.priority,
                                content=message.content.copy(),
                                timestamp=message.timestamp,
                                expires_at=message.expires_at,
                                correlation_id=message.correlation_id,
                                reply_to=message.reply_to,
                                headers=message.headers.copy()
                            )
                            
                            await self._route_to_agent(agent_message)
                            
                        except Exception as e:
                            self.logger.error(f"Failed to deliver message to subscriber {agent_id}: {str(e)}")
    
    async def receive_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """接收消息"""
        if agent_id not in self._agent_queues:
            return None
        
        queue = self._agent_queues[agent_id]
        message = await queue.get(timeout=timeout)
        
        if message:
            # 检查消息是否过期
            if message.is_expired():
                self.logger.debug(f"Message {message.message_id} expired")
                return await self.receive_message(agent_id, timeout)  # 递归获取下一条消息
            
            self._update_receive_stats(message)
        
        return message
    
    async def subscribe(self, agent_id: str, message_types: List[MessageType], 
                      callback: Optional[Callable[[Message], None]] = None,
                      filters: Dict[str, Any] = None) -> str:
        """订阅消息类型"""
        subscription_id = str(uuid.uuid4())
        
        # 如果没有提供回调，使用默认的队列投递
        if callback is None:
            callback = lambda msg: asyncio.create_task(self._route_to_agent(msg))
        
        subscription = Subscription(
            subscriber_id=agent_id,
            message_types=set(message_types),
            callback=callback,
            filters=filters or {}
        )
        
        async with self._subscription_lock:
            self._subscriptions[agent_id].append(subscription)
        
        self.logger.info(f"Agent {agent_id} subscribed to {[mt.value for mt in message_types]}")
        return subscription_id
    
    async def unsubscribe(self, agent_id: str, subscription_id: str = None):
        """取消订阅"""
        async with self._subscription_lock:
            if agent_id in self._subscriptions:
                if subscription_id:
                    # 取消特定订阅（这里简化实现，实际需要在Subscription中添加ID）
                    pass
                else:
                    # 取消所有订阅
                    del self._subscriptions[agent_id]
                    self.logger.info(f"Agent {agent_id} unsubscribed from all message types")
    
    async def publish(self, topic: str, content: Dict[str, Any], 
                     sender_id: str, priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """发布主题消息"""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=None,  # 广播消息
            message_type=MessageType.BROADCAST,
            priority=priority,
            content=content,
            timestamp=datetime.now(),
            headers={"topic": topic}
        )
        
        return await self.send_message(message)
    
    async def request_response(self, sender_id: str, receiver_id: str, 
                              content: Dict[str, Any], timeout: float = 30.0) -> Optional[Message]:
        """请求-响应模式"""
        correlation_id = str(uuid.uuid4())
        
        # 发送请求
        request = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            content=content,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            reply_to=sender_id,
            expires_at=datetime.now() + timedelta(seconds=timeout)
        )
        
        success = await self.send_message(request)
        if not success:
            return None
        
        # 等待响应
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            response = await self.receive_message(sender_id, timeout=1.0)
            if (response and 
                response.message_type == MessageType.RESPONSE and 
                response.correlation_id == correlation_id):
                return response
        
        return None  # 超时
    
    async def send_response(self, original_request: Message, content: Dict[str, Any], 
                           sender_id: str) -> bool:
        """发送响应消息"""
        if not original_request.reply_to or not original_request.correlation_id:
            return False
        
        response = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=original_request.reply_to,
            message_type=MessageType.RESPONSE,
            priority=original_request.priority,
            content=content,
            timestamp=datetime.now(),
            correlation_id=original_request.correlation_id
        )
        
        return await self.send_message(response)
    
    def _validate_message(self, message: Message) -> bool:
        """验证消息格式"""
        if not message.message_id or not message.sender_id:
            return False
        
        if not isinstance(message.content, dict):
            return False
        
        return True
    
    def _update_send_stats(self, message: Message):
        """更新发送统计"""
        self._message_stats["total_sent"] += 1
        self._message_stats["by_type"][message.message_type.value] += 1
        self._message_stats["by_priority"][message.priority.value] += 1
    
    def _update_receive_stats(self, message: Message):
        """更新接收统计"""
        self._message_stats["total_received"] += 1
        
        # 计算投递时间
        delivery_time = (datetime.now() - message.timestamp).total_seconds()
        current_avg = self._performance_metrics["avg_delivery_time"]
        total_received = self._message_stats["total_received"]
        
        # 更新平均投递时间
        self._performance_metrics["avg_delivery_time"] = (
            (current_avg * (total_received - 1) + delivery_time) / total_received
        )
    
    async def _handle_error(self, error: Exception, message: Message):
        """处理错误"""
        self._message_stats["total_dropped"] += 1
        
        # 调用错误处理器
        for handler in self._error_handlers:
            try:
                handler(error, message)
            except Exception as e:
                self.logger.error(f"Error handler failed: {str(e)}")
        
        # 记录到死信队列
        await self._dead_letter_queue.put(message)
    
    async def _message_worker(self, worker_id: str):
        """消息处理工作器"""
        self.logger.info(f"Message worker {worker_id} started")
        
        while self._running:
            try:
                # 从全局队列获取消息
                message = await self._global_queue.get(timeout=1.0)
                if message:
                    await self._process_global_message(message)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(1)
        
        self.logger.info(f"Message worker {worker_id} stopped")
    
    async def _process_global_message(self, message: Message):
        """处理全局消息"""
        # 这里可以添加全局消息处理逻辑
        # 例如：日志记录、监控、消息转换等
        pass
    
    async def _cleanup_expired_messages(self):
        """清理过期消息"""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                # 清理各个智能体队列中的过期消息
                for agent_id, queue in self._agent_queues.items():
                    # 这里需要实现队列的过期消息清理逻辑
                    pass
                
                # 清理死信队列中的过期消息
                # 实现类似逻辑
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取消息总线统计信息"""
        return {
            "message_stats": dict(self._message_stats),
            "performance_metrics": dict(self._performance_metrics),
            "queue_sizes": {
                agent_id: queue.size() 
                for agent_id, queue in self._agent_queues.items()
            },
            "global_queue_size": self._global_queue.size(),
            "dead_letter_queue_size": self._dead_letter_queue.size(),
            "active_agents": len(self._agent_queues),
            "active_subscriptions": sum(len(subs) for subs in self._subscriptions.values())
        }
    
    def add_error_handler(self, handler: Callable[[Exception, Message], None]):
        """添加错误处理器"""
        self._error_handlers.append(handler)
    
    def remove_error_handler(self, handler: Callable[[Exception, Message], None]):
        """移除错误处理器"""
        if handler in self._error_handlers:
            self._error_handlers.remove(handler)