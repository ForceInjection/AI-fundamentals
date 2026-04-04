# -*- coding: utf-8 -*-
"""
日志记录与审计系统

本模块实现了完整的日志记录、审计跟踪和合规性管理功能。
"""

import json
import time
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import structlog
from pythonjsonlogger import jsonlogger
import asyncio
from pathlib import Path

# ==================== 审计事件定义 ====================

class AuditEventType(Enum):
    """审计事件类型"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DIALOGUE_START = "dialogue_start"
    DIALOGUE_END = "dialogue_end"
    ENTITY_EXTRACTION = "entity_extraction"
    COREFERENCE_RESOLUTION = "coreference_resolution"
    STATE_UPDATE = "state_update"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ERROR = "system_error"
    SECURITY_VIOLATION = "security_violation"
    PERFORMANCE_ALERT = "performance_alert"
    CONFIGURATION_CHANGE = "configuration_change"

class SeverityLevel(Enum):
    """严重程度级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    dialogue_id: Optional[str]
    component: str
    action: str
    resource: Optional[str]
    result: str  # success, failure, partial
    severity: SeverityLevel
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data
    
    def get_hash(self) -> str:
        """获取事件哈希值（用于完整性验证）"""
        content = f"{self.event_id}{self.timestamp.isoformat()}{self.event_type.value}{self.action}"
        return hashlib.sha256(content.encode()).hexdigest()

# ==================== 审计日志管理器 ====================

class AuditLogger:
    """审计日志管理器"""
    
    def __init__(self, log_dir: str = "./logs", retention_days: int = 90):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.retention_days = retention_days
        
        # 配置审计日志记录器
        self.audit_logger = self._setup_audit_logger()
        
        # 事件缓冲区（用于批量写入）
        self.event_buffer: List[AuditEvent] = []
        self.buffer_size = 100
        self.last_flush = time.time()
        self.flush_interval = 60  # 60秒
    
    def _setup_audit_logger(self) -> logging.Logger:
        """设置审计日志记录器"""
        logger = logging.getLogger('audit')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        # 设置JSON格式化器
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def log_event(self, event: AuditEvent, immediate_flush: bool = False):
        """记录审计事件"""
        # 添加到缓冲区
        self.event_buffer.append(event)
        
        # 检查是否需要刷新
        if (immediate_flush or 
            len(self.event_buffer) >= self.buffer_size or
            time.time() - self.last_flush > self.flush_interval):
            self._flush_events()
    
    def _flush_events(self):
        """刷新事件缓冲区"""
        if not self.event_buffer:
            return
        
        for event in self.event_buffer:
            # 记录到审计日志
            self.audit_logger.info(
                "audit_event",
                extra=event.to_dict()
            )
        
        # 清空缓冲区
        self.event_buffer.clear()
        self.last_flush = time.time()
    
    def create_event(self, event_type: AuditEventType, component: str, 
                    action: str, result: str = "success",
                    severity: SeverityLevel = SeverityLevel.LOW,
                    **kwargs) -> AuditEvent:
        """创建审计事件"""
        return AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            component=component,
            action=action,
            result=result,
            severity=severity,
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id'),
            dialogue_id=kwargs.get('dialogue_id'),
            resource=kwargs.get('resource'),
            details=kwargs.get('details', {}),
            ip_address=kwargs.get('ip_address'),
            user_agent=kwargs.get('user_agent'),
            duration_ms=kwargs.get('duration_ms')
        )
    
    def log_dialogue_event(self, dialogue_id: str, action: str, 
                          user_id: Optional[str] = None, **kwargs):
        """记录对话相关事件"""
        event = self.create_event(
            event_type=AuditEventType.DIALOGUE_START if action == "start" else AuditEventType.DIALOGUE_END,
            component="dialogue_manager",
            action=action,
            dialogue_id=dialogue_id,
            user_id=user_id,
            **kwargs
        )
        self.log_event(event)
    
    def log_entity_extraction(self, dialogue_id: str, text: str, 
                             entities: List[Dict], processing_time: float,
                             user_id: Optional[str] = None):
        """记录实体提取事件"""
        event = self.create_event(
            event_type=AuditEventType.ENTITY_EXTRACTION,
            component="entity_recognition",
            action="extract_entities",
            dialogue_id=dialogue_id,
            user_id=user_id,
            duration_ms=processing_time * 1000,
            details={
                "input_text_length": len(text),
                "entities_count": len(entities),
                "entity_types": list(set(e.get('type') for e in entities))
            }
        )
        self.log_event(event)
    
    def log_coreference_resolution(self, dialogue_id: str, mention: str,
                                  resolved_entity: Optional[str], confidence: float,
                                  method: str, processing_time: float,
                                  user_id: Optional[str] = None):
        """记录指代消解事件"""
        result = "success" if resolved_entity else "failure"
        severity = SeverityLevel.LOW if confidence > 0.7 else SeverityLevel.MEDIUM
        
        event = self.create_event(
            event_type=AuditEventType.COREFERENCE_RESOLUTION,
            component="coreference_resolver",
            action="resolve_coreference",
            result=result,
            severity=severity,
            dialogue_id=dialogue_id,
            user_id=user_id,
            duration_ms=processing_time * 1000,
            details={
                "mention": mention,
                "resolved_entity": resolved_entity,
                "confidence": confidence,
                "method": method
            }
        )
        self.log_event(event)
    
    def log_security_event(self, event_type: str, description: str,
                          user_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          severity: SeverityLevel = SeverityLevel.HIGH):
        """记录安全事件"""
        event = self.create_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            component="security",
            action=event_type,
            result="detected",
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details={"description": description}
        )
        self.log_event(event, immediate_flush=True)
    
    def log_performance_alert(self, component: str, metric: str, 
                             value: float, threshold: float,
                             severity: SeverityLevel = SeverityLevel.MEDIUM):
        """记录性能告警"""
        event = self.create_event(
            event_type=AuditEventType.PERFORMANCE_ALERT,
            component=component,
            action="performance_threshold_exceeded",
            result="alert",
            severity=severity,
            details={
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "ratio": value / threshold if threshold > 0 else float('inf')
            }
        )
        self.log_event(event, immediate_flush=True)
    
    def cleanup_old_logs(self):
        """清理过期日志"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for log_file in self.log_dir.glob("audit_*.log"):
            try:
                # 从文件名提取日期
                date_str = log_file.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    print(f"已删除过期日志文件: {log_file}")
            except (ValueError, IndexError):
                # 跳过无法解析日期的文件
                continue

# ==================== 合规性管理器 ====================

class ComplianceManager:
    """合规性管理器"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """加载合规性规则"""
        return {
            "data_retention": {
                "personal_data_days": 365,
                "system_logs_days": 90,
                "audit_logs_days": 2555  # 7年
            },
            "access_control": {
                "max_failed_attempts": 5,
                "session_timeout_minutes": 30,
                "require_mfa": True
            },
            "data_protection": {
                "encrypt_at_rest": True,
                "encrypt_in_transit": True,
                "anonymize_logs": True
            },
            "monitoring": {
                "log_all_access": True,
                "alert_on_anomalies": True,
                "real_time_monitoring": True
            }
        }
    
    def check_data_retention_compliance(self, data_type: str, 
                                       creation_date: datetime) -> bool:
        """检查数据保留合规性"""
        retention_days = self.compliance_rules["data_retention"].get(
            f"{data_type}_days", 90
        )
        
        age_days = (datetime.now() - creation_date).days
        is_compliant = age_days <= retention_days
        
        if not is_compliant:
            self.audit_logger.log_event(
                self.audit_logger.create_event(
                    event_type=AuditEventType.DATA_ACCESS,
                    component="compliance_manager",
                    action="data_retention_violation",
                    result="violation_detected",
                    severity=SeverityLevel.HIGH,
                    details={
                        "data_type": data_type,
                        "age_days": age_days,
                        "retention_limit_days": retention_days
                    }
                ),
                immediate_flush=True
            )
        
        return is_compliant
    
    def validate_access_request(self, user_id: str, resource: str,
                               action: str, context: Dict[str, Any]) -> bool:
        """验证访问请求的合规性"""
        # 记录访问请求
        self.audit_logger.log_event(
            self.audit_logger.create_event(
                event_type=AuditEventType.DATA_ACCESS,
                component="compliance_manager",
                action=f"access_request_{action}",
                user_id=user_id,
                resource=resource,
                details=context
            )
        )
        
        # 这里可以添加具体的访问控制逻辑
        # 例如：检查用户权限、资源访问策略等
        
        return True  # 简化实现
    
    def anonymize_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """匿名化敏感数据"""
        sensitive_fields = ['user_id', 'ip_address', 'email', 'phone']
        anonymized_data = data.copy()
        
        for field in sensitive_fields:
            if field in anonymized_data:
                # 使用哈希进行匿名化
                original_value = str(anonymized_data[field])
                anonymized_data[field] = hashlib.sha256(
                    original_value.encode()
                ).hexdigest()[:8]
        
        return anonymized_data
    
    def generate_compliance_report(self, start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """生成合规性报告"""
        # 这里应该分析审计日志并生成报告
        # 简化实现
        return {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "compliance_status": "compliant",
            "violations": [],
            "recommendations": [
                "定期审查访问权限",
                "更新数据保留策略",
                "加强监控告警"
            ]
        }

# ==================== 结构化日志记录器 ====================

class StructuredLogger:
    """增强的结构化日志记录器"""
    
    def __init__(self, name: str, audit_logger: Optional[AuditLogger] = None):
        self.name = name
        self.audit_logger = audit_logger
        
        # 配置structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._add_correlation_id,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(name)
    
    def _add_correlation_id(self, logger, method_name, event_dict):
        """添加关联ID"""
        if 'correlation_id' not in event_dict:
            event_dict['correlation_id'] = str(uuid.uuid4())[:8]
        return event_dict
    
    def log_with_audit(self, level: str, message: str, 
                      audit_event_type: Optional[AuditEventType] = None,
                      **kwargs):
        """记录日志并同时记录审计事件"""
        # 记录结构化日志
        getattr(self.logger, level.lower())(message, **kwargs)
        
        # 记录审计事件
        if self.audit_logger and audit_event_type:
            event = self.audit_logger.create_event(
                event_type=audit_event_type,
                component=self.name,
                action=message,
                details=kwargs
            )
            self.audit_logger.log_event(event)
    
    def log_coreference_resolution(self, dialogue_id: str, mention: str,
                                 resolved_entity: Optional[str], confidence: float,
                                 processing_time: float, method: str,
                                 user_id: Optional[str] = None):
        """记录指代消解事件（包含审计）"""
        # 结构化日志
        self.logger.info(
            "coreference_resolution_completed",
            dialogue_id=dialogue_id,
            mention=mention,
            resolved_entity=resolved_entity,
            confidence=confidence,
            processing_time_ms=processing_time * 1000,
            method=method,
            user_id=user_id
        )
        
        # 审计日志
        if self.audit_logger:
            self.audit_logger.log_coreference_resolution(
                dialogue_id=dialogue_id,
                mention=mention,
                resolved_entity=resolved_entity,
                confidence=confidence,
                method=method,
                processing_time=processing_time,
                user_id=user_id
            )
    
    def log_entity_extraction(self, dialogue_id: str, text: str,
                            entities: List[Dict], processing_time: float,
                            user_id: Optional[str] = None):
        """记录实体提取事件（包含审计）"""
        entity_info = [
            {"text": e.get('text'), "type": e.get('type'), "confidence": e.get('confidence')}
            for e in entities
        ]
        
        # 结构化日志
        self.logger.info(
            "entity_extraction_completed",
            dialogue_id=dialogue_id,
            input_text_length=len(text),
            entities=entity_info,
            entity_count=len(entities),
            processing_time_ms=processing_time * 1000,
            user_id=user_id
        )
        
        # 审计日志
        if self.audit_logger:
            self.audit_logger.log_entity_extraction(
                dialogue_id=dialogue_id,
                text=text,
                entities=entities,
                processing_time=processing_time,
                user_id=user_id
            )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any],
                              severity: SeverityLevel = SeverityLevel.MEDIUM):
        """记录错误及其上下文"""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        # 结构化日志
        self.logger.error(
            "error_occurred",
            **error_context
        )
        
        # 审计日志
        if self.audit_logger:
            event = self.audit_logger.create_event(
                event_type=AuditEventType.SYSTEM_ERROR,
                component=self.name,
                action="error_occurred",
                result="error",
                severity=severity,
                details=error_context
            )
            self.audit_logger.log_event(event, immediate_flush=True)

# ==================== 日志分析器 ====================

class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
    
    def analyze_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """分析性能趋势"""
        # 简化实现 - 实际应该解析日志文件
        return {
            "avg_response_time": 150.5,
            "peak_response_time": 500.2,
            "error_rate": 0.02,
            "throughput_per_hour": 1200,
            "trend": "improving"
        }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常"""
        # 简化实现
        return [
            {
                "type": "high_error_rate",
                "description": "错误率超过阈值",
                "severity": "medium",
                "timestamp": datetime.now().isoformat()
            }
        ]
    
    def generate_usage_report(self) -> Dict[str, Any]:
        """生成使用报告"""
        return {
            "total_requests": 10000,
            "unique_users": 500,
            "avg_session_duration": 15.5,
            "most_active_hours": [9, 10, 14, 15],
            "popular_features": [
                "entity_extraction",
                "coreference_resolution"
            ]
        }

# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建审计日志管理器
    audit_logger = AuditLogger("./logs", retention_days=90)
    
    # 创建合规性管理器
    compliance_manager = ComplianceManager(audit_logger)
    
    # 创建结构化日志记录器
    structured_logger = StructuredLogger("test_component", audit_logger)
    
    # 模拟一些事件
    structured_logger.log_coreference_resolution(
        dialogue_id="test_001",
        mention="他",
        resolved_entity="张三",
        confidence=0.85,
        processing_time=0.15,
        method="neural_network",
        user_id="user_123"
    )
    
    # 记录安全事件
    audit_logger.log_security_event(
        event_type="suspicious_access",
        description="多次失败的登录尝试",
        user_id="user_456",
        ip_address="192.168.1.100",
        severity=SeverityLevel.HIGH
    )
    
    # 刷新缓冲区
    audit_logger._flush_events()
    
    print("日志记录和审计系统已初始化")