# -*- coding: utf-8 -*-
"""
系统集成与部署

本模块实现了基于FastAPI的微服务架构，包括主服务入口、任务处理引擎和系统配置管理。
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import logging
import time
from datetime import datetime
import uvicorn

# 导入自定义模块
from entity_recognition import EnhancedEntityRecognitionLayer, Entity
from coreference_resolution import AdvancedCoreferenceLayer, Mention, PronounType
from dialogue_state_manager import IntelligentStateManager

# 请求和响应模型
class DialogueRequest(BaseModel):
    """对话请求模型"""
    text: str = Field(..., description="用户输入文本")
    dialogue_id: str = Field(..., description="对话ID")
    task_type: str = Field(default="general_qa", description="任务类型")
    user_id: Optional[str] = Field(None, description="用户ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class EntityInfo(BaseModel):
    """实体信息模型"""
    id: str
    text: str
    type: str
    confidence: float
    start: int
    end: int

class CoreferenceInfo(BaseModel):
    """指代消解信息模型"""
    mention: str
    resolved_entity: Optional[str]
    confidence: float
    method: str

class DialogueResponse(BaseModel):
    """对话响应模型"""
    response: str = Field(..., description="系统回复")
    entities: List[EntityInfo] = Field(..., description="识别的实体")
    coreferences: List[CoreferenceInfo] = Field(..., description="指代消解结果")
    confidence: float = Field(..., description="整体置信度")
    processing_time: float = Field(..., description="处理时间（秒）")
    dialogue_state: Dict[str, Any] = Field(..., description="对话状态")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]

# 服务组件
class EntityRecognitionService:
    """实体识别服务"""
    
    def __init__(self):
        self.entity_layer = EnhancedEntityRecognitionLayer()
        self.logger = logging.getLogger(__name__)
    
    async def extract_entities(self, text: str, dialogue_id: str) -> List[Entity]:
        """异步提取实体"""
        try:
            dialogue_context = {
                'dialogue_id': dialogue_id,
                'timestamp': time.time()
            }
            
            # 在线程池中执行CPU密集型任务
            loop = asyncio.get_event_loop()
            entities = await loop.run_in_executor(
                None, 
                self.entity_layer.process, 
                text, 
                dialogue_context
            )
            
            self.logger.info(f"提取到{len(entities)}个实体: {[e.text for e in entities]}")
            return entities
            
        except Exception as e:
            self.logger.error(f"实体识别失败: {str(e)}")
            return []

class CoreferenceResolutionService:
    """指代消解服务"""
    
    def __init__(self):
        self.coref_layer = AdvancedCoreferenceLayer()
        self.logger = logging.getLogger(__name__)
    
    async def resolve_coreferences(self, text: str, entities: List[Entity], 
                                  dialogue_id: str) -> List[Any]:
        """异步指代消解"""
        try:
            # 简单的代词检测
            pronouns = self._detect_pronouns(text)
            if not pronouns:
                return []
            
            # 获取对话状态（这里简化处理）
            dialogue_state = {
                'current_turn': 1,
                'entity_salience': {e.id: 0.5 for e in entities},
                'entity_last_mention': {e.id: 1 for e in entities}
            }
            
            resolutions = []
            for mention in pronouns:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.coref_layer.resolve,
                    mention,
                    entities,
                    dialogue_state
                )
                resolutions.append(result)
            
            self.logger.info(f"完成{len(resolutions)}个指代消解")
            return resolutions
            
        except Exception as e:
            self.logger.error(f"指代消解失败: {str(e)}")
            return []
    
    def _detect_pronouns(self, text: str) -> List[Mention]:
        """简单的代词检测"""
        pronouns = ['他', '她', '它', '这', '那', 'he', 'she', 'it', 'this', 'that']
        mentions = []
        
        for pronoun in pronouns:
            start = text.find(pronoun)
            if start != -1:
                mention = Mention(
                    text=pronoun,
                    type=PronounType.PERSONAL if pronoun in ['他', '她', '它', 'he', 'she', 'it'] else PronounType.DEMONSTRATIVE,
                    start=start,
                    end=start + len(pronoun)
                )
                mentions.append(mention)
        
        return mentions

class DialogueStateService:
    """对话状态服务"""
    
    def __init__(self):
        self.state_managers: Dict[str, IntelligentStateManager] = {}
        self.logger = logging.getLogger(__name__)
    
    async def update_state(self, dialogue_id: str, user_input: str, 
                          system_response: str, entities: List[Entity], 
                          resolutions: List[Any]) -> Dict[str, Any]:
        """异步更新对话状态"""
        try:
            # 获取或创建状态管理器
            if dialogue_id not in self.state_managers:
                self.state_managers[dialogue_id] = IntelligentStateManager()
            
            state_manager = self.state_managers[dialogue_id]
            
            # 在线程池中执行状态更新
            loop = asyncio.get_event_loop()
            state = await loop.run_in_executor(
                None,
                state_manager.update_state,
                user_input,
                system_response,
                entities,
                resolutions
            )
            
            self.logger.info(f"对话{dialogue_id}状态已更新")
            return state
            
        except Exception as e:
            self.logger.error(f"状态更新失败: {str(e)}")
            return {}
    
    async def get_state(self, dialogue_id: str) -> Dict[str, Any]:
        """获取对话状态"""
        if dialogue_id in self.state_managers:
            return self.state_managers[dialogue_id].get_current_state()
        return {}

class TaskProcessingService:
    """任务处理服务"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def process_task(self, task_type: str, text: str, 
                          resolutions: List[Any], state: Dict[str, Any]) -> str:
        """异步任务处理"""
        try:
            # 简化的任务处理逻辑
            if task_type == "sports_query":
                return await self._process_sports_query(text, resolutions, state)
            elif task_type == "general_qa":
                return await self._process_general_qa(text, resolutions, state)
            else:
                return await self._process_default(text, resolutions, state)
                
        except Exception as e:
            self.logger.error(f"任务处理失败: {str(e)}")
            return "抱歉，处理您的请求时出现了错误。"
    
    async def _process_sports_query(self, text: str, resolutions: List[Any], 
                                   state: Dict[str, Any]) -> str:
        """处理体育查询"""
        # 模拟异步处理
        await asyncio.sleep(0.1)
        
        resolved_entities = [r.entity.text for r in resolutions if r.entity]
        if resolved_entities:
            return f"根据您的查询，关于{', '.join(resolved_entities)}的体育信息如下..."
        else:
            return "这是一个体育相关的查询，但我需要更多信息来提供准确答案。"
    
    async def _process_general_qa(self, text: str, resolutions: List[Any], 
                                 state: Dict[str, Any]) -> str:
        """处理一般问答"""
        await asyncio.sleep(0.1)
        
        if resolutions:
            resolved_text = text
            for resolution in resolutions:
                if resolution.entity:
                    resolved_text = resolved_text.replace(
                        resolution.mention.text, 
                        resolution.entity.text
                    )
            return f"根据上下文，您询问的是关于{resolved_text}的问题。"
        else:
            return "我理解了您的问题，正在为您查找相关信息。"
    
    async def _process_default(self, text: str, resolutions: List[Any], 
                              state: Dict[str, Any]) -> str:
        """默认处理"""
        await asyncio.sleep(0.1)
        return "感谢您的输入，我正在处理您的请求。"

# 日志记录
async def log_interaction(request: DialogueRequest, response: DialogueResponse):
    """异步记录交互日志"""
    logger = logging.getLogger("interaction")
    logger.info(f"对话{request.dialogue_id}: {request.text} -> {response.response}")

# 依赖注入
async def get_entity_service() -> EntityRecognitionService:
    return entity_service

async def get_coref_service() -> CoreferenceResolutionService:
    return coref_service

async def get_state_service() -> DialogueStateService:
    return state_service

async def get_task_service() -> TaskProcessingService:
    return task_service

# FastAPI应用
app = FastAPI(
    title="多轮指代消解系统",
    description="基于深度学习的多轮对话指代消解服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局服务实例
entity_service = EntityRecognitionService()
coref_service = CoreferenceResolutionService()
state_service = DialogueStateService()
task_service = TaskProcessingService()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        components={
            "entity_recognition": "ok",
            "coreference_resolution": "ok",
            "dialogue_state": "ok",
            "task_processing": "ok"
        }
    )

@app.post("/api/v1/process_dialogue", response_model=DialogueResponse)
async def process_dialogue(
    request: DialogueRequest,
    background_tasks: BackgroundTasks,
    entity_svc: EntityRecognitionService = Depends(get_entity_service),
    coref_svc: CoreferenceResolutionService = Depends(get_coref_service),
    state_svc: DialogueStateService = Depends(get_state_service),
    task_svc: TaskProcessingService = Depends(get_task_service)
):
    """处理对话请求"""
    start_time = time.time()
    
    try:
        # 1. 实体识别
        entities = await entity_svc.extract_entities(
            request.text, request.dialogue_id
        )
        
        # 2. 指代消解
        resolutions = await coref_svc.resolve_coreferences(
            request.text, entities, request.dialogue_id
        )
        
        # 3. 任务处理
        response_text = await task_svc.process_task(
            request.task_type, request.text, resolutions, {}
        )
        
        # 4. 状态更新
        state = await state_svc.update_state(
            request.dialogue_id, request.text, response_text, entities, resolutions
        )
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 构建响应
        response = DialogueResponse(
            response=response_text,
            entities=[
                EntityInfo(
                    id=e.id,
                    text=e.text,
                    type=e.type,
                    confidence=e.confidence,
                    start=e.start,
                    end=e.end
                ) for e in entities
            ],
            coreferences=[
                CoreferenceInfo(
                    mention=r.mention.text,
                    resolved_entity=r.entity.text if r.entity else None,
                    confidence=r.confidence,
                    method=r.method
                ) for r in resolutions
            ],
            confidence=sum(r.confidence for r in resolutions) / len(resolutions) if resolutions else 1.0,
            processing_time=processing_time,
            dialogue_state=state
        )
        
        # 异步日志记录
        background_tasks.add_task(log_interaction, request, response)
        
        return response
        
    except Exception as e:
        logging.error(f"处理对话时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dialogue/{dialogue_id}/state")
async def get_dialogue_state(
    dialogue_id: str,
    state_svc: DialogueStateService = Depends(get_state_service)
):
    """获取对话状态"""
    try:
        state = await state_svc.get_state(dialogue_id)
        return JSONResponse(content=state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/dialogue/{dialogue_id}")
async def reset_dialogue(
    dialogue_id: str,
    state_svc: DialogueStateService = Depends(get_state_service)
):
    """重置对话"""
    try:
        if dialogue_id in state_svc.state_managers:
            state_svc.state_managers[dialogue_id].reset_dialogue()
            return {"message": f"对话{dialogue_id}已重置"}
        else:
            return {"message": f"对话{dialogue_id}不存在"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动函数
def start_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """启动服务器"""
    uvicorn.run(
        "system_integration:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()