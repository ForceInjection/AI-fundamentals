# 多轮指代消解对话系统 v2

多轮对话里的「他」「那个项目」「上次说的事」要绑定到正确的实体上，对话才能继续。v2 用当前主流的方案重建了这个系统：**LLM 结构化输出 + 对话状态全量重建**，规则引擎降级为无 Key 时的对照路径。

v1（spaCy 规则管线）的问题与教训见文末「与 v1 的差异」。

## 架构

```text
用户消息 ──► 对话历史（全量）──► LLMResolver ──► DialogueState（pydantic 校验）
                     ▲                │  失败/无Key        │
                     │                ▼                   ▼
              entity_summary     RuleResolver      EntityStore.apply_state
              （实体速览回灌）    （规则对照路径）          │
                                                        ▼
                                          响应（消解结论 + 实体账本）
```

三个模块各管一件事：

| 模块              | 职责                                                                                                                                                                      | 行数 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| `schemas.py`      | pydantic 契约：`DialogueState` / `EntityRecord` / `Resolution`。代词不能成为实体名、`last_mention_turn` 不得早于 `first_seen_turn` 等不变量在 schema 层强制                 | 111 |
| `resolver.py`     | LLM 消解器：OpenAI 兼容接口（默认指向 DeepSeek，`COREF_BASE_URL`/`COREF_MODEL` 可配），prompt 内嵌 JSON Schema，校验失败回喂重试（最多 2 次），重试耗尽抛 `ResolverError` | 159 |
| `entity_store.py` | 服务端唯一事实源：ID 分配（顺序稳定）、跨轮合并（name 为逻辑键）、显著性 `exp(-0.3·轮距)·log1p(提及数)`（按轮，全系统统一单位）、属性保鲜（新值覆盖 + 历史留痕）          | 211 |
| `service.py`      | FastAPI：`POST /chat`（真实状态回灌闭环）、`GET /dialogues/{id}/state`、`DELETE`                                                                                          | 120 |
| `demo.py`         | 规则路径（`RuleResolver`：中文代词表 + 类型过滤 + 显著性加权，阈值 0.6 宁挂起不猜）+ 演示入口                                                                             | 260 |

LLM 每轮输出**完整状态**（而非增量补丁）是当前 DST 的主流做法：天然处理用户修正与话题切换，也把 v1 的「实体 ID 按轮内序号生成导致跨轮碰撞（李四被并进张三）」从机制上排除。

## 快速开始

```bash
# 1) 安装（任意一种）
uv sync                                          # 推荐
python -m pip install openai pydantic fastapi uvicorn httpx "spacy>=3.7,<3.9" pytest pytest-asyncio

# 2) 中文 NER 模型（可选；缺失时规则路径自动降级为纯代词表）
python -m spacy download zh_core_web_sm

# 3) 跑演示：有 Key 走 LLM 路径，无 Key 自动切规则路径
export DEEPSEEK_API_KEY=sk-...
python demo.py
python demo.py --rules        # 强制规则路径，对照两者的消解质量差距

# 4) 起服务
python service.py             # http://127.0.0.1:8600，POST /chat
#   curl -X POST http://127.0.0.1:8600/chat -H 'Content-Type: application/json' \
#     -d '{"dialogue_id":"d1","message":"张三加入了星辰项目，他是后端工程师。"}'

# 5) 测试（27 条断言，全部离线，不调真实 API）
python -m pytest tests/ -v
```

实测对照（v2.0.0，2026-11-14，DEEPSEEK_API_KEY 实跑 5 轮）：

| 轮次 | mention        | 规则路径                | LLM 路径                          |
| ---- | -------------- | ----------------------- | --------------------------------- |
| 1    | 他             | 张三（单候选）          | 张三（主语继承+性别）             |
| 2    | 她             | AMBIGUOUS（两候选 57%） | **李四**（同轮性别绑定）          |
| 3    | 他             | AMBIGUOUS               | **张三**（跨轮性别消歧）          |
| 4    | 那个项目 / 它  | AMBIGUOUS ×2            | **星辰项目** ×2（短语级指代）     |
| 5    | 上次会议说的事 | AMBIGUOUS               | AMBIGUOUS（历史无会议，诚实挂起） |

规则路径 1 对 4 挂起，LLM 路径 8 项消解全对 + 1 项诚实挂起——两条路径的差距就是「没有语义信号」与「有语义」的差距，也是本 demo 的教学核心。

## 与主流记忆框架的选型对照

本项目的实体记忆是**自实现的轻量账本**（211 行），刻意不引入外部服务。如果对话记忆需求超出「实体 + 属性 + 显著性」的范围，再考虑专用框架：

| 方案                                   | 定位                 | 何时选它                                                    |
| -------------------------------------- | -------------------- | ----------------------------------------------------------- |
| 本项目 EntityStore                     | 实体级状态，全量重建 | 教学演示；实体/指代是唯一记忆需求                           |
| [Mem0](https://github.com/mem0ai/mem0) | 通用语义记忆层       | 需要跨会话长期记忆 + 向量检索                               |
| Zep                                    | 时序知识图谱记忆     | 重时效推理（LongMemEval 2025：Zep 63.8 vs Mem0 49.0，厂商口径） |
| Letta                                  | 自带 agent 运行时    | 已在 Letta 生态；其文件系统方案在 LoCoMo 拿到 74%（官方口径）           |
| LangMem                                | LangGraph 内记忆     | 已绑定 LangGraph 时                                         |

生产环境的头号痛点是「信息保鲜」（实体属性随对话更新）而非存取——`EntityStore.apply_state` 的「LLM 全量视图 + 服务端合并校正 + 属性历史留痕」就是对这个问题最小可行的回答。

## 与 v1 的差异（教学点）

| 问题    | v1                                                                                                       | v2                                                                    |
| ------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 实体 ID | 轮内序号生成 → 跨轮碰撞（李四并入张三）                                                                  | store 顺序分配，name 为逻辑键                                         |
| 消解    | `max(salience)` 一行 / 规则打分，性别过滤是死代码                                                        | LLM 语义消解；规则路径保留为可解释对照                                |
| 状态    | API 层每轮伪造（current_turn 恒 1）                                                                      | 真实回灌，turn 单调递增，测试锁定                                     |
| 多余物  | config_management（import 即炸）、logging_and_audit（零调用）、multimodal_coref（随机数）等 6 个幻影模块 | 无：每个模块都被主链路触达                                            |
| 依赖    | 声明 22 个，11 个零 import                                                                               | 6 个运行依赖，全部真实使用                                            |
| 测试    | 「跑通即过」，无一条消解正确性断言                                                                       | 27 条：golden 消解断言、schema 拒绝用例、断链回归、AMBIGUOUS 语义自洽 |
| 文档    | 10+ 处 API 示例与代码不符                                                                                | 本 README 与代码逐字核对过                                            |

## 配套文章

[支持多轮对话指代消解的 ChatBot 系统：架构设计与实现详解](../deep-dives/04_coreference-resolution-dialogue-system.md) —— 架构设计与实现建议；§四 为 v2 架构的展开。
