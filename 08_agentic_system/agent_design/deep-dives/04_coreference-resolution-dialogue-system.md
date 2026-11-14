# 支持多轮对话指代消解的 ChatBot 系统：架构设计与实现详解

> 本文以配套 demo [coref-dialogue-system v2](../coref-dialogue-system/README.md)（pydantic v2 + OpenAI 兼容接口，源码与测试约 1300 行）为标本，2026-11-14 核对。LLM 路径经 DeepSeek API 实跑验证（5 轮对话），代码引用以 `文件:行号` 格式标注。

---

## 一、多轮对话死于一串代词

看一段真实的客服对话：

> 用户：张三加入了星辰项目，他是后端工程师。
> 用户：李四也在星辰项目，她负责前端。
> 用户：他的第一个任务是重构登录模块。
> 用户：那个项目月底要上线，它的负责人是谁？

第三轮的「他」是谁？第四轮的「那个项目」和「它」又指什么？单看每一句都语法完整，合起来却全是悬空引用。系统如果不能把「他」绑回张三、把「它」绑回星辰项目，从第三轮起回答就开始张冠李戴，对话在用户眼里已经死了，只是还没退出。

这就是多轮指代消解（coreference resolution）要解的问题：把代词、省略主语、模糊短语（「那个项目」「上次说的事」）正确绑定到对话历史中的实体上。它难在三个地方：

1. **跨轮依赖**：先行语可能在很多轮之前，且中间穿插着干扰实体
2. **语义信号**：中文「他/她/它」携带性别与有生性线索，「那个项目」则需要短语级语义匹配
3. **状态管理**：实体随对话增长、属性随对话更新（张三从「后端工程师」变成「后端负责人」），过期的状态比没有状态更危险

## 二、经典管线为什么在中文上走不通

指代消解的经典做法是两步管线：NER 提取实体，消解模型建立指代链——从 [End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045)（Lee et al., 2017）开山，到 [SpanBERT](https://arxiv.org/abs/1907.10529)（Joshi et al., 2020）把这条路线的精度推到高峰。这条路线在英文上工具成熟，在中文上却是一堵墙：

- [spaCy coreferee](https://github.com/richardpaulhudson/coreferee) 活跃维护（支持 Python 3.10-3.13 与 spaCy 3.7-3.8），但语言只有英、法、德、波，没有中文模型
- [fastcoref / LingMessCoref](https://github.com/shon-otmazgin/fastcoref)（[F-COREF 论文](https://arxiv.org/abs/2209.04280)实现，比 AllenNLP 快约 29 倍）同样只支持英文
- 中文指代消解的公开基准（CoNLL-2012 中文部分）与可用推理工具之间存在断层，自行训练的门槛把大多数工程团队挡在外面

本项目的 v1 版本正是在这堵墙上撞过之后写成的：spaCy 做中文 NER（这部分可行），指代消解用规则打分 + 显著性增量更新自行拼装。交付后经代码评审，暴露的问题比「效果不好」严重得多，值得原文照录，因为它们不是实现疏忽，而是增量管线的结构性风险：

**实体 ID 跨轮碰撞**。ID 按轮内序号生成（`entity_0`、`entity_1`…），第二轮回指「他」时新建的记录拿到 `entity_0`，与首轮「张三」相同；第四轮新实体「李四」也被并进 `entity_0`。实跑结果：李四从账本里消失，张三「提及 5 次」（实际只出现 1 次）。ID 分配这个动作发生在「轮内视角」里，看不到全局，碰撞是必然。

**状态闭环断裂**。API 层每个请求伪造一份状态：轮次恒为 1、显著性恒为 0.5，真实状态从不回灌消解器。多轮对话在服务层从未真正发生，系统只是把每个请求当第一轮处理。

**假能力模块**。「多模态指代消解」模块的跨模态匹配是对随机向量算余弦；「批处理优化」建了线程池后全部用随机数模拟；10 个模块里有 2 个从未被任何调用方引用（其一在 pydantic v2 下 import 即抛异常），被引用的模块里方法级死代码也随处可见。

测试套件 4/4 通过。但没有一条断言「他」应该解析到哪个实体：套件度量的是「不崩溃」，不是「解得对」。

## 三、v2 方案：LLM 结构化输出 + 状态全量重建

转向的两个依据，一个来自工具现实，一个来自实践共识：

- 中文指代消解没有可用的经典模型，LLM prompting 是现实路径（CRAC 2025 共享任务已设 LLM track）
- 对话状态跟踪（DST）的主流做法已经从增量更新转向**全量重建**：让 LLM 从对话历史重新生成完整状态，天然处理用户的修正、撤销与话题切换（参见 [Towards LLM-driven Dialogue State Tracking](https://arxiv.org/abs/2310.14970)，EMNLP 2023；[Enhancing DST Models through LLM](https://arxiv.org/html/2405.13037v1)）

### 3.1 架构总览：三个角色，两条入口

核心是三个角色：契约层（schemas）管不变量，消解器（resolver）管理解，账本（entity_store）管状态；service 与 demo 是两条入口，前者带 API 服务，后者是可独立运行的演示与规则对照路径。

```text
用户消息 ──► 对话历史（全量）──► LLMResolver ──► DialogueState（pydantic 校验）
                     ▲                │  失败/无Key        │
                     │                ▼                   ▼
              entity_summary     RuleResolver      EntityStore.apply_state
              （实体速览回灌）    （规则对照路径）          │
                                                        ▼
                                          响应（消解结论 + 实体账本）
```

| 模块              | 职责                                                                            |
| ----------------- | ------------------------------------------------------------------------------- |
| `schemas.py`      | pydantic 契约：`DialogueState` / `EntityRecord` / `Resolution`                  |
| `resolver.py`     | LLM 消解：OpenAI 兼容接口（默认 DeepSeek，base_url 可配）+ JSON 约束 + 校验重试 |
| `entity_store.py` | 服务端唯一事实源：ID 分配、跨轮合并、显著性、属性保鲜                           |
| `service.py`      | FastAPI `/chat`：真实状态回灌的多轮闭环                                         |
| `demo.py`         | 规则对照路径（无 Key 可跑）+ 演示入口                                           |

### 3.2 契约层：把领域不变量压进类型层

LLM 的输出不可信，边界处必须把关。v2 把领域不变量写成 pydantic 校验器，非法状态在进入系统之前就被拒绝：

```python
class EntityRecord(BaseModel):
    name: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def name_not_pronoun(cls, v: str) -> str:
        """代词不能成为实体本身——v1 的经典 bug：把「他」存成实体后它成了自己的先行语。"""
        pronouns = {"他", "她", "它", "他们", "她们", "它们", "这个", "那个", "上次那个"}
        if v.strip() in pronouns:
            raise ValueError(f"代词不能作为实体名: {v}")
        return v.strip()
```

`Resolution.resolved_to` 的取值约定是全系统的行为准则：实体名（回指成功）、`"NEW"`（引入新实体）、`"AMBIGUOUS"`（无法确定）。**宁可挂起也不猜**，错误消解对多轮对话的伤害比不消解大。这个约定同时写进 LLM 的 prompt 和规则路径的阈值判定。

### 3.3 消解器：schema 进 prompt，把关在代码，失败回喂

`resolver.py` 的请求侧把 JSON Schema 写进 system prompt，响应强制 JSON（`response_format: json_object`，任何 OpenAI 兼容端点可用，不依赖厂商受限解码）。真正的把关是 pydantic 校验：解析失败或校验失败时，把错误信息拼回对话重新请求，最多 2 次：

```python
except ValueError as e:          # pydantic ValidationError 的基类链
    last_err = e
messages.append({"role": "assistant", "content": raw})
messages.append({"role": "user",
                 "content": f"上面的输出有问题：{last_err}。请严格按 JSON Schema 重新输出…"})
```

重试耗尽抛 `ResolverError`，调用方降级到规则路径。LLM 与规则是并列的两条腿，不是绑定。

### 3.4 实体账本：LLM 出视图，服务端记账

全量重建的输出可能自相矛盾（计数漂移、轮次回退），直接采信会让状态随轮次漂移。`entity_store.py` 的分工是：LLM 出「当前视图」，store 负责记账校正——ID 顺序分配（`ent_0001`…），实体按 `name+type` 逻辑键合并，别名在回指时归并，`mention_count` 与轮次以本地观测为准。

显著性公式全系统统一按「轮」衰减（v1 一处按秒一处按轮，同名机制两种物理含义）：

```python
def salience(self, current_turn: int, decay: float = 0.3) -> float:
    gap = current_turn - max(self.mention_turns)
    return math.exp(-decay * gap) * math.log1p(len(self.mention_turns))
```

属性保鲜是这个模块最有生产价值的部分。agent 记忆系统的头号痛点是「信息随对话过期」而非存取——张三的角色从「后端工程师」变成「后端负责人」时，旧值必须被替换而不是叠加。v2 的做法：LLM 每轮输出最新属性视图，store 覆盖旧值并把历史写进 `attribute_history` 留痕。

### 3.5 服务化：闭环必须硬性成立

`service.py` 的 `/chat` 维持一条硬闭环：消息追加进全量历史 → 历史与实体速览一起交给消解器 → 输出经账本合入 → 下一轮自动携带更新后的历史与 `entity_summary`。轮次由历史中的用户消息计数，单调递增。v1 那种「每请求伪造 current_turn=1」的代码在这个结构里写不出来，并且有一条专门的回归测试锁住它。

## 四、实测：规则路径与 LLM 路径的对照

规则路径保留 v1 的骨架（中文代词表、类型过滤、显著性加权），修正了置信度公式与阈值（0.6：两个候选 50.2% 对 49.8% 时押注是赌博，宁可挂起）。五轮对话实测（2026-11-14，DeepSeek API 实跑）：

| 轮次 | mention             | 规则路径                | LLM 路径                                 |
| ---- | ------------------- | ----------------------- | ---------------------------------------- |
| 1    | 他                  | 张三（单候选）          | 张三（主语继承 + 性别）                  |
| 2    | 她                  | AMBIGUOUS（两候选 57%） | 李四（同轮性别绑定）                     |
| 3    | 他                  | AMBIGUOUS               | 张三（跨轮性别消歧）                     |
| 4    | 那个项目 / 它       | AMBIGUOUS ×2            | 星辰项目 ×2（短语级指代）                |
| 5    | 她 / 上次会议说的事 | AMBIGUOUS               | 李四 / AMBIGUOUS（历史无会议，诚实挂起） |

规则路径 1 对 4 挂起，LLM 路径 9 项消解中 8 项全对、1 项诚实挂起（表格每轮只列代表性 mention，第 2 轮 LLM 实际还输出了「李四」新实体注册与「星辰项目」回指）。两条路径的差距就是「没有语义信号」与「有语义」的差距，这也是保留规则路径的理由：它让「LLM 到底贡献了什么」变成可对照的事实。

第 5 轮的「上次会议说的事」值得单独说：对话历史里确实没有会议内容，LLM 返回 `AMBIGUOUS`（confidence 0.2）。消解器的价值不只是解对，还包括在解不对的时候说出来。

测试围绕三类断言组织（27 条，全部离线，LLM 调用走 stub）：golden 消解断言（「他」必须解析到「张三」）、schema 拒绝用例（代词做实体名等各一条）、v1 断链的回归测试（两轮之后 `GET /state` 的 turn 必须为 2）。

## 五、边界与选型

**这套方案的适用边界**：实体与指代是唯一记忆需求时，自实现的轻量账本（约 210 行）够用。记忆需求超出「实体 + 属性 + 显著性」（跨会话长期记忆、向量检索、时序推理）时，考虑专用框架：[Mem0](https://github.com/mem0ai/mem0)（[论文](https://arxiv.org/abs/2504.19413)，通用语义记忆层）、Zep（时序知识图谱）、[Letta](https://github.com/letta-ai/letta)（自带运行时，前身 [MemGPT](https://arxiv.org/abs/2310.08560)）。公开基准有 [LongMemEval](https://arxiv.org/abs/2410.10813)（长期交互记忆）与 [LoCoMo](https://arxiv.org/abs/2402.17753)（超长对话记忆）；各家公布的分数（Zep 63.8 对 Mem0 49.0、Letta 74%）均为厂商口径，选型前建议以自己场景复测。框架选型对照详见 demo README。

**成本与延迟**：每轮一次 LLM 调用，输入为全量历史加实体速览，随对话线性增长。长对话的成本控制方向是历史的滚动摘要 + 实体速览回灌（本 demo 已实现后者），这是下一版的自然扩展点。

**诚实挂起的代价**：AMBIGUOUS 策略把错误转移给了交互层——系统需要追问「你说的是哪个项目？」。这要求产品侧有承接澄清话术的设计，纯静默场景（日志分析、离线批处理）需要另行评估挂起率。

## 六、总评

**这套 v2 方案解决了什么**：中文多轮指代消解在经典工具链缺位的前提下，用「LLM 结构化输出 + 状态全量重建 + 服务端记账」达到了生产可用——5 轮实测 8 项消解全对，且在无法消解时诚实挂起。架构上每个模块都被主链路触达，27 条测试全部是对行为的断言。

**它没有解决什么**（折扣清单）：

- 长对话（几十轮以上）的全量重建成本未处理，滚动摘要是未实现的扩展点
- 实体消歧只到「对话内实体」，没有知识库链接（张三是哪个部门的张三）
- 多模态指代（「图里这个人」）无覆盖；v1 的多模态模块是对随机向量算余弦的假实现，重构时移除
- LLM 路径的效果依赖模型本身，换弱模型或超长历史时消解质量会回落，规则路径的挂起率会进一步上升
- 5 轮实测样本太小，不足以给出准确率数字；基准评估（如中文指代消解公开测试集）是未做项

**何时不用这套方案**：单轮问答无指代需求时，这是纯开销；已有 Mem0/Zep/Letta 落地的团队，实体账本的功能是其子集，应该复用框架而非自建。

## 源文件索引

demo 仓库 coref-dialogue-system v2.0.0（2026-11-14 核对）：

| 文件              | 关键内容                                    | 引用点                                |
| ----------------- | ------------------------------------------- | ------------------------------------- |
| `schemas.py`      | 契约与不变量：代词禁入实体名、轮次单调      | `schemas.py:52-63`                    |
| `resolver.py`     | schema 进 prompt、校验回喂重试、可降级      | `resolver.py:120-143`                 |
| `entity_store.py` | ID 分配、显著性（按轮衰减）、属性保鲜       | `entity_store.py:34`、`apply_state`   |
| `service.py`      | `/chat` 真实回灌闭环                        | `_rule_fallback` 与 `chat`            |
| `demo.py`         | 规则路径：代词表、类型过滤、0.6 阈值        | `resolve_mentions`                    |
| `tests/`          | golden 断言、schema 拒绝、断链回归（27 条） | `test_resolver.py`、`test_service.py` |

## 参考资料

**工具**：

- [spaCy coreferee](https://github.com/richardpaulhudson/coreferee) —— 活跃维护的 spaCy 指代消解组件，无中文模型
- [fastcoref](https://github.com/shon-otmazgin/fastcoref) —— 高速英文指代消解（F-COREF / LingMess 双模型）

**论文**：

- [End-to-end Neural Coreference Resolution（arXiv:1707.07045）](https://arxiv.org/abs/1707.07045) —— 端到端神经指代消解的开山之作（Lee et al., 2017）
- [SpanBERT（arXiv:1907.10529）](https://arxiv.org/abs/1907.10529) —— 指代消解与抽取任务的强基座（Joshi et al., 2020）
- [F-coref: Fast, Accurate and Easy to Use Coreference Resolution（arXiv:2209.04280）](https://arxiv.org/abs/2209.04280) —— 高速英文指代消解（Otmazgin et al., 2022）
- [Towards LLM-driven Dialogue State Tracking（arXiv:2310.14970，EMNLP 2023）](https://arxiv.org/abs/2310.14970) —— LLM 驱动 DST 的 LDST 框架
- [Enhancing Dialogue State Tracking Models through LLM（arXiv:2405.13037）](https://arxiv.org/html/2405.13037v1) —— LLM 增强 DST
- [MemGPT: Towards LLMs as Operating Systems（arXiv:2310.08560）](https://arxiv.org/abs/2310.08560) —— 记忆分层思想源头，Letta 前身
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory（arXiv:2504.19413）](https://arxiv.org/abs/2504.19413) —— 生产级 agent 长期记忆层

**基准与任务**：

- [LongMemEval（arXiv:2410.10813）](https://arxiv.org/abs/2410.10813) —— 长期交互记忆基准
- [LoCoMo（arXiv:2402.17753）](https://arxiv.org/abs/2402.17753) —— 超长对话记忆基准（ACL 2024）
- CRAC 2025 Shared Task on Multilingual Coreference Resolution（CODI-CRAC 2025 workshop）—— 本届新设 LLM track
