# 意图检测（Intent Detection）

意图检测是对话系统与智能客服的入口级任务：从用户的自然语言输入中识别出其真实意图（查询订单、投诉、闲聊等）。本目录收录两篇文档，从不同角度覆盖 LLM 时代的意图检测实践。

## 1. 核心文档

- **[基于 LLM 的意图检测](intent_detection_using_llm.zh-CN.md)** — 通用方法论：如何用 LLM 构建可靠的意图识别管线，覆盖 prompt 设计、多意图/模糊意图处理、误判归因与工程化落地中的常见陷阱。
- **[ChatBox 意图识别与语义理解](chatbox_intent_recognition_and_semantic_understanding.md)** — 以 ChatBox 为具体场景，深入语义理解层面，演示从用户偏好提取到多轮澄清的完整链路。

## 2. 阅读建议

1. 先看 **[基于 LLM 的意图检测](intent_detection_using_llm.zh-CN.md)**，建立对 LLM 意图识别的通用方法论认知。
2. 再对照 **[ChatBox 意图识别与语义理解](chatbox_intent_recognition_and_semantic_understanding.md)**，看一个具体场景下的落地实践。

## 3. 相关资源

- [LLM 基础概念总览](../README.md)
- [工作流编排与应用平台](../../workflow/README.md)
- [智能体系统架构](../../../08_agentic_system/README.md)
