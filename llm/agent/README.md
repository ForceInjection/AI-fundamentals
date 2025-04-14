# MCP 智能体演示项目

参考：[**LangChain + 模型上下文协议（MCP）：AI 智能体 Demo**](https://mp.weixin.qq.com/s/D5d3F3xKeqstBataPBVbFA)

## 介绍

MCP 是一个基于模型上下文协议的智能体演示项目，旨在展示如何使用模型上下文协议（MCP）来实现智能体的交互和通信。

### 项目结构

```text
├── README.md
├── client.py
├── client2.py
├── math_server.py
├── math_server2.py
├── requirements.txt
├── setup.sh
├── start_client.sh
└── start_server.sh
```

### 功能说明

- 使用模型上下文协议（MCP）实现智能体的交互和通信
- 支持两种传输模式：`Stdio`模式和`SSE`模式

### 传输模式说明

1. **Stdio模式** - 本地进程间通信
   - 服务端与客户端通过标准输入输出通信
   - 适用于本地开发调试
2. **SSE模式** - 远程`HTTP`通信
   - 基于`Server-Sent Events`的事件流通信
   - 适用于云原生/远程服务场景

## 项目演示

### 1）环境准备

```bash
./setup.sh
```

**注意**：服务端和客户端需要使用同一种 `transport` 模式！！！

### 2）运行 MCP 服务端

```bash
# 启动Stdio模式服务端
./start_server.sh stdio

# 启动SSE模式服务端（默认端口8080）
./start_server.sh sse
```

### 3）运行客户端

```bash
# 配置大模型参数，API_KEY需要到 https://platform.deepseek.com/api_keys 申请（以 deepseek 为例）
export OPENAI_API_BASE=https://api.deepseek.com/v1
export OPENAI_API_KEY=sk-xxxx

# 连接Stdio模式服务端
./start_client.sh stdio

# 连接SSE模式服务端
./start_client.sh sse
```

### 4）输出

#### 4.1）Stdio 模式

`Client` 输出：

```text
{'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='67b97765-81ea-4865-95fe-853adda05454'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_39633e94-865f-4edd-b0f8-62a115d7176f', 'function': {'arguments': '{"a": 3, "b": 5}', 'name': 'add'}, 'type': 'function', 'index': 0}, {'id': 'call_1_7854f3ff-cfc7-43a0-97bb-6528f2031409', 'function': {'arguments': '{"a": 8, "b": 12}', 'name': 'multiply'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 49, 'prompt_tokens': 218, 'total_tokens': 267, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 218}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'd4b4e8ee-6cd8-4da6-b86f-2c12e4d58908', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4cf92f57-8950-462e-86db-941b32dd700b-0', tool_calls=[{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_0_39633e94-865f-4edd-b0f8-62a115d7176f', 'type': 'tool_call'}, {'name': 'multiply', 'args': {'a': 8, 'b': 12}, 'id': 'call_1_7854f3ff-cfc7-43a0-97bb-6528f2031409', 'type': 'tool_call'}], usage_metadata={'input_tokens': 218, 'output_tokens': 49, 'total_tokens': 267, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}), ToolMessage(content='8', name='add', id='52c1fd46-b7c0-4713-95c3-49924b603c14', tool_call_id='call_0_39633e94-865f-4edd-b0f8-62a115d7176f'), ToolMessage(content='96', name='multiply', id='898b422a-5385-49b3-a3b0-45bb5922a9c9', tool_call_id='call_1_7854f3ff-cfc7-43a0-97bb-6528f2031409'), AIMessage(content='The result of \\((3 + 5) \\times 12\\) is \\(96\\).', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 276, 'total_tokens': 294, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 84}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'd0b664d2-6fef-49b4-b18f-199baf432822', 'finish_reason': 'stop', 'logprobs': None}, id='run-2636b93b-13a3-4d8a-9db1-ccdf7382b54b-0', usage_metadata={'input_tokens': 276, 'output_tokens': 18, 'total_tokens': 294, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}})]}
```

#### 4.2）SSE 模式输出

`Server` 输出：

```text
INFO:     Started server process [37445]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     127.0.0.1:62737 - "GET /sse HTTP/1.1" 200 OK
INFO:     127.0.0.1:62739 - "POST /messages/?session_id=b70ff264bb2545149f0efaef6d65ac6d HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:62740 - "POST /messages/?session_id=b70ff264bb2545149f0efaef6d65ac6d HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:62741 - "POST /messages/?session_id=b70ff264bb2545149f0efaef6d65ac6d HTTP/1.1" 202 Accepted
Processing request of type ListToolsRequest
INFO:     127.0.0.1:62806 - "POST /messages/?session_id=b70ff264bb2545149f0efaef6d65ac6d HTTP/1.1" 202 Accepted
Processing request of type CallToolRequest
```

`Client` 输出（开了 `debug` 模式）：

```text
[-1:checkpoint] State at the end of step -1:
{'messages': []}
[0:tasks] Starting 1 task for step 0:
- __start__ -> {'messages': "what's (3 + 5) x 12?"}
[0:writes] Finished step 0 with writes to 1 channel:
- messages -> "what's (3 + 5) x 12?"
[0:checkpoint] State at the end of step 0:
{'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='e3b25678-9085-49d4-85ee-65743211b6f3')]}
[1:tasks] Starting 1 task for step 1:
- agent -> {'is_last_step': False,
 'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='e3b25678-9085-49d4-85ee-65743211b6f3')],
 'remaining_steps': 24}
[1:writes] Finished step 1 with writes to 1 channel:
- messages -> [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'function': {'arguments': '{"a": 3, "b": 5}', 'name': 'add'}, 'type': 'function', 'index': 0}, {'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'function': {'arguments': '{"a": 12, "b": 12}', 'name': 'multiply'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 218, 'total_tokens': 266, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 26}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'cbca7535-abc8-41ac-8182-2fa4427809e5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4a52fe6f-f83f-4628-a299-30f96d76876a-0', tool_calls=[{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'type': 'tool_call'}, {'name': 'multiply', 'args': {'a': 12, 'b': 12}, 'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'type': 'tool_call'}], usage_metadata={'input_tokens': 218, 'output_tokens': 48, 'total_tokens': 266, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}})]
[1:checkpoint] State at the end of step 1:
{'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='e3b25678-9085-49d4-85ee-65743211b6f3'),
              AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'function': {'arguments': '{"a": 3, "b": 5}', 'name': 'add'}, 'type': 'function', 'index': 0}, {'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'function': {'arguments': '{"a": 12, "b": 12}', 'name': 'multiply'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 218, 'total_tokens': 266, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 26}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'cbca7535-abc8-41ac-8182-2fa4427809e5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4a52fe6f-f83f-4628-a299-30f96d76876a-0', tool_calls=[{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'type': 'tool_call'}, {'name': 'multiply', 'args': {'a': 12, 'b': 12}, 'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'type': 'tool_call'}], usage_metadata={'input_tokens': 218, 'output_tokens': 48, 'total_tokens': 266, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}})]}
[2:tasks] Starting 1 task for step 2:
- tools -> {'is_last_step': False,
 'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='e3b25678-9085-49d4-85ee-65743211b6f3'),
              AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'function': {'arguments': '{"a": 3, "b": 5}', 'name': 'add'}, 'type': 'function', 'index': 0}, {'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'function': {'arguments': '{"a": 12, "b": 12}', 'name': 'multiply'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 218, 'total_tokens': 266, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 26}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'cbca7535-abc8-41ac-8182-2fa4427809e5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4a52fe6f-f83f-4628-a299-30f96d76876a-0', tool_calls=[{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'type': 'tool_call'}, {'name': 'multiply', 'args': {'a': 12, 'b': 12}, 'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'type': 'tool_call'}], usage_metadata={'input_tokens': 218, 'output_tokens': 48, 'total_tokens': 266, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}})],
 'remaining_steps': 23}
[2:writes] Finished step 2 with writes to 1 channel:
- messages -> [ToolMessage(content='8', name='add', tool_call_id='call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7'),
 ToolMessage(content='144', name='multiply', tool_call_id='call_1_b1cb086a-3568-441e-a5c0-19783b1c0609')]
[2:checkpoint] State at the end of step 2:
{'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='e3b25678-9085-49d4-85ee-65743211b6f3'),
              AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'function': {'arguments': '{"a": 3, "b": 5}', 'name': 'add'}, 'type': 'function', 'index': 0}, {'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'function': {'arguments': '{"a": 12, "b": 12}', 'name': 'multiply'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 218, 'total_tokens': 266, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 26}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'cbca7535-abc8-41ac-8182-2fa4427809e5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4a52fe6f-f83f-4628-a299-30f96d76876a-0', tool_calls=[{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'type': 'tool_call'}, {'name': 'multiply', 'args': {'a': 12, 'b': 12}, 'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'type': 'tool_call'}], usage_metadata={'input_tokens': 218, 'output_tokens': 48, 'total_tokens': 266, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}}),
              ToolMessage(content='8', name='add', id='3b0c7226-f94e-4894-b02e-99dfd77a1a11', tool_call_id='call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7'),
              ToolMessage(content='144', name='multiply', id='263650f7-1de0-4bc3-a032-9dc386aaae8d', tool_call_id='call_1_b1cb086a-3568-441e-a5c0-19783b1c0609')]}
[3:tasks] Starting 1 task for step 3:
- agent -> {'is_last_step': False,
 'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='e3b25678-9085-49d4-85ee-65743211b6f3'),
              AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'function': {'arguments': '{"a": 3, "b": 5}', 'name': 'add'}, 'type': 'function', 'index': 0}, {'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'function': {'arguments': '{"a": 12, "b": 12}', 'name': 'multiply'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 218, 'total_tokens': 266, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 26}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'cbca7535-abc8-41ac-8182-2fa4427809e5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4a52fe6f-f83f-4628-a299-30f96d76876a-0', tool_calls=[{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'type': 'tool_call'}, {'name': 'multiply', 'args': {'a': 12, 'b': 12}, 'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'type': 'tool_call'}], usage_metadata={'input_tokens': 218, 'output_tokens': 48, 'total_tokens': 266, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}}),
              ToolMessage(content='8', name='add', id='3b0c7226-f94e-4894-b02e-99dfd77a1a11', tool_call_id='call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7'),
              ToolMessage(content='144', name='multiply', id='263650f7-1de0-4bc3-a032-9dc386aaae8d', tool_call_id='call_1_b1cb086a-3568-441e-a5c0-19783b1c0609')],
 'remaining_steps': 22}
[3:writes] Finished step 3 with writes to 1 channel:
- messages -> [AIMessage(content='The calculation is incorrect. Let me re-evaluate it properly.\n\nFirst, \\(3 + 5 = 8\\).\n\nThen, \\(8 \\times 12 = 96\\).\n\nSo, the correct answer is **96**.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 45, 'prompt_tokens': 276, 'total_tokens': 321, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 20}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': '1e6cd941-a5f8-45a9-a6d9-cad3e9236ef6', 'finish_reason': 'stop', 'logprobs': None}, id='run-dc5f9eac-a886-4a3d-977f-9d2bf9b699fc-0', usage_metadata={'input_tokens': 276, 'output_tokens': 45, 'total_tokens': 321, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}})]
[3:checkpoint] State at the end of step 3:
{'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='e3b25678-9085-49d4-85ee-65743211b6f3'),
              AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'function': {'arguments': '{"a": 3, "b": 5}', 'name': 'add'}, 'type': 'function', 'index': 0}, {'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'function': {'arguments': '{"a": 12, "b": 12}', 'name': 'multiply'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 218, 'total_tokens': 266, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 26}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'cbca7535-abc8-41ac-8182-2fa4427809e5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4a52fe6f-f83f-4628-a299-30f96d76876a-0', tool_calls=[{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'type': 'tool_call'}, {'name': 'multiply', 'args': {'a': 12, 'b': 12}, 'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'type': 'tool_call'}], usage_metadata={'input_tokens': 218, 'output_tokens': 48, 'total_tokens': 266, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}}),
              ToolMessage(content='8', name='add', id='3b0c7226-f94e-4894-b02e-99dfd77a1a11', tool_call_id='call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7'),
              ToolMessage(content='144', name='multiply', id='263650f7-1de0-4bc3-a032-9dc386aaae8d', tool_call_id='call_1_b1cb086a-3568-441e-a5c0-19783b1c0609'),
              AIMessage(content='The calculation is incorrect. Let me re-evaluate it properly.\n\nFirst, \\(3 + 5 = 8\\).\n\nThen, \\(8 \\times 12 = 96\\).\n\nSo, the correct answer is **96**.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 45, 'prompt_tokens': 276, 'total_tokens': 321, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 20}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': '1e6cd941-a5f8-45a9-a6d9-cad3e9236ef6', 'finish_reason': 'stop', 'logprobs': None}, id='run-dc5f9eac-a886-4a3d-977f-9d2bf9b699fc-0', usage_metadata={'input_tokens': 276, 'output_tokens': 45, 'total_tokens': 321, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}})]}
{'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='e3b25678-9085-49d4-85ee-65743211b6f3'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'function': {'arguments': '{"a": 3, "b": 5}', 'name': 'add'}, 'type': 'function', 'index': 0}, {'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'function': {'arguments': '{"a": 12, "b": 12}', 'name': 'multiply'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 218, 'total_tokens': 266, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 26}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'cbca7535-abc8-41ac-8182-2fa4427809e5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4a52fe6f-f83f-4628-a299-30f96d76876a-0', tool_calls=[{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7', 'type': 'tool_call'}, {'name': 'multiply', 'args': {'a': 12, 'b': 12}, 'id': 'call_1_b1cb086a-3568-441e-a5c0-19783b1c0609', 'type': 'tool_call'}], usage_metadata={'input_tokens': 218, 'output_tokens': 48, 'total_tokens': 266, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}}), ToolMessage(content='8', name='add', id='3b0c7226-f94e-4894-b02e-99dfd77a1a11', tool_call_id='call_0_5911ffbb-efd2-4753-af04-5841b3dfe1c7'), ToolMessage(content='144', name='multiply', id='263650f7-1de0-4bc3-a032-9dc386aaae8d', tool_call_id='call_1_b1cb086a-3568-441e-a5c0-19783b1c0609'), AIMessage(content='The calculation is incorrect. Let me re-evaluate it properly.\n\nFirst, \\(3 + 5 = 8\\).\n\nThen, \\(8 \\times 12 = 96\\).\n\nSo, the correct answer is **96**.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 45, 'prompt_tokens': 276, 'total_tokens': 321, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 20}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': '1e6cd941-a5f8-45a9-a6d9-cad3e9236ef6', 'finish_reason': 'stop', 'logprobs': None}, id='run-dc5f9eac-a886-4a3d-977f-9d2bf9b699fc-0', usage_metadata={'input_tokens': 276, 'output_tokens': 45, 'total_tokens': 321, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}})]}
```