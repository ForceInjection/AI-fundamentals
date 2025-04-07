# 说明

参考：[LangChain + 模型上下文协议（MCP）：AI 智能体 Demo](https://mp.weixin.qq.com/s/D5d3F3xKeqstBataPBVbFA)

## 环境准备

```bash
./setup.sh
```

## 运行 MCP 服务端

```bash
./start_server.sh
```

## 运行客户端

```bash

export OPENAI_API_BASE=https://api.deepseek.com/v1
export OPENAI_API_KEY=sk-xxxx

./start_client.sh
```

## 输出

```text
{'messages': [HumanMessage(content="what's (3 + 5) x 12?", additional_kwargs={}, response_metadata={}, id='67b97765-81ea-4865-95fe-853adda05454'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_0_39633e94-865f-4edd-b0f8-62a115d7176f', 'function': {'arguments': '{"a": 3, "b": 5}', 'name': 'add'}, 'type': 'function', 'index': 0}, {'id': 'call_1_7854f3ff-cfc7-43a0-97bb-6528f2031409', 'function': {'arguments': '{"a": 8, "b": 12}', 'name': 'multiply'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 49, 'prompt_tokens': 218, 'total_tokens': 267, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 218}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'd4b4e8ee-6cd8-4da6-b86f-2c12e4d58908', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4cf92f57-8950-462e-86db-941b32dd700b-0', tool_calls=[{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_0_39633e94-865f-4edd-b0f8-62a115d7176f', 'type': 'tool_call'}, {'name': 'multiply', 'args': {'a': 8, 'b': 12}, 'id': 'call_1_7854f3ff-cfc7-43a0-97bb-6528f2031409', 'type': 'tool_call'}], usage_metadata={'input_tokens': 218, 'output_tokens': 49, 'total_tokens': 267, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}), ToolMessage(content='8', name='add', id='52c1fd46-b7c0-4713-95c3-49924b603c14', tool_call_id='call_0_39633e94-865f-4edd-b0f8-62a115d7176f'), ToolMessage(content='96', name='multiply', id='898b422a-5385-49b3-a3b0-45bb5922a9c9', tool_call_id='call_1_7854f3ff-cfc7-43a0-97bb-6528f2031409'), AIMessage(content='The result of \\((3 + 5) \\times 12\\) is \\(96\\).', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 276, 'total_tokens': 294, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 84}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_3d5141a69a_prod0225', 'id': 'd0b664d2-6fef-49b4-b18f-199baf432822', 'finish_reason': 'stop', 'logprobs': None}, id='run-2636b93b-13a3-4d8a-9db1-ccdf7382b54b-0', usage_metadata={'input_tokens': 276, 'output_tokens': 18, 'total_tokens': 294, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}})]}
```