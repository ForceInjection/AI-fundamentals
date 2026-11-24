# 第 3 讲实验脚本

| 文件                    | 用途                                     | 对应实验 | 依赖                  |
| ----------------------- | ---------------------------------------- | -------- | --------------------- |
| `prefix_cache_probe.py` | 同前缀请求的命中观察（预热 + 计时）    | 实验 1   | openai 客户端（随 vLLM 安装） |

## prefix_cache_probe.py

```bash
# 终端 A(默认开 prefix caching)
vllm serve Qwen/Qwen3-1.7B --enforce-eager
# 终端 B
python3 prefix_cache_probe.py --model Qwen/Qwen3-1.7B
# 对照:关闭 prefix caching 再跑
# 终端 A: vllm serve Qwen/Qwen3-1.7B --enforce-eager --no-enable-prefix-caching
```

脚本行为：约 1.5K 字公共前缀(system)+ 6 个不同问题；先串行发 2 个预热请求
(让公共前缀的完整块写入缓存——HEAD 上哈希在 add_request 时就算好，提交发生
在 allocate_slots 末尾的 cache_blocks)；再逐条计时 6 个请求的 TTFT。

预期现象：命中生效时，计时轮各请求的 prefill 实际计算量只剩「问题段」,
TTFT 相对预热轮明显下降；关闭 prefix caching 后两组接近。

只测 TTFT（拿到首 token 即断开），对服务端压力小；数字带口径(模型变体、
prompt 长度、block_size、是否开启 prefix caching)。
