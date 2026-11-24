# 第 5 讲实验脚本

| 文件                    | 用途                                                | 对应实验 | 依赖         |
| ----------------------- | --------------------------------------------------- | -------- | ------------ |
| `capture_log_parser.py` | 解析 serve 日志中的 capture 耗时/显存/模式行        | 实验 2   | 仅标准库     |

另复用第 2 讲 `mixed_load_probe.py`（`cp ../02-scheduler/code/mixed_load_probe.py .`）做三态 TPOT 对照（实验 1）。

## capture_log_parser.py

```bash
vllm serve Qwen/Qwen3-1.7B 2>&1 | tee /tmp/vllm-capture.log
python3 capture_log_parser.py /tmp/vllm-capture.log
```

输出：capture 总耗时/显存(「Graph capturing finished in X secs, took Y GiB」,
官方注通常 5~20 秒)、图模式、后端选择行、编译完成行；无匹配时提示排查方向
（是否误加 --enforce-eager）。

正则按关键词匹配，行号随版本浮动属预期——脚本找的是「事实」，行号在讲义里。
