# KV Cache 八讲 — 实验环境

课程分两条实验轨：CPU 轨覆盖公式、分页、调度元数据和缓存模拟；GPU 轨只用于真实模型推理和 SGLang/vLLM 运行日志。没有 GPU 也能完成每讲的必做部分。

## 硬件与软件

| 项目     | CPU 轨                              | GPU 轨                                       |
| -------- | ----------------------------------- | -------------------------------------------- |
| 操作系统 | macOS 或 Linux                      | Linux                                        |
| Python   | 3.10–3.12（nano-vllm 要求 `<3.13`） | 3.10–3.12                                    |
| PyTorch  | 不要求                              | `>=2.4.0`，与 CUDA 对齐                      |
| CUDA     | 不要求                              | 与 PyTorch、flash-attn 对齐                  |
| 内存     | 8 GB+                               | 32 GB+                                       |
| 显存     | 不要求                              | 建议 24 GB+；跑 GLM-5.3-Flash 按模型规模准备 |
| 磁盘     | 5 GB+                               | 50 GB+（模型权重和日志）                     |

## 代码基线

| 仓库            | 本地路径                           | 版本           |
| --------------- | ---------------------------------- | -------------- |
| nano-vllm       | `~/Project/study/nano-vllm`        | `b9dd55a`      |
| vLLM            | `~/Project/ai-infra/vLLM`          | `8b365ff949`   |
| SGLang          | `~/Project/ai-infra/sglang`        | `f4de6abee6`   |
| ai-infra-course | `~/Project/study/ai-infra-course`  | 作为前置材料   |
| 本仓库          | `~/Project/wechat/AI-fundamentals` | 课程资料与文章 |

源码课统一按上述 commit 讲。不同版本的行号会漂移，实验前先 `git rev-parse --short HEAD` 核对。

## CPU 轨

### nano-vllm 脚本

L02 / L04 不需要 torch、flash-attn，也不需要模型权重：

```bash
cd ~/Project/study/nano-vllm
PYTHONPATH=. python docs/llm-inference-visual/scripts/L02_sequence.py
PYTHONPATH=. python docs/llm-inference-visual/scripts/L04_block_manager.py
```

如果只想跑这两个脚本，不必执行会拉取 GPU 依赖的完整安装。需要 `xxhash` 和 `numpy` 时：

```bash
pip install xxhash numpy
```

### L03 调度实验

L03 只构造 `Config` / `Scheduler` / `Sequence`，不加载模型权重；需要一个 Qwen3-0.6B 的 config 目录：

```bash
export NANOVLLM_MODEL_PATH=/path/to/Qwen3-0.6B
PYTHONPATH=. python docs/llm-inference-visual/scripts/L03_scheduler.py "$NANOVLLM_MODEL_PATH"
```

`L03_scheduler.py` 没有单场景开关，E/E2 随全脚本运行；它输出状态、块映射和计数器，不生成 token。

### 课程自带脚本

| 讲次    | 脚本                                                        | 用途                                                 |
| ------- | ----------------------------------------------------------- | ---------------------------------------------------- |
| 第 1 讲 | `01-kv-cache-basics/code/kv_estimator.py`                   | 正确的 Qwen3-0.6B / Qwen2.5-7B / Qwen2.5-72B KV 估算 |
| 第 2 讲 | `02-kv-cache-layout/code/block_table_demo.py`               | token → block → slot、内部碎片、共享前缀             |
| 第 5 讲 | `05-kv-cache-reuse/code/prefix_cache_sim.py`                | 共享前缀命中、lock_ref、驱逐的纯 Python 模拟         |
| 第 8 讲 | `08-kv-capacity-planning/code/capacity_plan.py`             | 容量规划算例（GLM / DeepSeek / nano-vllm 三条轨）    |

## GPU 轨

nano-vllm 端到端实验需要 CUDA 和模型权重：

```bash
cd ~/Project/study/nano-vllm
pip install -e ".[test]"
python -m pytest tests/ -v
```

注意：

- `flash-attn` 依赖 PyTorch / CUDA / ABI，安装失败时先对齐版本，不要直接改源码。
- 一个进程里多次创建 `LLM` 不会释放 KV 显存；GPU 实验用子进程隔离。
- `run_all.sh` 的默认 CPU 路径仍需要 `NANOVLLM_MODEL_PATH`；不要把它当成完全无依赖脚本。
- vLLM / SGLang 按各自官方安装说明准备；课程仓库不复制它们的源码。

## 每讲实验轨

| 讲次 | 必做（CPU）                         | 选做（GPU）                              |
| ---- | ----------------------------------- | ---------------------------------------- |
| 1    | KV 估算器与容量表                   | 无                                       |
| 2    | block table 模拟、内部碎片          | 无                                       |
| 3    | L03 调度元数据、CPU 测试            | `tests/verify_swap.py` 检查 token 一致性 |
| 4    | 读 vLLM 源码、看示例日志字段        | 小模型请求 + instrumentation             |
| 5    | prefix cache 模拟 / SGLang CPU 单测 | SGLang server + HiCache 指标             |
| 6    | 公式复算、源码阅读                  | SGLang 启动 GLM-5.3-Flash、看启动日志    |
| 7    | 公式复算、源码阅读                  | vLLM 示例日志（如有）                    |
| 8    | 容量算例 + 示例日志                 | 真实启动日志验证                         |

## 开课前 10 分钟自检

1. `python3 --version` 在 3.10–3.12。
2. `git -C ~/Project/study/nano-vllm rev-parse --short HEAD` 输出 `b9dd55a`。
3. 能跑 `PYTHONPATH=. python docs/llm-inference-visual/scripts/L02_sequence.py`。
4. 能跑课程第 1 讲的 `code/kv_estimator.py`。
5. 有 GPU 时，`python -c "import torch; print(torch.cuda.is_available())"` 为 True。

## 复用的既有材料

- `~/Project/study/ai-infra-course/06-kvcache-optimization/visuals/`：公式、张量、精度、offloading 四个 HTML。
- `~/Project/study/ai-infra-course/06-kvcache-optimization/code/calculate_qwen3_memory.py`：只作对照；Qwen3-0.6B 的 `head_dim` 以课程 `kv_estimator.py` 的 128 为准。
- `~/Project/study/ai-infra-course/05-vllm-inference/code/trace_nanovllm.py`：第 3 讲的 pre-swap 参考，不跟踪 `swap_out/swap_in`。
- `ai-infra-course` 的 V4-Pro 估算脚本不能用于 V4.1-Flash；第 7 讲使用 `02-deepseek-v41-flash.md` 的 890 B/token 口径。
