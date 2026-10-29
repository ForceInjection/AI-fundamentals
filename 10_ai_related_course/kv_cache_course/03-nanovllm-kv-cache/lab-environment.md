# 第 3 讲：实验环境

L03 的 §1–§7 只需要一个模型 config 目录和 `transformers`，不加载权重；GPU 只在真实模型推理时需要。

共用环境、依赖版本和代码基线见 [课程实验环境](../lab-environment.md)。

## 本讲额外要求

- nano-vllm 位于 `~/Project/study/nano-vllm`，HEAD `b9dd55a`；
- 准备一个 Qwen3-0.6B 的 config 目录，即包含 `config.json` 的目录。`NANOVLLM_MODEL_PATH` 指向这个目录；本讲说的“模型路径”和“config 目录”是同一个位置；
- L03 的 §1–§6 是纯元数据，§7 用 `Config` 读取 `config.json` 并构造真实的 `Scheduler`，两段都不启动模型、不读权重；
- 可选 GPU 实验 `tests/verify_swap.py` 才需要模型权重和 CUDA，用来检查 swap 前后 token 是否一致；
- 一个进程里多次创建 `LLM` 不会释放 KV 显存，GPU 实验用子进程隔离；
- `run_all.sh` 的默认 CPU 路径仍需要 `NANOVLLM_MODEL_PATH`，不要把它当成完全无依赖脚本。

## 依赖

- Python 3.10–3.12；
- `transformers`（读取 config）；
- `xxhash`、`numpy`（BlockManager 的哈希计算）；
- CPU 测试基线：`python -m pytest tests/ -v` 当前 43 个通过；换版本后以全部通过为准。

不需要 `torch` / `flash-attn` 就能完成 L03 的元数据部分。
