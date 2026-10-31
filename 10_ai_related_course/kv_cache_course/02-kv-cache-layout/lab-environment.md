# 第 2 讲：实验环境

本讲实验可以只用 CPU 完成，不需要 GPU 和模型权重。

共用环境、依赖版本和代码基线见 [课程实验环境](../lab-environment.md)。根目录的安装命令包含 `flash-attn`，那是端到端 GPU 实验才需要的；本讲走 CPU-only 路径。

## 本讲主脚本

`code/block_table_demo.py` 不依赖 nano-vllm，直接跑：

```bash
cd 10_ai_related_course/kv_cache_course/02-kv-cache-layout
python3 code/block_table_demo.py --self-check
python3 code/block_table_demo.py
python3 code/block_table_demo.py --block-size 4 --seq-len 10 --prefix-tokens 6
```

它打印 token → block → slot、block table、内部碎片、共享前缀的 `ref_count`，以及 A/B 碎片对照。L02/L04 只打印 block table，不打印 `slot_mapping`；需要看 slot 时以本脚本为准，或读 `nanovllm/engine/model_runner.py:153-186`。

## CPU-only 安装 nano-vllm

`BlockManager` 只依赖 `xxhash` 和 `numpy`，不需要 torch、triton、flash-attn。用 `--no-deps` 只安装 nano-vllm 包本身：

```bash
cd ~/Project/study/nano-vllm
pip install -e . --no-deps
pip install xxhash numpy
PYTHONPATH=. python -c "from nanovllm.engine.block_manager import BlockManager; print('ok')"
```

如果 `pip` 版本较旧不支持 `--no-deps`，先升级 pip，不要改成完整安装。`PYTHONPATH=.` 保证从仓库根目录导入 `nanovllm`，即使 editable install 没生效也能跑。

跑 L02/L04：

```bash
cd ~/Project/study/nano-vllm
PYTHONPATH=. python docs/llm-inference-visual/scripts/L02_sequence.py
PYTHONPATH=. python docs/llm-inference-visual/scripts/L04_block_manager.py
```

先 `git rev-parse --short HEAD` 确认 nano-vllm 在 `b9dd55a`，再运行脚本。不同版本的行号可能不同。

## 依赖边界

| 依赖 | 用途 | 本讲是否必需 |
| --- | --- | --- |
| Python 3.10+ | 跑主脚本 | 是 |
| xxhash / numpy | 跑 nano-vllm 的 `BlockManager` | 可选，仅对照源码时 |
| torch | nano-vllm 其余模块 | 否 |
| triton / flash-attn | 端到端 GPU 推理 | 否，第 3 讲起才可能用到 |

`flash-attn` 依赖 PyTorch、CUDA 和 ABI，安装失败时先对齐版本，不要直接改源码。第 1、2 讲不要求安装它。
