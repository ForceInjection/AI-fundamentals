# 共用实验环境

所有讲次的实验共用本环境。每讲的特有依赖写在各讲 `lab-environment.md`，共用部分只看这里。

## 源码仓（钉 commit）

| 仓库      | 本地路径                    | 基线 commit                              | 角色                |
| --------- | --------------------------- | ---------------------------------------- | ------------------- |
| nano-vllm | `~/Project/study/nano-vllm` | `b9dd55a`（先修课同款）                    | 每讲第 1 层回顾对照 |
| vLLM      | `~/Project/ai-infra/vLLM`   | `1acec5c009`(v0.34.0rc0-1153,2026-11-23) | 现状走读与实验      |

- vLLM 基线随课程推进重新钉；换 commit 时须重核课上全部 `file:line`，并在本表更新。
- 课程写作期发现旧文章（锚点 `8b365ff949`、`43d691ec6b` 等）与本基线的行号代差，属于预期；旧文章在本课程中只作演进脉络素材，不作为现状依据。

## Python 环境

- Python 3.10–3.12（nano-vllm 要求 `<3.13`；vLLM 官方 wheel 按 3.10–3.12 发布）。
- 建议两个独立 venv:nano-vllm 一个（依赖少，torch + triton + flash-attn）,vLLM 一个（直接 `pip install vllm` 安装稳定版，源码实验用 `-e` 安装本地仓）。
- 源码安装 vLLM 本地仓：

```bash
cd ~/Project/ai-infra/vLLM
git checkout 1acec5c009
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128  # 按本机 CUDA 调整
```

## 硬件分层

| 层级   | 要求                                      | 覆盖实验                                                              |
| ------ | ----------------------------------------- | --------------------------------------------------------------------- |
| 纯 CPU | 任意现代笔记本                            | 架构走读、进程观察、调度模拟、配置解析                                |
| 单 GPU | ≥ 8GB 显存（参考机型 RTX 3090/4070 Laptop） | 小模型部署与压测（Qwen3-0.6B/4B 级）、CUDA Graph、投机解码、CPU offload |
| 多 GPU | ≥ 2 卡                                    | TP=2、EP、PD 分离（第 6、9 讲部分实验）                                 |

GPU 不是全程必需：每讲的源码走读与纯 Python 演示不需要 GPU；标注「端到端」的实验才需要。

## 模型权重

课程统一使用小型模型控制下载量，生产级实验（第 10 讲）按需另下：

| 模型                             | 用途                                 |
| -------------------------------- | ------------------------------------ |
| Qwen3-0.6B                       | 默认实验模型，与 nano-vllm 先修课一致 |
| Qwen3-4B                         | 投机解码(MTP)、量化实验              |
| DeepSeek 蒸馏系或 Qwen3 MoE 小杯 | 第 6 讲 EP 实验（按显存选）            |

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ --local-dir-use-symlinks False
```

## 外部课程资源

- DeepLearning.AI 短课「Fast & Efficient LLM Inference with vLLM」（免费，官方背书）：第 8、10 讲的上手材料，与本课实验互补。
- docs.vllm.ai（stable,2026-11-06 版）：现状机制的权威校对；本课程 `file:line` 与文档冲突时，先确认文档对应的 vLLM 版本再判断。

## 口径约定

- 容量与性能数字同时写清：模型变体、精度、硬件、统计范围、`--max-num-len` 等关键参数。
- 吞吐与延迟数字分清「本机实测」与「官方口径」；官方口径一律标注。
- `MB` 按 `10^6 B` 计；换算 MiB 另标。
