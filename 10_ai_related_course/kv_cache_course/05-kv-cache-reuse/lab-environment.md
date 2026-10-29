# 第 5 讲：实验环境

实验分两段：前缀匹配与驱逐的模拟只需要 CPU；HiCache 的三层命中和强制驱逐需要可用的 SGLang 环境与 GPU。共用环境、依赖版本和代码基线见[课程实验环境](../lab-environment.md)。

## CPU 轨

- Python 3.10–3.12，只用标准库，不需要 torch 或模型权重。
- 脚本：`code/prefix_cache_sim.py`。
- 入口：

```bash
cd 10_ai_related_course/kv_cache_course/05-kv-cache-reuse
python3 code/prefix_cache_sim.py --demo abcd
python3 code/prefix_cache_sim.py --demo share
python3 code/prefix_cache_sim.py --demo evict
python3 code/prefix_cache_sim.py --self-check
```

`--self-check` 应该输出“self-check 通过”。如果失败，先看 Python 版本，再看脚本是否被改动。

SGLang 的官方单测是可选对照，需要先装好 SGLang 依赖：

```bash
cd ~/Project/ai-infra/sglang
python -m pytest test/registered/unit/mem_cache/test_radix_cache_unit.py -k "lock_ref or evict" -v
```

这组单测跑的是经典 `RadixCache`，HEAD 默认走 `UnifiedRadixCache`。两者的锁都沿路径累加，但字段和节点结构不同，不要把单测里的 `TreeNode` 当成默认实现的节点。

## GPU 轨

| 项目 | 要求 |
| --- | --- |
| SGLang | 本地 `~/Project/ai-infra/sglang`，HEAD `f4de6abee6` |
| 模型 | `Qwen/Qwen3-0.6B`，FP16，约 1.5 GB 权重 |
| 显存 | 24 GB 以上，给 KV 池留 8 GB 左右 |
| 主机内存 | 32 GB 以上；`--hicache-ratio 2` 会让 host pool 达到 device pool 的两倍 |
| 磁盘 | 5 GB 以上，L3 file backend 需要一个可写目录 |

启动命令和请求脚本见[课堂实验](hands-on-exercise.md)的 (b) 段。几个环境相关的点：

- `--enable-hierarchical-cache` 打开 L1/L2；不加 `--hicache-storage-backend` 时只有 host tier。
- L3 用 `--hicache-storage-backend file`，目录由 `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` 指定，默认 `/tmp/hicache`。
- `--enable-metrics` 打开 Prometheus 指标，`/metrics` 和 HTTP 服务同端口（默认 30000）。
- L1 HBM 和 L2 host DRAM 是单实例私有；L3 是否共享取决于 backend，file 只是本地目录，Mooncake / HF3FS 才是跨实例共享。
- `--hicache-write-policy` 可选 `write_through`、`write_through_selective`、`write_back`。三种策略的 backup 时机不同，实验固定用一种再比较。

## 记录要求

每次跑完至少记下：

- 启动参数，尤其是 `--max-total-tokens`、`--page-size`、`--hicache-ratio`、`--hicache-write-policy`；
- 每条请求的 `prompt_tokens` 和 `cached_tokens_details` 的 device / host / storage 三个数；
- TTFT 和 load_back 耗时；
- 如果发生驱逐，记下 `evicted tokens` 和当时的 device / host 使用量。

没有 GPU 时，(b) 段用教师提供的日志填写，(a) 和 (c) 的 CPU 部分照常完成。
