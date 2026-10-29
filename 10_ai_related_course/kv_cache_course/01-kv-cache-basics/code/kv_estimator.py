#!/usr/bin/env python3
"""KV Cache 容量估算器（第 1 讲）。

只依赖 Python 标准库。支持 MHA / GQA（MQA 是 H_kv=1 的 GQA 特例），
KV 精度支持 FP16 / BF16 / FP8 / INT8。

公式：
    per_token_bytes = 2 * b_kv * L * H_kv * D
    total_bytes = per_token_bytes * T * batch

MLA 不在本脚本实现。MLA 每层缓存的是压缩 latent 和共享 RoPE key：
    per_token_bytes_mla = b_kv * L * (kv_lora_rank + qk_rope_head_dim)
不乘 2，也不乘 H_kv；第 6 讲展开。

用法：
    python3 kv_estimator.py
    python3 kv_estimator.py --model qwen3-0.6b --seq 4096 --batch 8 --dtype fp16
    python3 kv_estimator.py --dtype all --seq 8192
    python3 kv_estimator.py --model qwen2.5-72b --gpu-gib 40
    python3 kv_estimator.py --model qwen2.5-72b --gpu-gib 80 --tp 8 --seq 4096
    python3 kv_estimator.py --self-check
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace

KIB = 1024
MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class ModelPreset:
    key: str
    name: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    params_billion: float

    @property
    def attention_type(self) -> str:
        if self.num_kv_heads == self.num_q_heads:
            return "MHA"
        if self.num_kv_heads == 1:
            return "MQA"
        return "GQA"

    @property
    def weights_bytes_fp16(self) -> int:
        # 权重按 FP16 参数估算，和 KV 精度分开记账。
        return int(round(self.params_billion * 1e9 * 2))


PRESETS: dict[str, ModelPreset] = {
    "qwen3-0.6b": ModelPreset(
        key="qwen3-0.6b",
        name="Qwen3-0.6B",
        num_layers=28,
        num_q_heads=16,
        num_kv_heads=8,
        head_dim=128,
        params_billion=0.6,
    ),
    "qwen2.5-7b": ModelPreset(
        key="qwen2.5-7b",
        name="Qwen2.5-7B",
        num_layers=28,
        num_q_heads=28,
        num_kv_heads=4,
        head_dim=128,
        params_billion=7.6,
    ),
    "qwen2.5-72b": ModelPreset(
        key="qwen2.5-72b",
        name="Qwen2.5-72B",
        num_layers=80,
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        params_billion=72.7,
    ),
}

# 每元素字节数。FP8/INT8 的 scale、INT4 的 0.5 B 元素本体和打包开销
# 都不在本脚本内；第 7 讲再展开 FP4 的 scale 与打包。
DTYPES: dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
}


def per_token_bytes(model: ModelPreset, dtype: str, tp: int = 1) -> int:
    """单请求、单 token 的 K+V 字节数。tp 只切 KV head，不改变公式。"""
    if model.num_kv_heads % tp != 0:
        raise ValueError(
            f"{model.name} 的 H_kv={model.num_kv_heads} 不能被 tp={tp} 整除"
        )
    kv_heads = model.num_kv_heads // tp
    return int(2 * DTYPES[dtype] * model.num_layers * kv_heads * model.head_dim)


def fmt_bytes(value: float) -> str:
    if value >= GIB:
        return f"{value / GIB:.2f} GiB"
    if value >= MIB:
        return f"{value / MIB:.1f} MiB"
    if value >= KIB:
        return f"{value / KIB:.1f} KiB"
    return f"{value:.0f} B"


def fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"


def selected_models(key: str, attention: str = "native") -> list[ModelPreset]:
    if key == "all":
        models = list(PRESETS.values())
    else:
        models = [PRESETS[key]]
    if attention == "mha":
        # 把预设的 H_kv 临时改成 H_q，用于对比 MHA；这不是真实模型配置。
        models = [
            replace(
                model,
                key=f"{model.key}-mha",
                name=f"{model.name} (MHA)",
                num_kv_heads=model.num_q_heads,
            )
            for model in models
        ]
    return models


def selected_dtypes(key: str) -> list[str]:
    if key == "all":
        return list(DTYPES)
    return [key]


def print_per_token_table(
    models: list[ModelPreset], dtypes: list[str], tp: int = 1
) -> None:
    suffix = "全模型，未按 tp 切分" if tp != 1 else "全模型"
    print(f"每 token KV（{suffix}；per_token_bytes = 2 × b_kv × L × H_kv × D）")
    print()
    if tp != 1:
        print(f"注意：本表每 token 值是全局口径；tp={tp} 时每卡值见后面的“每卡明细”。")
        print()
    print("| 模型 | 类型 | 精度 | L | H_kv | D | b_kv | 每 token 字节 | 每 token |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for model in models:
        for dtype in dtypes:
            size = per_token_bytes(model, dtype)
            print(
                f"| {model.name} | {model.attention_type} | {dtype.upper()} "
                f"| {model.num_layers} | {model.num_kv_heads} | {model.head_dim} "
                f"| {DTYPES[dtype]:g} | {fmt_int(size)} | {fmt_bytes(size)} |"
            )
    print()


def print_total_table(
    models: list[ModelPreset],
    dtypes: list[str],
    seq_len: int,
    batch: int,
    gpu_gib: float | None,
    tp: int,
    mem_util: float,
) -> None:
    title = f"总量（T={seq_len:,}, batch={batch}）"
    if gpu_gib is not None:
        title = (
            f"每卡总量（T={seq_len:,}, batch={batch}，预算 {gpu_gib:g} GiB/卡，"
            f"tp={tp}，mem_util={mem_util:g}）"
        )
    print(title)
    print()

    if gpu_gib is None:
        print("| 模型 | 精度 | 每请求 KV | 总 KV | 权重(FP16) | KV 与权重谁更大 |")
        print("| --- | --- | --- | --- | --- | --- |")
    else:
        print("| 模型 | 精度 | 每卡每请求 KV | 每卡总 KV | 每卡权重(FP16) | 并发上限 | 预算状态 |")
        print("| --- | --- | --- | --- | --- | --- | --- |")

    for model in models:
        for dtype in dtypes:
            if gpu_gib is None:
                token_bytes = per_token_bytes(model, dtype, tp=1)
                per_request = token_bytes * seq_len
                total = per_request * batch
                weights = model.weights_bytes_fp16
                bottleneck = "KV" if total > weights else "权重"
                print(
                    f"| {model.name} | {dtype.upper()} | {fmt_bytes(per_request)} "
                    f"| {fmt_bytes(total)} | {fmt_bytes(weights)} | {bottleneck} |"
                )
                continue

            budget = gpu_gib * GIB * mem_util
            weights_per_gpu = model.weights_bytes_fp16 / tp
            kv_budget = budget - weights_per_gpu
            token_bytes_gpu = per_token_bytes(model, dtype, tp=tp)
            per_request_gpu = token_bytes_gpu * seq_len
            total_gpu = per_request_gpu * batch

            if kv_budget <= 0:
                bottleneck = "权重（超出预算）"
                concurrency = "—"
            else:
                slots = int(kv_budget // per_request_gpu)
                concurrency = f"{slots:,}" if slots > 0 else "—"
                if total_gpu > kv_budget:
                    bottleneck = "KV（当前 batch 超出预算）"
                else:
                    bottleneck = "均未超预算"

            print(
                f"| {model.name} | {dtype.upper()} | {fmt_bytes(per_request_gpu)} "
                f"| {fmt_bytes(total_gpu)} | {fmt_bytes(weights_per_gpu)} "
                f"| {concurrency} | {bottleneck} |"
            )
    print()


def print_gpu_detail(
    models: list[ModelPreset],
    dtypes: list[str],
    seq_len: int,
    gpu_gib: float,
    tp: int,
    mem_util: float,
) -> None:
    print(f"每卡明细（gpu={gpu_gib:g} GiB, tp={tp}, mem_util={mem_util:g}）")
    print()
    print("| 模型 | 精度 | 每卡权重 | 可用 KV | 每卡每 token | 每卡每请求 KV |")
    print("| --- | --- | --- | --- | --- | --- |")
    budget = gpu_gib * GIB * mem_util
    for model in models:
        for dtype in dtypes:
            weights_per_gpu = model.weights_bytes_fp16 / tp
            kv_budget = budget - weights_per_gpu
            token_bytes_gpu = per_token_bytes(model, dtype, tp=tp)
            print(
                f"| {model.name} | {dtype.upper()} | {fmt_bytes(weights_per_gpu)} "
                f"| {fmt_bytes(kv_budget) if kv_budget > 0 else '权重已超预算'} "
                f"| {fmt_bytes(token_bytes_gpu)} | {fmt_bytes(token_bytes_gpu * seq_len)} |"
            )
    print()


def self_check() -> None:
    qwen3 = PRESETS["qwen3-0.6b"]
    qwen7b = PRESETS["qwen2.5-7b"]
    qwen72b = PRESETS["qwen2.5-72b"]

    assert per_token_bytes(qwen3, "fp16") == 112 * KIB
    assert per_token_bytes(qwen7b, "fp16") == 56 * KIB
    assert per_token_bytes(qwen72b, "fp16") == 320 * KIB
    assert per_token_bytes(qwen3, "fp8") == 56 * KIB
    assert per_token_bytes(qwen72b, "fp16", tp=8) == 40 * KIB
    assert per_token_bytes(qwen3, "fp16") * 4096 * 8 == 3584 * MIB

    print("self-check passed:")
    print(f"  Qwen3-0.6B FP16: {per_token_bytes(qwen3, 'fp16'):,} B/token = 112 KiB")
    print(f"  Qwen2.5-7B FP16: {per_token_bytes(qwen7b, 'fp16'):,} B/token = 56 KiB")
    print(f"  Qwen2.5-72B FP16: {per_token_bytes(qwen72b, 'fp16'):,} B/token = 320 KiB")
    print(f"  Qwen3-0.6B FP8: {per_token_bytes(qwen3, 'fp8'):,} B/token = 56 KiB")
    print("  Qwen3-0.6B FP16, T=4096, batch=8: 3.50 GiB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KV Cache 每 token 与总量估算（第 1 讲，纯标准库）",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=list(PRESETS) + ["all"],
        help="模型预设，默认全部",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=list(DTYPES) + ["all"],
        help="KV 精度，默认 fp16",
    )
    parser.add_argument(
        "--attention",
        default="native",
        choices=["native", "mha"],
        help="native 用预设的 H_kv；mha 把 H_kv 临时改成 H_q，用于对比",
    )
    parser.add_argument("--seq", type=int, default=4096, help="每请求序列长度 T")
    parser.add_argument("--batch", type=int, default=1, help="并发请求数")
    parser.add_argument(
        "--gpu-gib",
        type=float,
        default=None,
        help="每卡显存 GiB；给出后估算并发上限",
    )
    parser.add_argument("--tp", type=int, default=1, help="张量并行度，按 KV head 切分")
    parser.add_argument(
        "--mem-util",
        type=float,
        default=0.9,
        help="显存可用比例，默认 0.9",
    )
    parser.add_argument("--self-check", action="store_true", help="核对三个预设的每 token KV")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    if args.seq <= 0:
        parser.error("--seq 必须大于 0")
    if args.batch <= 0:
        parser.error("--batch 必须大于 0")
    if args.tp <= 0:
        parser.error("--tp 必须大于 0")
    if not 0 < args.mem_util <= 1:
        parser.error("--mem-util 必须在 (0, 1] 之间")
    if args.gpu_gib is not None and (
        not math.isfinite(args.gpu_gib) or args.gpu_gib <= 0
    ):
        parser.error("--gpu-gib 必须是有限正数")
    if args.tp != 1 and args.gpu_gib is None:
        parser.error("--tp 只在给出 --gpu-gib 时生效，请同时给 --gpu-gib")
    if args.mem_util != 0.9 and args.gpu_gib is None:
        parser.error("--mem-util 只在给出 --gpu-gib 时生效，请同时给 --gpu-gib")

    models = selected_models(args.model, args.attention)
    dtypes = selected_dtypes(args.dtype)

    compatible_models = []
    for model in models:
        if model.num_kv_heads % args.tp != 0:
            message = (
                f"{model.name} 的 H_kv={model.num_kv_heads} 不能被 tp={args.tp} 整除"
            )
            if args.model == "all":
                print(f"warning: 跳过 {message}", file=sys.stderr)
                continue
            parser.error(message)
        compatible_models.append(model)

    if not compatible_models:
        parser.error("没有模型能被当前 --tp 整除，请降低 --tp 或指定单个模型")
    models = compatible_models

    print_per_token_table(models, dtypes, args.tp)
    print_total_table(
        models,
        dtypes,
        args.seq,
        args.batch,
        args.gpu_gib,
        args.tp,
        args.mem_util,
    )

    if args.gpu_gib is not None:
        print_gpu_detail(
            models,
            dtypes,
            args.seq,
            args.gpu_gib,
            args.tp,
            args.mem_util,
        )

    print("说明：")
    print("- 权重按 FP16 参数估算（params × 2 B），KV 精度由 --dtype 单独选择。")
    print("- 并发上限 = (显存 × mem_util − 每卡权重) // (每卡每 token KV × T)，")
    print("  未扣激活、CUDA context、内存碎片和框架预留，只作量级参考。")
    print("- 权重已超预算时并发显示“—”，不输出负并发。")
    print(
        "- FP8/INT8 的 scale、MLA 与 INT4 的打包开销不在本脚本内；"
        "脚本输出是不含 scale 的下界，精确值见 README 精度表和第 7 讲。"
    )


if __name__ == "__main__":
    main()
