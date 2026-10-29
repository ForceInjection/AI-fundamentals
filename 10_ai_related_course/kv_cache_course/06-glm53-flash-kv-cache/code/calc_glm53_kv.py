#!/usr/bin/env python3
"""GLM-5.3-Flash 的 KV / indexer / KDA state 容量复算。

口径：单 rank、attn_tp=1、scaled FP8、64-token page 预留、无 MTP。
只用标准库。运行：python3 calc_glm53_kv.py
"""

from __future__ import annotations

MB = 1_000_000
MIB = 1024 * 1024

FULL_LAYERS = 11
KDA_LAYERS = 34
KV_LORA_RANK = 512
INDEX_HEAD_DIM = 128
QUANT_BLOCK = 128
SCALE_BYTES = 4
CONV_KERNEL = 4
NUM_HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64


def scaled_row(width: int) -> int:
    """每 128 个元素附 4 B scale 的行宽。"""
    return width + width // QUANT_BLOCK * SCALE_BYTES


def kda_slot_bytes(conv_itemsize: int, temporal_itemsize: int) -> int:
    """一个 state slot 的字节数，已经乘过 34 层。"""
    conv_elems = (CONV_KERNEL - 1) * (NUM_HEADS * HEAD_DIM + 2 * NUM_HEADS * HEAD_DIM)
    temporal_elems = NUM_HEADS * HEAD_DIM * HEAD_DIM
    return KDA_LAYERS * (conv_elems * conv_itemsize + temporal_elems * temporal_itemsize)


def main() -> None:
    kv_scaled_row = scaled_row(KV_LORA_RANK)
    kv_scaled = kv_scaled_row * FULL_LAYERS
    kv_raw = KV_LORA_RANK * FULL_LAYERS
    kv_bf16 = KV_LORA_RANK * 2 * FULL_LAYERS
    indexer_row = scaled_row(INDEX_HEAD_DIM)
    indexer = indexer_row * FULL_LAYERS
    total_scaled = kv_scaled + indexer
    total_raw = kv_raw + indexer
    total_bf16 = kv_bf16 + indexer

    print("== token 账（11 层 full）==")
    print(f"scaled FP8 行宽: {kv_scaled_row} B = {KV_LORA_RANK} + {KV_LORA_RANK}//{QUANT_BLOCK}×{SCALE_BYTES}")
    print(f"scaled FP8 KV: {kv_scaled_row} × {FULL_LAYERS} = {kv_scaled:,} B/token")
    print(f"indexer 行宽: {indexer_row} B = {INDEX_HEAD_DIM} + {INDEX_HEAD_DIM}//{QUANT_BLOCK}×{SCALE_BYTES}")
    print(f"indexer: {indexer_row} × {FULL_LAYERS} = {indexer:,} B/token")
    print(f"scaled FP8 合计: {total_scaled:,} B/token")
    print(f"TRT-LLM raw FP8 合计: {total_raw:,} B/token")
    print(f"BF16 合计: {total_bf16:,} B/token")
    print(f"indexer 64-token page: {PAGE_SIZE} × {indexer_row} = {PAGE_SIZE * indexer_row:,} B")
    effective_indexer = indexer_row // 4 * FULL_LAYERS
    print(f"indexer 有效写入（index_kpool=4 + compress）: 约 {effective_indexer} B/token")
    print()

    default = kda_slot_bytes(conv_itemsize=2, temporal_itemsize=4)
    all_fp32 = kda_slot_bytes(conv_itemsize=4, temporal_itemsize=4)
    bf16_flag = kda_slot_bytes(conv_itemsize=2, temporal_itemsize=2)

    conv_elems = (CONV_KERNEL - 1) * (NUM_HEADS * HEAD_DIM + 2 * NUM_HEADS * HEAD_DIM)
    temporal_elems = NUM_HEADS * HEAD_DIM * HEAD_DIM
    print("== KDA state 账（34 层合计）==")
    print(f"conv shape: ({CONV_KERNEL - 1}, {conv_elems // (CONV_KERNEL - 1):,})，元素 {conv_elems:,}/层")
    print(f"temporal shape: ({NUM_HEADS}, {HEAD_DIM}, {HEAD_DIM})，元素 {temporal_elems:,}/层")
    print(f"conv bf16 + temporal fp32: {default:,} B = {default / MB:.1f} MB/state slot")
    print(f"conv fp32 + temporal fp32: {all_fp32:,} B = {all_fp32 / MB:.1f} MB/state slot")
    print(f"conv bf16 + temporal bf16: {bf16_flag:,} B = {bf16_flag / MB:.1f} MB/state slot")
    print()

    print("== slots 账 ==")
    for slots in (5, 4, 3):
        print(
            f"{slots} slots: {slots * default / MB:.1f} MB/request（fp32 temporal），"
            f"{slots * bf16_flag / MB:.1f} MB/request（bf16 flag）"
        )
    print()

    print("== TP/attn_tp 均分（未计复制、padding、对齐）==")
    for tp in (1, 2, 4, 8):
        print(
            f"attn_tp={tp}: {default / MB / tp:.2f} MB/state slot，"
            f"{5 * default / MB / tp:.1f} MB/request（5 slots）"
        )
    print()

    tokens_128k = 131_072
    token_bytes = tokens_128k * total_scaled
    print("== 128K 上下文（只算 token KV + indexer）==")
    print(f"{tokens_128k:,} token × {total_scaled:,} B = {token_bytes:,} B")
    print(f"= {token_bytes / MB:.1f} MB = {token_bytes / 1e9:.2f} GB")
    print()

    print("== 单位换算 ==")
    print(f"{default / MB:.1f} MB = {default / MIB:.2f} MiB")
    print(f"5 slots = {5 * default / MB:.1f} MB = {5 * default / MIB:.2f} MiB")

    assert kv_scaled_row == 528
    assert kv_scaled == 5_808
    assert indexer_row == 132
    assert indexer == 1_452
    assert total_scaled == 7_260
    assert total_raw == 7_084
    assert total_bf16 == 12_716
    assert PAGE_SIZE * indexer_row == 8_448
    assert effective_indexer == 363
    assert round(default / MB, 1) == 147.6
    assert round(all_fp32 / MB, 1) == 152.6
    assert round(bf16_flag / MB, 1) == 76.3
    assert round(5 * default / MB, 1) == 738.1
    assert round(4 * default / MB, 1) == 590.5
    assert round(3 * default / MB, 1) == 442.9
    assert round(5 * bf16_flag / MB, 1) == 381.6
    assert round(default / MB / 8, 2) == 18.45
    print()
    print("断言全部通过。")


if __name__ == "__main__":
    main()
