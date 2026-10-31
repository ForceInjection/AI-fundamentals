#!/usr/bin/env python3
"""第 2 讲：block table / slot / 碎片演示（纯标准库）。

关键映射：
    logical_block_index = token_index // block_size
    physical_block_id  = block_table[logical_block_index]
    slot               = physical_block_id * block_size + offset

默认跑三个场景：
1. 单请求的 token -> block -> slot、block table 和内部碎片；
2. 两个请求共享前缀时的 block table 与 ref_count；
3. A/B 碎片对照：连续分配 max_model_len vs 固定 block 池。

用法：
    python3 block_table_demo.py --self-check
    python3 block_table_demo.py
    python3 block_table_demo.py --block-size 4 --seq-len 10 --prefix-tokens 6
"""

from __future__ import annotations

import argparse

KIB = 1024


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_kib(value: int) -> str:
    return f"{value / KIB:.1f} KiB"


def blocks_needed(seq_len: int, block_size: int) -> int:
    return (seq_len + block_size - 1) // block_size


def print_mapping(seq_len: int, block_size: int, per_token_bytes: int) -> None:
    num_blocks = blocks_needed(seq_len, block_size)
    block_table = list(range(num_blocks))

    print(f"场景 1：单请求 {seq_len} token，block_size={block_size}")
    print()
    print("| token | logical block | physical block | offset | slot |")
    print("| --- | --- | --- | --- | --- |")
    for token in range(seq_len):
        logical = token // block_size
        offset = token % block_size
        physical = block_table[logical]
        slot = physical * block_size + offset
        print(f"| {token} | {logical} | {physical} | {offset} | {slot} |")
    print()
    print(f"block table：{block_table}")
    print(f"手算一行：token {seq_len - 1} -> logical {num_blocks - 1}, "
          f"offset {(seq_len - 1) % block_size}, "
          f"slot = {block_table[-1]} × {block_size} + {(seq_len - 1) % block_size} "
          f"= {block_table[-1] * block_size + (seq_len - 1) % block_size}")
    print()

    block_bytes = block_size * per_token_bytes
    allocated = num_blocks * block_bytes
    valid = seq_len * per_token_bytes
    internal_tokens = num_blocks * block_size - seq_len
    internal = internal_tokens * per_token_bytes
    print(f"每 token KV = {fmt_kib(per_token_bytes)}，"
          f"block_bytes = {block_size} × {fmt_kib(per_token_bytes)} = {fmt_kib(block_bytes)}")
    print(f"分配 {num_blocks} 块 = {fmt_kib(allocated)}，"
          f"有效 KV = {seq_len} token = {fmt_kib(valid)}")
    print(f"内部碎片 = {num_blocks} × {block_size} − {seq_len} = {internal_tokens} token "
          f"= {fmt_kib(internal)}")
    print()


def print_shared_prefix(
    block_size: int,
    prefix_tokens: int,
    total_tokens: int,
    per_token_bytes: int,
) -> None:
    shared_blocks = prefix_tokens // block_size
    blocks_a = blocks_needed(total_tokens, block_size)
    blocks_b = blocks_needed(total_tokens, block_size)

    table_a = list(range(blocks_a))
    # 共享块用 A 的物理块号，其余从 blocks_a 开始新分配。
    table_b = table_a[:shared_blocks] + list(
        range(blocks_a, blocks_a + blocks_b - shared_blocks)
    )

    ref_count: dict[int, int] = {}
    for block_id in table_a:
        ref_count[block_id] = ref_count.get(block_id, 0) + 1
    for block_id in table_b:
        ref_count[block_id] = ref_count.get(block_id, 0) + 1

    print(f"场景 2：共享前缀，block_size={block_size}，"
          f"前 {prefix_tokens} 个 token 相同，第 {prefix_tokens + 1} 个起不同")
    print()
    print(f"请求 A：{total_tokens} token，block table = {table_a}")
    print(f"请求 B：{total_tokens} token，block table = {table_b}")
    print()
    print("| physical block | 引用请求 | ref_count |")
    print("| --- | --- | --- |")
    owners: dict[int, list[str]] = {}
    for block_id in table_a:
        owners.setdefault(block_id, []).append("A")
    for block_id in table_b:
        owners.setdefault(block_id, []).append("B")
    for block_id in sorted(owners):
        print(f"| {block_id} | {', '.join(owners[block_id])} | {ref_count[block_id]} |")
    print()

    shared_tokens = shared_blocks * block_size
    print(f"共享块数 = {prefix_tokens} // {block_size} = {shared_blocks}，"
          f"实际共享 {shared_tokens} 个 token。")
    if prefix_tokens % block_size:
        partial = prefix_tokens // block_size
        start = partial * block_size
        print(f"第 {prefix_tokens + 1} 个 token（0-based index {prefix_tokens}）已经不同，"
              f"它落在 block {partial}（覆盖 index {start}-{start + block_size - 1}）里，"
              f"该块不能共享。")
    print("前缀共享以完整 block 为粒度：块内只要有一个 token 不同，整块就不能复用。")
    print(f"共享块 ref_count = 2；私有块 ref_count = 1；"
          f"block 0 的槽位 {0}-{block_size - 1} 被两个请求共用。")
    print()


def print_fragmentation_ab(
    block_size: int,
    layers: int,
    per_token_layer_bytes: int,
    seq_len: int,
    max_model_len: int,
    arena_tokens: int,
    num_requests: int,
) -> None:
    per_token_bytes = layers * per_token_layer_bytes
    block_bytes = block_size * per_token_bytes

    print("场景 3：A/B 碎片对照")
    print()
    print(f"玩具模型：L={layers}，每层每 token 的 K+V = {fmt_kib(per_token_layer_bytes)}，"
          f"每 token = {fmt_kib(per_token_bytes)}，block_size={block_size}，"
          f"block_bytes = {fmt_kib(block_bytes)}。")
    print()

    # A：每个请求连续预留 max_model_len。
    reserved_tokens = num_requests * max_model_len
    useful_tokens = num_requests * seq_len
    reservation_waste_tokens = reserved_tokens - useful_tokens
    free_tokens = arena_tokens - reserved_tokens
    # 剩余空间按 max_model_len 切成可用的新请求，切不出的尾隙记外部碎片。
    usable_new_requests = free_tokens // max_model_len
    external_tokens = free_tokens % max_model_len
    if usable_new_requests:
        free_desc = f"{usable_new_requests} 个新请求（{free_tokens} token）"
    else:
        free_desc = f"0（剩余 {free_tokens} token，不够 {max_model_len} token）"

    # B：固定 block 池。
    pool_blocks = arena_tokens // block_size
    blocks_per_request = blocks_needed(seq_len, block_size)
    used_blocks = num_requests * blocks_per_request
    internal_tokens = used_blocks * block_size - useful_tokens
    free_blocks = pool_blocks - used_blocks

    print("| 方案 | 池容量 | 已分配 | 有效 KV | reservation waste | internal frag | external frag | 可用余量 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    print(
        f"| A 连续 max_model_len={max_model_len} "
        f"| {fmt_kib(arena_tokens * per_token_bytes)}（{arena_tokens} token） "
        f"| {fmt_kib(reserved_tokens * per_token_bytes)}（{reserved_tokens} token 预留） "
        f"| {fmt_kib(useful_tokens * per_token_bytes)}（{useful_tokens} token） "
        f"| {fmt_kib(reservation_waste_tokens * per_token_bytes)}（{reservation_waste_tokens} token） "
        f"| 0 "
        f"| {fmt_kib(external_tokens * per_token_bytes)}（{external_tokens} token） "
        f"| {free_desc} |"
    )
    print(
        f"| B 固定 block 池 "
        f"| {fmt_kib(pool_blocks * block_bytes)}（{pool_blocks} block） "
        f"| {fmt_kib(used_blocks * block_bytes)}（{used_blocks} block） "
        f"| {fmt_kib(useful_tokens * per_token_bytes)}（{useful_tokens} token） "
        f"| 0 "
        f"| {fmt_kib(internal_tokens * per_token_bytes)}（{internal_tokens} token） "
        f"| 0（按构造） "
        f"| {fmt_kib(free_blocks * block_bytes)}（{free_blocks} block） |"
    )
    print()
    if external_tokens:
        print(f"A 的尾隙 {external_tokens} token 总和够用，但不是 {max_model_len} token 的连续段，"
              "新请求放不进去，")
        print("所以记 external fragmentation。B 的空闲 block 都是固定单元，")
    else:
        print("A 的剩余空间恰好能按 max_model_len 切分，external fragmentation 为 0。")
        print("B 的空闲 block 都是固定单元，")
    print(f"任何请求都能直接使用，external fragmentation 按构造为 0；")
    print(f"B 的浪费只在每个请求最后一个 block 内，最多 block_size − 1 = {block_size - 1} 个 token。")
    print()


def self_check() -> None:
    assert blocks_needed(10, 4) == 3
    assert blocks_needed(4, 4) == 1
    assert blocks_needed(5, 4) == 2
    print("self-check passed:")
    print("  blocks_needed(10, 4) = 3")
    print("  blocks_needed(4, 4)  = 1")
    print("  blocks_needed(5, 4)  = 2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="block table、slot、内部/外部碎片演示（纯标准库）",
    )
    parser.add_argument("--self-check", action="store_true", help="核对 block 数计算")
    parser.add_argument("--block-size", type=int, default=4, help="每个 block 的 token 数")
    parser.add_argument("--seq-len", type=int, default=10, help="单请求 token 数")
    parser.add_argument("--prefix-tokens", type=int, default=6, help="两个请求相同的 token 数")
    parser.add_argument("--total-tokens", type=int, default=12, help="共享前缀场景的请求长度")
    parser.add_argument("--layers", type=int, default=2, help="玩具模型层数")
    parser.add_argument(
        "--per-token-layer-bytes",
        type=int,
        default=1024,
        help="玩具模型每层每 token 的 K+V 字节数",
    )
    parser.add_argument("--max-model-len", type=int, default=16, help="连续分配时的预留长度")
    parser.add_argument("--arena-tokens", type=int, default=72, help="A/B 对照的总 token 容量")
    parser.add_argument("--num-requests", type=int, default=4, help="A/B 对照的请求数")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    if args.block_size <= 0:
        parser.error("--block-size 必须大于 0")
    if args.seq_len <= 0:
        parser.error("--seq-len 必须大于 0")
    if args.prefix_tokens < 0:
        parser.error("--prefix-tokens 不能为负")
    if args.total_tokens <= 0:
        parser.error("--total-tokens 必须大于 0")
    if args.prefix_tokens > args.total_tokens:
        parser.error("--prefix-tokens 不能大于 --total-tokens")
    if args.layers <= 0 or args.per_token_layer_bytes <= 0:
        parser.error("--layers 和 --per-token-layer-bytes 必须大于 0")
    if args.max_model_len < args.seq_len:
        parser.error("--max-model-len 不能小于 --seq-len")
    if args.num_requests <= 0:
        parser.error("--num-requests 必须大于 0")
    if args.arena_tokens <= 0:
        parser.error("--arena-tokens 必须大于 0")
    if args.arena_tokens < args.num_requests * args.max_model_len:
        parser.error("--arena-tokens 至少要能放下 num_requests × max_model_len")

    per_token_bytes = args.layers * args.per_token_layer_bytes
    print_mapping(args.seq_len, args.block_size, per_token_bytes)
    print_shared_prefix(
        args.block_size,
        args.prefix_tokens,
        args.total_tokens,
        per_token_bytes,
    )
    print_fragmentation_ab(
        args.block_size,
        args.layers,
        args.per_token_layer_bytes,
        args.seq_len,
        args.max_model_len,
        args.arena_tokens,
        args.num_requests,
    )


if __name__ == "__main__":
    main()
