#!/usr/bin/env python3
"""前缀缓存模拟器（第 5 讲）。

只依赖 Python 标准库。模拟 SGLang 风格 radix tree 的四件事：

1. 共享前缀匹配，匹配落在节点中间时分裂节点；
2. `lock_ref` 从节点向根累加，root 不参与释放；
3. 只有 evictable leaf 能被驱逐，叶子删掉后父节点才回到候选；
4. `extra_key` / `cache_salt` 构成命名空间，同名 prompt 也不命中。

KV 数据本身不存，`value` 用 (起始槽位, 长度) 表示；`host_value` 同理，
表示 host pool 的槽位区间，不是 DRAM 指针。

用法：
    python3 prefix_cache_sim.py              # 跑全部演示
    python3 prefix_cache_sim.py --demo abcd  # 只看 ABCD 四步
    python3 prefix_cache_sim.py --demo share # 只看命中条件
    python3 prefix_cache_sim.py --demo evict # 只看驱逐
    python3 prefix_cache_sim.py --self-check # 断言关键行为
"""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

NodeId = count(1)

Namespace = tuple[Optional[str], Optional[str]]  # (extra_key, cache_salt)


def common_prefix_len(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def label_of(key: tuple[int, ...], limit: int = 10) -> str:
    """把 1..26 的 token 映射成字母，便于看 ABCD 例子。"""
    if not key:
        return "root"
    if all(1 <= t <= 26 for t in key) and len(key) <= limit:
        return "".join(chr(64 + t) for t in key)
    if len(key) <= 4:
        return "[" + ",".join(str(t) for t in key) + "]"
    return f"[{key[0]},{key[1]},...]({len(key)})"


@dataclass
class Node:
    key: tuple[int, ...]
    namespace: Namespace = ("", None)
    parent: Optional["Node"] = None
    children: dict[object, "Node"] = field(default_factory=dict)
    value: Optional[tuple[int, int]] = None       # device KV 槽位区间
    host_value: Optional[tuple[int, int]] = None  # host pool 槽位区间
    lock_ref: int = 0
    last_access: int = 0
    node_id: int = field(default_factory=lambda: next(NodeId))

    @property
    def length(self) -> int:
        return len(self.key)


class PrefixCache:
    """最小 radix tree，只保留与本讲有关的字段和动作。"""

    def __init__(self, capacity: Optional[int] = None, page_size: int = 1):
        self.root = Node(key=(), node_id=0)
        self.capacity = capacity
        self.page_size = page_size
        self.used = 0
        self.next_slot = 0
        self.clock = 0
        self.last_evicted: list[Node] = []

    # ---------- 树结构 ----------

    def _child(self, node: Node, token: int, ns: Namespace) -> Optional[Node]:
        key = (ns, token) if node is self.root else token
        return node.children.get(key)

    def _attach(self, parent: Node, child: Node) -> None:
        key = (child.namespace, child.key[0]) if parent is self.root else child.key[0]
        parent.children[key] = child
        child.parent = parent

    def _detach(self, parent: Node, child: Node) -> None:
        key = (child.namespace, child.key[0]) if parent is self.root else child.key[0]
        parent.children.pop(key, None)

    def _split(self, child: Node, split_len: int) -> Node:
        """把 child 的前 split_len 个 token 拆成新父节点。

        与 SGLang 一致：新节点复制 child.lock_ref；value/host_value 按
        split_len 切片，不复制数据。
        """
        parent = child.parent
        assert parent is not None
        old_key = child.key
        new_node = Node(
            key=old_key[:split_len],
            namespace=child.namespace,
            last_access=child.last_access,
        )
        new_node.lock_ref = child.lock_ref

        new_node.parent = parent
        new_node.children[old_key[split_len]] = child
        child.parent = new_node
        child.key = old_key[split_len:]

        if child.value is not None:
            start, length = child.value
            new_node.value = (start, split_len)
            child.value = (start + split_len, length - split_len)
        if child.host_value is not None:
            start, length = child.host_value
            new_node.host_value = (start, split_len)
            child.host_value = (start + split_len, length - split_len)

        parent_key = (
            (child.namespace, new_node.key[0]) if parent is self.root else new_node.key[0]
        )
        parent.children[parent_key] = new_node
        return new_node

    # ---------- 匹配与插入 ----------

    def match(
        self, tokens: tuple[int, ...], namespace: Namespace = ("", None)
    ) -> tuple[Node, int]:
        if self.page_size > 1:
            aligned = len(tokens) // self.page_size * self.page_size
            tokens = tokens[:aligned]

        node = self.root
        matched = 0
        while matched < len(tokens):
            child = self._child(node, tokens[matched], namespace)
            if child is None:
                break
            common = common_prefix_len(child.key, tokens[matched:])
            if common < child.length:
                # 与 SGLang 的 RadixKey.match 一致：命中长度按 page_size 向下取整。
                if self.page_size > 1:
                    common = common // self.page_size * self.page_size
                if common == 0:
                    break
                child = self._split(child, common)
            matched += common
            node = child
            if common == 0:
                break

        self.clock += 1
        node.last_access = self.clock
        return node, matched

    def insert(
        self, tokens: tuple[int, ...], namespace: Namespace = ("", None)
    ) -> Node:
        if self.page_size > 1:
            aligned = len(tokens) // self.page_size * self.page_size
            tokens = tokens[:aligned]

        node, matched = self.match(tokens, namespace)
        if matched == len(tokens):
            return node

        rest = tokens[matched:]
        if self.capacity is not None:
            while self.used + len(rest) > self.capacity:
                if not self.evict(len(rest)):
                    break

        new_node = Node(key=tuple(rest), namespace=namespace)
        self._attach(node, new_node)
        new_node.value = (self.next_slot, len(rest))
        self.next_slot += len(rest)
        self.used += len(rest)
        self.clock += 1
        new_node.last_access = self.clock
        return new_node

    # ---------- 引用与状态 ----------

    def inc_lock_ref(self, node: Node) -> None:
        cur = node
        while cur is not self.root:
            cur.lock_ref += 1
            cur = cur.parent

    def dec_lock_ref(self, node: Node) -> None:
        cur = node
        while cur is not self.root:
            assert cur.lock_ref > 0, f"node {label_of(cur.key)} lock_ref 已是 0"
            cur.lock_ref -= 1
            cur = cur.parent

    def state(self, node: Node) -> str:
        if node is self.root:
            return "root"
        if node.lock_ref > 0:
            return "locked"
        if node.value is None:
            return "evicted"
        if not any(child.value is not None for child in node.children.values()):
            return "evictable leaf"
        return "internal"

    def _is_evictable_leaf(self, node: Node) -> bool:
        return (
            node is not self.root
            and node.value is not None
            and node.lock_ref == 0
            and not any(child.value is not None for child in node.children.values())
        )

    def evictable_leaves(self) -> list[Node]:
        out: list[Node] = []

        def walk(node: Node) -> None:
            if self._is_evictable_leaf(node):
                out.append(node)
            for child in node.children.values():
                walk(child)

        walk(self.root)
        return out

    # ---------- 驱逐 ----------

    def evict(self, num_tokens: int) -> list[Node]:
        """按 LRU 驱逐 evictable leaf，直到释放够 num_tokens 或没有候选。"""
        heap = [
            (node.last_access, node.node_id, node) for node in self.evictable_leaves()
        ]
        heapq.heapify(heap)
        freed: list[Node] = []
        remaining = num_tokens

        while remaining > 0 and heap:
            _, _, node = heapq.heappop(heap)
            if not self._is_evictable_leaf(node):
                continue
            parent = node.parent
            assert parent is not None
            freed.append(node)
            remaining -= node.length
            self.used -= node.length
            self._detach(parent, node)
            node.value = None
            if self._is_evictable_leaf(parent):
                heapq.heappush(heap, (parent.last_access, parent.node_id, parent))

        self.last_evicted = freed
        return freed

    # ---------- 打印 ----------

    def describe(self, node: Node) -> str:
        kv = "-" if node.value is None else f"{node.value[0]}:{node.value[0] + node.value[1]}"
        return (
            f"{label_of(node.key)} [{self.state(node)}, "
            f"lock={node.lock_ref}, kv={kv}]"
        )

    def dump(self) -> None:
        print("root")

        def walk(node: Node, prefix: str) -> None:
            items = sorted(node.children.values(), key=lambda n: n.node_id)
            for i, child in enumerate(items):
                last = i == len(items) - 1
                print(prefix + ("`- " if last else "|- ") + self.describe(child))
                walk(child, prefix + ("   " if last else "|  "))

        walk(self.root, "")

    def reset_clock(self) -> None:
        self.clock = 0


# ---------- 演示 ----------


def demo_abcd() -> None:
    print("=" * 62)
    print("ABCD 例子：建树 -> 匹配 AB 并分裂 -> 加锁 -> 驱逐与解锁")
    print("=" * 62)

    cache = PrefixCache()
    cache.insert((1, 2, 3, 4))
    print("\n第 1 步：插入 ABCD，整段是一个节点")
    cache.dump()
    print(f"当前可驱逐节点：{[label_of(n.key) for n in cache.evictable_leaves()]}")

    print("\n第 2 步：匹配 AB，节点在匹配点分裂")
    node_ab, matched = cache.match((1, 2))
    print(f"命中长度 {matched}，AB 与 CD 成为两个节点")
    cache.dump()
    print(f"可驱逐节点：{[label_of(n.key) for n in cache.evictable_leaves()]}")

    print("\n第 3 步：对 AB 加锁，lock_ref 只累加到 AB")
    cache.inc_lock_ref(node_ab)
    cache.dump()
    print("AB 被锁住；CD 的 lock_ref 仍是 0，仍可驱逐")

    print("\n第 4 步：驱逐 2 个 token，再解锁 AB")
    freed = cache.evict(2)
    print(f"驱逐：{[label_of(n.key) for n in freed]}")
    cache.dec_lock_ref(node_ab)
    cache.dump()
    print("CD 已删除；AB 无子节点且 lock_ref 回到 0，成为 evictable leaf")


def demo_share() -> None:
    print("\n" + "=" * 62)
    print("命中条件：token 前缀、extra_key、cache_salt、page_size")
    print("=" * 62)

    system = tuple(range(1, 65))          # 64 token 的 system prompt
    q_a = tuple(range(1001, 1011))
    q_b = tuple(range(2001, 2011))
    q_c = tuple(range(3001, 3011))

    cache = PrefixCache()
    cache.insert(system + q_a, namespace=("lora-a", None))

    cases = [
        ("R1 同 namespace，同 system，不同问题", system + q_b, ("lora-a", None)),
        ("R2 同 prompt，换 LoRA", system + q_a, ("lora-b", None)),
        ("R3 同 prompt，同 LoRA，换 salt", system + q_a, ("lora-a", "tenant-2")),
        ("R4 完整重放 R1 的 prompt", system + q_a, ("lora-a", None)),
    ]

    for desc, tokens, ns in cases:
        _, hit = cache.match(tokens, namespace=ns)
        if hit == len(tokens):
            note = "全部命中"
        elif hit == len(system):
            note = "命中 system 前缀，问题部分重算"
        else:
            note = "不命中，整段重算"
        print(f"  {desc} | prompt={len(tokens)}, 命中={hit}, {note}")

    print("\npage_size 对齐：共享前缀 60 token，page_size=16")
    paged = PrefixCache(page_size=16)
    paged.insert(system[:60] + q_a)
    _, hit = paged.match(system[:60] + q_c)
    print(f"命中 {hit} token，60 向下对齐到 48，尾部 12 token 重算")


def demo_evict() -> None:
    print("\n" + "=" * 62)
    print("驱逐：容量 12 token，锁住 R2，再插入 R5")
    print("=" * 62)

    cache = PrefixCache(capacity=12)
    nodes = {}
    for i in range(1, 5):
        tokens = (100 + i * 10, 101 + i * 10, 102 + i * 10, 103 + i * 10)
        nodes[f"R{i}"] = cache.insert(tokens)

    cache.inc_lock_ref(nodes["R2"])
    print("\n插入 R1..R4 后（容量 12，已用 12），锁住 R2：")
    cache.dump()

    tokens = (150, 151, 152, 153)
    cache.insert(tokens)
    print("\n插入 R5 触发驱逐：")
    cache.dump()
    print(f"本轮驱逐：{[label_of(n.key) for n in cache.last_evicted]}")
    print(f"R2 状态：{cache.state(nodes['R2'])}，锁未释放，没有被驱逐")

    cache.dec_lock_ref(nodes["R2"])
    print(f"R2 解锁后状态：{cache.state(nodes['R2'])}")


def self_check() -> None:
    cache = PrefixCache()
    cache.insert((1, 2, 3, 4))
    assert len(cache.root.children) == 1

    node_ab, matched = cache.match((1, 2))
    assert matched == 2
    assert node_ab.key == (1, 2)
    assert node_ab.value == (0, 2)
    child_cd = next(iter(node_ab.children.values()))
    assert child_cd.key == (3, 4)
    assert child_cd.value == (2, 2)
    assert cache.state(child_cd) == "evictable leaf"

    cache.inc_lock_ref(node_ab)
    assert node_ab.lock_ref == 1
    assert cache.state(node_ab) == "locked"

    freed = cache.evict(2)
    assert [n.key for n in freed] == [(3, 4)]
    assert not node_ab.children
    assert cache.state(node_ab) == "locked"

    cache.dec_lock_ref(node_ab)
    assert node_ab.lock_ref == 0
    assert cache.state(node_ab) == "evictable leaf"

    ns_cache = PrefixCache()
    ns_cache.insert((1, 2, 3), namespace=("lora-a", None))
    _, hit = ns_cache.match((1, 2, 3), namespace=("lora-b", None))
    assert hit == 0

    salt_cache = PrefixCache()
    salt_cache.insert((1, 2, 3), namespace=("", "salt-1"))
    _, hit = salt_cache.match((1, 2, 3), namespace=("", "salt-2"))
    assert hit == 0

    paged = PrefixCache(page_size=16)
    paged.insert(tuple(range(1, 61)))
    _, hit = paged.match(tuple(range(1, 61)))
    assert hit == 48

    print("self-check 通过：split/lock_ref/evict/namespace/page_size 行为符合预期")


def main() -> None:
    parser = argparse.ArgumentParser(description="前缀缓存与驱逐的最小模拟")
    parser.add_argument(
        "--demo",
        choices=["all", "abcd", "share", "evict"],
        default="all",
        help="选择演示场景",
    )
    parser.add_argument("--self-check", action="store_true", help="只跑断言")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    if args.demo in ("all", "abcd"):
        demo_abcd()
    if args.demo in ("all", "share"):
        demo_share()
    if args.demo in ("all", "evict"):
        demo_evict()


if __name__ == "__main__":
    main()
