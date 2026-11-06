# 手搓 KV Cache 管理器：静态预留 -> 动态连续区 -> 分页
# 配套文章：hand_roll_kv_cache.md
# 运行：python hand_roll_kv_cache.py（纯标准库；无需 GPU）
# 用一个离散事件模拟器实测三种显存管理策略，验证文章的每一笔账：
#   版本 0  静态预留（按 max_len 一口价）：容量小、利用率低
#   版本 1  动态连续区：省显存但释放后产生外部碎片，新序列入住失败
#   版本 2  分页（block=16，vLLM GPU 口径）：无外部碎片，页内浪费 ~3%
import random

random.seed(42)
MAX_LEN = 1024               # 单序列最大长度（静态预留口径的上限）
MEAN_LEN = 220               # 序列实际长度的均值（长尾分布）
BLOCK = 16                   # 页大小（vLLM GPU 默认口径，vllm/_custom_ops.py:55）

# 生成 60 个序列请求：长度 32~MAX_LEN，均值约 220
lens = [min(MAX_LEN, max(32, int(random.gauss(MEAN_LEN, 160)))) for _ in range(60)]
lens_head = lens[:40]
# 显存预算恰好装下前 40 条入驻序列（消除尾部大片空白，让碎片现象真实显形）
TOTAL_SLOTS = sum(lens_head) + 32
print(f"60 个序列：均值 {sum(lens)/len(lens):.0f} tokens，最长 {max(lens)}，总需求 {sum(lens)}")
print(f"显存预算 {TOTAL_SLOTS} token 槽位，页大小 {BLOCK}\n")


# ---------- 版本 0：静态预留（按 max_len 一口价） ----------
def run_v0():
    capacity = TOTAL_SLOTS // MAX_LEN
    reserve = capacity * MAX_LEN
    used = sum(min(l, MAX_LEN) for l in lens[:capacity])
    return capacity, reserve, used


# ---------- 版本 1：动态连续区（三阶段：入驻 -> 释放 -> 再入驻） ----------
def run_v1():
    slots = [None] * TOTAL_SLOTS
    seqs = {}
    sid = 0
    for l in lens_head:                        # 阶段一：按到达顺序连续入驻
        run, start = 0, -1
        for i, v in enumerate(slots):
            run = run + 1 if v is None else 0
            if run == l:
                start = i - l + 1
                break
        if start >= 0:
            for i in range(start, start + l):
                slots[i] = sid
            seqs[sid] = start
            sid += 1
    n_phase1 = sid
    freed = [k for k in list(seqs) if k % 2 == 0]      # 阶段二：隔一放二地释放
    for k in freed:
        for i, v in enumerate(slots):
            if v == k:
                slots[i] = None
        del seqs[k]
    total_free = slots.count(None)
    largest_gap, run = 0, 0
    for v in slots:
        run = run + 1 if v is None else 0
        largest_gap = max(largest_gap, run)
    need = 800                                  # 阶段三：新序列想入住
    return n_phase1, len(freed), total_free, largest_gap, need, largest_gap >= need


# ---------- 版本 2：分页（block table 离散映射，同样三阶段） ----------
def run_v2():
    free_blocks = TOTAL_SLOTS // BLOCK
    pages, seq_len = {}, {}
    sid = 0
    for l in lens_head:                        # 阶段一：按页入驻
        need = (l + BLOCK - 1) // BLOCK
        if need <= free_blocks:
            pages[sid] = need
            seq_len[sid] = l
            free_blocks -= need
            sid += 1
    n_phase1 = sid
    freed = [k for k in list(pages) if k % 2 == 0]     # 阶段二：释放
    for k in freed:
        free_blocks += pages.pop(k)
        seq_len.pop(k)
    need = 800                                  # 阶段三：新序列按页凑
    need_blocks = (need + BLOCK - 1) // BLOCK
    ok = need_blocks <= free_blocks
    wasted = sum(pages[k] * BLOCK - seq_len[k] for k in pages)
    occupied = sum(pages[k] * BLOCK for k in pages)
    return n_phase1, len(freed), free_blocks * BLOCK, ok, wasted, occupied


if __name__ == "__main__":
    capacity, reserve, used = run_v0()
    print("== 版本 0：静态预留 ==")
    print(f"容量只有 {capacity} 条（{TOTAL_SLOTS}/{MAX_LEN}）；平均利用率 {used/reserve:.0%}")

    print("\n== 版本 1：动态连续区（三阶段）==")
    n1, freed1, total_free, largest_gap, need, ok1 = run_v1()
    print(f"入驻 {n1} 条，释放 {freed1} 条；一条 {need} token 的新序列想入住：")
    print(f"  总空闲 {total_free} 槽位（足够），最大连续空洞只有 {largest_gap} 槽位"
          f" -> {'成功' if ok1 else '入住失败（外部碎片）'}")

    print("\n== 版本 2：分页（block=16，同样三阶段）==")
    n2, freed2, free2, ok2, wasted, occupied = run_v2()
    print(f"入驻 {n2} 条，释放 {freed2} 条；同样 {need} token 新序列："
          f"空闲 {free2} 槽位，离散页随便挑 -> {'成功' if ok2 else '失败'}")
    print(f"在住序列页内浪费 {wasted} 槽位，占实际占用 {occupied} 的 {wasted/occupied:.1%}")

    print(f"\n容量对比：静态预留 {capacity} 条 vs 分页 {n2} 条"
          f"（{n2/capacity:.1f} 倍），且分页无外部碎片")
