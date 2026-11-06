# 手搓 BPE Tokenizer：字符级 -> 字节级 BPE -> tiktoken 对照
# 配套文章：hand_roll_bpe_tokenizer.md
# 运行：python hand_roll_bpe.py（纯标准库；最后的 tiktoken 对照段需 pip install tiktoken，
#       且首次运行会联网下载 cl100k 词表，离线环境会自动跳过）
# 验证三件事：
#   1. 字符级与 BPE 的编码-解码往返均无损
#   2. token 数随合并次数下降（压缩曲线），但合并过多会过拟合语料
#   3. tiktoken cl100k 在同一语料上的压缩率与字节级回退行为
CORPUS = (
    "the attention mechanism reads query keys and values. "
    "attention reads query keys values and caches them. "
    "transformer models read query key value caches. "
    "大模型的核心是注意力机制，注意力机制要读取 query、key 和 value。"
    "推理系统把 key 和 value 缓存在显存里，形成 KV cache。"
    "KV cache 越长，显存占用越大，推理越慢。"
) * 3   # 重复 3 份：真实训练语料总有重复，同时放大过拟合现象


# ---------- 版本 0：字符级 ----------
chars = sorted(set(CORPUS))
char_to_id = {c: i for i, c in enumerate(chars)}


def encode_v0(text):
    return [char_to_id[c] for c in text]


def decode_v0(ids):
    return "".join(chars[i] for i in ids)


# ---------- 版本 2：BPE（字节级起步，逐步合并最高频相邻对） ----------
def bpe_train(text, num_merges):
    ids = list(text.encode("utf-8"))
    merges = {}
    for step in range(num_merges):
        pairs = {}
        for a, b in zip(ids, ids[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        new_id = 256 + step
        i, out = 0, []
        while i < len(ids):            # 顺序扫描：处理重叠对（如 aaa）不踩坑
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
        merges[best] = new_id
    return merges


def bpe_encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        cand = {(a, b): merges[(a, b)] for a, b in zip(ids, ids[1:]) if (a, b) in merges}
        if not cand:
            break
        best = min(cand, key=lambda p: cand[p])   # 先学的合并优先
        best_id = cand[best]
        i, out = 0, []
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                out.append(best_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids


def bpe_decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), i in merges.items():
        vocab[i] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    n_bytes = len(CORPUS.encode("utf-8"))
    print(f"语料 {len(CORPUS)} 字符 / {n_bytes} UTF-8 字节（重复 3 份）")

    # 版本 0：字符级
    assert decode_v0(encode_v0(CORPUS)) == CORPUS
    print(f"v0 字符级：往返无损 ✓，词表 {len(chars)} 个字符，token 数 = {n_bytes}（字节级）")

    # 版本 2：BPE 压缩曲线与过拟合警告
    merges = bpe_train(CORPUS, 100)
    ids2 = bpe_encode(CORPUS, merges)
    assert bpe_decode(ids2, merges) == CORPUS
    print(f"v2 BPE 100 次合并：往返无损 ✓，词表 = {256 + len(merges)}，"
          f"token 数 = {len(ids2)}（压缩到 {len(ids2)/n_bytes:.0%}）")
    for k in [50, 100, 300]:
        m = bpe_train(CORPUS, k)
        n = len(bpe_encode(CORPUS, m))
        print(f"  {k:3d} 次合并: {n_bytes} 字节 -> {n} tokens（压缩到 {n/n_bytes:.0%}）")
    print("  注意：300 次合并时整段语料缩成 1 个 token：词表把语料背下来了（过拟合）")

    # tiktoken 对照（可选：需联网下载词表）
    print("\n== 与 tiktoken cl100k 对照 ==")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        t = enc.encode(CORPUS)
        print(f"tiktoken {tiktoken.__version__} | cl100k 词表 {enc.n_vocab} | "
              f"同语料 {len(t)} tokens（压缩到 {len(t)/n_bytes:.0%}）")
        for probe in ["KV cache", "注意力机制", "🤖"]:
            ids = enc.encode(probe)
            assert enc.decode(ids) == probe
            print(f"  {probe!r} -> {len(ids)} tokens -> 解码还原 ✓")
    except Exception as e:
        print(f"跳过（{type(e).__name__}）：pip install tiktoken 并联网后可复现此段")
