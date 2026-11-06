# 手搓 Attention：四个版本与一致性验证
# 配套文章：02_gpu_programming/02_cuda/21_write_attention_from_scratch.md
# 运行：python 21_attention_versions.py（依赖 numpy；无需 GPU）
# 验证四件事：
#   1. v0（三重循环，float64 参考基准）与 v2（矩阵化 + 减 max）一致
#   2. v1（朴素 softmax）在分数偏大时溢出（热数据下输出全为 inf/NaN）
#   3. v3（每块各自 softmax 的朴素分块）结果有偏差——分块教学的固有 bug
#   4. v4（online softmax，部分和滚动）与全量结果一致
import numpy as np

rng = np.random.default_rng(42)
N, M, D, DV = 16, 64, 128, 32   # 16 个 query，64 个 key，维度 128，输出维度 32
BLOCK = 16

# 常规数据
Q = rng.standard_normal((N, D)).astype(np.float32)
K = rng.standard_normal((M, D)).astype(np.float32)
V = rng.standard_normal((M, DV)).astype(np.float32)
# 溢出演示用的「热」数据：幅度放大 10 倍，最大分数可到 300 上下
Q_hot = (Q * 10).astype(np.float32)
K_hot = (K * 10).astype(np.float32)

# 必须显式 float32：np.sqrt 返回 float64 标量，会把整条分数矩阵静默升成 float64
SCALE = np.float32(1.0 / np.sqrt(D))


def softmax_naive(x):
    """不减 max 的 softmax（fp32 危险版）。"""
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_safe(x):
    """减 max 的 softmax（fp32 安全版）。"""
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention_v0(Q, K, V):
    """版本 0：三重循环（float64 累加器，作为参考基准）。"""
    n, m = Q.shape[0], K.shape[0]
    out = np.zeros((n, V.shape[1]), dtype=np.float64)
    for i in range(n):
        scores = np.zeros(m)
        for j in range(m):
            scores[j] = Q[i] @ K[j] * SCALE
        weights = softmax_safe(scores)
        for j in range(m):
            out[i] += weights[j] * V[j]
    return out


def attention_v1(Q, K, V):
    """版本 1：矩阵化 + 朴素 softmax。"""
    S = (Q @ K.T) * SCALE
    P = softmax_naive(S)
    return P @ V


def attention_v2(Q, K, V):
    """版本 2：矩阵化 + 减 max（日常正确版）。"""
    S = (Q @ K.T) * SCALE
    P = softmax_safe(S)
    return P @ V


def attention_v3(Q, K, V, block=BLOCK):
    """版本 3：朴素分块（每块各自 softmax）——有正确性 bug。"""
    outs = []
    for s in range(0, K.shape[0], block):
        S_blk = (Q @ K[s:s + block].T) * SCALE
        outs.append(softmax_safe(S_blk) @ V[s:s + block])
    return np.sum(outs, axis=0)


def attention_v4(Q, K, V, block=BLOCK):
    """版本 4：online softmax（分块 + 部分和滚动）。"""
    n = Q.shape[0]
    out = np.zeros((n, V.shape[1]), dtype=np.float32)
    m_run = np.full(n, -np.inf, dtype=np.float32)   # 滚动行最大值
    l_run = np.zeros(n, dtype=np.float32)           # 滚动分母
    for s in range(0, K.shape[0], block):
        S_blk = (Q @ K[s:s + block].T) * SCALE
        m_blk = np.max(S_blk, axis=1)
        m_new = np.maximum(m_run, m_blk)
        p = np.exp(S_blk - m_new[:, None])          # 本块未归一权重
        rescale = np.exp(m_run - m_new)             # 旧结果的重标定系数
        out = out * rescale[:, None] + p @ V[s:s + block]
        l_run = l_run * rescale + np.sum(p, axis=1)
        m_run = m_new
    return out / l_run[:, None]


if __name__ == "__main__":
    print("== 常规数据：各版与 float64 参考基准对比 ==")
    ref = attention_v0(Q, K, V)
    print("v2 vs v0:", f"{np.abs(attention_v2(Q, K, V) - ref).max():.1e}")
    print("v4 vs v0:", f"{np.abs(attention_v4(Q, K, V) - ref).max():.1e}")
    print("v3 vs v2:", f"{np.abs(attention_v3(Q, K, V) - attention_v2(Q, K, V)).max():.2f}", "（朴素分块的固有 bug）")

    print("\n== 溢出演示（热数据，fp32）==")
    smax = ((Q_hot @ K_hot.T) * SCALE).max()
    print(f"最大分数 = {smax:.0f}（> 88.7 即溢出 fp32 的 exp）")
    bad = attention_v1(Q_hot, K_hot, V)
    print(f"v1 朴素 softmax：非有限元素 {int(np.sum(~np.isfinite(bad)))} / {bad.size}")
    good = attention_v2(Q_hot, K_hot, V)
    print(f"v2 减 max：非有限元素 {int(np.sum(~np.isfinite(good)))}，"
          f"与 float64 参考最大偏差 {np.abs(good - attention_v0(Q_hot, K_hot, V)).max():.1e}")
