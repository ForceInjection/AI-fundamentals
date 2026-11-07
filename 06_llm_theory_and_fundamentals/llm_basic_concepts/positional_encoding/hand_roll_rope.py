"""手搓 RoPE：两种实现 + 四个实验（文章代码母本）。零依赖：纯 numpy。"""
import numpy as np

rng = np.random.default_rng(42)

# ---------- 实现 1：分块旋转（Su 论文原式，interleaved 配对） ----------
def rope_interleaved(x, pos, base=10000.0):
    """x: (..., d) 偶数维；pos: (...,)。配对 (x0,x1)(x2,x3)... 各自旋转 θ_i。"""
    d = x.shape[-1]
    half = d // 2
    theta = base ** (-np.arange(half) * 2 / d)
    cos, sin = np.cos(pos * theta), np.sin(pos * theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

# ---------- 实现 2：rotate_half（HF/主流框架，half 配对） ----------
def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)

def rope_half(x, pos, base=10000.0):
    d = x.shape[-1]
    half = d // 2
    theta = base ** (-np.arange(half) * 2 / d)
    cos, sin = np.cos(pos * theta), np.sin(pos * theta)
    x1, x2 = x[..., :half], x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x1 * sin + x2 * cos
    return out

# ---------- 对照：正弦绝对位置编码 ----------
def sinusoidal_pe(pos, d, base=10000.0):
    angles = np.outer(np.atleast_1d(pos), base ** (-np.arange(0, d, 2) * 2 / d))
    pe = np.zeros((len(np.atleast_1d(pos)), d))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe

d = 64
q = rng.standard_normal(d)
k = rng.standard_normal(d)

# ================= 实验 1：两种实现的「约定差异」 =================
print("== 实验 1：同一向量、同一位置，两种配对约定的输出 ==")
x = rng.standard_normal(d)
a = rope_interleaved(x, np.array([3]))
b = rope_half(x, np.array([3]))
print("interleaved 与 half 的输出逐元素差（非零=约定不同）:", f"{np.abs(a-b).max():.3f}")

# ================= 实验 2：相对性三组对照（同距、不同绝对位置） =================
print("\n== 实验 2：相对性三组对照（相对距离恒为 3，绝对位置 5/50/500/5000）==")
pe_all = sinusoidal_pe(np.arange(5006, dtype=np.float64), d)
vals = [float(pe_all[m] @ pe_all[m + 3]) for m in [5, 50, 500, 5000]]
print("  正弦编码直接内积  :", [f"{v:.6f}" for v in vals], "波动 0（相对性天然成立）")
rng_w = np.random.default_rng(42)
W_q = rng_w.standard_normal((d, d)) / np.sqrt(d)
W_k = rng_w.standard_normal((d, d)) / np.sqrt(d)
vals = [float((pe_all[m] @ W_q) @ (pe_all[m + 3] @ W_k)) for m in [5, 50, 500, 5000]]
print("  经可学习 W 变换后 :", [f"{v:.4f}" for v in vals], f"波动 {max(vals)-min(vals):.4f}（相对性被抹掉）")
for name, fn in [("interleaved", rope_interleaved), ("half", rope_half)]:
    vals = [float(np.dot(fn(q, np.array([m])), fn(k, np.array([m + 3]))))
            for m in [5, 50, 500, 5000]]
    spread = max(vals) - min(vals)
    print(f"  RoPE {name:12s}: {['%.10f' % v for v in vals]} 波动 {spread:.1e} ✓")

# ================= 实验 3：衰减真相（随机 vs 结构化） =================
print("\n== 实验 3：衰减真相（|score| 随相对距离）==")
trials = 20000
qb = rng.standard_normal((trials, d))
kb = rng.standard_normal((trials, d))
posq = rng.integers(0, 1000, size=(trials, 1))
for dist in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    scores = np.sum(qb * rope_half(kb, posq + dist), axis=1)
    print(f"  随机   q/k 距离 {dist:5d}: |score| 均值 {np.mean(np.abs(scores)):.4f}")
iq = np.arange(d // 2)
qs = np.concatenate([np.sin(iq / d * 8), np.cos(iq / d * 8)])
ks = np.concatenate([np.sin(iq / d * 8 + 0.7), np.cos(iq / d * 8 + 0.7)])
rows = [(dd, float(qs @ rope_half(ks, dd))) for dd in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]]
for dd, v in rows:
    print(f"  结构化 q/k 距离 {dd:5d}: score = {v:9.4f}")
half_mean = [np.mean([abs(v) for _, v in rows[:5]]), np.mean([abs(v) for _, v in rows[5:]])]
print(f"  结构化前半平均 |score| {half_mean[0]:.1f}，后半平均 {half_mean[1]:.1f}")

# ================= 实验 4：外推三策略的频率谱（4K→16K，s=4） =================
print("\n== 实验 4：外推三策略频率谱（d=64，base=10000，s=4）==")
h = d // 2
theta = 10000.0 ** (-np.arange(h) * 2 / d)          # 原始频率 θ_i
s = 4                                                # 目标外推倍数
pi_theta = theta / s                                 # 线性插值（PI）：全频压 4 倍
ntk_base = 10000.0 * s ** (d / (d - 2))              # NTK-aware：抬 base
ntk_theta = ntk_base ** (-np.arange(h) * 2 / d)
# YaRN 简化式：按波长 ramp，短波长不动、长波长插值
wavelength = 2 * np.pi / theta
ramp = np.clip((wavelength - 1) / (32 - 1), 0, 1)
yarn_theta = (1 - ramp) * theta + ramp * pi_theta

print(f"{'频率索引':>6s} {'原始 θ':>12s} {'PI θ/4':>12s} {'NTK':>12s} {'YaRN':>12s} {'波长':>10s}")
for i in [0, 4, 8, 16, 24, 31]:
    wl = 2 * np.pi / theta[i]
    print(f"{i:6d} {theta[i]:12.6f} {pi_theta[i]:12.6f} {ntk_theta[i]:12.6f} "
          f"{yarn_theta[i]:12.6f} {wl:10.2f}")
