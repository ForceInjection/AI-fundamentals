"""
量化推理演示：INT8 / INT4

纯 Python/NumPy 实现，演示量化原理。
用法: python3 quantization_demo.py
"""

import numpy as np


# ── 对称量化 ──

def symmetric_quantize(x: np.ndarray, bits: int = 8):
    """对称量化：浮点 → INT，零点固定在 0"""
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(x)) / qmax
    if scale == 0:
        scale = 1e-9
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int8 if bits <= 8 else np.int32)
    return q, scale


def symmetric_dequantize(q: np.ndarray, scale: float):
    """对称反量化：INT → 浮点"""
    return q.astype(np.float32) * scale


# ── 非对称量化 ──

def asymmetric_quantize(x: np.ndarray, bits: int = 8):
    """非对称量化：浮点 → UINT，引入 zero_point"""
    qmax = 2 ** bits - 1
    x_min, x_max = np.min(x), np.max(x)
    scale = (x_max - x_min) / qmax
    if scale == 0:
        scale = 1e-9
    zero_point = int(np.round(-x_min / scale))
    zero_point = max(0, min(qmax, zero_point))
    q = np.clip(np.round(x / scale) + zero_point, 0, qmax).astype(np.uint8 if bits <= 8 else np.uint32)
    return q, scale, zero_point


def asymmetric_dequantize(q: np.ndarray, scale: float, zero_point: int):
    """非对称反量化：UINT → 浮点"""
    return (q.astype(np.float32) - zero_point) * scale


# ── 评估 ──

def quantization_error(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """计算量化误差指标"""
    diff = original - reconstructed
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "relative_rmse": float(np.sqrt(np.mean(diff ** 2)) / (np.std(original) + 1e-9)),
    }


def print_metrics(name: str, metrics: dict):
    print(f"  {name}:")
    print(f"    最大绝对误差: {metrics['max_abs_error']:.6f}")
    print(f"    平均绝对误差: {metrics['mean_abs_error']:.6f}")
    print(f"    RMSE:         {metrics['rmse']:.6f}")
    print(f"    相对 RMSE:    {metrics['relative_rmse']:.4f}")


# ── 演示 ──

def demo_symmetric():
    """演示对称量化"""
    print("=" * 60)
    print("  1. 对称量化 (Symmetric INT8)")
    print("=" * 60)

    # 模拟权重矩阵
    np.random.seed(42)
    w = np.random.randn(128, 256) * 0.3
    w[0, 0] = 2.5  # 添加一个 outlier

    q, scale = symmetric_quantize(w, bits=8)
    w_hat = symmetric_dequantize(q, scale)

    print(f"  原始张量: shape={w.shape}, range=[{w.min():.4f}, {w.max():.4f}]")
    print(f"  Scale: {scale:.6f}")
    print(f"  量化级别: {2**8} (INT8)")
    print(f"  理论精度: {scale:.6f} (最小可分辨变化)")
    print_metrics("对称 INT8", quantization_error(w, w_hat))

    # Outlier 的影响
    print(f"\n  Outlier 的影响:")
    print(f"    99.9% 的值在 [-{3*0.3:.2f}, {3*0.3:.2f}] 之间")
    print(f"    Outlier 2.5 将 scale 拉大到 {scale:.4f}")
    print(f"    正常值约 0.3 → q ≈ {int(0.3/scale)}，仅用了 {2*int(0.3/scale)}/{256} 个量化级别")
    print()


def demo_asymmetric():
    """演示非对称量化"""
    print("=" * 60)
    print("  2. 非对称量化 (Asymmetric UINT8)")
    print("=" * 60)

    # 模拟 ReLU 后的激活值（全是非负数）
    np.random.seed(42)
    a = np.maximum(0, np.random.randn(128, 256) * 0.5 + 0.8)

    q, scale, zp = asymmetric_quantize(a, bits=8)
    a_hat = asymmetric_dequantize(q, scale, zp)

    print(f"  原始张量: shape={a.shape}, range=[{a.min():.4f}, {a.max():.4f}]")
    print(f"  Scale: {scale:.6f}, Zero Point: {zp}")
    print(f"  量化范围: [0, 255] (UINT8)")
    print_metrics("非对称 UINT8", quantization_error(a, a_hat))

    # 对比对称量化在这个数据上的表现
    q_sym, scale_sym = symmetric_quantize(a, bits=8)
    a_sym = symmetric_dequantize(q_sym, scale_sym)
    err_sym = quantization_error(a, a_sym)
    err_asym = quantization_error(a, a_hat)
    print(f"\n  对称 vs 非对称（偏斜数据）:")
    print(f"    对称 RMSE: {err_sym['rmse']:.6f}  ← 零点浪费了量化范围")
    print(f"    非对称 RMSE: {err_asym['rmse']:.6f}  ← zero_point 补偿偏移")
    print()


def demo_per_channel():
    """演示 Per-Tensor vs Per-Channel"""
    print("=" * 60)
    print("  3. Per-Tensor vs Per-Channel 量化")
    print("=" * 60)

    # 模拟不同 channel 差异很大的权重
    np.random.seed(42)
    w = np.random.randn(64, 256).astype(np.float32)
    w[0:16] *= 0.1    # channels 0-15: 值很小
    w[16:32] *= 0.5   # channels 16-31: 中等
    w[32:48] *= 1.0   # channels 32-47: 正常
    w[48:64] *= 3.0   # channels 48-63: 值很大

    # Per-tensor: 一个 scale 管所有 channel
    q_pt, scale_pt = symmetric_quantize(w, bits=8)
    w_pt = symmetric_dequantize(q_pt, scale_pt)
    err_pt = quantization_error(w, w_pt)

    # Per-channel: 每个 channel 自己的 scale
    w_pc = np.zeros_like(w)
    scales_pc = []
    for i in range(w.shape[0]):
        qi, si = symmetric_quantize(w[i], bits=8)
        w_pc[i] = symmetric_dequantize(qi, si)
        scales_pc.append(si)
    err_pc = quantization_error(w, w_pc)

    print(f"  Per-Tensor: scale={scale_pt:.4f}")
    print(f"    RMSE:  {err_pt['rmse']:.6f}")
    print(f"    通道 0-15  (小值): scale 对它们太粗糙")
    print(f"    通道 48-63 (大值): scale 刚好")
    print()
    print(f"  Per-Channel: {len(scales_pc)} 个 scale")
    print(f"    scale 范围: [{min(scales_pc):.4f}, {max(scales_pc):.4f}]")
    print(f"    RMSE:  {err_pc['rmse']:.6f}")
    print(f"    每个通道独立 scale，各自最优")
    print(f"    额外开销: {len(scales_pc)} × 4 bytes = {len(scales_pc)*4} bytes (可忽略)")
    print()


def demo_int4():
    """演示 INT4 量化"""
    print("=" * 60)
    print("  4. INT4 量化（分组量化）")
    print("=" * 60)

    np.random.seed(42)
    w = np.random.randn(256, 256) * 0.3

    # INT8
    q8, s8 = symmetric_quantize(w, bits=8)
    w8 = symmetric_dequantize(q8, s8)
    err8 = quantization_error(w, w8)

    # INT4 (per-tensor)
    q4, s4 = symmetric_quantize(w, bits=4)
    w4 = symmetric_dequantize(q4, s4)
    err4 = quantization_error(w, w4)

    # INT4 (group-wise: 每组 128 个值独立量化)
    group_size = 128
    w4g = np.zeros_like(w)
    for i in range(0, len(w.flatten()), group_size):
        group = w.flatten()[i:i + group_size]
        qg, sg = symmetric_quantize(group, bits=4)
        w4g.flat[i:i + group_size] = symmetric_dequantize(qg, sg)
    err4g = quantization_error(w, w4g)

    print(f"  INT8 (per-tensor):      RMSE={err8['rmse']:.6f}")
    print(f"  INT4 (per-tensor):      RMSE={err4['rmse']:.6f}  ← 精度大幅下降")
    print(f"  INT4 (group-wise, 128): RMSE={err4g['rmse']:.6f}  ← 分组恢复精度")
    print()
    print(f"  INT4 量化级别: {2**4} 个 (只有 16 个离散值)")
    print(f"  Group-wise 开销: {len(w.flatten()) // group_size} 个 scale × 4 bytes = {len(w.flatten()) // group_size * 4} bytes")
    print()


def demo_memory_savings():
    """演示显存节省"""
    print("=" * 60)
    print("  5. 显存节省估算")
    print("=" * 60)

    models = [
        ("Qwen2.5-0.5B", 0.5),
        ("Qwen2.5-7B", 7.6),
        ("Qwen2.5-14B", 14.2),
        ("Qwen2.5-72B", 72.7),
    ]

    print(f"  {'模型':<18} {'FP16':>8} {'INT8':>8} {'INT4':>8} {'FP16':>8} {'INT8':>8} {'INT4':>8}")
    print(f"  {'':-<18} {'':->8} {'':->8} {'':->8} {'':->8} {'':->8} {'':->8}")
    print(f"  {'':18} {'(GB)':>8} {'(GB)':>8} {'(GB)':>8} {'(910B3)':>8} {'(910B3)':>8} {'(910B3)':>8}")

    for name, params_b in models:
        fp16_gb = params_b * 2
        int8_gb = params_b * 1
        int4_gb = params_b * 0.5
        fp16_fit = "✓" if fp16_gb < 64 else "✗"
        int8_fit = "✓" if int8_gb < 64 else "✗"
        int4_fit = "✓" if int4_gb < 64 else "✗"
        print(f"  {name:<18} {fp16_gb:>7.1f}  {int8_gb:>7.1f}  {int4_gb:>7.1f}  {fp16_fit:>8}  {int8_fit:>8}  {int4_fit:>8}")

    print()
    print(f"  910B3 HBM: 64 GB")
    print(f"  当前 7B FP16 可用（~14 GB），INT4 后仅 ~3.5 GB")
    print(f"  INT4 量化后 72B 模型约 36 GB——可在单卡上运行！")


if __name__ == "__main__":
    demo_symmetric()
    demo_asymmetric()
    demo_per_channel()
    demo_int4()
    demo_memory_savings()
