# 手搓 MLA：MHA / MQA / GQA / MLA 四版注意力，与吸收变换的等价性验证
#
# 用法：
#   python hand_roll_mla.py                       # 随机权重跑四版对照（教学演示）
#   python hand_roll_mla.py <模型目录>            # 加载真实权重做吸收等价性验证
#                                            （如 HF 下载的 GLM-4.7-Flash / DeepSeek-V3 层目录）
#
# 依赖：numpy（真实权重路径另需 torch 与 safetensors 读 bf16）
# 配套文章：《手搓 MLA：从 MHA 到 weight absorption，看懂 DeepSeek 的 KV 压缩》
import numpy as np

# ---------- 公共件 ----------

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def rmsnorm(x, w, eps=1e-5):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w

def rope_interleave(x, pos, theta=1e6):
    """交错对旋转（x: [T, H, Dh]），q 与 k 用同一组角度"""
    T, _, Dh = x.shape
    inv = 1.0 / theta ** (np.arange(0, Dh // 2) * 2.0 / Dh)
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = a * sin + b * cos
    return out

def attention(q, k, v, causal=True):
    """标准逐头注意力。q/k: [T,H,D]，v: [T,H,Dv]，返回 [T,H,Dv]"""
    T = q.shape[0]
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.einsum("thd,shd->hts", q, k) * scale
    if causal:
        scores = scores + np.triu(np.ones((T, T)), 1) * -1e30
    return np.einsum("hts,shd->thd", softmax(scores), v)

# ---------- 四版注意力（单层，共享输入形状 [T, d_model]）----------
# 为可比性，四版用同一组超参：heads H，q/k 头维 dk（含 rope），v 头维 dv。
# 差别只在 K/V 的「头数」：MHA=H，MQA=1，GQA=G，MLA=1（但带 latent 上投影）。

def mha(x, Wq, Wk, Wv, Wo, H, dk, dv, pos):
    T, D = x.shape
    q = (x @ Wq.T).reshape(T, H, dk)
    k = (x @ Wk.T).reshape(T, H, dk)
    v = (x @ Wv.T).reshape(T, H, dv)
    q[..., dk - 64:], k[..., dk - 64:] = (rope_interleave(q[..., dk - 64:], pos),
                                          rope_interleave(k[..., dk - 64:], pos))
    return attention(q, k, v).reshape(T, H * dv) @ Wo.T

def mqa_gqa(x, Wq, Wk, Wv, Wo, H, G, dk, dv, pos):
    """MQA 取 G=1；GQA 取 1<G<H。K/V 只有 G 个头，广播到 H 个查询头"""
    T, D = x.shape
    q = (x @ Wq.T).reshape(T, H, dk)
    k = (x @ Wk.T).reshape(T, G, dk)
    v = (x @ Wv.T).reshape(T, G, dv)
    q[..., dk - 64:] = rope_interleave(q[..., dk - 64:], pos)
    k[..., dk - 64:] = rope_interleave(k[..., dk - 64:], pos)
    rep = H // G
    k = np.repeat(k, rep, axis=1)   # G 个 K/V 头复制给 H 个查询头
    v = np.repeat(v, rep, axis=1)
    return attention(q, k, v).reshape(T, H * dv) @ Wo.T

def mla_unabsorbed(x, W_DQ, W_QN, W_UQ, W_DKV, W_KR, W_KVN, W_UKV, Wo, H, pos,
                   d_lora=512, d_nope=192, d_rope=64, d_v=256, eps=1e-5):
    """论文式 (13)：还原出每头 k/v 再做 attention；KV cache 只需存 [c_kv | k_pe]"""
    T, D = x.shape
    q_c = rmsnorm(x @ W_DQ.T, W_QN, eps) @ W_UQ.T          # Q 侧低秩压缩
    q = q_c.reshape(T, H, d_nope + d_rope)
    q_nope, q_pe = q[..., :d_nope], q[..., d_nope:]
    q_pe = rope_interleave(q_pe, pos)
    kv_a = x @ W_DKV.T                                      # 每层每 token 只缓存的部分开始
    c_kv = rmsnorm(kv_a, W_KVN, eps)                        # [T, d_lora] ← 缓存它
    k_pe = rope_interleave((x @ W_KR.T).reshape(T, 1, d_rope), pos).squeeze(1)
    kv_up = c_kv @ W_UKV.T                                  # ← 还原开销只在计算时发生
    kv_up = kv_up.reshape(T, H, d_nope + d_v)
    k_nope, v = kv_up[..., :d_nope], kv_up[..., d_nope:]
    k_pe_h = np.repeat(k_pe[:, None, :], H, axis=1)
    k = np.concatenate([k_nope, k_pe_h], axis=-1)
    qq = np.concatenate([q_nope, q_pe], axis=-1)
    return attention(qq, k, v).reshape(T, H * d_v) @ Wo.T

def mla_absorbed(x, W_DQ, W_QN, W_UQ, W_DKV, W_KR, W_KVN, W_UKV, Wo, H, pos,
                 d_lora=512, d_nope=192, d_rope=64, d_v=256, eps=1e-5):
    """解码用吸收形式：W^UK 吸进 Q 侧，直接对 latent 做内积；输出侧 W^UV 再还原。
       与 mla_unabsorbed 数学严格等价（结合律），cache 同样只有 [c_kv | k_pe]"""
    T, D = x.shape
    q_c = rmsnorm(x @ W_DQ.T, W_QN, eps) @ W_UQ.T
    q = q_c.reshape(T, H, d_nope + d_rope)
    q_nope, q_pe = q[..., :d_nope], q[..., d_nope:]
    q_pe = rope_interleave(q_pe, pos)
    c_kv = rmsnorm(x @ W_DKV.T, W_KVN, eps)
    k_pe = rope_interleave((x @ W_KR.T).reshape(T, 1, d_rope), pos).squeeze(1)
    W_UK = W_UKV.reshape(H, d_nope + d_v, d_lora)[:, :d_nope, :]   # [H, d_nope, d_lora]
    W_UV = W_UKV.reshape(H, d_nope + d_v, d_lora)[:, d_nope:, :]   # [H, d_v,  d_lora]
    q_lat = np.einsum("thd,hdc->thc", q_nope, W_UK)                # 吸收：进 latent 空间
    scale = 1.0 / np.sqrt(d_nope + d_rope)
    scores = (np.einsum("thc,sc->hts", q_lat, c_kv)
              + np.einsum("thd,shd->hts", q_pe, np.repeat(k_pe[:, None, :], H, 1))) * scale
    scores += np.triu(np.ones((T, T)), 1) * -1e30
    out_lat = np.einsum("hts,sc->thc", softmax(scores), c_kv)      # 对 latent 聚合
    return np.einsum("thc,hdc->thd", out_lat, W_UV).reshape(T, H * d_v) @ Wo.T

# ---------- 演示与验证 ----------

def random_demo():
    """随机小权重：四版跑通 + 缓存元素数对照"""
    rng = np.random.default_rng(0)
    T, D, H, dk, dv, G = 6, 128, 4, 96, 96, 2
    pos = np.arange(T, dtype=np.float64)
    W = {n: rng.standard_normal((o, i)) * 0.05
         for n, o, i in [("Wq", H * dk, D), ("Wk", H * dk, D), ("Wv", H * dv, D),
                         ("Wo", D, H * dv), ("Wk1", dk, D), ("Wv1", dv, D),
                         ("WkG", G * dk, D), ("WvG", G * dv, D)]}
    mha_out = mha(X := rng.standard_normal((T, D)), W["Wq"], W["Wk"], W["Wv"], W["Wo"], H, dk, dv, pos)
    mqa_gqa(X, W["Wq"], W["Wk1"], W["Wv1"], W["Wo"], H, 1, dk, dv, pos)   # MQA
    mqa_gqa(X, W["Wq"], W["WkG"], W["WvG"], W["Wo"], H, G, dk, dv, pos)   # GQA
    print(f"四版跑通。本形状（H={H}, dk={dk}, dv={dv}, G={G}）每 token 每层缓存元素：")
    print(f"  MHA = {H * (dk + dv)}  GQA = {G * (dk + dv)}  MQA = {dk + dv}")
    print(f"  MLA 缓存 = d_lora + d_rope = {512 + 64}（与头数、d_nope、d_v 全都无关）")
    print(f"MHA 输出形状 {mha_out.shape}")

def real_weight_verify(model_dir):
    """真实权重：吸收 vs 未吸收的数值一致性（GLM-4.7-Flash / DeepSeek-V3 等 MLA 模型第 0 层）"""
    import json, torch
    from safetensors import safe_open
    cfg = json.load(open(f"{model_dir}/config.json"))
    idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    NH = cfg["num_attention_heads"]
    d_lora, d_rope = cfg["kv_lora_rank"], cfg["qk_rope_head_dim"]
    d_nope, d_v = cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    eps = cfg.get("rms_norm_eps", 1e-5)

    def load(name):
        with safe_open(f"{model_dir}/{idx[name]}", framework="pt") as fh:
            return fh.get_tensor(name).float().numpy().astype(np.float64)  # bf16→fp32 无损

    P = "model.layers.0.self_attn."
    W_DQ, W_QN = load(P + "q_a_proj.weight"), load(P + "q_a_layernorm.weight")
    W_UQ = load(P + "q_b_proj.weight")
    W_DKV_KR = load(P + "kv_a_proj_with_mqa.weight")     # [d_lora+d_rope, D]，两矩阵拼存
    W_DKV, W_KR = W_DKV_KR[:d_lora], W_DKV_KR[d_lora:]
    W_KVN = load(P + "kv_a_layernorm.weight")
    W_UKV = load(P + "kv_b_proj.weight")
    Wo = load(P + "o_proj.weight")

    rng = np.random.default_rng(0)
    T = 6
    X = rng.standard_normal((T, W_DQ.shape[1]))
    pos = np.arange(T, dtype=np.float64)
    kw = dict(H=NH, pos=pos, d_lora=d_lora, d_nope=d_nope, d_rope=d_rope, d_v=d_v, eps=eps)
    a = mla_unabsorbed(X, W_DQ, W_QN, W_UQ, W_DKV, W_KR, W_KVN, W_UKV, Wo, **kw).astype(np.float64)
    b = mla_absorbed(X, W_DQ, W_QN, W_UQ, W_DKV, W_KR, W_KVN, W_UKV, Wo, **kw).astype(np.float64)
    diff = np.abs(a - b).max()
    cos = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
    print(f"权重形状：q_b{W_UQ.shape} kv_a{W_DKV_KR.shape} kv_b{W_UKV.shape} o{Wo.shape}")
    print(f"吸收 vs 未吸收（真实权重，fp64）：max|Δ| = {diff:.3e}  cosine = {cos:.12f}")
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    print(f"吸收 vs 未吸收（fp32）：max|Δ| = {np.abs(a32 - b32).max():.3e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        real_weight_verify(sys.argv[1])
    else:
        random_demo()
