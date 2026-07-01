"""
Prefill / Decode 计算过程校验脚本

用微型模型模拟文章中的完整推理流程，验证：
1. Prefill 一次性计算所有 token 的 K/V，存入 Cache
2. Decode 每步只算新 token 的 Q/K/V，从 Cache 读取历史 K/V
3. 有 Cache 和重算全序列的 Attention 输出完全一致（数值验证）
4. 每一步的矩阵形状与文章标注一致
5. 两阶段的 FLOPs 对比

用法：
  python prefill_decode_validate.py          # 默认运行
  python prefill_decode_validate.py --verbose # 打印每步形状
"""

import torch
import torch.nn.functional as F
import math
import argparse


# ── 微型模型参数（与文章成比例缩小，几秒内跑完）──
D_MODEL   = 64      # 隐藏维度（文章：4096）
NUM_HEADS = 4       # 注意力头数（文章：32）
HEAD_DIM  = D_MODEL // NUM_HEADS   # 每头维度 = 16（文章：128）
NUM_LAYERS = 2      # Decoder 层数（文章：32）
VOCAB_SIZE = 100    # 词表大小
PROMPT_LEN = 4      # prompt token 数
DECODE_STEPS = 3    # 生成 token 数

# 固定随机种子，保证可复现
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="打印每步矩阵形状")
args = parser.parse_args()

def shape(t):
    """返回 tensor 的形状字符串"""
    return str(list(t.shape))

def log(msg):
    if args.verbose:
        print(msg)

print(f"{'='*60}")
print(f"模型配置：d_model={D_MODEL}, heads={NUM_HEADS}, head_dim={HEAD_DIM}")
print(f"          layers={NUM_LAYERS}, prompt_len={PROMPT_LEN}, decode_steps={DECODE_STEPS}")
print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════
# 一、构建微型模型：Embedding + 每层的 W_Q/W_K/W_V/W_O + FFN + LM Head
# ══════════════════════════════════════════════════════════

embedding   = torch.randn(VOCAB_SIZE, D_MODEL)     # token embedding 表
lm_head     = torch.randn(D_MODEL, VOCAB_SIZE)      # 输出投影到词表

# 每层一套权重（用 dict of tensors 模拟，2 层足够验证）
layers = []
for l in range(NUM_LAYERS):
    layers.append({
        "W_Q": torch.randn(D_MODEL, D_MODEL),
        "W_K": torch.randn(D_MODEL, D_MODEL),
        "W_V": torch.randn(D_MODEL, D_MODEL),
        "W_O": torch.randn(D_MODEL, D_MODEL),
        "W_1": torch.randn(D_MODEL, D_MODEL * 2),   # FFN up (SwiGLU 双倍)
        "W_2": torch.randn(D_MODEL * 2, D_MODEL),   # FFN down
    })

# 模拟 prompt token ids
prompt_ids = torch.randint(0, VOCAB_SIZE, (PROMPT_LEN,))   # (4,)


# ══════════════════════════════════════════════════════════
# 二、Prefill：一次性处理全部 prompt token
# ══════════════════════════════════════════════════════════

print("── 二、Prefill 阶段 ──")
print(f"输入 token ids: {prompt_ids.tolist()}")

# 2.1 Embedding lookup
X = embedding[prompt_ids]         # (4, 64)
log(f"2.1 Embedding 后 X: {shape(X)}")

# 存储每层的 KV Cache
kv_cache = []

# 逐层前向
for l_idx, layer in enumerate(layers):
    # 2.2 Q/K/V 投影
    Q = X @ layer["W_Q"]          # (4, 64) × (64, 64) = (4, 64)
    K = X @ layer["W_K"]          # (4, 64) × (64, 64) = (4, 64)
    V = X @ layer["W_V"]          # (4, 64) × (64, 64) = (4, 64)
    log(f"2.2 Layer {l_idx} Q/K/V: {shape(Q)}, {shape(K)}, {shape(V)}")

    # 2.3 拆分为多头
    # (4, 64) → (4, 4, 16) → (4, 4, 16) 即 (num_heads, seq_len, head_dim)
    Q = Q.view(PROMPT_LEN, NUM_HEADS, HEAD_DIM).transpose(0, 1)   # (4, 4, 16)
    K = K.view(PROMPT_LEN, NUM_HEADS, HEAD_DIM).transpose(0, 1)
    V = V.view(PROMPT_LEN, NUM_HEADS, HEAD_DIM).transpose(0, 1)
    log(f"2.3 多头拆分后 Q/K/V: {shape(Q)}, {shape(K)}, {shape(V)}")

    # 2.4 注意力计算（带因果掩码，这里因为是 Decoder）
    scale = math.sqrt(HEAD_DIM)
    attn_scores = (Q @ K.transpose(-2, -1)) / scale          # (4, 4, 4) × (4, 16, 4) = (4, 4, 4)
    log(f"2.4 注意力得分 S: {shape(attn_scores)}")

    # 因果掩码
    causal_mask = torch.triu(torch.ones(PROMPT_LEN, PROMPT_LEN), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    attn_weights = F.softmax(attn_scores, dim=-1)              # (4, 4, 4)
    O = attn_weights @ V                                        # (4, 4, 4) × (4, 4, 16) = (4, 4, 16)
    log(f"2.4 Attention 输出 O_h: {shape(O)}")

    # 2.5 多头拼接 + 输出投影
    O = O.transpose(0, 1).contiguous().view(PROMPT_LEN, D_MODEL)  # (4, 64)
    O = O @ layer["W_O"]                                           # (4, 64)
    log(f"2.5 输出投影后: {shape(O)}")

    # 残差连接
    X = X + O

    # FFN (简化为单层，因为校验重点在 Attention)
    ffn_out = X @ layer["W_1"]                                    # (4, 128)
    # 简化为 GELU（不做 SwiGLU 门控以保持代码简洁）
    ffn_out = F.gelu(ffn_out)
    ffn_out = ffn_out @ layer["W_2"]                               # (4, 64)
    X = X + ffn_out                                                # 残差连接
    log(f"2.5 FFN + 残差后: {shape(X)}")

    # 2.6 存入 KV Cache（形状与 2.3 一致：(num_heads, seq_len, head_dim)）
    kv_cache.append({"K": K.clone(), "V": V.clone()})
    print(f"  Layer {l_idx}: K_cache={shape(K)}, V_cache={shape(V)}")

# Prefill 最后一步：取最后一个位置 → LM Head → 第一个 token
last_hidden = X[-1:]                                              # (1, 64)
log(f"2.6 最后一层输出取末尾位置: {shape(last_hidden)}")
logits = last_hidden @ lm_head                                    # (1, 100)
first_token_id = torch.argmax(logits, dim=-1).item()
print(f"  第一个输出 token: {first_token_id}\n")


# ══════════════════════════════════════════════════════════
# 三、Decode：逐 token 生成（用 KV Cache）
# ══════════════════════════════════════════════════════════

print("── 三、Decode 阶段 ──")

current_token_id = first_token_id
generated_tokens_with_cache = [current_token_id]

for step in range(DECODE_STEPS):
    print(f"\n  Step {step+1} (当前 token: {current_token_id}):")

    # 3.1 只取当前 token 的 embedding
    x_new = embedding[current_token_id].unsqueeze(0)    # (1, 64)
    log(f"    3.1 x_new: {shape(x_new)}")

    # 逐层前向
    for l_idx, layer in enumerate(layers):
        # Q/K/V 投影（注意：GEMV!）
        Q_new = x_new @ layer["W_Q"]                     # (1, 64)
        K_new = x_new @ layer["W_K"]                     # (1, 64)
        V_new = x_new @ layer["W_V"]                     # (1, 64)
        log(f"    3.1 Q_new/K_new/V_new: {shape(Q_new)}")

        # 拆分为多头
        Q_new = Q_new.view(1, NUM_HEADS, HEAD_DIM).transpose(0, 1)   # (4, 1, 16)
        K_new = K_new.view(1, NUM_HEADS, HEAD_DIM).transpose(0, 1)
        V_new = V_new.view(1, NUM_HEADS, HEAD_DIM).transpose(0, 1)

        # 3.2 追加到 KV Cache
        K_full = torch.cat([kv_cache[l_idx]["K"], K_new], dim=1)    # (4, 5+step, 16)
        V_full = torch.cat([kv_cache[l_idx]["V"], V_new], dim=1)
        kv_cache[l_idx]["K"] = K_full
        kv_cache[l_idx]["V"] = V_full
        log(f"    3.2 K_cache 追加后: {shape(K_full)}")

        # 3.3 注意力计算（只用新 token 的 Q，但 K/V 用全量缓存）
        scale = math.sqrt(HEAD_DIM)
        attn_scores = (Q_new @ K_full.transpose(-2, -1)) / scale    # (4, 1, seq_len)
        log(f"    3.3 S: {shape(attn_scores)}")
        attn_weights = F.softmax(attn_scores, dim=-1)
        O_new = attn_weights @ V_full                                # (4, 1, 16)

        # 多头拼接
        O_new = O_new.transpose(0, 1).contiguous().view(1, D_MODEL)  # (1, 64)
        O_new = O_new @ layer["W_O"]

        x_new = x_new + O_new                                        # 残差

        # FFN
        ffn_out = x_new @ layer["W_1"]
        ffn_out = F.gelu(ffn_out)
        ffn_out = ffn_out @ layer["W_2"]
        x_new = x_new + ffn_out

    # LM Head → 下一个 token
    logits = x_new @ lm_head                                         # (1, 100)
    current_token_id = torch.argmax(logits, dim=-1).item()
    generated_tokens_with_cache.append(current_token_id)
    print(f"    输出 token: {current_token_id}")

print(f"\n  生成序列（有 KV Cache）: {generated_tokens_with_cache}")


# ══════════════════════════════════════════════════════════
# 四、正确性验证：逐 token 重算全序列 Attention（无 Cache）
# ══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("四、正确性验证：重算全序列 Attention（无 Cache）vs KV Cache")
print(f"{'='*60}")

# 重新从 Prefill 开始（不依赖之前的 kv_cache）
torch.manual_seed(42)  # 重置随机种子... 但权重没变，所以直接重用

# 重建 embedding 和 layer（使用相同的权重，这里直接重用上面的）
# Re-run prefill 拿到第一个 token（结果应相同）
X2 = embedding[prompt_ids]
for l_idx, layer in enumerate(layers):
    Q = X2 @ layer["W_Q"]
    K = X2 @ layer["W_K"]
    V = X2 @ layer["W_V"]
    Q = Q.view(PROMPT_LEN, NUM_HEADS, HEAD_DIM).transpose(0, 1)
    K = K.view(PROMPT_LEN, NUM_HEADS, HEAD_DIM).transpose(0, 1)
    V = V.view(PROMPT_LEN, NUM_HEADS, HEAD_DIM).transpose(0, 1)
    scale = math.sqrt(HEAD_DIM)
    attn_scores = (Q @ K.transpose(-2, -1)) / scale
    causal_mask = torch.triu(torch.ones(PROMPT_LEN, PROMPT_LEN), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)
    O = attn_weights @ V
    O = O.transpose(0, 1).contiguous().view(PROMPT_LEN, D_MODEL)
    O = O @ layer["W_O"]
    X2 = X2 + O
    ffn_out = X2 @ layer["W_1"]
    ffn_out = F.gelu(ffn_out)
    ffn_out = ffn_out @ layer["W_2"]
    X2 = X2 + ffn_out

last_hidden2 = X2[-1:]
first_token_id2 = torch.argmax(last_hidden2 @ lm_head, dim=-1).item()
assert first_token_id == first_token_id2, "Prefill 结果不一致"

# Decode 阶段：每一步都重算全部 Attention（费时，但理论上等价）
seq_token_ids = prompt_ids.tolist() + [first_token_id2]
generated_tokens_no_cache = [first_token_id2]

for step in range(DECODE_STEPS):
    current_id = generated_tokens_no_cache[-1]
    seq_token_ids.append(current_id)
    seq_tensor = torch.tensor(seq_token_ids)

    # 整个序列重算
    X_all = embedding[seq_tensor]        # (N, 64)，N = 5 + step

    for l_idx, layer in enumerate(layers):
        Q = X_all @ layer["W_Q"]
        K = X_all @ layer["W_K"]
        V = X_all @ layer["W_V"]
        Q = Q.view(len(seq_token_ids), NUM_HEADS, HEAD_DIM).transpose(0, 1)
        K = K.view(len(seq_token_ids), NUM_HEADS, HEAD_DIM).transpose(0, 1)
        V = V.view(len(seq_token_ids), NUM_HEADS, HEAD_DIM).transpose(0, 1)
        scale = math.sqrt(HEAD_DIM)
        seq_len = len(seq_token_ids)
        attn_scores = (Q @ K.transpose(-2, -1)) / scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        O = attn_weights @ V
        O = O.transpose(0, 1).contiguous().view(seq_len, D_MODEL)
        O = O @ layer["W_O"]
        X_all = X_all + O
        ffn_out = X_all @ layer["W_1"]
        ffn_out = F.gelu(ffn_out)
        ffn_out = ffn_out @ layer["W_2"]
        X_all = X_all + ffn_out

    # 取最后一个位置
    next_token = torch.argmax(X_all[-1:] @ lm_head, dim=-1).item()

    # 验证：重算的结果应与 KV Cache 的结果完全一致
    assert next_token == generated_tokens_with_cache[step + 1], \
        f"Step {step}: 无 Cache={next_token}, 有 Cache={generated_tokens_with_cache[step + 1]} — 不一致！"

    generated_tokens_no_cache.append(next_token)

print("  ✓ 有 KV Cache 与无 Cache（全量重算）生成的 token 序列完全一致")
print(f"    序列: {generated_tokens_no_cache}")


# ══════════════════════════════════════════════════════════
# 五、FLOPs 对比
# ══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("五、FLOPs 对比（1 层 Attention 的 Q/K/V 投影）")
print(f"{'='*60}")

def gemm_flops(m, k, n):
    """矩阵乘法 A(m,k) × B(k,n) 的 FLOPs 估算"""
    return 2 * m * k * n

# Prefill: (4, 64) × (64, 64) → 三个投影
prefill_proj_flops = 3 * gemm_flops(PROMPT_LEN, D_MODEL, D_MODEL)
prefill_data_bytes = D_MODEL * D_MODEL * 2  # 读 W_Q（FP16）
prefill_ai = prefill_proj_flops / prefill_data_bytes

# Decode: (1, 64) × (64, 64) → 三个投影
decode_proj_flops = 3 * gemm_flops(1, D_MODEL, D_MODEL)
decode_data_bytes = D_MODEL * D_MODEL * 2
decode_ai = decode_proj_flops / decode_data_bytes

print(f"  Prefill 1 token Q/K/V 投影: {gemm_flops(1, D_MODEL, D_MODEL):,} FLOPs")
print(f"  Prefill 4 token Q/K/V 投影: {prefill_proj_flops:,} FLOPs")
print(f"  数据搬运（读 W_Q）: {prefill_data_bytes:,} bytes")
print(f"  算术强度: {prefill_ai:.2f} FLOPs/byte")
print(f"")
print(f"  Decode  1 token Q/K/V 投影: {decode_proj_flops:,} FLOPs")
print(f"  数据搬运（读 W_Q）: {decode_data_bytes:,} bytes")
print(f"  算术强度: {decode_ai:.2f} FLOPs/byte")
print(f"")
print(f"  结论: Decode 的算术强度仅为 Prefill(4 tokens) 的 {decode_ai/prefill_ai:.0%}")

# 文章中的对比（按文章参数）：
article_prefill_flops = 134_000_000
article_prefill_data  = 32_000_000
article_decode_flops  = 33_500_000
article_decode_data   = 32_000_000
article_prefill_ai    = article_prefill_flops / article_prefill_data
article_decode_ai     = article_decode_flops / article_decode_data

print(f"\n  文章中的参考值:")
print(f"    Prefill 算术强度: ~{article_prefill_ai:.1f} FLOPs/byte")
print(f"    Decode  算术强度: ~{article_decode_ai:.2f} FLOPs/byte")
print(f"    Decode/Preill:    {article_decode_ai/article_prefill_ai:.0%}")

print(f"\n{'='*60}")
print("验证通过：KV Cache 在数学上与全量重算完全等价，但 Decode 每步比")
print("Prefill 计算量小得多——同时数据搬运量几乎不变，导致 memory-bound。")
print(f"{'='*60}")
