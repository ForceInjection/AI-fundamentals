#!/usr/bin/env python3
"""Estimate DeepSeek V4 (Pro/Flash) inference memory usage.

Based on config.json from HuggingFace:
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json

Key differences from traditional MHA/GQA/MLA:
  - K=V sharing (no 2x coefficient) + inverse RoPE
  - Cross-token compression: c4a (1/4) and c128a (1/128)
  - Heterogeneous layers: 30 c4a + 31 c128a
  - c4a layers have additional Indexer Cache
  - MoE: 384 routed experts + 1 shared, top-6 per token
  - Mixed precision: FP8 weights, FP8 KV, FP4 indexer (production)

Reference:
  - memory_analysis.md section 4.6
  - vllm/module_analysis/vllm_deepseek_v4.md
"""

import argparse, json, os, sys

def format_size(b):
    if b < 1024: return f"{b:,.0f} B"
    elif b < 1024**2: return f"{b/1024:,.2f} KiB"
    elif b < 1024**3: return f"{b/1024**2:,.2f} MiB"
    else: return f"{b/1024**3:,.2f} GiB"

def parse_args():
    p = argparse.ArgumentParser(description="DeepSeek V4 memory estimator")
    p.add_argument("--config", default=None, help="Path to config.json")
    p.add_argument("--gpu-mem-gib", type=float, default=80.0, help="GPU memory budget (GiB)")
    p.add_argument("--overhead-gib", type=float, default=2.0, help="Reserved overhead (GiB)")
    p.add_argument("--scenarios", type=int, nargs="*", default=[512,4096,32768,131072,1048576], help="Sequence lengths for B_max")
    p.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs")
    return p.parse_args()

DEFAULT_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json"
DEEPSEEK_V4_PRO = {"total_params": 1.6e12, "active_params_per_token": 37e9}

def load_config(path):
    if path and os.path.exists(path):
        with open(path) as f: return json.load(f)
    local = os.path.join(os.path.dirname(__file__), "deepseek_v4_pro_config.json")
    if os.path.exists(local):
        with open(local) as f: return json.load(f)
    try:
        import urllib.request
        print(f"Downloading config from {DEFAULT_URL} ...")
        return json.loads(urllib.request.urlopen(DEFAULT_URL).read())
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr); sys.exit(1)

def classify_layers(ratios):
    c4a, c128a, swa, other = [], [], [], []
    for i, r in enumerate(ratios):
        if r == 4: c4a.append(i)
        elif r == 128: c128a.append(i)
        elif r == 0: swa.append(i)
        else: other.append((i, r))
    return c4a, c128a, swa, other

def calculate_deepseek_v4_memory(args):
    cfg = load_config(args.config)

    L_total = cfg["num_hidden_layers"]
    H = cfg["hidden_size"]
    N_attn = cfg["num_attention_heads"]
    N_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("kv_lora_rank", cfg.get("head_dim", 512))
    d_kv = head_dim
    d_idx = cfg.get("index_head_dim", 128)
    idx_n_heads = cfg.get("index_n_heads", 64)
    idx_topk = cfg.get("index_topk", 1024)
    W = cfg.get("sliding_window", 128)
    max_S = cfg["max_position_embeddings"]
    vocab_size = cfg.get("vocab_size", 129280)

    compress_ratios = cfg.get("compress_ratios", [])
    if compress_ratios:
        c4a, c128a, swa_only, other = classify_layers(compress_ratios)
        L_c4a = len(c4a)
        L_c128a = len(c128a)
    else:
        L_c4a = 30
        L_c128a = 31

    print("=" * 72)
    print("  DeepSeek V4 — KV Cache & Memory Estimator")
    print("=" * 72)
    print()
    print("Model Configuration (from config.json)")
    print("-" * 44)
    print(f"  hidden_size (H)            = {H:,}")
    print(f"  num_hidden_layers (L)      = {L_total}  (c4a={L_c4a}, c128a={L_c128a})")
    print(f"  num_attention_heads        = {N_attn}")
    print(f"  num_key_value_heads (MLA)  = {N_kv}")
    print(f"  head_dim (kv_lora_rank)    = {d_kv}")
    print(f"  index_head_dim (d_idx)     = {d_idx}")
    print(f"  index_n_heads              = {idx_n_heads}")
    print(f"  index_topk                 = {idx_topk}")
    print(f"  sliding_window (W)         = {W}")
    print(f"  max_position_embeddings    = {max_S:,}  (~1M)")
    print(f"  vocab_size                 = {vocab_size:,}")
    if swa_only:
        print(f"  [WARN] {len(swa_only)} SWA-only layers (compress_ratio=0) not counted in KV.")
    if other:
        print(f"  [WARN] {len(other)} layers have unexpected compress_ratio: {other[:3]}...")
    print()

    # --- 1. Model Weights ---
    print("--- 1. Model Weights ---")
    print()
    notes = DEEPSEEK_V4_PRO
    P_total = notes["total_params"]
    P_active = notes["active_params_per_token"]
    b_w = 2
    mem_total_bf16 = P_total * b_w
    print(f"  Total params (MoE, 384 experts)       ≈ {P_total / 1e12:.1f}T")
    print(f"  Weight memory (BF16, total)            = {format_size(mem_total_bf16)}")
    print(f"    (requires expert-parallel + quantization)")

    b_w_fp8 = 1
    mem_total_fp8 = P_total * b_w_fp8
    per_gpu_fp8 = mem_total_fp8 / args.num_gpus
    print(f"  Weight memory (FP8, total)             = {format_size(mem_total_fp8)}")
    print(f"    With {args.num_gpus} GPUs: ~{format_size(per_gpu_fp8)}/GPU (expert-parallel)")

    mem_active_bf16 = P_active * b_w
    topk = cfg.get("num_experts_per_tok", 6)
    nrouted = cfg.get("n_routed_experts", 384)
    print(f"  Active params per token (~{P_active / 1e9:.0f}B)")
    print(f"    = {format_size(mem_active_bf16)} (BF16)")
    print(f"    (top-{topk} experts out of {nrouted} per token)")
    print()

    # --- 2. KV Cache: Per-Layer Formulas ---
    print("--- 2. KV Cache: Per-Layer Formulas (BF16) ---")
    print()
    b_kv = 2
    C_c4a = 4
    C_c128a = 128
    print(f"  KV precision   : BF16 ({b_kv} bytes)")
    print(f"  c4a  compress  : 1/{C_c4a}  ({L_c4a} layers, with Indexer)")
    print(f"  c128a compress : 1/{C_c128a}  ({L_c128a} layers, no Indexer)")
    print()

    sample_lengths = [
        ("1 token  (unit)    ", 1),
        ("4K tokens          ", 4096),
        ("32K tokens         ", 32768),
        ("128K tokens        ", 131072),
        ("1M tokens (max)    ", max_S),
    ]
    def kv_per_layer_bf16(S, C, has_indexer):
        window = min(S, W)
        base = b_kv * (window + S / C) * d_kv
        if has_indexer:
            base += b_kv * (S / C) * d_idx
        return base

    for label, S_val in sample_lengths:
        m_c4a = kv_per_layer_bf16(S_val, C_c4a, True)
        m_c128a = kv_per_layer_bf16(S_val, C_c128a, False)
        total_kv = L_c4a * m_c4a + L_c128a * m_c128a
        print(f"  {label} : KV = {format_size(total_kv)}  (B=1)")

    mem_c4a_1M = kv_per_layer_bf16(max_S, C_c4a, True)
    mem_c128a_1M = kv_per_layer_bf16(max_S, C_c128a, False)
    total_kv_1M_bf16 = L_c4a * mem_c4a_1M + L_c128a * mem_c128a_1M
    kv_per_token_avg = total_kv_1M_bf16 / max_S  # asymptotic per-token cost

    mla_per_layer = (d_kv + cfg.get("qk_rope_head_dim", 64)) * 2 + d_idx * 2
    mla_1M_total = L_total * mla_per_layer * max_S
    saving_pct = 100 * (1 - total_kv_1M_bf16 / mla_1M_total)

    print()
    print(f"  Asymp. per-token KV (BF16, S>>W)       = {format_size(kv_per_token_avg)}")
    print(f"  1M context total (BF16)                = {format_size(total_kv_1M_bf16)}")
    print(f"  Classic MLA equivalent (61-layer, 1M)  = {format_size(mla_1M_total)}")
    print(f"  Savings vs classic MLA                 = ~{saving_pct:.1f}%")
    print()

    # --- 3. Mixed Precision: FP8 KV + FP4 Indexer (Production) ---
    print("--- 3. Mixed Precision: FP8 KV + FP4 Indexer (Production) ---")
    print()

    b_kv_fp8 = 1.0
    b_idx_fp4 = 0.5

    def kv_per_layer_mixed(S, C, has_indexer):
        window = min(S, W)
        base = b_kv_fp8 * (window + S / C) * d_kv
        if has_indexer:
            base += b_idx_fp4 * (S / C) * d_idx
        return base

    mem_c4a_mixed_1M = kv_per_layer_mixed(max_S, C_c4a, True)
    mem_c128a_mixed_1M = kv_per_layer_mixed(max_S, C_c128a, False)
    total_kv_1M_mixed = L_c4a * mem_c4a_mixed_1M + L_c128a * mem_c128a_mixed_1M
    kv_per_token_mixed = total_kv_1M_mixed / max_S  # asymptotic

    print(f"  Asymp. per-token KV (fp8+fp4)          = {format_size(kv_per_token_mixed)}")
    print(f"  1M context (fp8 KV + fp4 indexer)      = {format_size(total_kv_1M_mixed)}")
    print(f"  vs BF16 all-precision                  = {format_size(total_kv_1M_bf16)}")
    print()

    # --- 4. Inventory Recap (KV-only) ---
    print("--- 4. KV Cache Capacity at Scale ---")
    print()

    print(f"  Unlike traditional models, DeepSeek V4's KV cache is")
    print(f"  dramatically compressed. For a data-parallel setup,")
    print(f"  the bottleneck is weight memory, not KV.")
    print()

    total_mixed_128K = (
        L_c4a * kv_per_layer_mixed(131072, C_c4a, True)
        + L_c128a * kv_per_layer_mixed(131072, C_c128a, False)
    )
    print(f"  KV per sequence at 1M (mixed)  = {format_size(total_kv_1M_mixed)}")
    print(f"  KV per sequence at 128K        = {format_size(total_mixed_128K)}")
    print()

    if total_kv_1M_mixed > 0:
        scenarios = sorted(set(s for s in args.scenarios if s <= max_S))
        print(f"  Token budget example (assume 40 GiB available for KV):")
        budget_kv = 40 * 1024**3
        for S in scenarios:
            kv_total_at_S = L_c4a * kv_per_layer_mixed(S, C_c4a, True) + L_c128a * kv_per_layer_mixed(S, C_c128a, False)
            num_reqs = int(budget_kv / kv_total_at_S) if kv_total_at_S > 0 else 999999
            print(f"  S={S:>10,}: KV/s = {format_size(kv_total_at_S)}, max concurrent requests ≈ {num_reqs:,}")
        print()

    # --- 5. Summary Comparison ---
    print("--- 5. Comparison: Old MLA vs DeepSeek V4 ---")
    print()

    print(f"  {'Metric':<35} {'Classic MLA (V3.2)':<22} {'DeepSeek V4':<22}")
    print(f"  {'-'*35} {'-'*22} {'-'*22}")
    print(f"  {'KV per layer per token':<35} "
          f"{mla_per_layer:<22,d} "
          f"{'4~128 (compressed)':<22}")
    print(f"  {'Asymp. KV/token (BF16)':<35} "
          f"{format_size(mla_per_layer):<22} "
          f"{format_size(kv_per_token_avg):<22}")
    print(f"  {'Total KV 1M context (BF16)':<35} "
          f"{format_size(mla_1M_total):<22} "
          f"{format_size(total_kv_1M_bf16):<22}")
    print(f"  {'KV 1M (fp8/fp4 prod.)':<35} "
          f"{'-':<22} "
          f"{format_size(total_kv_1M_mixed):<22}")
    print(f"  {'KV saving (BF16)':<35} "
          f"{'-':<22} "
          f"{saving_pct:.1f}%{'':<18}")

    print()
    print("=" * 72)
    print("  References:")
    print("    memory_analysis.md (§4.6)")
    print("    vllm/module_analysis/vllm_deepseek_v4.md")
    print("=" * 72)


if __name__ == "__main__":
    calculate_deepseek_v4_memory(parse_args())
