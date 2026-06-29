#!/usr/bin/env python3
"""
KV Cache size calculator for all attention types covered in
attention_kv_cache_formats.md.

Each function takes model config and returns per-token-per-layer sizes plus
full-model estimates. Run with no arguments to verify the article's numbers.

Usage:
  python3 kv_cache_calc.py                    # verify article numbers
  python3 kv_cache_calc.py --model llama2-70b # specific model
  python3 kv_cache_calc.py --seq-len 131072   # custom context length
"""

import argparse
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Model configs (from article and public model cards)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    name: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int      # 1 for MQA, < num_q_heads for GQA, = for MHA
    head_dim: int
    kv_lora_rank: int = 0  # MLA latent dim (0 = not MLA)
    qk_rope_head_dim: int = 0  # MLA decoupled RoPE dim
    dtype_bytes: int = 2    # FP16/BF16 = 2, FP8 = 1, FP4 = 0.5

    @property
    def is_mla(self):
        return self.kv_lora_rank > 0

    @property
    def is_gqa(self):
        return not self.is_mla and self.num_kv_heads < self.num_q_heads

    @property
    def is_mqa(self):
        return not self.is_mla and self.num_kv_heads == 1

    @property
    def attn_type(self):
        if self.is_mla:
            return "MLA"
        if self.num_kv_heads == 1:
            return "MQA"
        if self.num_kv_heads < self.num_q_heads:
            return "GQA"
        return "MHA"


CONFIGS = {
    "llama2-70b-mha": Config(
        name="LLaMA-2 70B (hypothetical MHA)",
        num_layers=80, num_q_heads=64, num_kv_heads=64, head_dim=128),
    "llama2-70b": Config(
        name="LLaMA-2 70B (GQA)",
        num_layers=80, num_q_heads=64, num_kv_heads=8, head_dim=128),
    "mqa-example": Config(
        name="MQA example (32 heads)",
        num_layers=80, num_q_heads=32, num_kv_heads=1, head_dim=128),
    "deepseek-v3": Config(
        name="DeepSeek-V3 (MLA)",
        num_layers=61, num_q_heads=128, num_kv_heads=128, head_dim=128,
        kv_lora_rank=512, qk_rope_head_dim=64),
}

# ---------------------------------------------------------------------------
# Calculators
# ---------------------------------------------------------------------------

def mha_gqa_kv(cfg: Config):
    """Standard MHA/GQA/MQA: K and V stored directly as (num_kv_heads, head_dim)."""
    k_size = cfg.num_kv_heads * cfg.head_dim * cfg.dtype_bytes
    v_size = cfg.num_kv_heads * cfg.head_dim * cfg.dtype_bytes
    total = k_size + v_size
    return {
        "k_per_token": k_size,
        "v_per_token": v_size,
        "total_per_token_layer": total,
        "description": f"K({cfg.num_kv_heads},{cfg.head_dim}) + V({cfg.num_kv_heads},{cfg.head_dim})",
    }


def mla_kv(cfg: Config):
    """MLA: KV latent (compressed) + decoupled RoPE K."""
    latent_size = cfg.kv_lora_rank * cfg.dtype_bytes
    rope_k_size = cfg.num_q_heads * cfg.qk_rope_head_dim * cfg.dtype_bytes
    total = latent_size + rope_k_size
    return {
        "kv_latent_per_token": latent_size,
        "rope_k_per_token": rope_k_size,
        "total_per_token_layer": total,
        "description": f"KV latent({cfg.kv_lora_rank}) + RoPE K({cfg.num_q_heads},{cfg.qk_rope_head_dim})",
    }


def compute_kv(cfg: Config, seq_len: int = 4096):
    """Compute per-token and full-model KV Cache for a config."""
    if cfg.is_mla:
        d = mla_kv(cfg)
    else:
        d = mha_gqa_kv(cfg)

    per_token = d["total_per_token_layer"]
    full_model = per_token * cfg.num_layers * seq_len

    d.update({
        "config": cfg.name,
        "type": cfg.attn_type,
        "num_layers": cfg.num_layers,
        "seq_len": seq_len,
        "full_model_bytes": full_model,
        "full_model_gb": full_model / (1024**3),
        "per_token_kb": per_token / 1024,
    })
    return d


def print_row(d):
    """Print one model's KV cache stats."""
    pt = d["per_token_kb"]
    fm = d["full_model_gb"]
    print(f"  {d['type']:<5s} | {d['description'][:45]:<45s} | "
          f"{pt:>7.2f} KB | {fm:>7.2f} GB | seq_len={d['seq_len']:,}")


# ---------------------------------------------------------------------------
# CSA/HCA estimation
# ---------------------------------------------------------------------------

def csa_estimate(mla_per_token: float, compression_ratio: int, stride: int = 0):
    """Estimate CSA compressed KV Cache.

    Args:
        mla_per_token: MLA per-token-per-layer KV size in bytes
        compression_ratio: number of tokens merged into 1 compressed token
        stride: sliding window stride (0 = no overlap)
    """
    overlap_factor = compression_ratio / stride if stride > 0 else 1
    return mla_per_token / compression_ratio * overlap_factor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KV Cache size calculator for attention_kv_cache_formats.md")
    parser.add_argument("--model", "-m", default="all",
                        choices=list(CONFIGS.keys()) + ["all"],
                        help="Model config to compute")
    parser.add_argument("--seq-len", "-s", type=int, default=4096,
                        help="Sequence length (default: 4096)")
    parser.add_argument("--extended", "-x", type=int, default=1_000_000,
                        help="Extended context for 1M scenario (default: 1000000)")
    args = parser.parse_args()

    models = list(CONFIGS.values()) if args.model == "all" else [CONFIGS[args.model]]

    # ── Per-token / per-layer ──
    print("═══ Per-Token KV Cache (single layer) ═══")
    print(f"  {'Type':<5s} | {'Storage':<45s} | {'per token':>7s} | "
          f"{'full model':>7s} |")
    print(f"  {'':5s} | {'':45s} | {'(fp16)':>7s} | {'(fp16)':>7s} |")
    print("-" * 80)
    for cfg in models:
        d = compute_kv(cfg, args.seq_len)
        print_row(d)

    # ── CSA/HCA estimates (from vLLM blog: https://vllm.ai/blog/2026-04-24-deepseek-v4) ──
    print()
    print("═══ CSA/HCA Compression (DeepSeek V4, from vLLM blog) ═══")
    # Blog: per compressed entry = 64 bytes (shared-KV) + 8 bytes (indexer for c4a)
    # c4a effective compression = 4:1 (8:1 with stride 4 overlap)
    # c128a compression = 128:1
    entry_kv = 64   # shared-KV per compressed entry (bytes)
    entry_idx = 8   # indexer per c4a compressed entry (bytes)
    c4a_effective = 4  # 8:1 compression with stride 4 → 2× redundancy → effective 4:1
    c128a_ratio = 128

    c4a_per_token = (entry_kv + entry_idx) / c4a_effective
    c128a_per_token = entry_kv / c128a_ratio
    blended = (30 * c4a_per_token + 31 * c128a_per_token) / 61

    print(f"  Source: vLLM blog (bf16, 61 layers: 30×c4a + 31×c128a)")
    print(f"  Per compressed entry: {entry_kv}B (shared-KV) + {entry_idx}B (c4a indexer)")
    print(f"  c4a  per original token: {c4a_per_token:.1f} B = {c4a_per_token/1024:.4f} KB")
    print(f"  c128a per original token: {c128a_per_token:.1f} B = {c128a_per_token/1024:.4f} KB")
    print(f"  Blended average: {blended:.1f} B = {blended/1024:.4f} KB")
    print(f"  Full model @ 1M context (blog): {9.62} GiB")
    print(f"  Per-token-per-layer from total: "
          f"{9.62*1024**3/(61*1_000_000):.0f} B = {9.62*1024**3/(61*1_000_000)/1024:.3f} KB")
    print()

    # Also show raw fp16 MLA baseline for comparison
    mla_base = compute_kv(CONFIGS["deepseek-v3"], args.extended)
    mla_pt = mla_base["total_per_token_layer"]
    print(f"  MLA fp16 baseline: {mla_pt/1024:.2f} KB/token/layer")
    print(f"  MLA fp16 full @ 1M: {mla_pt * CONFIGS['deepseek-v3'].num_layers * args.extended / (1024**3):.1f} GB")
    print(f"  V3.2 deployed (FP8, blog): 576 B/token/layer → ~83.9 GiB @ 1M")

    # ── Section VI comparison table ──
    print()
    print("═══ §VI Comparison Table Verification ═══")
    print(f"  {'Type':<8s} | {'per token':>10s} | {'vs MHA':>8s} | "
          f"{'1M context':>12s} |")
    print("-" * 50)
    rows = [
        ("MHA", CONFIGS["llama2-70b-mha"]),
        ("GQA", CONFIGS["llama2-70b"]),
        ("MQA", CONFIGS["mqa-example"]),
        ("MLA", CONFIGS["deepseek-v3"]),
    ]
    mha_pt = None
    for label, cfg in rows:
        d = compute_kv(cfg, 1)  # per-token only
        pt_kb = d["per_token_kb"]
        if label == "MHA":
            mha_pt = d["total_per_token_layer"]
        ratio = mha_pt / d["total_per_token_layer"] if mha_pt else 0
        full_1m = d["total_per_token_layer"] * cfg.num_layers * args.extended / (1024**3)
        print(f"  {label:<8s} | {pt_kb:>7.2f} KB | {ratio:>6.1f}× | "
              f"{full_1m:>9.1f} GB |")
    # CSA/HCA (from vLLM blog, bf16)
    csa_blended = 169  # bytes/token/layer (9.62 GiB / 61 layers / 1M tokens)
    csa_pt_kb = csa_blended / 1024
    csa_ratio = mha_pt / csa_blended if mha_pt else 0
    print(f"  {'CSA/HCA':<8s} | {csa_pt_kb:>7.2f} KB | {csa_ratio:>6.0f}× | "
          f"{'~9.6 (bf16)':>9s} |")
    print()
    print("  Note: 1M context assumes 80 layers for MHA/GQA/MQA, "
          "61 for MLA/CSA-HCA (MoE).")
    print("  CSA/HCA from vLLM blog (https://vllm.ai/blog/2026-04-24-deepseek-v4):")
    print("  9.62 GiB / 61 layers / 1M tokens = 169 B/token/layer (bf16).")
    print("  With FP8/fp4 quantization: ~5 GB total, ~84 B/token/layer.")


if __name__ == "__main__":
    main()
