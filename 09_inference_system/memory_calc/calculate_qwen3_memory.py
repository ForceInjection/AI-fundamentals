import argparse

def format_size(bytes_val):
    """Helper to format bytes to human readable string"""
    if bytes_val < 1024:
        return f"{bytes_val} Bytes"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.2f} KiB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.2f} MiB"
    else:
        return f"{bytes_val/1024**3:.2f} GiB"

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate Qwen3-0.6B inference memory usage.")
    parser.add_argument("--gpu-mem-gib", type=float, default=40.0, help="GPU memory budget in GiB.")
    parser.add_argument(
        "--overhead-gib",
        type=float,
        default=1.4,
        help="Reserved overhead in GiB (CUDA context, activations, fragmentation buffer).",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        nargs="*",
        default=[512, 4096, 32768],
        help="Sequence lengths to estimate B_max for.",
    )
    return parser.parse_args()


def calculate_qwen3_memory(args):
    # Qwen3-0.6B Configuration
    # Ref: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
    config = {
        "model_name": "Qwen3-0.6B",
        "hidden_size": 1024,        # H
        "num_hidden_layers": 28,    # L
        "num_attention_heads": 16,  # N_attn
        "num_key_value_heads": 8,   # N_kv (GQA)
        "vocab_size": 151936,
        "torch_dtype": "bfloat16",  # 2 bytes
        "param_count_approx": 0.6 * 10**9 # 0.6B
    }

    print(f"--- Analysis for {config['model_name']} ---")
    print(f"Config: L={config['num_hidden_layers']}, H={config['hidden_size']}, "
          f"N_attn={config['num_attention_heads']}, N_kv={config['num_key_value_heads']}")
    
    # 1. Model Weights
    # Precision: BF16 -> 2 bytes
    b_w = 2
    mem_weights = config["param_count_approx"] * b_w
    print(f"\n1. Model Weights (approx):")
    print(f"   {mem_weights:,.0f} Bytes")
    print(f"   ≈ {format_size(mem_weights)}")
    
    # 2. KV Cache Per Token
    # Formula: 2 * b_kv * L * 1 * 1 * H * (N_kv / N_attn)
    # b_kv = 2 (BF16)
    b_kv = 2
    L = config["num_hidden_layers"]
    H = config["hidden_size"]
    ratio = config["num_key_value_heads"] / config["num_attention_heads"]
    
    kv_per_token = 2 * b_kv * L * 1 * 1 * H * ratio
    print(f"\n2. KV Cache Per Token:")
    print(f"   Formula: 2 * {b_kv} * {L} * {H} * ({config['num_key_value_heads']}/{config['num_attention_heads']})")
    print(f"   Value: {kv_per_token:,.0f} Bytes")
    print(f"   ≈ {format_size(kv_per_token)}")
    
    # Compare with Non-GQA (if N_kv = N_attn = 16)
    kv_per_token_no_gqa = 2 * b_kv * L * 1 * 1 * H * 1.0
    print(f"   (Without GQA: {format_size(kv_per_token_no_gqa)})")

    # 3. Case Study: GPU Memory Budget
    gpu_mem_total = args.gpu_mem_gib * 1024**3
    mem_overhead = args.overhead_gib * 1024**3
    mem_available = gpu_mem_total - mem_weights - mem_overhead
    
    print(f"\n3. Case Study: GPU Budget ({format_size(gpu_mem_total)})")
    print(f"   Reserved Overhead: {format_size(mem_overhead)}")
    print(f"   Available for KV: {format_size(mem_available)}")
    
    max_tokens = mem_available / kv_per_token
    print(f"   Max Token Capacity (Total): {max_tokens:,.0f} tokens")
    
    # Scenarios
    scenarios = args.scenarios
    print(f"\n   Max Concurrency (Batch Size) for different Sequence Lengths (S):")
    for S in scenarios:
        b_max = max_tokens / S
        print(f"   - S = {S:<5}: B_max ≈ {b_max:,.1f} -> {int(b_max)}")

if __name__ == "__main__":
    calculate_qwen3_memory(parse_args())
