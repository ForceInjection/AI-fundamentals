#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-V3-MoE模型基于vLLM + Ascend 910B2的推理部署方案 - 计算验证脚本

本脚本验证文档中所有关键计算公式的准确性，包括：
1. Dense权重计算
2. MLA KV Cache优化计算
3. 并发用户数计算
4. 吞吐量计算
5. Expert存储策略
6. 通信优化分析
7. 成本分析
"""

import math

def verify_dense_weights():
    """验证Dense权重计算"""
    print("=== Dense权重计算验证 ===")
    
    # 基础参数
    vocab_size = 129280
    hidden_size = 7168
    num_layers = 61
    intermediate_size = 18432
    bytes_per_param = 1.9  # FP16约2字节，考虑量化优化
    
    # Embedding层
    embedding_params = vocab_size * hidden_size
    
    # 每层参数（基于MLA架构）
    # MLA注意力机制
    mla_attention_params = (
        hidden_size * hidden_size +  # 输出投影
        hidden_size * (512 + 1536) +  # Down投影（KV和Q LoRA）
        512 * 16384 +  # KV Up投影
        1536 * 24576   # Q/K Up投影
    )
    
    # FFN层
    ffn_params = hidden_size * intermediate_size * 2
    
    # 单层总参数
    params_per_layer = mla_attention_params + ffn_params
    
    # 所有层参数
    all_layers_params = params_per_layer * num_layers
    
    # 输出层
    output_params = hidden_size * vocab_size
    
    # 总Dense参数
    total_dense_params = embedding_params + all_layers_params + output_params
    total_dense_gb = total_dense_params * bytes_per_param / (1024**3)
    total_dense_gib = total_dense_params * bytes_per_param / (1000**3) * (1000**3) / (1024**3)
    
    print(f"Embedding参数: {embedding_params/1e9:.1f}B")
    print(f"单层参数: {params_per_layer/1e6:.1f}M")
    print(f"所有层参数: {all_layers_params/1e9:.1f}B")
    print(f"输出层参数: {output_params/1e9:.1f}B")
    print(f"总Dense参数: {total_dense_params/1e9:.1f}B")
    print(f"Dense权重存储: {total_dense_gib:.1f}GiB")
    print(f"文档中的值: 46.2GiB")
    print(f"计算准确性: {'✓' if abs(total_dense_gib - 46.2) < 1.0 else '✗'}")
    print()
    
    return total_dense_gib

def verify_kv_cache():
    """验证MLA KV Cache计算"""
    print("=== MLA KV Cache计算验证 ===")
    
    # MLA架构参数
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    num_layers = 61
    bytes_per_element = 2  # FP16
    
    # MLA KV Cache计算
    mla_kv_per_token_bytes = (kv_lora_rank + qk_rope_head_dim) * num_layers * bytes_per_element
    mla_kv_per_token_mib = mla_kv_per_token_bytes / (1024**2)
    
    print(f"Hidden size: {7168}")
    print(f"Layers: {num_layers}")
    print(f"KV LoRA rank: {kv_lora_rank}")
    print(f"QK RoPE head dim: {qk_rope_head_dim}")
    print(f"MLA KV Cache/token: {mla_kv_per_token_bytes:,} bytes = {mla_kv_per_token_mib:.3f}MiB")
    print(f"文档中的值: 0.070MiB/token")
    print(f"计算准确性: {'✓' if abs(mla_kv_per_token_mib - 0.070) < 0.005 else '✗'}")
    print()
    
    # 传统Attention对比
    hidden_size = 7168
    traditional_kv_per_token_bytes = 2 * hidden_size * num_layers * bytes_per_element
    traditional_kv_per_token_mib = traditional_kv_per_token_bytes / (1024**2)
    compression_ratio = traditional_kv_per_token_mib / mla_kv_per_token_mib
    
    print(f"传统KV Cache/token: {traditional_kv_per_token_mib:.1f}MiB")
    print(f"MLA压缩比: {compression_ratio:.1f}×")
    print(f"文档中的压缩比: 约24×")
    print(f"压缩比准确性: {'✓' if abs(compression_ratio - 24) < 2 else '✗'}")
    print()
    
    return mla_kv_per_token_mib

def verify_expert_storage():
    """验证Expert存储计算"""
    print("=== Expert存储计算验证 ===")
    
    # Expert参数
    hidden_size = 7168
    moe_intermediate_size = 2048
    expert_layers = 58  # 61层总数 - 3层Dense层
    total_experts = 257  # 256路由专家 + 1共享专家
    bytes_per_param = 1.9
    
    # 单个Expert参数（SwiGLU结构）
    single_expert_params = hidden_size * moe_intermediate_size * 3  # Gate + Up + Down
    single_expert_storage_mib = single_expert_params * bytes_per_param / (1024**2)
    
    print(f"单个Expert参数: {single_expert_params/1e6:.2f}M")
    print(f"Expert数量: {total_experts}")
    print(f"总Expert参数: {single_expert_params * total_experts / 1e9:.1f}B")
    print(f"单个Expert存储: {single_expert_storage_mib:.1f}MiB")
    print(f"文档中的值: 84.0MiB")
    print(f"计算准确性: {'✓' if abs(single_expert_storage_mib - 84.0) < 5.0 else '✗'}")
    print()
    
    # 分层存储策略
    gpu_hot_experts = 16
    cpu_warm_experts = 64
    ssd_cold_experts = 176
    
    gpu_hot_cache_gib = gpu_hot_experts * single_expert_storage_mib / 1024
    cpu_warm_cache_gib = cpu_warm_experts * single_expert_storage_mib / 1024
    ssd_cold_storage_gib = ssd_cold_experts * single_expert_storage_mib / 1024
    
    print(f"GPU热缓存({gpu_hot_experts}个): {gpu_hot_cache_gib:.2f}GiB")
    print(f"CPU温缓存({cpu_warm_experts}个): {cpu_warm_cache_gib:.2f}GiB")
    print(f"SSD冷存储({ssd_cold_experts}个): {ssd_cold_storage_gib:.2f}GiB")
    print(f"文档中的值: 1.31GiB, 5.25GiB, 14.44GiB")
    print(f"存储计算准确性: {'✓' if abs(gpu_hot_cache_gib - 1.31) < 0.1 else '✗'}")
    print()
    
    return gpu_hot_cache_gib

def verify_throughput_calculation():
    """验证吞吐量计算"""
    print("=== 吞吐量计算验证 ===")
    
    # 基础参数
    activated_params = 37e9  # 37B激活参数
    flops_per_param = 2  # 每参数2 FLOPs
    single_card_tflops = 376  # Ascend 910B2算力
    tp_size = 8
    moe_efficiency = 0.5  # MoE效率
    
    # 单副本计算
    single_replica_effective_tflops = single_card_tflops * tp_size * moe_efficiency
    single_replica_throughput = single_replica_effective_tflops * 1e12 / (activated_params * flops_per_param)
    
    print(f"激活参数: {activated_params/1e9:.0f}B")
    print(f"FLOPs/参数: {flops_per_param}")
    print(f"单卡算力: {single_card_tflops} TFLOPS")
    print(f"MoE效率: {moe_efficiency*100:.1f}%")
    print(f"单副本有效算力: {single_replica_effective_tflops:.1f} TFLOPS")
    print(f"单副本吞吐量: {single_replica_throughput:,.0f} tokens/s")
    print(f"文档中的值: 20,324 tokens/s")
    print(f"计算准确性: {'✓' if abs(single_replica_throughput - 20324) < 500 else '✗'}")
    print()
    
    # DP副本计算
    dp_replicas = 3  # 24卡配置
    total_theoretical_throughput = single_replica_throughput * dp_replicas
    
    print(f"DP副本数: {dp_replicas}")
    print(f"总理论吞吐量: {total_theoretical_throughput:,.0f} tokens/s")
    print(f"文档中的值: 60,972 tokens/s")
    print(f"计算准确性: {'✓' if abs(total_theoretical_throughput - 60972) < 500 else '✗'}")
    print()
    
    # 实际效率
    actual_efficiency = 0.8
    actual_throughput = total_theoretical_throughput * actual_efficiency
    
    print(f"实际效率: {actual_efficiency*100:.1f}%")
    print(f"实际预期吞吐量: {actual_throughput:,.0f} tokens/s")
    print(f"文档中的值: 48,778 tokens/s")
    print(f"计算准确性: {'✓' if abs(actual_throughput - 48778) < 500 else '✗'}")
    print()
    
    return single_replica_throughput, total_theoretical_throughput, actual_throughput

def verify_memory_requirements():
    """验证显存需求计算"""
    print("=== 显存需求计算验证 ===")
    
    # 基础参数
    total_memory_gib = 59.6  # 显存效率（保守估计，与H20方案统一）
    memory_efficiency = 0.85
    available_memory_gib = total_memory_gib * memory_efficiency
    
    print(f"单卡总显存: {total_memory_gib}GiB")
    print(f"显存效率: {memory_efficiency*100:.1f}%")
    print(f"可用显存: {available_memory_gib:.2f}GiB")
    print()
    
    # 权重显存需求（统一使用24.8B Dense参数计算）
    dense_params_b = 24.8  # 24.8B Dense参数
    bytes_per_param = 2  # FP16
    dense_weight_gib = dense_params_b * bytes_per_param
    expert_hot_cache_gib = 1.31
    total_weight_gib = dense_weight_gib + expert_hot_cache_gib
    weight_per_card_gib = total_weight_gib / 8  # TP=8
    
    print(f"Dense参数: {dense_params_b}B")
    print(f"Dense权重: {dense_weight_gib}GiB")
    print(f"Expert热缓存: {expert_hot_cache_gib}GiB")
    print(f"总权重显存: {total_weight_gib:.2f}GiB")
    print(f"TP=8单卡权重: {weight_per_card_gib:.2f}GiB")
    print(f"文档中的值: 6.36GiB (基于统一计算)")
    print(f"计算准确性: {'✓' if abs(weight_per_card_gib - 6.36) < 0.1 else '✗'}")
    print()
    
    # KV Cache显存需求
    kv_per_token_mib = 0.070
    seq_len = 2048
    kv_per_user_gib = kv_per_token_mib * seq_len / 1024
    
    # 基于统一计算：可用KV显存 = (59.6GiB - 6.2GiB) × 8卡 × 0.8
    available_kv_memory = (total_memory_gib - weight_per_card_gib) * 8 * 0.8
    max_concurrent_users = available_kv_memory / kv_per_user_gib
    
    print(f"KV Cache/token: {kv_per_token_mib}MiB")
    print(f"序列长度: {seq_len}")
    print(f"KV Cache/用户: {kv_per_user_gib:.3f}GiB")
    print(f"可用KV显存: {available_kv_memory:.1f}GiB")
    print(f"最大并发用户: {max_concurrent_users:.0f}")
    print(f"文档中的值: 2,461用户")
    print(f"计算准确性: {'✓' if abs(max_concurrent_users - 2461) < 100 else '✗'}")
    print()
    
    return weight_per_card_gib, max_concurrent_users

def verify_communication_optimization():
    """验证通信优化分析"""
    print("=== 通信优化分析验证 ===")
    
    # 基础通信时间（TP=8）
    base_comm_time_ms = 45.9  # 原始通信时间
    hccs_optimized_time_ms = 43.1  # HCCS优化后通信时间
    mla_compression = 25  # MLA压缩比
    compressed_comm_time_ms = hccs_optimized_time_ms / mla_compression
    
    print(f"原始通信时间(TP=8): {base_comm_time_ms}ms")
    print(f"HCCS优化后时间: {hccs_optimized_time_ms}ms")
    print(f"MLA压缩比: {mla_compression}×")
    print(f"最终优化通信时间: {compressed_comm_time_ms:.2f}ms")
    print()
    
    # Overlap率分析
    overlap_rates = [0, 0.3, 0.5, 0.8, 0.95]
    print("Overlap率分析:")
    
    for overlap_rate in overlap_rates:
        effective_comm_time = compressed_comm_time_ms * (1 - overlap_rate)
        tokens_per_second = 1000 / effective_comm_time if effective_comm_time > 0 else float('inf')
        print(f"    {overlap_rate*100:2.0f}% overlap: {effective_comm_time:.2f}ms -> {tokens_per_second:8.0f} tokens/s")
    
    # 80% overlap的有效通信时间
    target_overlap = 0.8
    effective_comm_time_80 = compressed_comm_time_ms * (1 - target_overlap)
    
    print(f"\n80% overlap有效通信时间: {effective_comm_time_80:.2f}ms")
    print(f"文档中的值: 0.37ms")
    print(f"计算准确性: {'✓' if abs(effective_comm_time_80 - 0.37) < 0.05 else '✗'}")
    print()
    
    return effective_comm_time_80

def verify_cost_analysis():
    """验证成本分析"""
    print("=== 成本分析验证 ===")
    
    # 32卡配置成本（基于文档数据）
    hardware_cost_32_cards = 1280000  # 32卡硬件成本（元）
    annual_operating_cost = 345600  # 年运营成本（元）
    three_year_tco = hardware_cost_32_cards + annual_operating_cost * 3
    
    # 性能指标
    theoretical_throughput = 81296  # tokens/s
    actual_throughput = 65037  # tokens/s
    
    # 性能成本比
    cost_per_tokens_per_second = three_year_tco / actual_throughput
    
    print(f"32卡硬件成本: {hardware_cost_32_cards:,}元")
    print(f"年运营成本: {annual_operating_cost:,}元")
    print(f"3年TCO: {three_year_tco:,}元")
    print(f"理论吞吐量: {theoretical_throughput:,} tokens/s")
    print(f"实际吞吐量: {actual_throughput:,} tokens/s")
    print(f"每tokens/s成本: {cost_per_tokens_per_second:.1f}元")
    print(f"注：文档中未包含详细成本分析，此为基于假设数据的计算")
    print()
    
    return cost_per_tokens_per_second

def main():
    """主验证函数"""
    print("DeepSeek-V3-MoE模型基于vLLM + Ascend 910B2的推理部署方案 - 计算验证")
    print("=" * 80)
    print()
    
    # 执行所有验证
    verify_dense_weights()
    verify_kv_cache()
    verify_expert_storage()
    verify_throughput_calculation()
    verify_memory_requirements()
    verify_communication_optimization()
    verify_cost_analysis()
    
    print("=" * 50)
    print("验证完成！")
    print()
    print("注意事项:")
    print("1. 所有计算基于文档中的假设和参数")
    print("2. 实际部署时可能因硬件差异有所不同")
    print("3. MLA压缩比和overlap率是关键性能因素")
    print("4. 建议在实际环境中进行性能测试验证")

if __name__ == "__main__":
    main()