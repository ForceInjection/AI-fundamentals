"""
LLM 推理 on NPU: 加载 Qwen2.5-0.5B-Instruct 进行本地推理

用法:
  ASCEND_RT_VISIBLE_DEVICES=7 python3 llm_inference.py infer "什么是 NPU？"
  ASCEND_RT_VISIBLE_DEVICES=7 python3 llm_inference.py chat
  ASCEND_RT_VISIBLE_DEVICES=7 python3 llm_inference.py benchmark
"""

import argparse
import time

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(device: str = "npu:0"):
    """加载模型和 tokenizer 到 NPU"""
    if not torch.npu.is_available():
        raise RuntimeError(
            "NPU 不可用，请检查: 1) torch_npu 是否安装 2) CANN 环境变量是否加载 "
            "3) ASCEND_RT_VISIBLE_DEVICES 是否设置"
        )
    print(f"加载模型: {MODEL_NAME}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()

    elapsed = time.time() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e6
    mem = torch.npu.memory_allocated() / 1024**3
    print(f"模型加载完成 ({elapsed:.0f}s), 参数: {params:.0f}M, HBM: {mem:.1f} GB")
    return tokenizer, model


def run_inference(tokenizer, model, prompt: str,
                  max_new_tokens: int = 256,
                  temperature: float = 0.7,
                  device: str = "npu:0") -> tuple[str, float, int]:
    """单次推理，返回 (回答, 耗时, token数)"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.npu.synchronize()
    elapsed = time.time() - t0

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer, elapsed, len(generated_ids)


def interactive_chat(tokenizer, model, device: str = "npu:0"):
    """交互式对话"""
    print("\nLLM 交互对话 (输入 /quit 退出, /clear 清除历史)")
    messages = []

    while True:
        try:
            prompt = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt == "/clear":
            messages = []
            print("对话历史已清除")
            continue

        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        print("AI> ", end="", flush=True)
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.npu.synchronize()
        elapsed = time.time() - t0

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(answer)
        print(f"({len(generated_ids)} tokens, {elapsed:.1f}s, "
              f"{len(generated_ids)/elapsed:.1f} tok/s)")

        messages.append({"role": "assistant", "content": answer})


def benchmark(tokenizer, model, device: str = "npu:0"):
    """性能测试：首 token 延迟、生成速度、HBM 占用"""
    print(f"\n{'='*60}")
    print(f"  性能 Benchmark")
    print(f"{'='*60}")

    prompts = [
        "什么是深度学习？",
        "请用 Python 写一个快速排序算法。",
        "介绍一下华为昇腾 NPU 的特点。",
    ]

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        torch.npu.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.npu.synchronize()
        elapsed = time.time() - t0

        gen_tokens = outputs.shape[1] - input_len
        answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        preview = answer[:60].replace("\n", " ")

        print(f"\n  Prompt: {prompt}")
        print(f"  回答: {preview}...")
        print(f"  Input: {input_len} tokens | Generated: {gen_tokens} tokens")
        print(f"  Time: {elapsed:.1f}s | Speed: {gen_tokens/elapsed:.1f} tok/s")

    peak_mem = torch.npu.max_memory_allocated() / 1024**3
    print(f"\n  HBM 峰值: {peak_mem:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="LLM 推理 on NPU")
    sub = parser.add_subparsers(dest="cmd")

    p_infer = sub.add_parser("infer", help="单次推理")
    p_infer.add_argument("prompt", help="输入提示")
    p_infer.add_argument("--max-tokens", type=int, default=256)
    p_infer.add_argument("--temperature", type=float, default=0.7)
    p_infer.add_argument("--device", default="npu:0")

    p_chat = sub.add_parser("chat", help="交互对话")
    p_chat.add_argument("--device", default="npu:0")

    p_bench = sub.add_parser("benchmark", help="性能测试")
    p_bench.add_argument("--device", default="npu:0")

    args = parser.parse_args()

    if args.cmd not in ("infer", "chat", "benchmark"):
        parser.print_help()
        return

    tokenizer, model = load_model(args.device)

    if args.cmd == "infer":
        answer, elapsed, n_tokens = run_inference(
            tokenizer, model, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device,
        )
        print(f"\n{answer}")
        print(f"\n({n_tokens} tokens, {elapsed:.1f}s, "
              f"{n_tokens/elapsed:.1f} tok/s)")

    elif args.cmd == "chat":
        interactive_chat(tokenizer, model, args.device)

    elif args.cmd == "benchmark":
        benchmark(tokenizer, model, args.device)


if __name__ == "__main__":
    main()
