"""
从 Ascend 文档生成指令微调数据集

用法:
  1. 生成问题: ASCEND_RT_VISIBLE_DEVICES=7 python3 gen_sft_data.py generate
  2. 仅统计:   python3 gen_sft_data.py stats

输出格式 (JSONL):
  {"instruction": "什么是达芬奇架构？", "input": "", "output": "昇腾 AI Core 是...", "source": "02-ascend-architecture.md"}
"""

import argparse
import json
import os
import re
import time
from pathlib import Path


def load_and_chunk_docs(docs_dir: str) -> list[dict]:
    """加载文档并按段落分块"""
    chunks = []
    for f in sorted(Path(docs_dir).glob("*.md")):
        content = f.read_text(encoding="utf-8", errors="replace")
        # 移除 YAML frontmatter
        content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
        # 按 ## 标题分段
        sections = re.split(r"\n(?=## )", content)
        for section in sections:
            section = section.strip()
            if len(section) < 100:  # 跳过太短的
                continue
            # 提取段落标题作为上下文
            title_match = re.match(r"## (.+)", section)
            title = title_match.group(1) if title_match else f.name
            chunks.append({
                "text": section,
                "title": title,
                "source": f.name,
            })
    return chunks


def generate_questions(chunks: list[dict], output_path: str,
                       model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                       device: str = "npu:0",
                       questions_per_chunk: int = 3):
    """用 7B 模型为每个 chunk 生成问题"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    samples = []
    for i, chunk in enumerate(chunks):
        print(f"\n[{i+1}/{len(chunks)}] {chunk['source']} - {chunk['title'][:50]}")

        # 构造 prompt：让模型基于文本生成问题
        prompt = (
            "你是一个训练数据生成助手。请基于下面给出的技术文档片段，生成 {} 个中文问题。"
            "问题应该覆盖文档中的关键信息，用中文提问。"
            "只输出问题，每行一个，不要编号，不要多余内容。\n\n"
            "文档片段:\n{}\n\n"
            "生成的问题:".format(questions_per_chunk, chunk["text"][:1500])
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7,
                do_sample=True, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 解析生成的问题（每行一个）
        questions = [q.strip().lstrip("0123456789.、- ") for q in response.split("\n") if q.strip()]
        questions = [q for q in questions if len(q) > 5 and ("？" in q or "?" in q)]

        # 取前 N 个有效问题
        for q in questions[:questions_per_chunk]:
            samples.append({
                "instruction": q,
                "input": "",
                "output": chunk["text"],
                "source": chunk["source"],
            })
            print(f"  + {q[:80]}")

        # 每 10 个 chunk 保存一次
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            with open(output_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"  已保存 {len(samples)} 条 (进度 {i+1}/{len(chunks)})")

    print(f"\n完成! 总计 {len(samples)} 条指令数据 → {output_path}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="生成 Ascend 文档指令微调数据集")
    parser.add_argument("cmd", choices=["generate", "stats"], help="generate: 生成数据; stats: 查看统计")
    parser.add_argument("--docs-dir", default="./train-docs/", help="文档目录")
    parser.add_argument("--output", default="./sft-data.jsonl", help="输出文件")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--questions-per-chunk", type=int, default=2, help="每个 chunk 生成的问题数")
    args = parser.parse_args()

    chunks = load_and_chunk_docs(args.docs_dir)
    print(f"文档分块: {len(chunks)} 个段落")

    if args.cmd == "stats":
        total_chars = sum(len(c["text"]) for c in chunks)
        print(f"总字符: {total_chars:,}")
        for c in chunks[:5]:
            print(f"  [{c['source']}] {c['title'][:60]}: {len(c['text'])} chars")
        return

    # generate
    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(output_dir, exist_ok=True)
    samples = generate_questions(
        chunks, args.output, args.model, args.device, args.questions_per_chunk
    )

    # 检查结果
    total_chars = sum(len(s["instruction"]) + len(s["output"]) for s in samples)
    print(f"\n{'='*60}")
    print(f"  生成 {len(samples)} 条指令数据, ~{total_chars/1024:.0f} KB")
    print(f"  保存到: {os.path.abspath(args.output)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
