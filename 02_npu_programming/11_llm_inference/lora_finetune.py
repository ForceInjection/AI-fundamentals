"""
LoRA 微调 Qwen2.5-7B on Ascend NPU

使用 7B BF16 模型 + LoRA 在 Ascend 学习文档上做参数高效微调。
训练数据为 docs/*.md 的文本块，目标是让模型学习 Ascend 领域的表达风格。

用法:
  ASCEND_RT_VISIBLE_DEVICES=7 python3 lora_finetune.py

可选参数:
  --data-dir ./docs/        训练数据目录
  --epochs 3                训练轮数
  --batch-size 1            批大小（配合梯度累积）
  --grad-accum 4            梯度累积步数
  --lr 2e-4                 学习率
  --lora-r 8                LoRA rank
  --save-path ./lora-adapter 适配器保存路径
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch_npu
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType


# ── 配置 ──

def get_lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """Qwen2.5 LoRA 配置：微调 Attention 的 Q/K/V/O 投影"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


# ── 数据准备 ──

class TextDataset(Dataset):
    """将文档文本切分为固定长度的训练样本"""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 512):
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.samples = []
        for text in texts:
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, add_special_tokens=True)
            # 切分为 max_length 的块，相邻块重叠 1/4
            stride = max_length // 4
            for start in range(0, len(tokens), max_length - stride):
                chunk = tokens[start:start + max_length]
                if len(chunk) < 32:
                    continue
                if len(chunk) < max_length:
                    chunk = chunk + [pad_id] * (max_length - len(chunk))
                self.samples.append(torch.tensor(chunk, dtype=torch.long))
        self.pad_id = pad_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        input_ids = tokens[:-1]
        labels = tokens[1:].clone()
        # Padding 位置不参与 loss 计算
        non_pad_mask = (tokens[1:] != self.pad_id)
        labels[~non_pad_mask] = -100
        attention_mask = (input_ids != self.pad_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def load_training_texts(data_dir: str, tokenizer=None) -> list[str]:
    """递归加载目录下所有 .md 文件，按段落分块（以 token 长度为参考）"""
    texts = []
    for f in sorted(Path(data_dir).rglob("*.md")):
        try:
            content = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = f.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            continue
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        chunk = ""
        for para in paragraphs:
            test_chunk = chunk + para + "\n\n"
            chunk_len = len(tokenizer.encode(test_chunk)) if tokenizer else len(test_chunk)
            if chunk_len < 1024:
                chunk = test_chunk
            else:
                if chunk.strip():
                    texts.append(chunk.strip())
                chunk = para + "\n\n"
        if chunk.strip():
            texts.append(chunk.strip())
    return texts


# ── 训练 ──

def train(args):
    device = torch.device(args.device)

    # 1. 加载 tokenizer
    print(f"加载 tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 加载数据
    print(f"加载训练数据: {args.data_dir}")
    texts = load_training_texts(args.data_dir, tokenizer)
    print(f"  文档段落数: {len(texts)}")
    total_chars = sum(len(t) for t in texts)
    print(f"  总字符数: {total_chars:,}")

    dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    print(f"  训练样本数: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError(f"未找到有效训练数据，请检查 --data-dir: {args.data_dir}")

    # 设置随机种子保证可复现性
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(args.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # 3. 加载模型 + LoRA
    print(f"加载模型: {args.model_name} (BF16)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    load_time = time.time() - t0

    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6
    mem = torch.npu.memory_allocated() / 1024**3
    print(f"  模型加载: {load_time:.0f}s, {params_m:.0f}M 参数, HBM: {mem:.1f} GB")

    # 显存优化：启用梯度检查点
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 注入 LoRA
    lora_config = get_lora_config(r=args.lora_r, alpha=args.lora_alpha)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/params:.4f}%)")

    # 4. 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    steps_per_epoch = (len(dataloader) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(50, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 5. 训练循环
    print(f"\n开始训练: {args.epochs} epochs, {steps_per_epoch} steps/epoch, "
          f"batch={args.batch_size}×{args.grad_accum}")
    print(f"  lr={args.lr}, warmup={warmup_steps} steps")
    print(f"{'='*60}")

    model.train()
    torch.npu.reset_peak_memory_stats()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        accumulated_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # BF16 混合精度前向
            with torch.npu.amp.autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                loss = outputs.loss

            loss.backward()
            accumulated_loss += loss.item()

            is_last_step = (step + 1) == len(dataloader)
            is_accum_step = (step + 1) % args.grad_accum == 0

            if is_accum_step or is_last_step:
                # 根据实际累积步数进行梯度平均
                actual_accum = args.grad_accum if is_accum_step else ((step + 1) % args.grad_accum or args.grad_accum)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(actual_accum)

                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_steps += 1
                epoch_loss += accumulated_loss
                accumulated_loss = 0.0

                if global_step % args.log_interval == 0:
                    avg_loss = epoch_loss / epoch_steps
                    mem_now = torch.npu.memory_allocated() / 1024**3
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch+1} | Step {global_step:4d}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} | lr: {lr_now:.2e} | HBM: {mem_now:.1f} GB")

        epoch_avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1} 完成 | Avg Loss: {epoch_avg_loss:.4f}")

        # 保存最佳模型
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            save_path = os.path.join(args.save_path, "best")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            print(f"  Best 适配器已保存: {save_path} (loss={best_loss:.4f})")

    # 6. 最终保存
    final_path = os.path.join(args.save_path, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n训练完成! 最终适配器: {final_path}")
    print(f"  Best loss: {best_loss:.4f}")

    peak_mem = torch.npu.max_memory_allocated() / 1024**3
    print(f"  HBM 峰值: {peak_mem:.1f} GB")
    print(f"  总 steps: {global_step}")


def parse_args():
    p = argparse.ArgumentParser(description="LoRA 微调 Qwen2.5-7B on NPU")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data-dir", default="./docs/")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--save-path", default="./lora-adapter")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader 加载进程数")
    p.add_argument("--pin-memory", action="store_true", help="启用 pin_memory")
    p.add_argument("--drop-last", action="store_true", help="丢弃最后一个不完整 batch")
    return p.parse_args()


if __name__ == "__main__":
    if not torch.npu.is_available():
        raise RuntimeError("NPU 不可用，请检查环境")
    train(parse_args())
