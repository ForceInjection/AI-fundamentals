"""
多卡分布式训练 (DDP + HCCL) on Ascend NPU

用法:
  # 2 卡训练
  ASCEND_RT_VISIBLE_DEVICES=0,1 python3 ddp_train.py --epochs 2 --batch-size 1 --grad-accum 4

  # 8 卡训练
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ddp_train.py --epochs 2 --batch-size 1 --grad-accum 2

原理:
  - 每张 NPU 启动一个进程，通过 HCCL 通信
  - 各进程独立计算 forward/backward，梯度通过 AllReduce 同步
  - 等效 batch_size = per_gpu_batch × grad_accum × world_size
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType


# ── 配置 ──

def setup_distributed():
    """初始化 HCCL 分布式环境"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="hccl")
        torch.npu.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lora_config(r=8, alpha=16, dropout=0.05):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


# ── 数据准备 ──

class SFTDataset(Dataset):
    """指令微调数据集"""

    def __init__(self, data_path, tokenizer, max_length=512):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer 未设置 pad_token_id 且 eos_token_id 为 None")
        self.pad_id = pad_id
        self.samples = []
        self.label_mask_positions = []

        SYSTEM_PROMPT = "你是一个华为昇腾 NPU 技术专家，请根据你的知识回答问题。"
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = item.get("instruction", "")
                output = item.get("output", "")
                if not instruction or not output:
                    continue

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ]
                tokenized = tokenizer.apply_chat_template(
                    messages, tokenize=True, return_tensors="pt",
                    padding=False, truncation=False,
                )
                token_ids = tokenized[0].tolist()

                prefix_ids = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": instruction}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt",
                    padding=False, truncation=False,
                )[0].tolist()
                assistant_start = len(prefix_ids)

                if assistant_start >= max_length:
                    continue

                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                elif len(token_ids) < max_length:
                    token_ids = token_ids + [pad_id] * (max_length - len(token_ids))

                self.samples.append(torch.tensor(token_ids, dtype=torch.long))
                self.label_mask_positions.append(assistant_start)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        assistant_start = self.label_mask_positions[idx]
        input_ids = tokens[:-1]
        labels = tokens[1:].clone()
        effective_start = min(assistant_start, len(tokens) - 1)
        label_start = max(0, effective_start - 1)
        labels[:label_start] = -100
        labels[labels == self.pad_id] = -100
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ── 训练 ──

def train(args):
    rank, world_size, local_rank = setup_distributed()

    # 只在 rank 0 打印
    def log(msg):
        if rank == 0:
            print(msg)

    device = torch.device(f"npu:{local_rank}")

    # Tokenizer
    log(f"[Rank {rank}] 加载 tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Data
    dataset = SFTDataset(args.sft, tokenizer, max_length=args.max_length)
    log(f"[Rank {rank}] 指令样本数: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError(f"SFT 数据为空: {args.sft}")

    # DistributedSampler: 每个进程处理不同的数据子集
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
    ) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Model
    log(f"[Rank {rank}] 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    lora_config = get_lora_config(r=args.lora_r, alpha=args.lora_alpha)
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # DDP 包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                     find_unused_parameters=False)
    trainable_model = model.module if world_size > 1 else model

    trainable_params = sum(p.numel() for p in trainable_model.parameters() if p.requires_grad)
    log(f"[Rank {rank}] 可训练参数: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )

    steps_per_epoch = (len(dataloader) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(50, max(1, total_steps // 10))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Training loop
    effective_batch = args.batch_size * args.grad_accum * world_size
    log(f"\n开始 DDP 训练: {args.epochs} epochs, {steps_per_epoch} steps/epoch/rank")
    log(f"  world_size={world_size}, batch={args.batch_size}×{args.grad_accum}×{world_size}={effective_batch}")
    log(f"  lr={args.lr}, warmup={warmup_steps} steps, device=npu:{local_rank}")
    log(f"{'='*60}")

    model.train()
    torch.npu.reset_peak_memory_stats()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_steps = 0
        accumulated_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.npu.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss

            loss.backward()
            accumulated_loss += loss.item() / args.grad_accum

            is_last_step = (step + 1) == len(dataloader)
            is_accum_step = (step + 1) % args.grad_accum == 0

            if is_accum_step or is_last_step:
                actual_accum = args.grad_accum if is_accum_step else (step + 1) % args.grad_accum
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

                if global_step % args.log_interval == 0 and rank == 0:
                    avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
                    lr_now = scheduler.get_last_lr()[0]
                    mem_now = torch.npu.memory_allocated(local_rank) / 1024**3
                    log(f"  Epoch {epoch+1} | Step {global_step:4d}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | lr: {lr_now:.2e} | HBM: {mem_now:.1f} GB")

        epoch_avg = epoch_loss / max(epoch_steps, 1)
        log(f"  Epoch {epoch+1} 完成 | Avg Loss: {epoch_avg:.4f}")

        if rank == 0 and epoch_avg < best_loss:
            best_loss = epoch_avg
            save_path = os.path.join(args.save_path, "best")
            if world_size > 1:
                dist.barrier()
            os.makedirs(save_path, exist_ok=True)
            trainable_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            log(f"  Best 适配器已保存: {save_path} (loss={best_loss:.4f})")
            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        final_path = os.path.join(args.save_path, "final")
        os.makedirs(final_path, exist_ok=True)
        trainable_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        peak_mem = torch.npu.max_memory_allocated(local_rank) / 1024**3
        log(f"\n训练完成! 最终适配器: {final_path}")
        log(f"  Best loss: {best_loss:.4f}")
        log(f"  HBM 峰值: {peak_mem:.1f} GB / 卡")
        log(f"  总 steps: {global_step}")
        log(f"  等效 batch: {effective_batch}")

    if world_size > 1:
        dist.barrier()
    cleanup_distributed()


def parse_args():
    p = argparse.ArgumentParser(description="DDP 多卡 LoRA 微调 on NPU")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--sft", default="./sft-data-380.jsonl")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-path", default="./ddp-lora-adapter")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--drop-last", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
