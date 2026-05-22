"""
Mini-GPT: 手写 GPT-2 风格 Transformer 在 NPU 上训练

从零实现 decoder-only Transformer，字符级编码，单 NPU 训练。
用法:
  ASCEND_RT_VISIBLE_DEVICES=7 python3 train_gpt.py --train-docs ./docs/
  ASCEND_RT_VISIBLE_DEVICES=7 python3 train_gpt.py --generate "NPU 是" --checkpoint ./ckpt.pt
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu


# ── CharTokenizer ──

class CharTokenizer:
    """字符级编码器：每个唯一字符映射为一个 token ID"""

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.vocab_size: int = 0

    def fit(self, text: str):
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars)
        print(f"  Tokenizer: {self.vocab_size} 个唯一字符")

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char.get(i, "?") for i in ids)


# ── Causal Self-Attention ──

class CausalSelfAttention(nn.Module):
    """多头 causal self-attention

    Q/K/V 通过一次 Linear 投影得到，然后 split 成多个 head。
    Causal mask 确保每个位置只能看到当前位置及之前的内容。
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Q/K/V 合并为一个 Linear，效率更高
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: 上三角为 -inf
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, n_embd

        # Q/K/V 投影并 split heads
        qkv = self.c_attn(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=-1)
        # reshape: [B, T, C] → [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale  # [B, n_head, T, T]

        # Causal mask
        att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum + merge heads
        y = att @ v  # [B, n_head, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


# ── Transformer Block ──

class TransformerBlock(nn.Module):
    """一个 Transformer block: Attention + FFN，各带残差连接和 pre-norm"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── MiniGPT ──

class MiniGPT(nn.Module):
    """GPT-2 风格 decoder-only Transformer"""

    def __init__(self, vocab_size: int, n_embd: int = 384, n_head: int = 6,
                 n_layer: int = 6, block_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # 权重初始化
        self.apply(self._init_weights)
        # LM head 和 token embedding 共享权重（减少参数量）
        self.lm_head.weight = self.token_embedding.weight

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  MiniGPT 参数量: {n_params / 1e6:.2f}M")
        print(f"  block_size={block_size}, n_layer={n_layer}, "
              f"n_head={n_head}, n_embd={n_embd}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = x.shape
        assert T <= self.block_size, f"seq_len {T} > block_size {self.block_size}"

        # Token + Position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # LM head
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        """自回归生成文本"""
        self.eval()
        for _ in range(max_new_tokens):
            # 截断到 block_size
            ctx = token_ids[:, -self.block_size:]
            logits, _ = self(ctx)
            # 取最后一个位置的 logits
            logits = logits[:, -1, :] / temperature
            # Top-K 过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat((token_ids, next_id), dim=1)
        self.train()
        return token_ids


# ── Trainer ──

class Trainer:
    """训练循环：数据准备、梯度更新、loss 记录"""

    def __init__(self, model: MiniGPT, tokenizer: CharTokenizer,
                 data: torch.Tensor, device: str = "npu:0",
                 batch_size: int = 32, block_size: int = 128,
                 lr: float = 3e-4, max_iters: int = 2000,
                 eval_interval: int = 200, save_path: str = "./ckpt.pt"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.data = data.to(device)
        self.device = device
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.save_path = save_path

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01,
        )
        self.loss_history: list[tuple[int, float]] = []

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """随机采样一个 batch 的 (x, y) 对"""
        ix = torch.randint(0, len(self.data) - self.block_size - 1,
                           (self.batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def estimate_loss(self, num_batches: int = 5) -> float:
        """评估 loss（取多个 batch 平均）"""
        self.model.eval()
        total = 0.0
        for _ in range(num_batches):
            x, y = self.get_batch()
            _, loss = self.model(x, y)
            total += loss.item()
        self.model.train()
        return total / num_batches

    def train(self):
        print(f"\n{'='*60}")
        print(f"  开始训练")
        print(f"{'='*60}")
        print(f"  数据: {len(self.data):,} tokens, "
              f"batch_size={self.batch_size}, block_size={self.block_size}")
        print(f"  max_iters={self.max_iters}, lr={self.optimizer.param_groups[0]['lr']}")

        self.model.train()
        t_start = time.time()

        for it in range(1, self.max_iters + 1):
            x, y = self.get_batch()
            self.optimizer.zero_grad()
            _, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()

            if it % self.eval_interval == 0 or it == 1 or it == self.max_iters:
                eval_loss = self.estimate_loss()
                elapsed = time.time() - t_start
                self.loss_history.append((it, eval_loss))
                print(f"  iter {it:5d}/{self.max_iters} | "
                      f"loss: {eval_loss:.4f} | "
                      f"time: {elapsed:.0f}s | "
                      f"{self.max_iters * elapsed / it - elapsed:.0f}s remaining")

        print(f"\n  训练完成! 总耗时: {time.time() - t_start:.0f}s")

    def save_checkpoint(self):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "tokenizer_vocab": {
                "char_to_id": self.tokenizer.char_to_id,
                "id_to_char": self.tokenizer.id_to_char,
            },
            "config": {
                "vocab_size": self.tokenizer.vocab_size,
                "block_size": self.block_size,
            },
            "loss_history": self.loss_history,
        }, self.save_path)
        print(f"  Checkpoint 已保存: {self.save_path}")


# ── CLI ──

def load_docs(docs_dir: str) -> str:
    """加载目录下所有 .md 文件，拼接为训练语料"""
    text = ""
    for f in sorted(Path(docs_dir).glob("*.md")):
        content = f.read_text(encoding="utf-8")
        text += content + "\n"
    print(f"  加载 {len(list(Path(docs_dir).glob('*.md')))} 个文档, "
          f"共 {len(text):,} 字符")
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Mini-GPT: 手写 GPT-2 风格 Transformer 在 NPU 上训练"
    )
    parser.add_argument("--train-docs", help="训练文档目录")
    parser.add_argument("--generate", help="生成文本的起始提示")
    parser.add_argument("--checkpoint", default="./ckpt.pt", help="checkpoint 路径")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gen-tokens", type=int, default=300,
                        help="生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="生成温度 (越低越保守)")

    args = parser.parse_args()

    if args.train_docs:
        # ── 训练模式 ──
        print("=" * 60)
        print("  Mini-GPT 训练模式")
        print("=" * 60)

        text = load_docs(args.train_docs)
        tokenizer = CharTokenizer()
        tokenizer.fit(text)

        # 编码全部数据
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        model = MiniGPT(
            vocab_size=tokenizer.vocab_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            block_size=args.block_size,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data=data,
            device=args.device,
            batch_size=args.batch_size,
            block_size=args.block_size,
            lr=args.lr,
            max_iters=args.max_iters,
            save_path=args.checkpoint,
        )
        trainer.train()
        trainer.save_checkpoint()

    elif args.generate:
        # ── 生成模式 ──
        print("=" * 60)
        print("  Mini-GPT 生成模式")
        print("=" * 60)

        if not os.path.exists(args.checkpoint):
            print(f"错误: checkpoint 不存在: {args.checkpoint}")
            return

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = ckpt["config"]
        vocab_info = ckpt["tokenizer_vocab"]

        # 重建 tokenizer
        tokenizer = CharTokenizer()
        tokenizer.char_to_id = vocab_info["char_to_id"]
        tokenizer.id_to_char = {int(k): v for k, v in vocab_info["id_to_char"].items()}
        tokenizer.vocab_size = cfg["vocab_size"]

        # 重建模型
        model = MiniGPT(
            vocab_size=cfg["vocab_size"],
            block_size=cfg["block_size"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(args.device)

        # 编码 prompt
        prompt_ids = tokenizer.encode(args.generate)
        x = torch.tensor([prompt_ids], dtype=torch.long).to(args.device)

        print(f"  Prompt: {args.generate}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Max tokens: {args.gen_tokens}")

        # 生成
        output_ids = model.generate(
            x, max_new_tokens=args.gen_tokens,
            temperature=args.temperature,
        )
        generated = tokenizer.decode(output_ids[0].tolist())
        print(f"\n{'-'*40}")
        print(generated)
        print(f"{'-'*40}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
