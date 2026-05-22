"""
RAG 检索增强生成 on Ascend NPU

Embedding 在 NPU 本地运行，LLM 由第三方 API 提供。
用法:
  ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_pipeline.py index --docs ./docs/
  ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_pipeline.py query --top-k 5
  ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_pipeline.py ask "什么是 NPU?"
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import torch_npu
from sentence_transformers import SentenceTransformer


# ── Document Loader ──

class DocumentLoader:
    """加载本地文档，提取纯文本"""

    def __init__(self, supported_suffixes=(".md", ".txt", ".rst")):
        self.supported_suffixes = supported_suffixes

    def load_dir(self, path: str) -> list[dict]:
        """加载目录下所有支持的文档，返回 [{"path": str, "content": str}]"""
        docs = []
        for f in sorted(Path(path).rglob("*")):
            if f.suffix in self.supported_suffixes and f.is_file():
                content = self._read(f)
                if content.strip():
                    docs.append({"path": str(f), "content": content})
        return docs

    def load_file(self, path: str) -> dict:
        """加载单个文档"""
        p = Path(path)
        return {"path": str(p), "content": self._read(p)}

    def _read(self, filepath: Path) -> str:
        try:
            return filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return filepath.read_text(encoding="gbk", errors="ignore")


# ── Text Chunker ──

class TextChunker:
    """滑动窗口文本分块"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, meta: Optional[dict] = None) -> list[dict]:
        """将文本切分为重叠块，返回 [{"text": str, "meta": dict}]"""
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) <= self.chunk_size:
                current += para + "\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = para + "\n"
        if current.strip():
            chunks.append(current.strip())

        # 滑动窗口重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    prev_tail = chunks[i - 1][-self.chunk_overlap:]
                    chunk = prev_tail + "\n" + chunk
                overlapped.append(chunk)
            chunks = overlapped

        return [{"text": c, "meta": meta or {}} for c in chunks]

    def _split_paragraphs(self, text: str) -> list[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# ── Embedding Engine ──

class EmbeddingEngine:
    """在 NPU 上运行 embedding 模型

    将 BGE/BERT 类模型的 transformer 部分移到 NPU 进行推理，
    使用 mean pooling 生成句向量并 L2 归一化。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 device: str = "npu:0", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Optional[SentenceTransformer] = None
        self._dim: int = 0
        self._tokenizer = None
        self._transformer = None

    def load(self):
        print(f"加载 embedding 模型: {self.model_name}")
        t0 = time.time()
        self._model = SentenceTransformer(self.model_name, device="cpu")
        self._dim = self._model.get_sentence_embedding_dimension()
        self._tokenizer = self._model.tokenizer

        # 提取底层 transformer 并移到 NPU
        # sentence-transformers 内部结构: model._first_module() 通常是 Transformer
        for module in self._model.modules():
            if hasattr(module, "auto_model"):
                self._transformer = module.auto_model.to(self.device).eval()
                break

        if self._transformer is None:
            raise RuntimeError("无法提取模型 transformer 层")

        print(f"模型加载完成 ({time.time() - t0:.1f}s), 设备: {self.device}")
        return self

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """在 NPU 上编码文本列表，返回 [N, dim] ndarray"""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            emb = self._encode_batch(batch)
            all_embeddings.append(emb)
            if show_progress:
                print(f"\r  NPU 编码中... {min(i + self.batch_size, len(texts))}/{len(texts)}",
                      end="")
        if show_progress:
            print()
        return np.concatenate(all_embeddings, axis=0)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """对一个 batch 进行 NPU 推理 + mean pooling + 归一化"""
        encoded = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self._transformer(input_ids=input_ids,
                                        attention_mask=attention_mask)
            # outputs[0]: last_hidden_state [B, L, D]
            token_embeddings = outputs[0]
            # Mean pooling: 对 attention_mask 做加权平均
            mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            # L2 归一化
            mean_embeddings = torch.nn.functional.normalize(
                mean_embeddings, p=2, dim=1
            )

        return mean_embeddings.cpu().numpy()

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询文本（BGE 需要加前缀）"""
        if "bge" in self.model_name.lower():
            query = f"为这个句子生成表示以用于检索相关文章：{query}"
        return self.encode([query], show_progress=False)


# ── Vector Store ──

class VectorStore:
    """FAISS 向量库"""

    def __init__(self, dim: int, index_type: str = "FlatIP"):
        self.dim = dim
        if index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(dim)
        elif index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unknown index_type: {index_type}")
        self.chunks: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[dict]):
        self.index.add(embeddings.astype(np.float32))
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            chunk["idx"] = start_idx + i
        self.chunks.extend(chunks)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> list[dict]:
        scores, indices = self.index.search(query_emb.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(score)
                results.append(chunk)
        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"索引已保存: {path}.index + {path}.chunks.json ({len(self.chunks)} 条)")

    @classmethod
    def load(cls, path: str, dim: int) -> "VectorStore":
        store = cls(dim)
        store.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.chunks.json", "r", encoding="utf-8") as f:
            store.chunks = json.load(f)
        print(f"索引已加载: {len(store.chunks)} 条, dim={store.dim}")
        return store


# ── LLM Client ──

class LLMClient:
    """OpenAI 兼容 API 客户端"""

    def __init__(self, endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo"):
        self.endpoint = endpoint or os.environ.get("RAG_LLM_ENDPOINT", "")
        self.api_key = api_key or os.environ.get("RAG_LLM_API_KEY", "")
        self.model = model or os.environ.get("RAG_LLM_MODEL", "gpt-3.5-turbo")

    def chat(self, messages: list[dict], temperature: float = 0.3,
             max_tokens: int = 1024) -> str:
        """调用 LLM API，返回回答文本"""
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(self.endpoint, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"LLM API 调用失败: {e}") from e


# ── RAG Pipeline ──

class RAGPipeline:
    """编排完整的 RAG 流程"""

    SYSTEM_PROMPT = (
        "你是一个基于参考资料回答问题的助手。请根据下面提供的参考资料回答问题。"
        "如果参考资料中没有相关信息，请如实告知，不要编造。"
        "回答时请引用参考资料的来源路径。"
    )

    def __init__(self, embedder: EmbeddingEngine, store: VectorStore,
                 llm: LLMClient, top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k

    def index_documents(self, docs_dir: str, chunk_size: int = 512,
                        chunk_overlap: int = 128):
        """构建索引"""
        print(f"索引目录: {docs_dir}")
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        docs = loader.load_dir(docs_dir)
        print(f"找到 {len(docs)} 个文档")

        all_chunks = []
        for doc in docs:
            chunks = chunker.chunk(doc["content"], meta={"source": doc["path"]})
            all_chunks.extend(chunks)

        print(f"切分为 {len(all_chunks)} 个文本块 (size={chunk_size}, overlap={chunk_overlap})")

        # 提取文本编码
        texts = [c["text"] for c in all_chunks]
        print("正在编码文本块...")
        t0 = time.time()
        embeddings = self.embedder.encode(texts)
        encode_time = time.time() - t0
        print(f"编码完成 ({encode_time:.1f}s, {len(texts) / encode_time:.0f} 条/s)")

        # 存入向量库
        self.store.add(embeddings, all_chunks)
        print(f"索引构建完成: {len(self.store.chunks)} 条")

    def query(self, question: str) -> dict:
        """执行 RAG 查询，返回检索结果 + LLM 回答"""
        # 1. 编码查询
        t0 = time.time()
        q_emb = self.embedder.encode_query(question)
        encode_time = time.time() - t0

        # 2. 检索
        t0 = time.time()
        hits = self.store.search(q_emb, top_k=self.top_k)
        search_time = time.time() - t0

        # 3. 构建 prompt
        context_parts = []
        for h in hits:
            src = h["meta"].get("source", "unknown")
            context_parts.append(f"[来源: {src}]\n{h['text']}")
        context = "\n\n---\n\n".join(context_parts)

        user_msg = f"参考资料:\n\n{context}\n\n问题: {question}"

        # 4. 调用 LLM
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        answer = self.llm.chat(messages)

        return {
            "question": question,
            "answer": answer,
            "sources": [{"source": h["meta"].get("source", ""), "score": h["score"]} for h in hits],
            "timing": {"encode_ms": round(encode_time * 1000, 1), "search_ms": round(search_time * 1000, 1)},
        }

    def interactive(self):
        """交互式查询模式"""
        print("\nRAG 交互式查询 (输入 /quit 退出, /topk N 设置检索数量)")
        while True:
            try:
                q = input("\n提问> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q == "/quit":
                break
            if q.startswith("/topk"):
                try:
                    self.top_k = int(q.split()[1])
                    print(f"  top_k = {self.top_k}")
                except (IndexError, ValueError):
                    print("  用法: /topk N")
                continue

            result = self.query(q)
            print(f"\n检索到 {len(result['sources'])} 条相关文档:")
            for i, s in enumerate(result["sources"], 1):
                print(f"  [{i}] {Path(s['source']).name} (相关度: {s['score']:.4f})")
            print(f"\nLLM 回答:\n{result['answer']}")
            print(f"(编码 {result['timing']['encode_ms']}ms, 检索 {result['timing']['search_ms']}ms)")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="RAG 检索增强生成 on Ascend NPU")
    sub = parser.add_subparsers(dest="cmd")

    # index
    p_idx = sub.add_parser("index", help="索引文档")
    p_idx.add_argument("--docs", default="./docs/", help="文档目录")
    p_idx.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="embedding 模型名")
    p_idx.add_argument("--chunk-size", type=int, default=512)
    p_idx.add_argument("--chunk-overlap", type=int, default=128)
    p_idx.add_argument("--index-path", default="./rag_index", help="索引保存路径前缀")
    p_idx.add_argument("--device", default="npu:0")

    # search
    p_s = sub.add_parser("search", help="仅检索（不需要 LLM API）")
    p_s.add_argument("question", help="搜索查询")
    p_s.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="embedding 模型名")
    p_s.add_argument("--index-path", default="./rag_index", help="索引路径前缀")
    p_s.add_argument("--top-k", type=int, default=5)
    p_s.add_argument("--device", default="npu:0")
    p_s.add_argument("--show-text", action="store_true", help="显示检索到的文本片段")

    # query
    p_q = sub.add_parser("query", help="交互式查询 (需要 LLM API)")
    p_q.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="embedding 模型名")
    p_q.add_argument("--index-path", default="./rag_index", help="索引路径前缀")
    p_q.add_argument("--top-k", type=int, default=5)
    p_q.add_argument("--device", default="npu:0")

    # ask
    p_a = sub.add_parser("ask", help="单次查询 (需要 LLM API)")
    p_a.add_argument("question", help="问题")
    p_a.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="embedding 模型名")
    p_a.add_argument("--index-path", default="./rag_index", help="索引路径前缀")
    p_a.add_argument("--top-k", type=int, default=5)
    p_a.add_argument("--device", default="npu:0")

    args = parser.parse_args()

    # 检查 LLM API 配置 (query/ask 才需要)
    if args.cmd in ("query", "ask"):
        missing = []
        if not os.environ.get("RAG_LLM_ENDPOINT"):
            missing.append("RAG_LLM_ENDPOINT")
        if not os.environ.get("RAG_LLM_API_KEY"):
            missing.append("RAG_LLM_API_KEY")
        if missing:
            print(f"错误: 缺少环境变量 {', '.join(missing)}")
            print("用法: export RAG_LLM_ENDPOINT='https://...' RAG_LLM_API_KEY='sk-...'")
            sys.exit(1)

    # 加载 embedding 模型
    embedder = EmbeddingEngine(model_name=args.model, device=args.device).load()

    if args.cmd == "index":
        store = VectorStore(dim=embedder.dim)
        pipeline = RAGPipeline(embedder, store, LLMClient(), top_k=5)
        pipeline.index_documents(
            docs_dir=args.docs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        store.save(args.index_path)

    elif args.cmd == "search":
        store = VectorStore.load(args.index_path, dim=embedder.dim)
        pipeline = RAGPipeline(embedder, store, LLMClient(), top_k=args.top_k)

        t0 = time.time()
        q_emb = embedder.encode_query(args.question)
        hits = store.search(q_emb, top_k=args.top_k)
        elapsed = time.time() - t0

        print(f"\n查询: {args.question}")
        print(f"检索到 {len(hits)} 条结果 ({elapsed * 1000:.1f}ms):\n")
        for i, h in enumerate(hits, 1):
            src = Path(h["meta"].get("source", "")).name
            text_preview = h["text"][:100].replace("\n", " ")
            print(f"  [{i}] {src} (相关度: {h['score']:.4f})")
            print(f"      {text_preview}...")
            if args.show_text:
                print(f"\n{h['text']}\n")

    elif args.cmd == "query":
        store = VectorStore.load(args.index_path, dim=embedder.dim)
        llm = LLMClient()
        pipeline = RAGPipeline(embedder, store, llm, top_k=args.top_k)
        pipeline.interactive()

    elif args.cmd == "ask":
        store = VectorStore.load(args.index_path, dim=embedder.dim)
        llm = LLMClient()
        pipeline = RAGPipeline(embedder, store, llm, top_k=args.top_k)
        result = pipeline.query(args.question)
        print(f"\n检索到 {len(result['sources'])} 条相关文档:")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {Path(s['source']).name} (相关度: {s['score']:.4f})")
        print(f"\nLLM 回答:\n{result['answer']}")
        print(f"(编码 {result['timing']['encode_ms']}ms, 检索 {result['timing']['search_ms']}ms)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
