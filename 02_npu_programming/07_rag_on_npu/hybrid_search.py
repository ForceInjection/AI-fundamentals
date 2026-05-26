"""
RAG 混合检索 + 重排序

BM25 关键词检索 + FAISS 向量检索 → RRF 融合 → CrossEncoder 重排序

用法:
  ASCEND_RT_VISIBLE_DEVICES=7 python3 hybrid_search.py search "什么是 NPU？"
  ASCEND_RT_VISIBLE_DEVICES=7 python3 hybrid_search.py eval        # 对比评估
  ASCEND_RT_VISIBLE_DEVICES=7 python3 hybrid_search.py benchmark   # 性能基准
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
from rag_pipeline import EmbeddingEngine, VectorStore


# ── BM25 关键词检索 ──

class BM25Retriever:
    """BM25 关键词检索器"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freq = {}
        self.avg_dl = 0
        self.N = 0

    def _tokenize(self, text: str) -> list[str]:
        """jieba 中文分词 + 2-gram 英文"""
        try:
            import jieba
            tokens = list(jieba.cut(text))
        except ImportError:
            tokens = list(text)
        # 英文/数字部分追加字符级 bigram 以匹配缩写
        extra = []
        for token in tokens:
            if len(token) <= 4 and token.isascii():
                extra.append(token)
        return tokens + extra

    def index(self, documents: list[dict]):
        """构建 BM25 索引"""
        self.corpus = documents
        self.N = len(documents)
        total_len = 0

        for doc in documents:
            tokens = self._tokenize(doc["text"])
            doc["_tokens"] = tokens
            doc["_len"] = len(tokens)
            total_len += doc["_len"]

            seen = set()
            for token in tokens:
                if token not in seen:
                    self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
                    seen.add(token)

        self.avg_dl = total_len / max(self.N, 1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """BM25 搜索，返回 [(doc_idx, score)]"""
        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = doc["_len"]
            doc_tokens = doc["_tokens"]

            # 统计 query term 在文档中的频率
            tf = {}
            for token in doc_tokens:
                tf[token] = tf.get(token, 0) + 1

            for token in query_tokens:
                if token not in self.doc_freq:
                    continue
                df = self.doc_freq[token]
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                f = tf.get(token, 0)
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
                score += idf * numerator / max(denominator, 1e-9)

            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ── RRF 融合 ──

def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 10,
    top_k: int = 10,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    """加权 RRF 融合两路检索结果"""
    rrf_scores = {}

    for rank, result in enumerate(vector_results, 1):
        idx = result["idx"]
        rrf_scores[idx] = rrf_scores.get(idx, 0) + vector_weight / (k + rank)

    for rank, result in enumerate(bm25_results, 1):
        idx = result["idx"]
        rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_weight / (k + rank)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    fused = []
    for idx, score in sorted_items[:top_k]:
        result = None
        for r in vector_results + bm25_results:
            if r["idx"] == idx:
                result = dict(r)
                break
        if result:
            result["rrf_score"] = score
            fused.append(result)

    return fused


# ── Hybrid Search Pipeline ──

class HybridSearcher:
    """混合检索器：BM25 + FAISS → RRF"""

    def __init__(self, embedder: EmbeddingEngine, store: VectorStore):
        self.embedder = embedder
        self.store = store
        self.bm25 = BM25Retriever()
        self.bm25.index(store.chunks)

    def search(self, query: str, top_k: int = 5, rrf_k: int = 10) -> list[dict]:
        """混合检索：向量为主（权重 0.7），BM25 为辅（权重 0.3）"""
        t0 = time.time()

        # 1. 向量检索
        q_emb = self.embedder.encode_query(query)
        vector_hits = self.store.search(q_emb, top_k=max(top_k * 2, 10))

        # 2. BM25 检索
        bm25_raw = self.bm25.search(query, top_k=max(top_k * 2, 10))
        bm25_hits = []
        for idx, score in bm25_raw:
            chunk = dict(self.store.chunks[idx])
            chunk["bm25_score"] = score
            bm25_hits.append(chunk)

        # 3. RRF 融合（向量权重 0.7, BM25 权重 0.3）
        fused = reciprocal_rank_fusion(
            vector_hits, bm25_hits, k=rrf_k, top_k=top_k,
            vector_weight=0.7, bm25_weight=0.3,
        )

        elapsed = time.time() - t0
        for h in fused:
            h["search_ms"] = round(elapsed * 1000, 1)

        return fused


# ── 评估 ──

EVAL_QUERIES = [
    {"question": "NPU 的 HBM 带宽是多少？", "gt_ids": [83]},
    {"question": "什么是达芬奇架构？", "gt_ids": [15]},
    {"question": "如何安装 torch_npu？", "gt_ids": [2]},
    {"question": "npu-smi 如何查看 NPU 之间的拓扑连接？", "gt_ids": [12]},
    {"question": "Ascend 910B3 的算力是多少 TFLOPS？", "gt_ids": [84]},
    {"question": "NPU 的 AI Core 包含哪些计算单元？", "gt_ids": [15]},
    {"question": "什么是 RAG？", "gt_ids": []},  # 之前 bge-small 检索失败的查询
    {"question": "什么是 FlashAttention？", "gt_ids": []},
    {"question": "什么是 KV Cache？", "gt_ids": []},
    {"question": "BGE 模型查询时需要加什么前缀？", "gt_ids": []},
]


def evaluate(searcher: HybridSearcher, queries: list[dict], top_k: int = 5) -> dict:
    """评估混合检索效果"""
    print(f"\n{'='*70}")
    print(f"  混合检索评估 (BM25 + FAISS + RRF, top_k={top_k})")
    print(f"{'='*70}")

    mrr_sum = 0
    recall_hits = {1: 0, 3: 0, 5: 0}
    valid_count = 0

    for i, q in enumerate(queries):
        hits = searcher.search(q["question"], top_k=top_k)
        ranked_ids = [h["idx"] for h in hits]

        if q["gt_ids"]:
            # 计算 MRR
            for rank, rid in enumerate(ranked_ids, 1):
                if rid in q["gt_ids"]:
                    mrr_sum += 1.0 / rank
                    break
            # 计算 Recall
            gt_set = set(q["gt_ids"])
            for k_val in [1, 3, 5]:
                if gt_set & set(ranked_ids[:k_val]):
                    recall_hits[k_val] += 1
            valid_count += 1

        print(f"  [{i+1}] {q['question'][:45]}")
        for j, h in enumerate(hits[:3]):
            rrf = h.get("rrf_score", 0)
            vec = h.get("score", 0)
            bm = h.get("bm25_score", "-")
            print(f"      {j+1}. idx={h['idx']} rrf={rrf:.4f} vec={vec:.4f} bm25={bm}")

    results = {}
    if valid_count > 0:
        results["mrr"] = mrr_sum / valid_count
        results["recall@1"] = recall_hits[1] / valid_count
        results["recall@3"] = recall_hits[3] / valid_count
        results["recall@5"] = recall_hits[5] / valid_count
        print(f"\n  MRR: {results['mrr']:.4f} | "
              f"R@1: {results['recall@1']:.2f} | "
              f"R@3: {results['recall@3']:.2f} | "
              f"R@5: {results['recall@5']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="RAG 混合检索 + 重排序")
    sub = parser.add_subparsers(dest="cmd")

    p_search = sub.add_parser("search", help="搜索")
    p_search.add_argument("question", help="查询")
    p_search.add_argument("--top-k", type=int, default=5)

    p_eval = sub.add_parser("eval", help="对比评估")

    args = parser.parse_args()

    if args.cmd not in ("search", "eval"):
        parser.print_help()
        return

    # 加载 embedding 和索引
    embedder = EmbeddingEngine(device="npu:0").load()
    store = VectorStore.load("./rag_index", dim=embedder.dim)
    searcher = HybridSearcher(embedder, store)

    if args.cmd == "search":
        hits = searcher.search(args.question, top_k=args.top_k)
        print(f"\n查询: {args.question}")
        print(f"检索到 {len(hits)} 条结果:\n")
        for i, h in enumerate(hits, 1):
            src = Path(h["meta"].get("source", "")).name
            rrf = h.get("rrf_score", 0)
            print(f"  [{i}] {src} (RRF={rrf:.4f})")
            print(f"      {h['text'][:120]}...")

    elif args.cmd == "eval":
        # 1. 纯向量检索（baseline）
        vector_results = {"mrr": 0, "recall@1": 0, "recall@3": 0, "recall@5": 0}
        valid = 0
        for q in EVAL_QUERIES:
            if q["gt_ids"]:
                q_emb = embedder.encode_query(q["question"])
                hits = store.search(q_emb, top_k=5)
                ranked = [h["idx"] for h in hits]
                gt = set(q["gt_ids"])
                for rank, rid in enumerate(ranked, 1):
                    if rid in gt:
                        vector_results["mrr"] += 1.0 / rank
                        break
                for k_val in [1, 3, 5]:
                    if gt & set(ranked[:k_val]):
                        vector_results[f"recall@{k_val}"] += 1
                valid += 1
        n = max(valid, 1)
        vector_results["mrr"] /= n
        for k_val in [1, 3, 5]:
            vector_results[f"recall@{k_val}"] /= n

        # 2. 混合检索
        hybrid_results = evaluate(searcher, EVAL_QUERIES)

        # 3. 对比
        print(f"\n{'='*70}")
        print(f"  对比总结")
        print(f"{'='*70}")
        print(f"  {'指标':<15} {'纯向量':<12} {'混合检索':<12} {'变化':<10}")
        print(f"  {'-'*49}")
        for key in ["mrr", "recall@1", "recall@3", "recall@5"]:
            v_val = vector_results.get(key, 0)
            h_val = hybrid_results.get(key, 0)
            delta = h_val - v_val
            sign = "+" if delta > 0 else ""
            print(f"  {key:<15} {v_val:<12.4f} {h_val:<12.4f} {sign}{delta:<9.4f}")


if __name__ == "__main__":
    main()
