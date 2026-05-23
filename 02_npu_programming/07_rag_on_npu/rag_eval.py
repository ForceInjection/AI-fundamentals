"""
RAG 检索质量评估

基于 FAISS + numpy 计算检索指标（MRR、Recall@k、Precision@k）。
依赖 rag_pipeline 中的 EmbeddingEngine 和 VectorStore。

用法:
  ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_eval.py
"""

import time
from pathlib import Path

# ── 评估查询数据集 ──
# 每个查询标注了 ground_truth 的 chunk ID（从 FAISS 索引中确认）

# Ground truth 基于远端 FAISS 索引实际检索结果 + 人工确认 chunk 内容
# 每个查询的 gt_ids 为 top-5 搜索结果中包含正确答案的 chunk ID

EVAL_QUERIES = [
    {
        "question": "NPU 的 HBM 带宽是多少？",
        "gt_ids": [83],      # chunk 83: ascend-dmi HBM bandwidth 1538 GB/s
    },
    {
        "question": "什么是达芬奇架构？",
        "gt_ids": [15],      # chunk 15: 达芬奇 (Da Vinci) 架构核心设计
    },
    {
        "question": "如何安装 torch_npu？",
        "gt_ids": [2],       # chunk 2: pip install torch-npu 命令
    },
    {
        "question": "npu-smi 如何查看 NPU 之间的拓扑连接？",
        "gt_ids": [12],      # chunk 12: HCCS topology 矩阵输出
    },
    {
        "question": "Ascend 910B3 的算力是多少 TFLOPS？",
        "gt_ids": [84],      # chunk 84: ascend-dmi FP16 313.7 TFLOPS
    },
    {
        "question": "torch_npu 安装需要 numpy 什么版本？",
        "gt_ids": [2],       # chunk 2: numpy<2 安装命令
    },
    {
        "question": "NPU 的 AI Core 包含哪些计算单元？",
        "gt_ids": [15],      # chunk 15: Cube/Vector/Scalar 三单元
    },
    {
        "question": "什么是达芬奇架构的计算核心？",
        "gt_ids": [15],      # chunk 15: 昇腾 AI Core 是达芬奇架构的计算核心单元
    },
]


class RetrievalEvaluator:
    """检索质量评估器"""

    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def mrr(self, ranked_ids, gt_ids):
        """Mean Reciprocal Rank: 第一个命中的排名的倒数"""
        for rank, rid in enumerate(ranked_ids, 1):
            if rid in gt_ids:
                return 1.0 / rank
        return 0.0

    def recall_at_k(self, ranked_ids, gt_ids, k):
        """Recall@k: top-k 中命中了多少 ground truth"""
        top_k = set(ranked_ids[:k])
        hits = len(top_k & set(gt_ids))
        return hits / len(gt_ids) if gt_ids else 0.0

    def precision_at_k(self, ranked_ids, gt_ids, k):
        """Precision@k: top-k 中 ground truth 的占比"""
        if k <= 0:
            return 0.0
        effective_k = min(k, len(ranked_ids))
        if effective_k <= 0:
            return 0.0
        top_k = set(ranked_ids[:effective_k])
        hits = len(top_k & set(gt_ids))
        return hits / effective_k

    def evaluate(self, queries, top_k=5):
        """对一组查询运行评估，返回聚合指标"""
        results = {
            "mrr": 0.0,
            "recall_at": {1: 0.0, 3: 0.0, 5: 0.0},
            "precision_at": {1: 0.0, 3: 0.0, 5: 0.0},
        }
        per_query = []
        max_idx = len(self.store.chunks) - 1

        for i, q in enumerate(queries):
            # 校验 ground truth chunk ID 有效性
            invalid_ids = [idx for idx in q.get("gt_ids", []) if idx < 0 or idx > max_idx]
            if invalid_ids:
                print(f"警告: 查询 {i} 包含无效 gt_ids: {invalid_ids} (索引范围 0-{max_idx})，已跳过")
                continue

            t0 = time.perf_counter()
            q_emb = self.embedder.encode_query(q["question"])
            hits = self.store.search(q_emb, top_k=top_k)
            elapsed = time.perf_counter() - t0

            ranked_ids = [h["idx"] for h in hits]
            gt_ids = q["gt_ids"]
            mrr_val = self.mrr(ranked_ids, gt_ids)

            pq = {
                "id": i,
                "question": q["question"],
                "mrr": mrr_val,
                "elapsed_ms": elapsed * 1000,
            }
            for k in [1, 3, 5]:
                pq[f"recall@{k}"] = self.recall_at_k(ranked_ids, gt_ids, k)
                pq[f"prec@{k}"] = self.precision_at_k(ranked_ids, gt_ids, k)
                results["recall_at"][k] += pq[f"recall@{k}"]
                results["precision_at"][k] += pq[f"prec@{k}"]

            results["mrr"] += mrr_val
            per_query.append(pq)

        n = len(per_query)
        if n == 0:
            return results, per_query
        results["mrr"] /= n
        for k in [1, 3, 5]:
            results["recall_at"][k] /= n
            results["precision_at"][k] /= n

        return results, per_query


def print_results(results, per_query):
    """格式化输出评估结果"""
    print("\n" + "=" * 70)
    print("  RAG 检索评估结果")
    print("=" * 70)
    print(f"  查询数量: {len(per_query)}")
    print(f"  MRR:       {results['mrr']:.4f}")
    print(f"  Recall@1:  {results['recall_at'][1]:.4f}")
    print(f"  Recall@3:  {results['recall_at'][3]:.4f}")
    print(f"  Recall@5:  {results['recall_at'][5]:.4f}")
    print(f"  Precision@1: {results['precision_at'][1]:.4f}")
    print(f"  Precision@3: {results['precision_at'][3]:.4f}")
    print(f"  Precision@5: {results['precision_at'][5]:.4f}")

    print(f"\n  {'#':<4} {'MRR':<8} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'耗时':<8}  问题")
    print("  " + "-" * 73)
    for pq in per_query:
        q_display = pq["question"][:40]
        print(f"  {pq['id']:<4} {pq['mrr']:<8.4f} "
              f"{pq['recall@1']:<8.4f} {pq['recall@3']:<8.4f} "
              f"{pq['recall@5']:<8.4f} {pq['elapsed_ms']:>6.1f}ms {q_display:<40}")
    if per_query:
        total_elapsed = sum(pq["elapsed_ms"] for pq in per_query)
        print(f"\n  总耗时: {total_elapsed:.1f}ms (均值 {total_elapsed/len(per_query):.1f}ms)")
    else:
        print("\n  无有效查询结果")
    print()


def main():
    from rag_pipeline import EmbeddingEngine, VectorStore
    import argparse
    parser = argparse.ArgumentParser(description="RAG 检索评估")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--index-path", default="./rag_index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    print(f"加载 embedding 模型: {args.model}")
    embedder = EmbeddingEngine(model_name=args.model, device=args.device)
    embedder.load()
    store = VectorStore.load(args.index_path, dim=embedder.dim)

    evaluator = RetrievalEvaluator(embedder, store)
    results, per_query = evaluator.evaluate(EVAL_QUERIES, top_k=args.top_k)
    print_results(results, per_query)


if __name__ == "__main__":
    main()
