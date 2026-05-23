"""
RAGAS 评估集成：对比 7B BF16 vs 0.5B BF16 的 RAG 回答质量

需要先安装 ragas: pip install ragas rapidfuzz

用法:
  ASCEND_RT_VISIBLE_DEVICES=0 python3 rag_eval_ragas.py
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rag_pipeline import EmbeddingEngine, VectorStore, LocalLLMClient

try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
        from ragas.metrics import NonLLMContextRecall, NonLLMContextPrecisionWithReference
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False

# ── 评估数据集（含参考答案，用于 Context Recall） ──

EVAL_DATASET = [
    {
        "question": "NPU 的 HBM 带宽是多少？",
        "reference": "Ascend 910B3 的 HBM 带宽为 1538 GB/s。",
        "gt_chunks": [83],
    },
    {
        "question": "什么是达芬奇架构？",
        "reference": "达芬奇架构是华为昇腾 AI 处理器的核心架构，包含 Cube、Vector、Scalar 三种计算单元。",
        "gt_chunks": [15],
    },
    {
        "question": "如何安装 torch_npu？",
        "reference": "先安装 PyTorch 2.1.0，然后 pip install torch-npu==2.1.0.post13，注意 numpy 版本需 <2。",
        "gt_chunks": [2],
    },
    {
        "question": "npu-smi 如何查看 NPU 之间的拓扑连接？",
        "reference": "使用 npu-smi info -l 命令查看 8 卡之间的 HCCS 互联拓扑。",
        "gt_chunks": [12],
    },
    {
        "question": "Ascend 910B3 的算力是多少 TFLOPS？",
        "reference": "Ascend 910B3 的 FP16 算力为 313.7 TFLOPS。",
        "gt_chunks": [84],
    },
    {
        "question": "NPU 的 AI Core 包含哪些计算单元？",
        "reference": "AI Core 包含 Cube 单元（矩阵运算）、Vector 单元（向量运算）和 Scalar 单元（标量运算）。",
        "gt_chunks": [15],
    },
    {
        "question": "torch_npu 安装需要 numpy 什么版本？",
        "reference": "numpy 版本需要 < 2，推荐 1.26.4，因为 torch_npu 2.1.0 使用 NumPy 1.x C API 编译。",
        "gt_chunks": [2],
    },
]


def load_rag(device, embedding_model="BAAI/bge-small-zh-v1.5",
             index_path="./rag_index"):
    """加载 RAG 组件"""
    embedder = EmbeddingEngine(model_name=embedding_model, device=device).load()
    store = VectorStore.load(index_path, dim=embedder.dim)
    return embedder, store


def run_query_with_llm(embedder, store, llm_client, question, top_k=5, hits=None, reference=""):
    """执行单次 RAG 查询，返回检索结果 + LLM 回答"""
    t0 = time.time()
    if hits is None:
        q_emb = embedder.encode_query(question)
        hits = store.search(q_emb, top_k=top_k)

    context_parts = []
    for h in hits:
        src = h["meta"].get("source", "unknown")
        context_parts.append("[来源: {}]\n{}".format(src, h["text"]))
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "你是一个基于参考资料回答问题的助手。请根据下面提供的参考资料回答问题。"
        "如果参考资料中没有相关信息，请如实告知，不要编造。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "参考资料:\n\n{}\n\n问题: {}".format(context, question)},
    ]
    answer = llm_client.chat(messages, temperature=0.1)
    elapsed = time.time() - t0

    retrieved_contexts = [h["text"] for h in hits]
    return {
        "question": question,
        "answer": answer,
        "contexts": retrieved_contexts,
        "ground_truth": reference,
        "timing": elapsed,
    }


def compute_retrieval_overlap(hits, gt_ids, top_k=5):
    """计算检索结果与 ground truth chunk IDs 的重叠"""
    ranked_ids = [h["idx"] for h in hits[:top_k]]
    gt_set = set(gt_ids)
    hits_set = set(ranked_ids)
    overlap = len(hits_set & gt_set)

    # MRR
    mrr = 0.0
    for rank, rid in enumerate(ranked_ids, 1):
        if rid in gt_set:
            mrr = 1.0 / rank
            break

    return {
        "recall@5": overlap / len(gt_set) if gt_set else 0,
        "mrr": mrr,
        "hit": overlap > 0,
    }


def main():
    device = "npu:0"

    # 加载 RAG
    print("加载 RAG pipeline...")
    embedder, store = load_rag(device=device)

    results = {}

    for model_label, model_name in [
        ("7B", "Qwen/Qwen2.5-7B-Instruct"),
        ("0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    ]:

        print("\n加载 {} (BF16)...".format(model_name))
        try:
            llm = LocalLLMClient(model_name=model_name, device=device).load()
        except Exception as e:
            print("  加载 {} 失败: {}".format(model_name, e))
            continue

        model_results = []
        for i, item in enumerate(EVAL_DATASET):
            q = item["question"]
            try:
                hits = store.search(embedder.encode_query(q), top_k=5)
                ret = compute_retrieval_overlap(hits, item["gt_chunks"])
                ans = run_query_with_llm(embedder, store, llm, q, hits=hits, reference=item["reference"])
                ans["retrieval"] = ret
                model_results.append(ans)
                print("  [{}] {}: {}... (MRR={:.2f}, {:.1f}s)".format(
                    i + 1, model_label,
                    ans["answer"][:60].replace("\n", " "),
                    ret["mrr"], ans["timing"]))
            except Exception as e:
                print("  [{}] {} 查询失败: {}".format(i + 1, model_label, e))
                model_results.append({
                    "question": q, "error": str(e),
                    "answer": "", "contexts": [], "timing": 0.0,
                    "ground_truth": item["reference"],
                    "retrieval": {"recall@5": 0, "mrr": 0.0, "hit": False},
                })

        results[model_label] = model_results

        # 释放模型内存
        allocated = torch.npu.memory_allocated() / 1024**3
        llm.release()
        del llm
        allocated_after = torch.npu.memory_allocated() / 1024**3
        print("  {} 已释放 (显存: {:.1f} GB → {:.1f} GB)".format(model_label, allocated, allocated_after))

    if "7B" not in results or "0.5B" not in results:
        print("错误: 至少一个模型加载失败，无法生成对比报告")
        sys.exit(1)

    results_7b = results["7B"]
    results_05b = results["0.5B"]

    # ── 汇总对比 ──

    print("\n" + "=" * 70)
    print("  对比报告")
    print("=" * 70)

    # 检索指标（两个模型相同，因为检索由 embedding 驱动）
    avg_mrr = np.mean([r["retrieval"]["mrr"] for r in results_7b])
    avg_recall = np.mean([r["retrieval"]["recall@5"] for r in results_7b])

    print("\n  检索质量（bge-small-zh-v1.5）:")
    print("    MRR:       {:.4f}".format(avg_mrr))
    print("    Recall@5:  {:.4f}".format(avg_recall))

    # 生成质量对比
    print("\n  {: <36} | {: <8} | {: <8} | {: <50}".format(
        "问题", "7B时间", "0.5B时间", "7B 回答"))
    print("  " + "-" * 105)
    for i, item in enumerate(EVAL_DATASET):
        t7 = results_7b[i]["timing"]
        t05 = results_05b[i]["timing"]
        a7 = results_7b[i]["answer"][:50].replace("\n", " ")
        print("  {:<36} | {:5.1f}s  | {:5.1f}s  | {}".format(
            item["question"][:36], t7, t05, a7))

    # 速度对比
    times_7b = [r["timing"] for r in results_7b]
    times_05b = [r["timing"] for r in results_05b]
    print("\n  速度对比:")
    print("    7B:   均值 {:.1f}s, 范围 {:.1f}-{:.1f}s".format(
        np.mean(times_7b), min(times_7b), max(times_7b)))
    print("    0.5B: 均值 {:.1f}s, 范围 {:.1f}-{:.1f}s".format(
        np.mean(times_05b), min(times_05b), max(times_05b)))

    # 长度对比
    len_7b = [len(r["answer"]) for r in results_7b]
    len_05b = [len(r["answer"]) for r in results_05b]
    print("\n  回答长度对比 (字符):")
    print("    7B:   均值 {:.0f}, 范围 {:.0f}-{:.0f}".format(
        np.mean(len_7b), min(len_7b), max(len_7b)))
    print("    0.5B: 均值 {:.0f}, 范围 {:.0f}-{:.0f}".format(
        np.mean(len_05b), min(len_05b), max(len_05b)))

    # RAGAS 指标（需 eval-env: langchain>=0.3,<0.4 + ragas + rapidfuzz）
    if HAS_RAGAS:
        print("\n  RAGAS 指标:")
        try:
            from ragas import SingleTurnSample
            cr = NonLLMContextRecall()
            for model_label, model_results in [("7B", results_7b), ("0.5B", results_05b)]:
                cr_scores = []
                for i, item in enumerate(EVAL_DATASET):
                    if "error" in model_results[i]:
                        continue
                    ref_ctxs = []
                    for cid in item["gt_chunks"]:
                        if 0 <= cid < len(store.chunks) and "text" in store.chunks[cid]:
                            ref_ctxs.append(store.chunks[cid]["text"])
                    if not ref_ctxs:
                        continue
                    cr_scores.append(float(cr.single_turn_score(SingleTurnSample(
                        retrieved_contexts=model_results[i]["contexts"],
                        reference_contexts=ref_ctxs,
                    ))))
                print("    Context Recall ({}): {:.3f}".format(model_label, np.mean(cr_scores)))
        except Exception as e:
            import traceback
            print("    RAGAS 失败: {}".format(e))
            traceback.print_exc()
    else:
        print("\n  RAGAS 未安装。在独立 venv 中安装以启用指标:")
        print("    python3 -m venv eval-env && source eval-env/bin/activate")
        print("    pip install ragas rapidfuzz 'langchain-core>=0.3,<0.4' 'langchain-community>=0.3,<0.4'")

    # 保存结果
    output = {
        "retrieval_mrr": float(avg_mrr),
        "retrieval_recall@5": float(avg_recall),
        "7b_avg_time": float(np.mean(times_7b)),
        "0.5b_avg_time": float(np.mean(times_05b)),
        "7b_avg_length": float(np.mean(len_7b)),
        "0.5b_avg_length": float(np.mean(len_05b)),
        "queries": [
            {
                "question": item["question"],
                "reference": item["reference"],
                "7b": {"answer": results_7b[i]["answer"], "time": results_7b[i]["timing"]},
                "0.5b": {"answer": results_05b[i]["answer"], "time": results_05b[i]["timing"]},
            }
            for i, item in enumerate(EVAL_DATASET)
        ],
    }
    with open("rag_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n  详细结果已保存到 rag_eval_results.json")


if __name__ == "__main__":
    main()
