import math
import argparse
from collections import defaultdict

def load_qrels(path):
    """Standard TREC qrels: qid iter docid rel (tab/space separated)."""
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[:4]
            rel = int(rel)

            if rel > 0:
                qrels[qid][docid] = rel
    return qrels
def load_run(path):
    run =  defaultdict(list)
    with open(path, "r", encoding="utf") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            qid, _, docid, rank, score, _ = parts[:6]

            run[qid].append(
                (int(rank), docid, float(score))

            )
    for qid in run:
        run[qid].sort(key=lambda x: x[0])
    return run
def precision_at_k(ranked_docs, relevant, k):
    hits = sum(
        1 for doc in ranked_docs[:k]
        if relevant(doc)
    )
    return hits / k
def mrr(ranked_docs, relevant):
    for rank, doc in enumerate(ranked_docs, start=1):
        if relevant(doc):
            return 1 / rank
    return 0.0
def dcg_at_k(ranked_docs, qrels, k):
    score = 0.0 
    for rank, doc in enumerate(ranked_docs[:k], start=1):
        rel = qrels.get(doc, 0)
        gain = (2 ** rel) - 1
        score += gain / math.log2(rank + 1)
    return score

def ndcg_at_k(ranked_docs, qrels, k):
    actual = dcg_at_k(ranked_docs, qrels, k)
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    ideal = sum(
        ((2 ** rel) - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_rels, start=1)
    )
    return actual / ideal if ideal > 0 else 0.0
def evaluate(qrels, run):
    metrics = defaultdict(list)
    for qid, judgements in qrels.items():
        ranked_docs = [docid for _, docid, _ in run.get(qid, [])]
        relevant_any = lambda d: judgements.get(d, 0) >= 1
        relevant_grade2 = lambda d: judgements.get(d, 0) == 2
        metrics["nDCG@5"].append(ndcg_at_k(ranked_docs, judgements, 5))
        metrics["nDCG@10"].append(ndcg_at_k(ranked_docs, judgements, 10))
        metrics["MRR"].append(mrr(ranked_docs, relevant_any))
        metrics["P@5"].append(precision_at_k(ranked_docs, relevant_any, 5))
        metrics["P@10"].append(precision_at_k(ranked_docs, relevant_any, 10))
    print(f"\nEvaluated queries: {len(qrels)}")
    print("-" * 48)
    for name, values in metrics.items():
        average = sum(values) / len(values) if values else 0
        print(f"{name:<28} {average:.4f}")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qrels", default="jobe_paper_dense_study.tsv")
    p.add_argument("--run", required=True)
    args = p.parse_args()
    qrels=load_qrels(args.qrels)
    run = load_run(args.run)
    evaluate(qrels,run)