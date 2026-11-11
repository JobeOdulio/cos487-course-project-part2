# python -m pip install "accelerate==1.2.1"
# python -m pip install "datasets==3.6.0"
# CUDA_VISIBLE_DEVICES=0 python fine_tune_sbert.py

import json
import re
import html
import random
import argparse
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from tqdm import tqdm


def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def make_query(t: dict) -> str:
    return (clean_html(t["Title"]) + " " + clean_html(t.get("Body", ""))).strip()


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


def retrieve_and_write(model, query_ids, query_texts, answer_ids, answer_emb,
                       output, run_name, top_k, batch_size):
    query_emb = model.encode(query_texts, batch_size=batch_size, convert_to_tensor=True,
                             normalize_embeddings=True, show_progress_bar=True)
    k = min(top_k, len(answer_ids))
    chunk = 256
    with open(output, "w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(query_ids), chunk), desc="Searching"):
            q = query_emb[start:start + chunk]
            scores = util.dot_score(q, answer_emb)
            top_scores, top_idx = torch.topk(scores, k=k, dim=1)
            for i in range(q.shape[0]):
                qid = query_ids[start + i]
                for rank in range(k):
                    aid = answer_ids[top_idx[i, rank].item()]
                    out.write(f"{qid} Q0 {aid} {rank + 1} {top_scores[i, rank].item():.6f} {run_name}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", default="Answers.json")
    p.add_argument("--topics_train", default="topics_1.json")
    p.add_argument("--topics_test", default="topics_2.json")
    p.add_argument("--qrels", default="qrel_1.tsv")
    p.add_argument("--base_model", default="sentence-transformers/all-distilroberta-v1")
    p.add_argument("--model_out", default="best_sbert_model")
    p.add_argument("--output", default="run_topics2_finetuned.txt")
    p.add_argument("--run_name", default="sbert_distilroberta_ft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_corpus_size", type=int, default=0,
                   help="0 = full corpus for validation; otherwise val-relevant docs + random distractors up to this size")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------------- Load data ----------------
    with open(args.answers, encoding="utf-8") as f:
        answers = json.load(f)
    with open(args.topics_train, encoding="utf-8") as f:
        topics1 = json.load(f)
    with open(args.topics_test, encoding="utf-8") as f:
        topics2 = json.load(f)

    answer_text = {a["Id"]: clean_html(a["Text"]) for a in answers}
    qrels = load_qrels(args.qrels)

    # Only keep topics that have at least one relevant, existing answer
    topics1 = [t for t in topics1
               if t["Id"] in qrels and any(d in answer_text for d in qrels[t["Id"]])][:5]
    print(f"Topics with usable judgments: {len(topics1)}")

    # 90/10 split by query
    random.shuffle(topics1)
    n_val = max(1, int(len(topics1) * args.val_ratio))
    val_topics, train_topics = topics1[:n_val], topics1[n_val:]
    print(f"Train queries: {len(train_topics)} | Validation queries: {len(val_topics)}")

    # Training pairs (query, relevant answer)
    train_examples = []
    for t in train_topics:
        q = make_query(t)
        for docid in qrels[t["Id"]]:
            if docid in answer_text:
                train_examples.append(InputExample(texts=[q, answer_text[docid]]))
    print(f"Training pairs: {len(train_examples)}")

    # Validation evaluator
    val_queries = {t["Id"]: make_query(t) for t in val_topics}
    val_rel = {t["Id"]: {d for d in qrels[t["Id"]] if d in answer_text} for t in val_topics}

    if args.val_corpus_size > 0:
        keep = set().union(*val_rel.values())
        others = [d for d in answer_text if d not in keep]
        random.shuffle(others)
        keep.update(others[: max(0, args.val_corpus_size - len(keep))])
        val_corpus = {d: answer_text[d] for d in keep}
    else:
        val_corpus = answer_text

    evaluator = InformationRetrievalEvaluator(
        queries=val_queries,
        corpus=val_corpus,
        relevant_docs=val_rel,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[100],
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[10, 100],
        score_functions={"cosine": util.cos_sim},
        name="val",
        show_progress_bar=False,
    )
    # Select best checkpoint by nDCG@10 (honoured by sentence-transformers >= 3.0)
    try:
        evaluator.primary_metric = "cosine_ndcg@10"
    except Exception:
        pass

    # Model + loss
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_length

    # NoDuplicatesDataLoader avoids the same query/answer appearing twice in a batch
    # (which would create false in-batch negatives).
    train_loader = NoDuplicatesDataLoader(train_examples, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_loader) * args.epochs * 0.1)

    # Train (evaluates each epoch, keeps best)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        # evaluate at end of every epoch
        evaluation_steps=0,
        output_path=args.model_out,
        save_best_model=True,
        use_amp=(device == "cuda"),
        show_progress_bar=True,
    )

    # Inference on topics_2 with the best saved model
    print(f"Loading best model from {args.model_out}")
    best = SentenceTransformer(args.model_out, device=device)
    best.max_seq_length = args.max_seq_length

    answer_ids = list(answer_text.keys())
    answer_emb = best.encode([answer_text[a] for a in answer_ids], batch_size=64,
                             convert_to_tensor=True, normalize_embeddings=True,
                             show_progress_bar=True)

    retrieve_and_write(
        best,
        [t["Id"] for t in topics2],
        [make_query(t) for t in topics2],
        answer_ids, answer_emb,
        args.output, args.run_name, args.top_k, 64,
    )
    print(f"Done. Final TREC run written to {args.output}")


if __name__ == "__main__":
    main()