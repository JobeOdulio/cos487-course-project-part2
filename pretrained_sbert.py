# python -m pip install \
#   "sentence-transformers==5.1.2" \
#   "transformers==4.51.3" \
#   "huggingface-hub==0.36.2"

import json
import re
import html
import argparse
import os
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from preprocessing import preprocess, cleantext


# This part is for cleaning the Answers.json HTML - For project you need to carefully design this
# Also note that, the models have their own tokenizer; so you are not tokenizing the text here


def main():
    # All your code should take inputs as parameters with default values
    p = argparse.ArgumentParser()
    p.add_argument("--papers", default="JSON Files.zip")
    p.add_argument("--topics", default="Train.json")
    p.add_argument("--output", default="results/sbert_paper.tsv")
    p.add_argument("--mode", choices=["paper", "paragraph"], default= "paper")
    p.add_argument("--model", default="sentence-transformers/all-distilroberta-v1")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--run_name", default="sbert_distilroberta")
    args = p.parse_args()

    # this checks if gpu is available or not
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = preprocess()
    if args.mode == "paper":
        collection = processor.load_paper(args.papers)
    else:
        collection = processor.load_paragraph(args.papers)
    print("Starting preprocessing...", flush=True)
    # Used for loading collection and queries -- modify for your project
    
    answer_ids = list(collection.keys())
    answer_texts = [
        collection[doc_id]["text"]
        for doc_id in answer_ids
    ]
    print(f"Loaded {len(answer_ids)} {args.mode}s")
    with open(args.topics, "r", encoding="utf-8") as f:
        topics = __import__("json").load(f)
    query_ids = [str(t["query_id"]) for t in topics]
    query_texts = [
            cleantext(t["query"])
            for t in topics
        ]
    print(f"Loaded {len(query_texts)} queries")
    # Iterating through answers and passing them through pre-processing pipeline

    # Loading the model
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length

    print(f"Encoding {len(answer_texts)} answers...")
    # Encoding your collection
    answer_emb = model.encode(
        answer_texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Encoding {len(query_texts)} queries...")
    # encoding your queries
    query_emb = model.encode(
        query_texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Retrieval (chunked over queries to limit memory)
    k = min(args.top_k, len(answer_ids))
    chunk = 128
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(query_ids), chunk), desc="Searching"):
            q_chunk = query_emb[start:start + chunk]
            # cosine (normalized)
            scores = util.dot_score(q_chunk, answer_emb)
            top_scores, top_idx = torch.topk(scores, k=k, dim=1)

            for i in range(q_chunk.shape[0]):
                
                qid = query_ids[start + i]
                for rank in range(k):
                    
                    doc_id = answer_ids[top_idx[i, rank].item()]
                    score = top_scores[i, rank].item()
                    # TREC: qid Q0 docid rank score run_name
                    out.write(f"{qid} Q0 {doc_id} {rank + 1} {score:.6f} {args.run_name}\n")

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
