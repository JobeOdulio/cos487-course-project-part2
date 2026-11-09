# python -m pip install \
#   "sentence-transformers==5.1.2" \
#   "transformers==4.51.3" \
#   "huggingface-hub==0.36.2"

import json
import re
import html
import argparse

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# This part is for cleaning the Answers.json HTML - For project you need to carefully design this
# Also note that, the models have their own tokenizer; so you are not tokenizing the text here
def clean_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    # All your code should take inputs as parameters with default values
    p = argparse.ArgumentParser()
    p.add_argument("--answers", default="Answers.json")
    p.add_argument("--topics", default="topics_1.json")
    p.add_argument("--output", default="run.txt")
    p.add_argument("--model", default="sentence-transformers/all-distilroberta-v1")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--run_name", default="sbert_distilroberta")
    args = p.parse_args()

    # this checks if gpu is available or not
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Used for loading collection and queries -- modify for your project
    with open(args.answers, "r", encoding="utf-8") as f:
        answers = json.load(f)[:2000]
    with open(args.topics, "r", encoding="utf-8") as f:
        topics = json.load(f)

    # Iterating through answers and passing them through pre-processing pipeline
    answer_ids = [a["Id"] for a in answers]
    answer_texts = [clean_html(a["Text"]) for a in answers]

    query_ids = [t["Id"] for t in topics]
    # Query = title + body (title repeated info is fine; body adds context)
    query_texts = [
        (clean_html(t["Title"]) + " " + clean_html(t.get("Body", ""))).strip()
        for t in topics
    ]

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
    chunk = 256

    with open(args.output, "w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(query_ids), chunk), desc="Searching"):
            q_chunk = query_emb[start:start + chunk]
            # cosine (normalized)
            scores = util.dot_score(q_chunk, answer_emb)
            top_scores, top_idx = torch.topk(scores, k=k, dim=1)

            for i in range(q_chunk.shape[0]):
                qid = query_ids[start + i]
                for rank in range(k):
                    aid = answer_ids[top_idx[i, rank].item()]
                    score = top_scores[i, rank].item()
                    # TREC: qid Q0 docid rank score run_name
                    out.write(f"{qid} Q0 {aid} {rank + 1} {score:.6f} {args.run_name}\n")

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
