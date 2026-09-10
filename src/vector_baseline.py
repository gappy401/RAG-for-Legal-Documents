"""
Vector-search-only retrieval baseline, run across three embedding models,
measured against eval_set.json with the same recall@k methodology as
bm25_baseline.py so results are directly comparable.
"""
import json
import sys

import faiss
import numpy as np

sys.path.insert(0, "src")
from chunking import chunk_contract
from config import load_config
from embeddings import load_embedding_model, embed_texts

MODELS_TO_TEST = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "nlpaueb/legal-bert-base-uncased",
]


def overlaps(chunk: dict, gold_start: int, gold_end: int) -> bool:
    chunk_start = chunk["child_start"]
    chunk_end = chunk_start + len(chunk["child_text"])
    return chunk_start < gold_end and gold_start < chunk_end


def evaluate_model(model_name: str, all_chunks: list[dict], title_to_text: dict,
                    present_items: list[dict], top_k_values: list[int]) -> dict:
    print(f"\n=== {model_name} ===")
    model = load_embedding_model(model_name)

    child_texts = [c["child_text"] for c in all_chunks]
    chunk_embeddings = embed_texts(model, child_texts)

    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
    index.add(chunk_embeddings.astype(np.float32))

    hits_at = {k: 0 for k in top_k_values}
    max_k = max(top_k_values)

    for item in present_items:
        query_embedding = embed_texts(model, [item["natural_question"]]).astype(np.float32)

        # search wider than max_k since we filter out other-contract chunks after
        search_k = min(len(all_chunks), max_k * 10)
        scores, indices = index.search(query_embedding, search_k)

        gold_start = item["gold_answer_span"]["answer_start"]
        gold_end = gold_start + len(item["gold_answer_span"]["text"])

        found_rank = None
        rank = 0
        for idx in indices[0]:
            chunk = all_chunks[idx]
            if chunk["parent_text"] not in title_to_text[item["contract_file"]]:
                continue
            rank += 1
            if overlaps(chunk, gold_start, gold_end):
                found_rank = rank
                break

        status = f"found at rank {found_rank}" if found_rank else "NOT in top results"
        print(f"  {item['id']:<25} {status}")

        for k in hits_at:
            if found_rank is not None and found_rank <= k:
                hits_at[k] += 1

    return hits_at


def main():
    config = load_config()
    chunking_cfg = config["chunking"]
    top_k_values = config["retrieval"]["top_k_values"]

    with open(config["paths"]["eval_set"]) as f:
        eval_set = json.load(f)
    with open(config["paths"]["cuad_json"]) as f:
        cuad = json.load(f)

    title_to_text = {c["title"]: c["paragraphs"][0]["context"] for c in cuad["data"]}

    all_chunks = []
    for title, text in title_to_text.items():
        if title not in {item["contract_file"] for item in eval_set}:
            continue
        chunks = chunk_contract(
            text,
            parent_size=chunking_cfg["parent_size"],
            parent_overlap=chunking_cfg["parent_overlap"],
            child_size=chunking_cfg["child_size"],
            child_overlap=chunking_cfg["child_overlap"],
        )
        all_chunks.extend(chunks)

    present_items = [item for item in eval_set if item["expected_answer"] == "present"]
    n = len(present_items)

    results = {}
    for model_name in MODELS_TO_TEST:
        results[model_name] = evaluate_model(model_name, all_chunks, title_to_text, present_items, top_k_values)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<45}" + "".join(f"R@{k:<8}" for k in top_k_values))
    for model_name, hits_at in results.items():
        row = f"{model_name:<45}"
        for k in top_k_values:
            row += f"{hits_at[k]}/{n} ({hits_at[k]/n:.0%}) "
        print(row)


if __name__ == "__main__":
    main()