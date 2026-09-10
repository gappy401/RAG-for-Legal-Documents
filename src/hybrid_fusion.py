"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF): combines BM25 and
vector search rankings without needing to normalize their incompatible
score scales. Reuses build_index/tokenize from bm25_baseline.py and
load_embedding_model/embed_texts from embeddings.py rather than
duplicating that logic.
"""
import json
import sys

import faiss
import numpy as np

sys.path.insert(0, "src")
from chunking import chunk_contract
from config import load_config
from bm25_base import tokenize, build_index as build_bm25_index
from embeddings import load_embedding_model, embed_texts

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # chosen from vector_baseline.py results


def overlaps(chunk: dict, gold_start: int, gold_end: int) -> bool:
    chunk_start = chunk["child_start"]
    chunk_end = chunk_start + len(chunk["child_text"])
    return chunk_start < gold_end and gold_start < chunk_end


def rrf_fuse(bm25_ranked_indices: list[int], vector_ranked_indices: list[int], k: int) -> list[int]:
    """
    Takes two ranked lists of chunk indices (best first) and returns one
    fused ranking. Rank is 1-indexed inside the formula, per the standard
    RRF definition.
    """
    scores = {}
    for rank, idx in enumerate(bm25_ranked_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(vector_ranked_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    return sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)


def main():
    config = load_config()
    chunking_cfg = config["chunking"]
    top_k_values = config["retrieval"]["top_k_values"]
    rrf_k = config["retrieval"]["rrf_k"]

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

    print(f"Indexed {len(all_chunks)} child chunks\n")

    # --- build BM25 index ---
    bm25 = build_bm25_index(all_chunks)

    # --- build vector index ---
    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    model = load_embedding_model(EMBEDDING_MODEL)
    child_texts = [c["child_text"] for c in all_chunks]
    chunk_embeddings = embed_texts(model, child_texts).astype(np.float32)
    dim = chunk_embeddings.shape[1]
    vector_index = faiss.IndexFlatIP(dim)
    vector_index.add(chunk_embeddings)

    present_items = [item for item in eval_set if item["expected_answer"] == "present"]
    n = len(present_items)

    bm25_only_hits = {k: 0 for k in top_k_values}
    vector_only_hits = {k: 0 for k in top_k_values}
    hybrid_hits = {k: 0 for k in top_k_values}
    bm25_ranked_by_item = {}
    vector_ranked_by_item = {}

    for item in present_items:
        query_tokens = tokenize(item["natural_question"])
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_ranked = sorted(range(len(all_chunks)), key=lambda i: bm25_scores[i], reverse=True)

        query_embedding = embed_texts(model, [item["natural_question"]]).astype(np.float32)
        _, vector_ranked_arr = vector_index.search(query_embedding, len(all_chunks))
        vector_ranked = vector_ranked_arr[0].tolist()

        fused_ranked = rrf_fuse(bm25_ranked, vector_ranked, k=rrf_k)
        bm25_ranked_by_item[item["id"]] = bm25_ranked
        vector_ranked_by_item[item["id"]] = vector_ranked

        gold_start = item["gold_answer_span"]["answer_start"]
        gold_end = gold_start + len(item["gold_answer_span"]["text"])

        def find_rank(ranked_indices):
            rank = 0
            for idx in ranked_indices:
                chunk = all_chunks[idx]
                if chunk["parent_text"] not in title_to_text[item["contract_file"]]:
                    continue
                rank += 1
                if overlaps(chunk, gold_start, gold_end):
                    return rank
            return None

        bm25_rank = find_rank(bm25_ranked)
        vector_rank = find_rank(vector_ranked)
        hybrid_rank = find_rank(fused_ranked)

        print(f"{item['id']:<25} BM25: {bm25_rank or 'miss':<6} "
              f"Vector: {vector_rank or 'miss':<6} Hybrid: {hybrid_rank or 'miss'}")

        for k in top_k_values:
            if bm25_rank is not None and bm25_rank <= k:
                bm25_only_hits[k] += 1
            if vector_rank is not None and vector_rank <= k:
                vector_only_hits[k] += 1
            if hybrid_rank is not None and hybrid_rank <= k:
                hybrid_hits[k] += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<12}" + "".join(f"R@{k:<10}" for k in top_k_values))
    for label, hits in [("BM25", bm25_only_hits), ("Vector", vector_only_hits), ("Hybrid", hybrid_hits)]:
        row = f"{label:<12}"
        for k in top_k_values:
            row += f"{hits[k]}/{n} ({hits[k]/n:.0%}) "
        print(row)

    sweep_rrf_k(
        all_chunks, title_to_text, present_items,
        bm25_ranked_by_item, vector_ranked_by_item,
        top_k_values, k_values_to_test=[1, 5, 10, 20, 40, 60],
    )


def sweep_rrf_k(all_chunks, title_to_text, present_items, bm25_ranked_by_item,
                 vector_ranked_by_item, top_k_values, k_values_to_test):
    """
    Reuses already-computed BM25/vector rankings (expensive part) and only
    re-runs the cheap RRF fusion math for each candidate k value.
    """
    print(f"\n{'RRF k':<10}" + "".join(f"R@{k:<10}" for k in top_k_values))

    for rrf_k in k_values_to_test:
        hits = {k: 0 for k in top_k_values}
        n = len(present_items)

        for item in present_items:
            bm25_ranked = bm25_ranked_by_item[item["id"]]
            vector_ranked = vector_ranked_by_item[item["id"]]
            fused_ranked = rrf_fuse(bm25_ranked, vector_ranked, k=rrf_k)

            gold_start = item["gold_answer_span"]["answer_start"]
            gold_end = gold_start + len(item["gold_answer_span"]["text"])

            rank = 0
            found_rank = None
            for idx in fused_ranked:
                chunk = all_chunks[idx]
                if chunk["parent_text"] not in title_to_text[item["contract_file"]]:
                    continue
                rank += 1
                if overlaps(chunk, gold_start, gold_end):
                    found_rank = rank
                    break

            for k in top_k_values:
                if found_rank is not None and found_rank <= k:
                    hits[k] += 1

        row = f"{rrf_k:<10}"
        for k in top_k_values:
            row += f"{hits[k]}/{n} ({hits[k]/n:.0%}) "
        print(row)


if __name__ == "__main__":
    main()