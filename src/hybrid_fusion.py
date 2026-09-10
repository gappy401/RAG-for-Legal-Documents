"""
Hybrid retrieval via Reciprocal Rank Fusion (BM25 + BGE embeddings).

Two corpus scopes, controlled by --full-corpus:
  - default: indexes only the 4 contracts your eval set references (fast,
    good for quick iteration on chunking/RRF-k changes)
  - --full-corpus: indexes all 510 CUAD contracts (~68k chunks) -- the
    realistic-scale test. Embeddings are cached to disk after first run
    since embedding 68k chunks is the expensive step; delete the cache
    file to force a rebuild after a config change.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, "src")
from chunking import chunk_contract
from config import load_config
from bm25_base import tokenize, build_index as build_bm25_index
from embeddings import load_embedding_model, embed_texts

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # chosen from vector_baseline.py results
FULL_CORPUS_CACHE = Path("data/full_corpus_cache.pkl")


def overlaps(chunk: dict, gold_start: int, gold_end: int) -> bool:
    chunk_start = chunk["child_start"]
    chunk_end = chunk_start + len(chunk["child_text"])
    return chunk_start < gold_end and gold_start < chunk_end


def rrf_fuse(bm25_ranked_indices: list[int], vector_ranked_indices: list[int], k: int) -> list[int]:
    scores = {}
    for rank, idx in enumerate(bm25_ranked_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(vector_ranked_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)


def find_rank(ranked_indices, all_chunks, title_to_text, contract_file, gold_start, gold_end):
    """Global rank position -- every chunk from every document counts
    toward the rank, even ones from other contracts we skip checking.
    This matters: it's what makes cross-document competition real."""
    for rank, idx in enumerate(ranked_indices, start=1):
        chunk = all_chunks[idx]
        if chunk["parent_text"] not in title_to_text[contract_file]:
            continue
        if overlaps(chunk, gold_start, gold_end):
            return rank
    return None


def chunk_all(contracts, chunking_cfg):
    all_chunks = []
    for c in contracts:
        text = c["paragraphs"][0]["context"]
        chunks = chunk_contract(
            text,
            parent_size=chunking_cfg["parent_size"],
            parent_overlap=chunking_cfg["parent_overlap"],
            child_size=chunking_cfg["child_size"],
            child_overlap=chunking_cfg["child_overlap"],
        )
        all_chunks.extend(chunks)
    return all_chunks


def build_index(all_chunks, use_cache: bool):
    """Builds BM25 + vector index over all_chunks. If use_cache is True,
    saves/loads the result from disk (only worth it at full-corpus scale --
    embedding 68k chunks is slow; embedding ~500 is not)."""
    if use_cache and FULL_CORPUS_CACHE.exists():
        print(f"Loading cached index from {FULL_CORPUS_CACHE} ...")
        with open(FULL_CORPUS_CACHE, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    bm25 = build_bm25_index(all_chunks)
    print(f"BM25 index built in {time.time() - t0:.1f}s")

    t0 = time.time()
    model = load_embedding_model(EMBEDDING_MODEL)
    child_texts = [c["child_text"] for c in all_chunks]
    chunk_embeddings = embed_texts(model, child_texts).astype(np.float32)
    print(f"Embedded {len(all_chunks)} chunks in {time.time() - t0:.1f}s")

    dim = chunk_embeddings.shape[1]
    vector_index = faiss.IndexFlatIP(dim)
    vector_index.add(chunk_embeddings)

    result = {"bm25": bm25, "vector_index": vector_index}

    if use_cache:
        FULL_CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(FULL_CORPUS_CACHE, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached to {FULL_CORPUS_CACHE} for future runs")

    return result


def sweep_rrf_k(all_chunks, title_to_text, present_items, bm25_ranked_by_item,
                 vector_ranked_by_item, top_k_values, k_values_to_test):
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
            rank = find_rank(fused_ranked, all_chunks, title_to_text, item["contract_file"], gold_start, gold_end)
            for k in top_k_values:
                if rank is not None and rank <= k:
                    hits[k] += 1
        row = f"{rrf_k:<10}"
        for k in top_k_values:
            row += f"{hits[k]}/{n} ({hits[k]/n:.0%}) "
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-corpus", action="store_true",
                         help="Index all 510 CUAD contracts instead of just the 4 in eval_set.json")
    parser.add_argument("--sweep-k", action="store_true",
                         help="Also run an RRF k-sweep after the main comparison")
    args = parser.parse_args()

    config = load_config()
    chunking_cfg = config["chunking"]
    top_k_values = config["retrieval"]["top_k_values"]
    rrf_k = config["retrieval"]["rrf_k"]

    with open(config["paths"]["eval_set"]) as f:
        eval_set = json.load(f)
    with open(config["paths"]["cuad_json"]) as f:
        cuad = json.load(f)

    title_to_text = {c["title"]: c["paragraphs"][0]["context"] for c in cuad["data"]}
    eval_contract_titles = {item["contract_file"] for item in eval_set}

    if args.full_corpus:
        print(f"Mode: FULL CORPUS ({len(cuad['data'])} contracts)")
        contracts = cuad["data"]
    else:
        print(f"Mode: eval-set contracts only ({len(eval_contract_titles)} contracts)")
        contracts = [c for c in cuad["data"] if c["title"] in eval_contract_titles]

    all_chunks = chunk_all(contracts, chunking_cfg)
    print(f"Indexed {len(all_chunks)} child chunks\n")

    index_data = build_index(all_chunks, use_cache=args.full_corpus)
    bm25 = index_data["bm25"]
    vector_index = index_data["vector_index"]
    model = load_embedding_model(EMBEDDING_MODEL)

    present_items = [item for item in eval_set if item["expected_answer"] == "present"]
    n = len(present_items)

    bm25_hits = {k: 0 for k in top_k_values}
    vector_hits = {k: 0 for k in top_k_values}
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

        bm25_rank = find_rank(bm25_ranked, all_chunks, title_to_text, item["contract_file"], gold_start, gold_end)
        vector_rank = find_rank(vector_ranked, all_chunks, title_to_text, item["contract_file"], gold_start, gold_end)
        hybrid_rank = find_rank(fused_ranked, all_chunks, title_to_text, item["contract_file"], gold_start, gold_end)

        print(f"{item['id']:<25} BM25: {bm25_rank or 'miss':<6} "
              f"Vector: {vector_rank or 'miss':<6} Hybrid: {hybrid_rank or 'miss'}")

        for k in top_k_values:
            if bm25_rank is not None and bm25_rank <= k:
                bm25_hits[k] += 1
            if vector_rank is not None and vector_rank <= k:
                vector_hits[k] += 1
            if hybrid_rank is not None and hybrid_rank <= k:
                hybrid_hits[k] += 1

    print(f"\n{'='*60}\nSUMMARY ({len(all_chunks)} chunks)\n{'='*60}")
    print(f"{'Method':<12}" + "".join(f"R@{k:<10}" for k in top_k_values))
    for label, hits in [("BM25", bm25_hits), ("Vector", vector_hits), ("Hybrid", hybrid_hits)]:
        row = f"{label:<12}"
        for k in top_k_values:
            row += f"{hits[k]}/{n} ({hits[k]/n:.0%}) "
        print(row)

    if args.sweep_k:
        sweep_rrf_k(all_chunks, title_to_text, present_items,
                    bm25_ranked_by_item, vector_ranked_by_item,
                    top_k_values, k_values_to_test=[1, 5, 10, 20, 40, 60])


if __name__ == "__main__":
    main()