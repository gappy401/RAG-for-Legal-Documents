"""
BM25-only retrieval baseline, measured against eval_set.json.
Config-driven: chunk sizes and paths come from config.yaml, not
hardcoded here.
"""
import json
import re
import sys

from rank_bm25 import BM25Okapi

sys.path.insert(0, "src")
from chunking import chunk_contract
from config import load_config


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index(all_chunks: list[dict]):
    tokenized_corpus = [tokenize(c["child_text"]) for c in all_chunks]
    return BM25Okapi(tokenized_corpus)


def overlaps(chunk: dict, gold_start: int, gold_end: int) -> bool:
    chunk_start = chunk["child_start"]
    chunk_end = chunk_start + len(chunk["child_text"])
    return chunk_start < gold_end and gold_start < chunk_end


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

    print(f"Indexed {len(all_chunks)} child chunks "
          f"(child_size={chunking_cfg['child_size']} from config.yaml)\n")

    bm25 = build_index(all_chunks)
    present_items = [item for item in eval_set if item["expected_answer"] == "present"]
    hits_at = {k: 0 for k in top_k_values}

    for item in present_items:
        query_tokens = tokenize(item["natural_question"])
        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(all_chunks)), key=lambda i: scores[i], reverse=True)

        gold_start = item["gold_answer_span"]["answer_start"]
        gold_end = gold_start + len(item["gold_answer_span"]["text"])

        found_rank = None
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = all_chunks[idx]
            if chunk["parent_text"] not in title_to_text[item["contract_file"]]:
                continue
            if overlaps(chunk, gold_start, gold_end):
                found_rank = rank
                break

        status = f"found at rank {found_rank}" if found_rank else "NOT in top results"
        print(f"{item['id']:<25} {status}")

        for k in hits_at:
            if found_rank is not None and found_rank <= k:
                hits_at[k] += 1

    print()
    n = len(present_items)
    for k, hits in hits_at.items():
        print(f"Recall@{k}: {hits}/{n} = {hits/n:.0%}")


if __name__ == "__main__":
    main()