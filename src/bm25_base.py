"""
BM25-only retrieval baseline, measured against eval_set.json.

This is deliberately the weakest link in the pipeline -- pure lexical
matching, no embeddings, no reranking -- so later stages (vector search,
hybrid fusion, reranking) have a real number to beat, not a guess.
"""
import json
import re
import sys

from rank_bm25 import BM25Okapi

sys.path.insert(0, "src")
from chunking import chunk_contract

CHILD_SIZE = 600  # locked in from the chunking experiments


def tokenize(text: str) -> list[str]:
    """Simplest possible tokenizer: lowercase, split on non-alphanumeric.
    BM25 doesn't need anything fancier than this to start."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index(all_chunks: list[dict]):
    """Tokenizes every child chunk and builds a BM25 index over them."""
    tokenized_corpus = [tokenize(c["child_text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def overlaps(chunk: dict, gold_start: int, gold_end: int) -> bool:
    """True if this chunk's character range overlaps the gold span at all
    (not full containment -- a partial hit still counts as 'found it')."""
    chunk_start = chunk["child_start"]
    chunk_end = chunk_start + len(chunk["child_text"])
    return chunk_start < gold_end and gold_start < chunk_end


def main():
    with open("eval/eval_set.json") as f:
        eval_set = json.load(f)
    with open("data/cuad-repo/unzipped/CUADv1.json") as f:
        cuad = json.load(f)

    title_to_text = {c["title"]: c["paragraphs"][0]["context"] for c in cuad["data"]}

    # build ONE combined index across all 4 contracts -- this matters:
    # a real system doesn't know in advance which document holds the
    # answer, so retrieval has to search across everything, not just
    # the one contract we happen to be testing against.
    all_chunks = []
    for title, text in title_to_text.items():
        if title not in {item["contract_file"] for item in eval_set}:
            continue  # only index the 4 contracts our eval set actually covers
        chunks = chunk_contract(text, child_size=CHILD_SIZE, child_overlap=CHILD_SIZE // 8)
        all_chunks.extend(chunks)

    print(f"Indexed {len(all_chunks)} child chunks\n")

    bm25 = build_index(all_chunks)

    present_items = [item for item in eval_set if item["expected_answer"] == "present"]

    hits_at = {1: 0, 3: 0, 5: 0}

    for item in present_items:
        query_tokens = tokenize(item["natural_question"])
        scores = bm25.get_scores(query_tokens)

        # rank chunk indices by score, descending
        ranked_indices = sorted(range(len(all_chunks)), key=lambda i: scores[i], reverse=True)

        gold_start = item["gold_answer_span"]["answer_start"]
        gold_end = gold_start + len(item["gold_answer_span"]["text"])

        # find the rank (1-indexed) of the first chunk that overlaps the gold span
        found_rank = None
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = all_chunks[idx]
            # only compare against chunks from the SAME contract --
            # overlap check on char offsets is meaningless across documents
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