"""
Sweep child_size values and measure how many gold spans stay intact
at the child level. Run this yourself and share the printed table.
"""
import json
import sys
sys.path.insert(0, "src")  # adjust if you run this from a different directory
from chunking import chunk_contract

with open("eval/eval_set.json") as f:
    eval_set = json.load(f)
with open("data/cuad-repo/unzipped/CUADv1.json") as f:
    cuad = json.load(f)
    
title_to_text = {c["title"]: c["paragraphs"][0]["context"] for c in cuad["data"]}

present_items = [item for item in eval_set if item["expected_answer"] == "present"]

child_sizes_to_test = [300, 400, 600, 800, 1000]

print(f"{'child_size':<12}{'intact':<10}{'split':<10}{'total children (all 4 contracts)':<10}")

for child_size in child_sizes_to_test:
    chunk_cache = {}
    intact_count = 0
    split_count = 0
    total_children = 0

    for item in present_items:
        title = item["contract_file"]
        text = title_to_text[title]

        if title not in chunk_cache:
            chunk_cache[title] = chunk_contract(text, child_size=child_size, child_overlap=child_size // 8)
        chunks = chunk_cache[title]

        gold_start = item["gold_answer_span"]["answer_start"]
        gold_end = gold_start + len(item["gold_answer_span"]["text"])

        matches = [c for c in chunks if c["child_start"] <= gold_start < c["child_start"] + len(c["child_text"])]
        intact = any(gold_end <= c["child_start"] + len(c["child_text"]) for c in matches)

        if intact:
            intact_count += 1
        else:
            split_count += 1

    # count total children across all 4 contracts (only counted once per contract, not per eval item)
    total_children = sum(len(v) for v in chunk_cache.values())

    print(f"{child_size:<12}{intact_count:<10}{split_count:<10}{total_children:<10}")