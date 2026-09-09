import json
import sys
sys.path.insert(0, "src")
from chunking import chunk_contract

with open("eval/eval_set.json") as f:
    eval_set = json.load(f)
with open("data/cuad-repo/unzipped/CUADv1.json") as f:
    cuad = json.load(f)

title_to_text = {c["title"]: c["paragraphs"][0]["context"] for c in cuad["data"]}
present_items = [item for item in eval_set if item["expected_answer"] == "present"]

for size in [600, 800]:
    print(f"--- child_size={size} ---")
    chunk_cache = {}
    for item in present_items:
        title = item["contract_file"]
        text = title_to_text[title]
        if title not in chunk_cache:
            chunk_cache[title] = chunk_contract(text, child_size=size, child_overlap=size // 8)
        chunks = chunk_cache[title]

        gold_start = item["gold_answer_span"]["answer_start"]
        gold_end = gold_start + len(item["gold_answer_span"]["text"])
        matches = [c for c in chunks if c["child_start"] <= gold_start < c["child_start"] + len(c["child_text"])]
        intact = any(gold_end <= c["child_start"] + len(c["child_text"]) for c in matches)
        if not intact:
            for c in matches:
                child_end = c["child_start"] + len(c["child_text"])
                print(f"  SPLIT: {item['id']} -- child covers [{c['child_start']}:{child_end}] "
                      f"({child_end - c['child_start']} chars), gold needs up to {gold_end} "
                      f"({gold_end - child_end} chars short)")
    print()