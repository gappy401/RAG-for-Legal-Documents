"""
Sweep child_size values and measure how many gold spans stay intact.
Paths come from config.yaml; the sizes to test are a CLI argument,
since a sweep list is something you want to vary per-run, not a fixed
setting that belongs in version-controlled config.

Usage:
    python test_chunk_sizes.py
    python test_chunk_sizes.py --child-sizes 550 600 650
"""
import argparse
import json
import sys

sys.path.insert(0, "src")
from chunking import chunk_contract
from config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--child-sizes", type=int, nargs="+",
        default=[300, 400, 600, 800, 1000],
        help="child_size values to test",
    )
    args = parser.parse_args()

    config = load_config()

    with open(config["paths"]["eval_set"]) as f:
        eval_set = json.load(f)
    with open(config["paths"]["cuad_json"]) as f:
        cuad = json.load(f)

    title_to_text = {c["title"]: c["paragraphs"][0]["context"] for c in cuad["data"]}
    present_items = [item for item in eval_set if item["expected_answer"] == "present"]

    print(f"{'child_size':<12}{'intact':<10}{'split':<10}{'total children':<10}")

    for child_size in args.child_sizes:
        chunk_cache = {}
        intact_count = 0
        split_count = 0

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

        total_children = sum(len(v) for v in chunk_cache.values())
        print(f"{child_size:<12}{intact_count:<10}{split_count:<10}{total_children:<10}")


if __name__ == "__main__":
    main()