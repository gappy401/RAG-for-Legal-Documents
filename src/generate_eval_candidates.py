"""
LLM-assisted eval set expansion, WITH a mandatory human review gate.

Samples new contracts (not already in eval_set.json), picks a mix of
present/absent clause categories per contract, and asks the Anthropic API
to draft ONE natural-language question per item -- using your own existing
hand-written eval items as few-shot style examples, so drafts match the
phrasing variety you already established (direct + scenario-based, not
CUAD's formal style).

Output goes to eval/eval_candidates.json -- a SEPARATE file from
eval_set.json. Nothing here is trusted ground truth until a human reviews
and moves accepted items into eval_set.json by hand. This script never
writes to eval_set.json directly.
"""
import argparse
import csv
import json
import random
import sys
import time

import anthropic

sys.path.insert(0, "src")
from config import load_config

MODEL = "claude-sonnet-5"
RANDOM_SEED = 42  # fixed seed so contract sampling is reproducible


def load_category_descriptions(path: str) -> dict:
    """CUAD's category_descriptions.csv has label prefixes baked into every
    cell ('Category: X', 'Description: Y') plus a BOM character on the first
    column name -- clean both before using. Keys are lowercased because
    CUAD's internal category IDs (e.g. 'Termination For Convenience') don't
    always match the CSV's capitalization (e.g. 'Termination for
    Convenience') -- a case-sensitive lookup would silently return nothing
    for several categories."""
    descriptions = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category_raw = row["Category (incl. context and answer)"]
            category = category_raw.split("Category:", 1)[-1].strip()
            description = row["Description"].split("Description:", 1)[-1].strip()
            descriptions[category.lower()] = description
    return descriptions


def pick_items_for_contract(contract: dict, max_present: int = 3, max_absent: int = 1) -> list[dict]:
    """Selects a mix of present + absent clause categories for one contract."""
    para = contract["paragraphs"][0]
    present = [qa for qa in para["qas"] if not qa["is_impossible"]]
    absent = [qa for qa in para["qas"] if qa["is_impossible"]]

    random.shuffle(present)
    random.shuffle(absent)

    selected = []
    for qa in present[:max_present]:
        category = qa["id"].split("__")[-1]
        ans = qa["answers"][0]
        selected.append({
            "category": category, "present": True,
            "text": ans["text"], "answer_start": ans["answer_start"],
        })
    for qa in absent[:max_absent]:
        category = qa["id"].split("__")[-1]
        selected.append({"category": category, "present": False, "text": None, "answer_start": None})

    return selected


def build_prompt(item: dict, category_desc: str, few_shot_examples: list[dict]) -> str:
    examples_text = "\n".join(
        f'- Category: {ex["clause_category"]} | Natural question: "{ex["natural_question"]}"'
        for ex in few_shot_examples
    )

    if item["present"]:
        context = (
            f'The contract contains a "{item["category"]}" clause. '
            f'Category meaning: {category_desc}\n'
            f'The actual clause text: "{item["text"][:400]}"'
        )
    else:
        context = (
            f'The contract does NOT contain a "{item["category"]}" clause. '
            f'Category meaning: {category_desc}'
        )

    return f"""You are drafting ONE realistic question a non-lawyer (e.g. a business
person or paralegal) would type into a contract Q&A search tool -- NOT the
formal legal phrasing a law dataset would use.

Examples of the phrasing style already used in this project:
{examples_text}

{context}

Write exactly one natural-language question matching this style. Respond
with ONLY the question text, nothing else -- no quotes, no preamble."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-contracts", type=int, default=30)
    parser.add_argument("--max-present", type=int, default=3)
    parser.add_argument("--max-absent", type=int, default=1)
    parser.add_argument("--output", default="eval/eval_candidates.json")
    args = parser.parse_args()

    config = load_config()
    with open(config["paths"]["cuad_json"]) as f:
        cuad = json.load(f)
    with open(config["paths"]["eval_set"]) as f:
        existing_eval_set = json.load(f)

    already_used = {item["contract_file"] for item in existing_eval_set}
    category_descriptions = load_category_descriptions("data/cuad-repo/category_descriptions.csv")

    # use your own hand-written items as few-shot style anchors
    few_shot_examples = existing_eval_set[:4]

    candidate_contracts = [c for c in cuad["data"] if c["title"] not in already_used]
    random.seed(RANDOM_SEED)
    sampled_contracts = random.sample(candidate_contracts, min(args.num_contracts, len(candidate_contracts)))

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    candidates = []
    for i, contract in enumerate(sampled_contracts, start=1):
        items = pick_items_for_contract(contract, args.max_present, args.max_absent)
        print(f"[{i}/{len(sampled_contracts)}] {contract['title'][:60]} -- {len(items)} items")

        for item in items:
            category_desc = category_descriptions.get(item["category"].lower(), "")
            prompt = build_prompt(item, category_desc, few_shot_examples)

            response = client.messages.create(
                model=MODEL,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            natural_question = response.content[0].text.strip()

            candidate = {
                "id": f"candidate-{len(candidates)+1:04d}",
                "contract_file": contract["title"],
                "natural_question": natural_question,
                "clause_category": item["category"],
                "expected_answer": "present" if item["present"] else "not_present",
                "gold_answer_span": (
                    {"text": item["text"], "answer_start": item["answer_start"]}
                    if item["present"] else None
                ),
                "_reviewed": False,  # human review gate -- flip to true only after checking by hand
            }
            candidates.append(candidate)
            print(f"    [{candidate['expected_answer']:<12}] {item['category']:<30} -> {natural_question}")

            time.sleep(0.5)  # gentle rate limiting

    with open(args.output, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"\nWrote {len(candidates)} UNREVIEWED candidates to {args.output}")
    print("Nothing here is trusted ground truth yet -- review each one, fix any wrong")
    print("phrasing or wrong category matches, THEN manually move accepted items into eval_set.json.")


if __name__ == "__main__":
    main()