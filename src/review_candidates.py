"""
Prints a side-by-side review view for one or more candidates: the
question, the claimed gold span, an automated offset-match check against
the real contract text, and surrounding context -- the same manual
verification process, generalized to any candidate.

Usage:
    python src/review_candidate.py --id candidate-0001
    python src/review_candidate.py --category "Agreement Date"
    python src/review_candidate.py --all
"""
import argparse
import json


def load_data():
    with open("data/cuad-repo/unzipped/CUADv1.json") as f:
        cuad = json.load(f)
    with open("eval/eval_candidates.json") as f:
        candidates = json.load(f)
    title_to_text = {c["title"]: c["paragraphs"][0]["context"] for c in cuad["data"]}
    return title_to_text, candidates


def review_one(candidate: dict, title_to_text: dict):
    print("=" * 80)
    print(f"ID: {candidate['id']}  |  Category: {candidate['clause_category']}  |  "
          f"Expected: {candidate['expected_answer']}  |  Reviewed: {candidate.get('_reviewed', False)}")
    print(f"Contract: {candidate['contract_file']}")
    print(f"Question: {candidate['natural_question']!r}")
    print("-" * 80)

    span = candidate.get("gold_answer_span")
    if span is None:
        print("(no gold span -- this is a 'not_present' item; just judge whether the")
        print(" question is a plausible, realistic thing to ask about this contract)")
        return

    text = title_to_text.get(candidate["contract_file"])
    if text is None:
        print("!! Contract title not found in CUAD data -- can't verify offset.")
        return

    answer_start = span["answer_start"]
    gold_text = span["text"]
    actual_slice = text[answer_start:answer_start + len(gold_text)]
    matches = actual_slice == gold_text

    print(f"Gold span text: {gold_text!r}")
    print(f"Offset matches real contract text? {'YES' if matches else 'NO -- MISMATCH, investigate'}")
    print()
    print("Context (150 chars before/after):")
    before = text[max(0, answer_start - 150):answer_start]
    after = text[answer_start + len(gold_text):answer_start + len(gold_text) + 150]
    print(f"...{before}>>>{actual_slice}<<<{after}...")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", help="Review a single candidate by id, e.g. candidate-0001")
    group.add_argument("--category", help="Review all candidates in one category, e.g. 'Agreement Date'")
    group.add_argument("--all", action="store_true", help="Review every candidate, one at a time (pauses between each)")
    args = parser.parse_args()

    title_to_text, candidates = load_data()

    if args.id:
        candidate = next((c for c in candidates if c["id"] == args.id), None)
        if candidate is None:
            print(f"No candidate found with id {args.id!r}")
            return
        review_one(candidate, title_to_text)

    elif args.category:
        matches = [c for c in candidates if c["clause_category"].lower() == args.category.lower()]
        print(f"{len(matches)} candidates in category {args.category!r}\n")
        for c in matches:
            review_one(c, title_to_text)
            print()

    elif args.all:
        for i, c in enumerate(candidates, start=1):
            review_one(c, title_to_text)
            if i < len(candidates):
                input("\n[Enter for next candidate, Ctrl+C to stop] ")


if __name__ == "__main__":
    main()