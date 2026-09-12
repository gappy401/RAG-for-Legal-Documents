"""
Auto-flags likely-problematic items in eval_candidates.json so manual
review can focus on flagged items instead of reading all of them.

Two checks:
1. Duplicate phrasing -- same natural_question text reused across
   different contracts within the same category (defeats the point of
   varied, realistic phrasing).
2. Garbled/OCR text -- gold_answer_span text containing words a spell
   checker doesn't recognize, which tends to catch OCR errors like
   'UQITDITV' (should be 'LIQUIDITY'). Not perfect -- legal jargon and
   proper nouns will trigger some false positives -- but narrows 120
   items down to a much smaller list worth a closer look.
"""
import json
import re
from collections import Counter, defaultdict

from spellchecker import SpellChecker

spell = SpellChecker()


def find_duplicate_phrasing(candidates: list[dict]) -> list[str]:
    by_category = defaultdict(list)
    for c in candidates:
        by_category[c["clause_category"]].append(c["id"])

    # group by (category, question) instead to find the actual duplicate ids
    by_category_question = defaultdict(list)
    for c in candidates:
        by_category_question[(c["clause_category"], c["natural_question"])].append(c["id"])

    flagged = []
    for (category, question), ids in by_category_question.items():
        if len(ids) > 1:
            flagged.extend(ids)
    return flagged


def find_garbled_text(candidates: list[dict], min_unknown_ratio: float = 0.3) -> list[str]:
    flagged = []
    for c in candidates:
        span = c.get("gold_answer_span")
        if not span:
            continue
        text = span["text"]
        words = re.findall(r"[A-Za-z]+", text)
        if len(words) < 2:
            continue
        # only check longer words -- short words trigger too many false
        # positives (abbreviations, legal shorthand)
        words_to_check = [w for w in words if len(w) >= 5]
        if not words_to_check:
            continue
        unknown = spell.unknown([w.lower() for w in words_to_check])
        ratio = len(unknown) / len(words_to_check)
        if ratio >= min_unknown_ratio:
            flagged.append((c["id"], ratio, list(unknown)[:5]))
    return flagged


def main():
    with open("eval/eval_candidates.json") as f:
        candidates = json.load(f)

    print(f"Checking {len(candidates)} candidates...\n")

    dup_ids = find_duplicate_phrasing(candidates)
    print(f"=== Duplicate phrasing: {len(dup_ids)} items flagged ===")
    for cid in dup_ids:
        item = next(c for c in candidates if c["id"] == cid)
        print(f"  {cid}: [{item['clause_category']}] {item['natural_question']!r}")

    garbled = find_garbled_text(candidates)
    print(f"\n=== Possibly garbled/OCR text: {len(garbled)} items flagged ===")
    for cid, ratio, sample_unknown in garbled:
        item = next(c for c in candidates if c["id"] == cid)
        print(f"  {cid}: [{item['clause_category']}] {ratio:.0%} unknown words -- e.g. {sample_unknown}")
        print(f"      text: {item['gold_answer_span']['text'][:100]!r}")

    all_flagged_ids = set(dup_ids) | {cid for cid, _, _ in garbled}
    print(f"\nTotal unique items needing review: {len(all_flagged_ids)} / {len(candidates)}")

    flagged_items = [c for c in candidates if c["id"] in all_flagged_ids]
    with open("eval/eval_candidates_flagged.json", "w") as f:
        json.dump(flagged_items, f, indent=2)
    print(f"Wrote flagged items to eval/eval_candidates_flagged.json for focused review")


if __name__ == "__main__":
    main()