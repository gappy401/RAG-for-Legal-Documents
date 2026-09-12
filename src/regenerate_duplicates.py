"""
Fixes duplicate-phrasing items in eval_candidates.json by regenerating
just the duplicates (keeping the first occurrence of each phrase as-is),
feeding the LLM a growing list of already-used phrasings per category so
it can't just recreate the same duplicate problem.

Run flag_candidates.py first to see what's duplicated; this script acts
on eval_candidates.json directly (not the flagged subset file).
"""
import json
import sys
import time
from collections import defaultdict

import anthropic

sys.path.insert(0, "src")
from generate_eval_candidates import load_category_descriptions, build_prompt, MODEL


def main():
    with open("eval/eval_candidates.json") as f:
        candidates = json.load(f)
    with open("eval/eval_set.json") as f:
        existing_eval_set = json.load(f)

    category_descriptions = load_category_descriptions("data/cuad-repo/category_descriptions.csv")
    few_shot_examples = existing_eval_set[:4]

    # group indices by (category, question) to find duplicates
    by_phrase = defaultdict(list)
    for i, c in enumerate(candidates):
        by_phrase[(c["clause_category"], c["natural_question"])].append(i)

    # track phrasings already "claimed" per category (starts with the
    # kept first-occurrence of every duplicate group, plus every already
    # -unique question, since regenerated ones shouldn't collide with
    # ANY existing phrasing, not just other duplicates)
    used_phrasings_by_category = defaultdict(list)
    for c in candidates:
        used_phrasings_by_category[c["clause_category"]].append(c["natural_question"])

    client = anthropic.Anthropic()
    regenerated_count = 0

    for (category, question), indices in by_phrase.items():
        if len(indices) <= 1:
            continue  # not a duplicate, leave alone

        # keep the FIRST occurrence as-is, regenerate the rest
        for idx in indices[1:]:
            candidate = candidates[idx]
            category_desc = category_descriptions.get(category.lower(), "")

            item = {
                "category": category,
                "present": candidate["expected_answer"] == "present",
                "text": candidate["gold_answer_span"]["text"] if candidate["gold_answer_span"] else None,
            }

            avoid_list = list(set(used_phrasings_by_category[category]))
            prompt = build_prompt(item, category_desc, few_shot_examples, avoid_phrasings=avoid_list)

            response = client.messages.create(
                model=MODEL,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            new_question = response.content[0].text.strip()

            print(f"{candidate['id']}: {question!r}\n  -> {new_question!r}")

            candidate["natural_question"] = new_question
            candidate["_reviewed"] = False  # still needs human review, just like any generated item
            used_phrasings_by_category[category].append(new_question)  # so the NEXT regeneration avoids this one too
            regenerated_count += 1

            time.sleep(0.5)

    with open("eval/eval_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"\nRegenerated {regenerated_count} duplicate items in place.")
    print("Rerun flag_candidates.py to confirm duplicates are resolved, then continue manual review.")


if __name__ == "__main__":
    main()