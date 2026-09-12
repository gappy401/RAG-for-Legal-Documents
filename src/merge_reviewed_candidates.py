"""
Merges only human-reviewed candidates (_reviewed: true) from
eval_candidates.json into eval_set.json. Anything still marked false is
left behind -- this script will never promote an unreviewed item into
the trusted eval set, even accidentally.
"""
import json

with open("eval/eval_set.json") as f:
    eval_set = json.load(f)
with open("eval/eval_candidates.json") as f:
    candidates = json.load(f)

reviewed = [c for c in candidates if c.get("_reviewed") is True]
still_unreviewed = [c for c in candidates if c.get("_reviewed") is not True]

print(f"Reviewed and ready to merge: {len(reviewed)}")
print(f"Still unreviewed (left behind): {len(still_unreviewed)}")

# strip the _reviewed field before merging -- it's provenance for the
# review process, not part of the eval_set.json schema itself
for c in reviewed:
    c.pop("_reviewed", None)

eval_set.extend(reviewed)

with open("eval/eval_set.json", "w") as f:
    json.dump(eval_set, f, indent=2)

# keep only the not-yet-reviewed ones in the candidates file, so you can
# pick up where you left off next time instead of re-reviewing everything
with open("eval/eval_candidates.json", "w") as f:
    json.dump(still_unreviewed, f, indent=2)

print(f"\neval_set.json now has {len(eval_set)} total items.")
print(f"eval_candidates.json now has {len(still_unreviewed)} items left to review.")