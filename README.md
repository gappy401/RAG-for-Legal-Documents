# Ask My Docs — Legal Contract RAG

A domain-specific retrieval-augmented QA system for legal/commercial contracts,
built with hybrid retrieval (BM25 + vector search), cross-encoder reranking,
citation enforcement, and a CI-gated evaluation pipeline.

## Why legal/contracts

Legal contract review is a high-stakes, high-volume domain where hallucination
carries real cost (a wrong answer about a termination clause or liability cap
isn't just annoying — it's a liability risk). That makes citation enforcement
a core requirement here, not an add-on.

## Dataset

Built on [CUAD](https://github.com/TheAtticusProject/cuad) (Contract Understanding
Atticus Dataset) — 510 real commercial contracts, expert-annotated across 41 clause
categories, with character-offset answer spans (SQuAD 2.0 style, including
unanswerable/negative cases).

## Eval set methodology

The eval set (`eval/eval_set.json`) is hand-built from CUAD, not scraped from
production usage — there is no live system yet, so there are no real user queries
to mine. This is a deliberate, standard first phase:

1. **Seed set (this phase):** natural-language questions hand-written by a human
   reviewing real contracts, using CUAD's expert-labeled spans as ground truth,
   with the CUAD team's own `category_descriptions.csv` as domain reference for
   what each clause category actually means.
2. Each item includes a positive or negative case — some questions map to a clause
   that genuinely exists in the contract (`expected_answer: "present"`), others to
   one that's confirmed absent (`expected_answer: "not_present"`) — since correctly
   declining to answer when nothing exists is as important as correct retrieval.

### Known limitation

This eval set reflects *hypothesized* user questions, not real usage. In a
production deployment, the standard next step is mining real queries — especially
ones that got negative feedback or required rephrasing — and folding validated
failures back into the eval set as permanent regression tests. That mining step
doesn't exist here because there's no live traffic yet; it's the natural next
phase once/if this is deployed, not an oversight.

## Status

🚧 Early stage — eval set under construction. See `eval/eval_set.json`.