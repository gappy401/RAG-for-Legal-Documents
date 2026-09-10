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

We're also aware of [LegalBench-RAG](https://github.com/zeroentropy-ai/legalbenchrag),
a peer-reviewed benchmark built from CUAD + 3 other legal datasets with the same
character-offset ground truth methodology. Planned as a future validation step
once the pipeline is complete, to check results against an external, citable
benchmark rather than only our own hand-built set.

## Findings log

Numbers below are measured against the current 10-item eval set (7 positive,
3 negative cases across 4 contracts) unless noted otherwise. Config (chunk
sizes, etc.) lives in `config.yaml`.

### Chunking: parent-child hierarchical, child_size sweep

Tested whether gold answer spans stay intact within a single child chunk at
various sizes, using `RecursiveCharacterTextSplitter` (LangChain) for
boundary-aware splitting rather than naive fixed-character cuts.

| child_size | intact | split | total children |
|---|---|---|---|
| 300 | 4/7 | 3/7 | 895 |
| 400 | 5/7 | 2/7 | 698 |
| **600** | **6/7** | **1/7** | 494 |
| 800 | 5/7 | 2/7 | 389 |
| 1000 | 6/7 | 1/7 | 314 |

**Chose child_size=600.** Ties the best intact rate (6/7) at a smaller, cheaper
chunk size than 1000. The result is non-monotonic (800 is worse than both 600
and 1000) — `RecursiveCharacterTextSplitter` cuts at the nearest natural
paragraph/sentence boundary near the target size, not a fixed offset, so a
larger `chunk_size` doesn't guarantee a strictly better cut point for every span.

The one remaining split (`loha-supply-002`, a 661-character Warranty Duration
clause) is longer than any tested child_size below 800, and even at 800 the
splitter's chosen boundary didn't fully capture it — a known limitation of
this specific clause rather than something further size-tuning fixes cleanly.

All 7 spans that split at the child level were confirmed **fully intact at the
parent level** (2000-char parent chunks) — this is the actual justification for
parent-child retrieval: precise matching on small chunks, full context
preserved via the parent swap at generation time.

Switching from naive character-slicing to `RecursiveCharacterTextSplitter`
(boundary-aware) at the same child_size=400 improved intact spans from 3/7 to
5/7 with no other changes — a measurable win from boundary-aware splitting alone.

### Retrieval baseline: BM25 only

| Metric | Result |
|---|---|
| Recall@1 | 2/7 (29%) |
| Recall@3 | 5/7 (71%) |
| Recall@5 | 5/7 (71%) |

BM25 performs well on queries with exact lexical overlap with the clause text
(Governing Law questions using "law"/"state"/"govern" → rank 1), but fails hard
on paraphrased/indirect questions with no shared vocabulary:
- `limeenergy-002` ("how long before it needs renewing?" vs. clause text "ten
  years... commence...") → rank 50
- `metlife-remarketing-001` ("cap on liability?" vs. legal boilerplate about
  "contribute any amount in excess of...") → rank 99

This selective, explainable failure pattern is the direct motivation for adding
vector search.

### Retrieval baseline: vector search, 3 embedding models compared

| Model | R@1 | R@3 | R@5 |
|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 14% | 57% | 86% |
| **BAAI/bge-small-en-v1.5** | **57%** | 71% | 71% |
| nlpaueb/legal-bert-base-uncased | 14% | 14% | 57% |

**Confirmed hypothesis:** both general-purpose embedding models fixed the two
BM25 paraphrase failures — `metlife-remarketing-001` went from rank 99 (BM25)
to rank 1 (BGE) / rank 2 (MiniLM); `limeenergy-002` went from rank 50 to rank 2
(both models).

**Unexpected finding:** MiniLM regressed on cases BM25 handled trivially —
`limeenergy-001` (exact-term Governing Law match) dropped from BM25's rank 1 to
MiniLM's rank 20. BGE did not show this regression (stayed at rank 1). This is
the concrete evidence for hybrid retrieval: BM25 and embeddings fail on
*different* query types, so neither alone is sufficient.

**Legal-BERT underperformed both general-purpose models across the board**,
including missing `metlife-remarketing-001` entirely (not in top 50 candidates
searched — worse than BM25's rank 99). This is a deliberate, reported negative
result: legal-BERT is a masked-language-model checkpoint, not contrastively
trained for sentence similarity, so mean-pooling its hidden states does not
produce a strong retrieval embedding space out of the box. Domain-specific
pretraining did not outperform general-purpose retrieval-tuned models here.

**Chose BGE (`bge-small-en-v1.5`)** as the embedding model carried forward into
hybrid fusion — best R@1, and no regression on the easy exact-match case
MiniLM showed.

### Hybrid fusion (BM25 + BGE via Reciprocal Rank Fusion)

RRF combines two ranked lists using only rank position (not raw scores, which
aren't on comparable scales between BM25 and cosine similarity):
`score = 1/(k + rank_bm25) + 1/(k + rank_vector)`.

**First run, using RRF's standard default `k=60`, performed *worse* than
either method alone at R@3/R@5:**

| Method | R@1 | R@3 | R@5 |
|---|---|---|---|
| BM25 only | 29% | 71% | 71% |
| Vector (BGE) only | 57% | 71% | 71% |
| Hybrid (k=60) | 57% | **57%** | **57%** |

Root cause: `k=60` comes from the original RRF paper's benchmarks against
corpora of thousands of documents, where the gap between rank 1 and rank 20 is
small relative to the whole ranking. At this project's scale (494 chunks), a
rank-1-vs-rank-73 gap (`metlife-remarketing-001`, BM25 rank 73 → Vector rank 1)
is proportionally enormous, and `k=60` over-dampens that signal, letting
weaker-but-more-"average" chunks crowd out a genuinely strong single-method hit.

**Swept `k` across [1, 5, 10, 20, 40, 60] to find the right value for this
corpus size, rather than assuming the paper's default applies:**

| RRF k | R@1 | R@3 | R@5 |
|---|---|---|---|
| **1** | **71%** | **71%** | **86%** |
| **5** | **71%** | **71%** | **86%** |
| 10 | 57% | 71% | 86% |
| 20 | 57% | 57% | 71% |
| 40 | 57% | 57% | 71% |
| 60 | 57% | 57% | 57% |

**Chose k=5** (ties with k=1, but less extreme/brittle a value to generalize
from a 10-item eval set). At k=5, hybrid retrieval beats *every* individual
method at *every* cutoff: R@1 71% (vs. BM25's 29%, Vector's 57%), R@5 86%
(vs. both baselines' 71%).

**Takeaway for anyone tuning RRF elsewhere:** the standard `k=60` default is
not universal — it should be validated against your own corpus size and eval
set, not assumed. A small corpus needs a smaller `k` to avoid washing out
strong single-method signals.

## Status

🚧 Early stage — retrieval baselines and hybrid fusion (RRF, k=5) complete.
Next: cross-encoder reranking.