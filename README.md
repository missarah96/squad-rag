# SQuAD RAG: comparing dense, BM25 and hybrid retrieval

A small retrieval-augmented question-answering system over Wikipedia passages, evaluated
end to end against SQuAD's answer keys, so no hand-written test set is needed.

## Problem

A RAG system can fail in two places: the retriever doesn't find the right passage, or the
LLM gets the right passage and still answers badly, or invents an answer when the passages don't
contain one. This project measures each stage separately:

1. **Retrieval:** how often is the passage that contains the answer in the top 3?
2. **Answering:** given the top 3, does the LLM produce the gold answer, cite the right
   passage, and say "not in the documents" when it should?

## Data

[SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad) validation split (CC BY-SA 4.0):
10,570 crowd-written questions over **2,067 unique Wikipedia paragraphs** from 48 articles.

- **Documents:** the 2,067 paragraphs. They're short (~120 words), so they aren't split further.
- **Test set:** 300 questions sampled at random (seed 42). Each question's gold passage is
  the paragraph it was written about, and its gold answers are SQuAD's answer spans.
- **Unanswerable set:** 10 hand-written questions ([rag/data.py](rag/data.py)), 5 off-topic
  ("How many moons does Jupiter have?") and 5 near misses on topics the corpus does cover
  ("What was the final score of Super Bowl 51?"). Each was checked by searching all
  2,067 passages for its key terms.

## Method

| Stage | Implementation |
|---|---|
| Dense | `all-MiniLM-L6-v2` embeddings, normalized, FAISS `IndexFlatIP` (cosine) |
| BM25 | `rank-bm25` Okapi over lowercased word tokens |
| Hybrid | Reciprocal Rank Fusion of the dense and BM25 top 50: `score = Σ 1/(60 + rank)` |
| Answering | Top 3 passages, numbered, sent to `claude-opus-5-5` (effort `low`) with instructions to answer only from them, cite `[n]`, or reply "not in the documents" |
| Scoring | SQuAD normalization (lowercase, strip punctuation and articles); an answer is correct if any gold answer appears in the response |

## Results

### Retrieval (300 questions)

| Retriever | Hit@1 | **Hit@3** | Hit@5 | MRR@5 |
|---|---|---|---|---|
| Dense (MiniLM + FAISS) | 63.7% | **79.0%** | 85.7% | 0.720 |
| BM25 | 75.0% | **86.3%** | 90.0% | 0.811 |
| Hybrid (RRF) | 70.0% | **87.0%** | 92.7% | 0.791 |

- **BM25 beats dense by 7 points at Hit@3.** SQuAD annotators wrote questions while looking
  at the paragraph, so questions reuse the passage's exact words ("Obersee", "Ted Ginn Jr.").
  Keyword matching rewards that. A small general-purpose embedding model blurs it.
- **Hybrid is best at Hit@3 and Hit@5 but worse than BM25 at Hit@1.** The two retrievers
  miss different questions (dense misses 46 that BM25 gets, BM25 misses 24 that dense gets),
  so fusing them recovers more passages within a few ranks. But RRF gives the weaker dense
  ranking equal weight, which pushes some BM25 rank-1 hits down to rank 2 or 3. Hybrid is
  used for the answer stage because the LLM sees the top 3.

### Answering (hybrid top 3)

_Fill in from `results/answers.md` after running `python eval_answers.py`._

| Metric | Result |
|---|---|
| Answer contains gold text (300 answerable) | |
| … when gold passage was in top 3 | |
| … when gold passage was not in top 3 | |
| Cited the gold passage (when in top 3) | |
| Correctly said "not in the documents" (10 unanswerable) | |

## Failure notes

### Retrieval misses (39 of 300 for hybrid)

1. **The right article, the wrong paragraph.** In 30 of 39 misses, the top result comes from the
   same Wikipedia article as the gold passage. Neighbouring paragraphs share the article's
   vocabulary, and the question doesn't say which section it means.
2. **Questions that assume the reader has the paragraph.** SQuAD questions were written with
   the passage in view, so many are underspecified on their own: "Where was Tesla's property
   sent?" (the passage says *estate*, not *property*), "What kind of bureaucracy did the
   Normans institute?" Different wording and missing context hurt both retrievers.
3. **Some misses aren't really misses.** In 7 of 39 cases, a gold answer string appears in a
   passage that *was* retrieved, e.g. Tesla's birth year is in two paragraphs. A
   single-gold-passage metric understates retrieval quality, and the answer stage can still
   get these right.

### Wrong answers

_Run `python show_failures.py` and write three lines on why they fail, for example the gold
passage wasn't retrieved, the answer was correct but worded differently from the gold span, or
the model abstained even though the answer was present._

## How to run

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate   # or python -m venv .venv
uv pip install -r requirements.txt                          # or pip install -r requirements.txt

python eval_retrieval.py                  # retrieval table -> results/retrieval.md (no API key needed)

export ANTHROPIC_API_KEY=sk-ant-...       # from https://console.anthropic.com
python eval_answers.py --limit 5          # smoke test: 5 questions + the 10 unanswerable
python eval_answers.py                    # full run -> results/answers.md (cached; reruns are free)
python show_failures.py                   # print 10 wrong answers with their passages

streamlit run app.py                      # optional demo UI with cited passages
```

The first run downloads SQuAD (~35 MB) and the MiniLM model (~90 MB). Passage embeddings are
cached in `cache/`.

## Layout

```
rag/data.py          load SQuAD, build passage corpus, sample test set, unanswerable questions
rag/retrievers.py    DenseRetriever, BM25Retriever, HybridRetriever (RRF)
rag/llm.py           Claude prompt + answer scoring
eval_retrieval.py    retrieval comparison
eval_answers.py      end-to-end answer evaluation
show_failures.py     inspect wrong answers
app.py               Streamlit demo
```
