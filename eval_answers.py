"""Answer the 300 test questions + 10 unanswerable ones with Claude from the top-3 passages.

Responses are cached in results/answers.jsonl, so re-running only calls the API for
questions that don't have an answer yet. Writes the summary to results/answers.md.

    python eval_answers.py --limit 5     # cheap smoke test
    python eval_answers.py               # full run
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from rag import llm
from rag.data import UNANSWERABLE, load_corpus, sample_test_set
from rag.retrievers import build_all

OUT = Path("results")
CACHE = OUT / "answers.jsonl"
TOP_K = 3


def load_cache():
    if not CACHE.exists():
        return {}
    with open(CACHE) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--limit", type=int, help="only the first N answerable questions")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    passages, questions = load_corpus()
    test = sample_test_set(questions)[: args.limit]
    items = [{"id": q["id"], "question": q["question"], "answers": q["answers"],
              "gold_passage": q["passage_id"], "answerable": True} for q in test]
    items += [{"id": f"unanswerable-{i}", "question": q, "answers": [], "gold_passage": None,
               "answerable": False} for i, q in enumerate(UNANSWERABLE)]

    retriever = build_all(passages)[args.retriever]
    for item, hits in zip(items, retriever.search_many([i["question"] for i in items], TOP_K)):
        item["retrieved"] = [pid for pid, _ in hits]

    cache = load_cache()
    todo = [i for i in items if i["id"] not in cache]
    print(f"{len(items)} questions, {len(todo)} need an API call ({llm.MODEL}, effort={args.effort})")

    def run(item):
        out = llm.answer(item["question"], [passages[p] for p in item["retrieved"]], args.effort)
        return {**item, "response": out["text"], "stop_reason": out["stop_reason"], "model": out["model"]}

    OUT.mkdir(exist_ok=True)
    with open(CACHE, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run, i): i for i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except anthropic.AuthenticationError:
                raise SystemExit("Authentication failed: set ANTHROPIC_API_KEY (see README).")
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                print(f"  skipped {futures[fut]['id']}: {e}")
                continue
            cache[rec["id"]] = rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")

    report(items, cache, args.retriever)


def report(items, cache, retriever_name):
    recs = [cache[i["id"]] | {"retrieved": i["retrieved"]} for i in items if i["id"] in cache]
    ans = [r for r in recs if r["answerable"]]
    unans = [r for r in recs if not r["answerable"]]
    for r in ans:
        r["correct"] = llm.contains_gold(r["response"], r["answers"])
        r["gold_in_top3"] = r["gold_passage"] in r["retrieved"]
        gold_num = r["retrieved"].index(r["gold_passage"]) + 1 if r["gold_in_top3"] else None
        r["cited_gold"] = gold_num in llm.cited(r["response"])

    def pct(xs):
        return f"{sum(xs) / len(xs):.1%} ({sum(xs)}/{len(xs)})" if xs else "n/a"

    hit = [r for r in ans if r["gold_in_top3"]]
    miss = [r for r in ans if not r["gold_in_top3"]]
    lines = [
        f"Answers from `{llm.MODEL}` using the top {TOP_K} `{retriever_name}` passages.\n",
        "| Metric | Result |", "|---|---|",
        f"| Answer contains gold text (all answerable) | **{pct([r['correct'] for r in ans])}** |",
        f"| … when gold passage was in top 3 | {pct([r['correct'] for r in hit])} |",
        f"| … when gold passage was not in top 3 | {pct([r['correct'] for r in miss])} |",
        f"| Cited the gold passage (when in top 3) | {pct([r['cited_gold'] for r in hit])} |",
        f"| Said \"not in the documents\" on an answerable question | {pct([llm.abstained(r['response']) for r in ans])} |",
        f"| **Correctly said \"not in the documents\" (unanswerable)** | **{pct([llm.abstained(r['response']) for r in unans])}** |",
        f"| Refusals (safety classifier) | {sum(r['stop_reason'] == 'refusal' for r in recs)} |",
    ]
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    (OUT / "answers.md").write_text(text)
    with open(OUT / "answers_scored.jsonl", "w") as f:
        for r in ans + unans:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
