"""Compare dense, BM25 and hybrid retrieval on 300 SQuAD questions.

Writes results/retrieval.md (the comparison table) and
results/retrieval_per_question.jsonl (top-5 ids per question, for failure analysis).
"""
import json
from pathlib import Path

from rag.data import load_corpus, sample_test_set
from rag.retrievers import build_all

KS = (1, 3, 5)
OUT = Path("results")


def main():
    passages, questions = load_corpus()
    test = sample_test_set(questions)
    print(f"{len(passages)} passages, {len(test)} test questions")

    retrievers = build_all(passages)
    queries = [q["question"] for q in test]
    rows, per_question = [], {q["id"]: {"question": q["question"], "gold": q["passage_id"]} for q in test}

    for name, retriever in retrievers.items():
        results = retriever.search_many(queries, max(KS))
        ranks = []
        for q, hits in zip(test, results):
            ids = [pid for pid, _ in hits]
            per_question[q["id"]][name] = ids
            ranks.append(ids.index(q["passage_id"]) + 1 if q["passage_id"] in ids else None)
        row = {"retriever": name}
        for k in KS:
            row[f"hit@{k}"] = sum(r is not None and r <= k for r in ranks) / len(ranks)
        row["mrr@5"] = sum(1 / r for r in ranks if r) / len(ranks)
        rows.append(row)

    header = "| Retriever | Hit@1 | **Hit@3** | Hit@5 | MRR@5 |\n|---|---|---|---|---|\n"
    table = header + "".join(
        f"| {r['retriever']} | {r['hit@1']:.1%} | **{r['hit@3']:.1%}** | {r['hit@5']:.1%} | {r['mrr@5']:.3f} |\n"
        for r in rows)
    print("\n" + table)

    OUT.mkdir(exist_ok=True)
    (OUT / "retrieval.md").write_text(
        f"Retrieval over {len(passages)} SQuAD v1.1 validation passages, "
        f"{len(test)} random questions (seed 42).\n\n{table}")
    with open(OUT / "retrieval_per_question.jsonl", "w") as f:
        for qid, rec in per_question.items():
            f.write(json.dumps({"id": qid, **rec}) + "\n")


if __name__ == "__main__":
    main()
