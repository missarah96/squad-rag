"""Print wrong answers next to the gold answer and the retrieved passages, for writing failure notes.

    python show_failures.py          # 10 random wrong answers
    python show_failures.py -n 20
"""
import argparse
import json
import random

from rag.data import load_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    passages, _ = load_corpus()
    with open("results/answers_scored.jsonl") as f:
        wrong = [r for r in map(json.loads, f) if r["answerable"] and not r["correct"]]
    print(f"{len(wrong)} wrong answers; showing {min(args.n, len(wrong))}\n")

    for r in random.Random(args.seed).sample(wrong, min(args.n, len(wrong))):
        print("=" * 100)
        print(f"Q:        {r['question']}")
        print(f"Gold:     {r['answers']}")
        print(f"Response: {r['response']}")
        print(f"Gold passage in top 3: {r['gold_in_top3']}")
        for i, pid in enumerate(r["retrieved"], 1):
            mark = "  <-- gold" if pid == r["gold_passage"] else ""
            print(f"  [{i}] {passages[pid]['title']}{mark}: {passages[pid]['text'][:300]}...")
        if not r["gold_in_top3"]:
            print(f"  gold ({passages[r['gold_passage']]['title']}): {passages[r['gold_passage']]['text'][:300]}...")
        print()


if __name__ == "__main__":
    main()
