"""Load SQuAD v1.1 validation and turn it into a passage corpus + test questions."""
import random

from datasets import load_dataset

SEED = 42

# Questions the corpus can't answer. The first five are off-topic; the last five are
# "near misses" about articles that ARE in the corpus but ask for a fact that isn't.
# Each was checked with a grep over all 2,067 passages (see README).
UNANSWERABLE = [
    "Who directed the film Inception?",
    "How many moons does Jupiter have?",
    "What is the boiling point of mercury in degrees Celsius?",
    "Who wrote the novel One Hundred Years of Solitude?",
    "What programming language was created by Guido van Rossum?",
    "What was the final score of Super Bowl 51?",
    "How many Apollo astronauts were born in Ohio?",
    "What was Nikola Tesla's favourite food?",
    "What is the melting point of chlorophyll?",
    "How many students attend Fresno's largest high school?",
]


def load_corpus():
    """Return (passages, questions).

    passages: list of {"id", "title", "text"}, one per unique SQuAD context.
    questions: list of {"id", "question", "answers", "passage_id"}.
    """
    ds = load_dataset("rajpurkar/squad", split="validation")
    passages, pid_by_text, questions = [], {}, []
    for row in ds:
        text = row["context"]
        if text not in pid_by_text:
            pid_by_text[text] = len(passages)
            passages.append({"id": len(passages), "title": row["title"], "text": text})
        questions.append({
            "id": row["id"],
            "question": row["question"],
            "answers": sorted(set(row["answers"]["text"])),
            "passage_id": pid_by_text[text],
        })
    return passages, questions


def sample_test_set(questions, n=300, seed=SEED):
    return random.Random(seed).sample(questions, n)
