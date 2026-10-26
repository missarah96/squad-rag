"""Answer a question from retrieved passages with Claude, and score the answer."""
import re
import string

import anthropic

MODEL = "claude-opus-5-5"
NOT_FOUND = "not in the documents"

SYSTEM = f"""You answer questions using only the numbered passages the user provides.

- Answer in as few words as possible (a name, date, number or short phrase), then cite the passage you used in square brackets, e.g. "Denver Broncos [2]".
- Use only information stated in the passages, not your own knowledge.
- If the passages do not contain the answer, reply with exactly: {NOT_FOUND}"""

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def format_passages(passages):
    return "\n\n".join(f"[{i}] ({p['title'].replace('_', ' ')})\n{p['text']}"
                       for i, p in enumerate(passages, start=1))


def answer(question, passages, effort="low"):
    """Return {"text", "stop_reason", "model"} for one question."""
    response = client().beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": effort},
        # If a safety classifier declines, re-run on Anthropic's recommended fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user",
                   "content": f"{format_passages(passages)}\n\nQuestion: {question}"}],
    )
    if response.stop_reason == "refusal":
        return {"text": "", "stop_reason": "refusal", "model": response.model}
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return {"text": text, "stop_reason": response.stop_reason, "model": response.model}


# --- scoring -------------------------------------------------------------------------

def normalize(s):
    """SQuAD-style normalization: lowercase, drop punctuation and articles, squash spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_gold(response_text, gold_answers):
    norm = normalize(response_text)
    return any(normalize(g) in norm for g in gold_answers if normalize(g))


def abstained(response_text):
    return NOT_FOUND in response_text.lower()


def cited(response_text):
    """Passage numbers cited as [n]."""
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", response_text)})
