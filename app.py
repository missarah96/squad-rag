"""Streamlit demo: ask a question, see Claude's answer and the passages it cited.

    streamlit run app.py
"""
import streamlit as st

from rag import llm
from rag.data import load_corpus
from rag.retrievers import build_all


@st.cache_resource
def setup():
    passages, _ = load_corpus()
    return passages, build_all(passages)


passages, retrievers = setup()
st.title("SQuAD RAG")
st.caption(f"{len(passages):,} Wikipedia passages from 48 articles (SQuAD v1.1 validation)")

question = st.text_input("Question", placeholder="Which NFL team represented the AFC at Super Bowl 50?")
retriever = st.radio("Retriever", list(retrievers), index=2, horizontal=True)

if question:
    hits = retrievers[retriever].search(question, 3)
    top = [passages[pid] for pid, _ in hits]
    with st.spinner("Asking Claude…"):
        out = llm.answer(question, top)
    st.subheader("Answer")
    st.write(out["text"] or "_(declined by safety classifier)_")

    cited = llm.cited(out["text"])
    st.subheader("Retrieved passages")
    for i, p in enumerate(top, 1):
        label = f"[{i}] {p['title'].replace('_', ' ')}" + ("  ✅ cited" if i in cited else "")
        with st.expander(label, expanded=i in cited):
            st.write(p["text"])
