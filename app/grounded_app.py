import streamlit as st
from app.retrieval import search
from pathlib import Path

st.set_page_config(page_title="Kam-GPT (Grounded)")
st.title("Kam-GPT (Grounded)")
st.write("Answers grounded strictly in /data (resume, LinkedIn, portfolio).")

policy = ""
if Path("prompts/answer_policy.md").exists():
    policy = Path("prompts/answer_policy.md").read_text()

q = st.text_input("Ask about your background:")
if st.button("Search") and q.strip():
    hits = search(q, k=5, index_dir="index")
    if not hits:
        st.warning("I don’t find that in my portfolio data. Add details to /data and rebuild index.")
    else:
        st.subheader("Answer (grounded)")
        st.write(hits[0]["text"][:800])
        st.subheader("Evidence")
        for h in hits: st.write(f"- {h['path']}")
        if policy:
            with st.expander("Answer Policy"):
                st.code(policy)
