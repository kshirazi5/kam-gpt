"""Streamlit interface for the Kam-GPT experience generator."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kam_gpt import generate_engineer_response

st.set_page_config(page_title="Kam-GPT", page_icon="🤖")

st.title("Kam-GPT")
st.caption("An interactive portfolio for Kamran Shirazi")

with st.expander("How it works", expanded=False):
    st.markdown(
        textwrap.dedent(
            """
            Ask any question about Kamran's work and this app will surface the
            most relevant experience snippet along with suggested follow-up
            prompts. The underlying logic is lightweight and deterministic, so
            it's easy to deploy anywhere Streamlit runs.
            """
        ).strip()
    )

prompt = st.text_area(
    "What would you like to know?",
    placeholder="Tell me about Android reliability work",
    help="Questions are matched to curated experience stories using keyword "
    "signals.",
)

if prompt:
    response = generate_engineer_response(prompt)
    st.subheader("Answer")
    st.write(response["answer"])

    follow_ups = response.get("follow_up_questions")
    if follow_ups:
        st.subheader("Follow-up questions")
        st.markdown("\n".join(f"- {question}" for question in follow_ups))
else:
    st.info("Enter a question to explore Kamran's experience portfolio.")
