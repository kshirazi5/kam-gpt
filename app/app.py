"""Streamlit entry point for the Kam-GPT conversational portfolio."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List

import streamlit as st


st.set_page_config(
    page_title="Kam-GPT",
    page_icon="🤖",
    layout="wide",
    menu_items={
        "Report a bug": "mailto:kamran@example.com",
        "About": "Kam-GPT is an AI guide to Kamran Shirazi's experience.",
    },
)


@dataclass
class KnowledgeEntry:
    """Represents a lightweight knowledge record for rule-based responses."""

    keywords: Iterable[str]
    response_builder: Callable[[str], str]
    match_type: str = "any"

    def matches(self, normalized_text: str, tokens: set[str]) -> bool:
        """Return ``True`` when the entry should handle the prompt."""

        def _keyword_in(keyword: str) -> bool:
            if " " in keyword:
                return keyword in normalized_text
            return keyword in tokens

        if self.match_type == "all":
            return all(_keyword_in(keyword) for keyword in self.keywords)
        return any(_keyword_in(keyword) for keyword in self.keywords)


def _default_response(prompt: str) -> str:
    cleaned_prompt = prompt.strip()
    if cleaned_prompt:
        return (
            "I'm Kam-GPT, Kamran Shirazi's AI guide. I might not have details on "
            f"“{cleaned_prompt}” yet. Try asking about his experience, skills, "
            "favourite projects, or how to get in touch."
        )
    return (
        "I'm Kam-GPT, Kamran Shirazi's AI guide. Ask me about his experience, "
        "skills, favourite projects, or how to get in touch and I'll share what I know."
    )


def _experience_response(_: str) -> str:
    return (
        "Kamran has over six years of experience designing scalable data and ML platforms. "
        "He has led initiatives that productionize machine learning models and build MLOps "
        "foundations for cross-functional teams."
    )


def _focus_response(_: str) -> str:
    return (
        "His recent focus areas include applied machine learning, MLOps automation, and "
        "shipping AI features end-to-end with product engineering partners."
    )


def _location_response(_: str) -> str:
    return (
        "Kamran works remotely from Toronto, Canada, and collaborates easily across time zones."
    )


def _timeline_response(_: str) -> str:
    timeline = [
        "2024 – Leads AI platform initiatives that unlock faster experimentation.",
        "2022 – Drove data-informed growth experiments at a high-growth startup.",
        "2019 – Scaled analytics tooling for fintech clients across North America.",
        "2016 – Graduated with a B.Sc. in Computer Science and entered data engineering.",
    ]
    return "\n".join(f"• {item}" for item in timeline)


def _contact_response(_: str) -> str:
    return (
        "You can reach Kamran at kamran@example.com or connect with him on LinkedIn to talk "
        "about collaboration opportunities."
    )


def _projects_response(_: str) -> str:
    return (
        "He enjoys building intelligent data products – think ML-powered user onboarding, "
        "recommendation systems, and analytics pipelines that keep stakeholders in the loop."
    )


def _acknowledgement_response(_: str) -> str:
    return (
        "Happy to help! Let me know what else you'd like to learn about Kamran's "
        "experience, projects, or how he collaborates."
    )


KNOWLEDGE_BASE: List[KnowledgeEntry] = [
    KnowledgeEntry(
        ("experience", "background", "expertise", "profile", "yourself", "resume", "cv"),
        _experience_response,
    ),
    KnowledgeEntry(
        (
            "focus",
            "specialty",
            "speciality",
            "specialities",
            "specialties",
            "interests",
            "mlops",
        ),
        _focus_response,
    ),
    KnowledgeEntry(("machine", "learning"), _focus_response, match_type="all"),
    KnowledgeEntry(("toronto", "canada", "where", "based", "location"), _location_response),
    KnowledgeEntry(("timeline", "history", "journey", "career", "resume"), _timeline_response),
    KnowledgeEntry(("contact", "email", "reach", "connect", "linkedin"), _contact_response),
    KnowledgeEntry(("project", "projects", "work", "built", "building"), _projects_response),
    KnowledgeEntry(
        (
            "ok",
            "okay",
            "cool",
            "great",
            "thanks",
            "thank",
            "awesome",
            "nice",
        ),
        _acknowledgement_response,
    ),
]


def _normalize_prompt(prompt: str) -> tuple[str, set[str]]:
    normalized_text = re.sub(r"[^a-z0-9\s]", " ", prompt.lower()).strip()
    tokens = {token for token in normalized_text.split() if token}
    return normalized_text, tokens


def generate_response(prompt: str) -> str:
    """Return a rule-based response that emulates a friendly chat assistant."""

    normalized_text, tokens = _normalize_prompt(prompt)
    for entry in KNOWLEDGE_BASE:
        if entry.matches(normalized_text, tokens):
            return entry.response_builder(prompt)
    return _default_response(prompt)


def render_sidebar() -> None:
    """Render supporting information and controls in the sidebar."""

    with st.sidebar:
        st.header("Meet Kamran 👋")
        st.markdown(
            """
            **Role:** Senior machine learning engineer \\
            **Specialities:** Data platforms, MLOps, applied AI \\
            **Based in:** Toronto, Canada
            """
        )
        st.caption("Curious what Kamran has worked on? Ask away in the chat!")

        if st.button("Reset conversation"):
            st.session_state.pop("messages", None)
            st.experimental_rerun()


def render_chat_interface() -> None:
    """Render the conversational UI backed by the lightweight knowledge base."""

    st.title("Kam-GPT")
    st.subheader("Your AI guide to Kamran Shirazi")
    st.write(
        "Use the chat below to learn about Kamran's experience, favourite projects, and how "
        "he collaborates on AI initiatives."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi there! I'm Kam-GPT. Ask me anything about Kamran Shirazi's background, "
                    "skills, or availability."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about Kamran's expertise"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = generate_response(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    """Application entry point."""

    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
