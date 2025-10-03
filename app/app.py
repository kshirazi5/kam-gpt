"""Streamlit entry point for the Kam-GPT conversational portfolio."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, List

from openai import OpenAI, AzureOpenAI
import streamlit as st


LOGGER = logging.getLogger(__name__)

LLM_BASE_SYSTEM_PROMPT = """
You are Kam-GPT, an AI guide to data analyst and applied data science graduate student
Kamran Shirazi. Use the conversation so far and the portfolio facts below to answer with
concise, friendly guidance that highlights Kamran's experience, values, and
availability.

Portfolio facts:
- Kamran has 3+ years of experience turning messy datasets into decision-ready insights
  across telematics, healthcare, e-commerce, and automotive programs.
- He currently leads analytics initiatives for Lytx telematics operations from San Diego,
  building dashboards, optimizing SQL pipelines, and automating monitoring for KPIs.
- Previous roles include analytics work at Integra LifeSciences, Cox Automotive, and
  Amazon where he improved reporting cadences, surfaced revenue opportunities, and led
  operations teams.
- Kamran is based in San Diego, California, and collaborates seamlessly with remote
  stakeholders across the U.S.
- He values transparent collaboration, psychological safety, and data-informed decision
  making, and enjoys storytelling that empowers cross-functional partners.
- You can reach Kamran at kamran@example.com or connect via LinkedIn for collaboration
  opportunities.
""".strip()


DEFAULT_OPENAI_MODEL = "gpt-4.1"


def _get_model_name() -> str:
    """Return the deployment/model name for OpenAI or Azure OpenAI."""

    return os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv(
        "OPENAI_MODEL", DEFAULT_OPENAI_MODEL
    )


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI | AzureOpenAI | None:
    """Return an OpenAI-compatible client when credentials are present."""

    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_api_key and azure_endpoint:
        return AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )

    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()


st.set_page_config(
    page_title="Kam-GPT",
    page_icon="🤖",
    layout="wide",
    menu_items={
        "Report a bug": "mailto:kamran@example.com",
        "About": "Kam-GPT is an AI guide to Kamran Shirazi's experience.",
    },
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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
            "I'm Kam-GPT, Kamran Shirazi's AI guide. I might not have a tailored answer "
            f"for “{cleaned_prompt}” yet, but you can ask about his experience, skills, "
            "favourite projects, collaboration style, values, or how to connect with him."
        )
    return (
        "I'm Kam-GPT, Kamran Shirazi's AI guide. Ask me about his experience, skills, "
        "tech stack, favourite projects, collaboration style, values, or how to connect."
    )


def _experience_response(_: str) -> str:
    return (
        "Kamran is a San Diego–based data analyst with 3+ years of experience translating "
        "messy datasets into decision-ready insights for telematics, healthcare, "
        "e-commerce, and automotive teams."
    )


def _focus_response(_: str) -> str:
    return (
        "His recent focus areas include business intelligence, SQL and Python automation, "
        "and scaling Snowflake and Redshift pipelines that keep stakeholders informed."
    )


def _location_response(_: str) -> str:
    return (
        "Kamran is based in San Diego, California, and partners with remote stakeholders "
        "across the United States."
    )


def _timeline_response(_: str) -> str:
    timeline = [
        "2024 – Leads Lytx telematics analytics to deliver executive-ready dashboards and "
        "faster KPI monitoring.",
        "2023 – Streamlined compliance and revenue reporting at Integra LifeSciences with "
        "Power BI and SQL automation.",
        "2022 – Surfaced $1.5M in incremental marketing ROI as a content performance "
        "analyst at Cox Automotive.",
        "2021 – Drove operations analytics and people leadership as an Amazon Area Manager.",
    ]
    return "\n".join(f"• {item}" for item in timeline)


def _contact_response(_: str) -> str:
    return (
        "You can reach Kamran at kamran@example.com or connect with him on LinkedIn to talk "
        "about collaboration opportunities."
    )


def _resume_response(_: str) -> str:
    return (
        "You can download Kamran's latest resume from the sidebar, which also includes a "
        "preview of the key highlights."
    )


def _linkedin_response(_: str) -> str:
    return (
        "There's a LinkedIn profile export available in the sidebar if you'd like to review "
        "Kamran's roles and accomplishments in more detail."
    )


def _projects_response(_: str) -> str:
    return (
        "He enjoys building data products that surface actionable insights — think fleet "
        "operations dashboards, compliance scorecards, and marketing performance analyses."
    )


def _acknowledgement_response(_: str) -> str:
    return (
        "Happy to help! Let me know what else you'd like to learn about Kamran's "
        "experience, projects, or how he collaborates."
    )


def _skills_response(_: str) -> str:
    return (
        "Kamran works across SQL, Python, and modern data tooling like dbt, Airflow, and "
        "Snowflake. He pairs visualization platforms such as Tableau, Power BI, QuickSight, "
        "and Looker Studio with reproducible analytics workflows."
    )


def _collaboration_response(_: str) -> str:
    return (
        "He believes in transparent collaboration — partnering closely with operations, "
        "product, and business leaders to ensure analytics work drives measurable impact. "
        "Expect frequent demos, async updates, and thoughtful documentation."
    )


def _availability_response(_: str) -> str:
    return (
        "Kamran is open to remote-friendly analytics roles, contract projects, and "
        "fractional engagements. He typically responds within a business day across U.S. "
        "time zones."
    )


def _values_response(_: str) -> str:
    return (
        "He values curiosity, psychological safety, and data-informed decision making. "
        "Teams that celebrate experimentation and learn from fast feedback loops are the "
        "ones he thrives in."
    )


def _education_response(_: str) -> str:
    return (
        "Kamran is pursuing an M.S. in Applied Data Science at the University of San Diego, "
        "completed the UC Berkeley Data Analytics Bootcamp, and advanced his undergraduate "
        "studies during the COVID era."
    )


def _impact_response(_: str) -> str:
    return (
        "Highlights include trimming KPI latency by ~40% at Lytx, surfacing $1.5M in "
        "incremental marketing ROI at Cox Automotive, and driving 20% defect reduction as "
        "an Amazon operations leader."
    )


KNOWLEDGE_BASE: List[KnowledgeEntry] = [
    KnowledgeEntry(
        ("experience", "background", "expertise", "profile", "yourself"),
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
    KnowledgeEntry(("san", "diego", "california", "where", "based", "location"), _location_response),
    KnowledgeEntry(("timeline", "history", "journey", "career"), _timeline_response),
    KnowledgeEntry(("contact", "email", "reach", "connect"), _contact_response),
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
    KnowledgeEntry(("skill", "skills", "tech", "stack", "tools", "technology"), _skills_response),
    KnowledgeEntry(("collaborate", "collaboration", "partner", "team", "communication"), _collaboration_response),
    KnowledgeEntry(("availability", "available", "open", "hire", "hiring", "contract"), _availability_response),
    KnowledgeEntry(("values", "culture", "principles", "approach"), _values_response),
    KnowledgeEntry(("education", "degree", "school", "university", "study", "learning"), _education_response),
    KnowledgeEntry(("impact", "results", "outcome", "wins", "success", "achievement"), _impact_response),
    KnowledgeEntry(("resume", "cv"), _resume_response),
    KnowledgeEntry(("linkedin", "profile"), _linkedin_response),
]


def _normalize_prompt(prompt: str) -> tuple[str, set[str]]:
    normalized_text = re.sub(r"[^a-z0-9\s]", " ", prompt.lower()).strip()
    tokens = {token for token in normalized_text.split() if token}
    return normalized_text, tokens


def _generate_rule_based_response(prompt: str) -> str:
    """Return a rule-based response that emulates a friendly chat assistant."""

    normalized_text, tokens = _normalize_prompt(prompt)
    matched_responses: list[str] = []

    for entry in KNOWLEDGE_BASE:
        if entry.matches(normalized_text, tokens):
            response = entry.response_builder(prompt)
            if response not in matched_responses:
                matched_responses.append(response)

    if matched_responses:
        return "\n\n".join(matched_responses)
    return _default_response(prompt)


def _coalesce_llm_text(response: object) -> str:
    """Extract the text content from an OpenAI responses payload."""

    text_parts: list[str] = []
    output = getattr(response, "output", [])

    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "text":
                text_value = getattr(getattr(content, "text", None), "value", "")
                if text_value:
                    text_parts.append(text_value)

    if not text_parts and hasattr(response, "output_text"):
        return str(response.output_text)
    return "\n".join(part.strip() for part in text_parts if part).strip()


def _generate_llm_response(prompt: str, history: list[dict[str, str]]) -> str:
    """Use GPT-4.1 to create a conversational response when configured."""

    client = _get_openai_client()
    if client is None:
        return _generate_rule_based_response(prompt)

    messages = [{"role": "system", "content": _get_system_prompt()}]
    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": prompt})

    try:
        response = client.responses.create(
            model=_get_model_name(),
            input=messages,
            temperature=0.3,
            max_output_tokens=600,
        )
    except Exception:  # pragma: no cover - network errors fall back gracefully
        LOGGER.exception("Falling back to rule-based response due to OpenAI failure")
        return _generate_rule_based_response(prompt)

    llm_text = _coalesce_llm_text(response)
    if not llm_text:
        return _generate_rule_based_response(prompt)
    return llm_text


def generate_response(prompt: str, history: list[dict[str, str]]) -> str:
    """Return the best available response for the current configuration."""

    if _get_openai_client() is None:
        return _generate_rule_based_response(prompt)
    return _generate_llm_response(prompt, history)


@lru_cache(maxsize=None)
def _load_document(filename: str) -> str:
    """Return the raw text for a document located in the data directory."""

    path = DATA_DIR / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        LOGGER.warning("Document %s could not be found", filename)
    except OSError:
        LOGGER.exception("Unable to read document %s", filename)
    return ""


def _strip_front_matter(content: str) -> str:
    """Remove YAML front matter from markdown documents for preview rendering."""

    if content.startswith("---"):
        end_marker = content.find("\n---", 3)
        if end_marker != -1:
            return content[end_marker + 4 :].lstrip("\n")
    return content


def _truncate_document(content: str, limit: int = 2400) -> str:
    """Return a trimmed version of ``content`` that respects the token budget."""

    if len(content) <= limit:
        return content

    trimmed = content[:limit]
    last_break = trimmed.rfind("\n")
    if last_break != -1 and last_break > limit - 400:
        trimmed = trimmed[:last_break]
    return trimmed.rstrip() + "\n…"


@lru_cache(maxsize=1)
def _get_system_prompt() -> str:
    """Combine the base system prompt with resume and LinkedIn context."""

    prompt_sections = [LLM_BASE_SYSTEM_PROMPT]

    resume = _strip_front_matter(_load_document("resume.md"))
    if resume:
        prompt_sections.append(
            "Resume excerpts (Markdown):\n" + _truncate_document(resume)
        )

    linkedin = _strip_front_matter(_load_document("linkedin.md"))
    if linkedin:
        prompt_sections.append(
            "LinkedIn profile excerpts (Markdown):\n" + _truncate_document(linkedin)
        )

    return "\n\n".join(section.strip() for section in prompt_sections if section.strip())


def _render_document_controls(filename: str, label: str, description: str) -> None:
    """Render download and preview controls for a markdown document."""

    content = _load_document(filename)
    if not content:
        st.caption(f"{label} is currently unavailable.")
        return

    st.markdown(f"**{label}:** {description}")
    st.download_button(
        label=f"Download {label.lower()} (Markdown)",
        data=content,
        file_name=f"kamran-shirazi-{filename}",
        mime="text/markdown",
    )
    with st.expander(f"Preview {label.lower()}"):
        st.markdown(_strip_front_matter(content))


def render_sidebar() -> None:
    """Render supporting information and controls in the sidebar."""

    with st.sidebar:
        st.header("Meet Kamran 👋")
        st.markdown(
            """
            **Role:** Data analyst & applied data science graduate student \\
            **Specialities:** Business intelligence, SQL/Python automation, cloud data warehousing \\
            **Based in:** San Diego, California
            """
        )
        st.caption("Curious what Kamran has worked on? Ask away in the chat!")

        if st.button("Reset conversation"):
            st.session_state.pop("messages", None)
            st.experimental_rerun()

        st.subheader("Documents")
        _render_document_controls(
            "resume.md",
            "Resume",
            "Download the latest copy or skim the highlights.",
        )
        _render_document_controls(
            "linkedin.md",
            "LinkedIn export",
            "Review Kamran's recent roles, education, and skills.",
        )


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
                    "skills, tech stack, collaboration style, or availability."
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

        response = generate_response(prompt, st.session_state.messages)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    """Application entry point."""

    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
